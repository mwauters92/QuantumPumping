#------------------------------------------------------------------------------
# open_jja_qobs.py
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# import necessary libraries
#------------------------------------------------------------------------------

import sys
sys.path.append('../')

import numpy as np 
from qutip import Qobj, mesolve, expect, propagator, floquet_modes, ket2dm  
from modules import * 
from types import *
from scipy.integrate import simpson

#------------------------------------------------------------------------------
# qobs class
#------------------------------------------------------------------------------

class Qdynamics():

    def __init__(self, model):
        
        model_pars = model.__dict__
        self.L = model_pars['system_size']
        self.omega = model_pars['omega']
        self.n_t = model_pars['n_t']
        self.dt = model_pars['dt']
        self.period = model_pars['period']
        self.n_periods = model_pars['n_p']
        self.t = model_pars['time_domain'] 
        self.model_pars = model_pars

        # time-dependent Hamiltonian - should be qutip.Qobj 
        self.H = model.hamiltonian

        # initial state of the system - should be qutip.Qobj 
        self.rho0 = model.initial_state()

        # dictionary of time-dependent operators
        self.ops_dict = model.get_ops()


    def instantaneous_energies(self):
        """

        Obtain the instantaneous energies as a function of time. 
        Valid results only when we are in the adiabatic regime.

        Returns:
        --------
        res: dict
            a dictionary with the time domain and the instantaneous energies

        """

        n_en = 2 ** self.L
        energies = np.zeros( (self.n_t * self.n_periods, n_en) )
        for i, t_i in enumerate( np.nditer( self.t ) ): 
            ham = self.H(t_i) 
            energies[i] = np.real( ham.eigenenergies() )

        energies = np.asarray( energies )
        self.inst_energies = energies 

        res = {'time': self.t, 'inst_energies': energies}
        return res 


    def find_gap(self, ni=0, nf=1):
        """

        Evaluates the minimum many-body gap, given the indices of the states
        involved.   

        Arguments:
        ----------
        ni : int    (default: 0)
            the index of the lower-energy state 
        nf : int    (default: 1)
            the index of the higher-energy state 

        Returns:
        --------
        eg : float
            the numerical value of the gap

        """
        
        error_msg = 'Instantaneous energies were not calculated.'
        assert "inst_energies" in self.__dict__ , error_msg 

        ie = self.inst_energies 
        eg = np.min( ie[:,nf] - ie[:, ni] )
        return eg


    def instdiag_withN(self, n_states):
        """
        
        Calculates the particle number for a given number of instantaneous 
        eigenstates of the many-body Hamiltonian. The particle number is
        given as a function of time and number of eigenstate. 

        Also, it evaluates the fluctuations of the particle number.
            \delta N = \sqrt{ <N^2> - <N>^2 } 

        Arguments:
        ---------
        n_states: int
            number of states to examine 

        Returns:
        --------
        res: dict
            dictionary of results

        """

        occ_ops = self.ops_dict['occupation']

        dN_t_n, N_t_nj = [], []
        for k, t_k in enumerate( np.nditer( self.t ) ): 

            # instantaneous diagonalization
            ham = self.H(t_k)
            _, states = ham.eigenstates()
         
            # occupation numbers 
            N_n = [] 
            for n_i in range(n_states):

                expval = [expect( op_i(0.0), states[n_i]) for op_i in occ_ops]
                N_n.append( expval )

            N_t_nj.append( N_n )
            
            # particle number fluctuations  
            N_op = Qobj()
            for op_i in occ_ops:
                N_op = N_op + op_i(0.0)
 
            dN_n = []
            for n_i in range(n_states):

                N2_expval = expect( N_op * N_op, states[n_i] )
                N_expval = expect( N_op, states[n_i] ) 
                dN_expval = N2_expval - N_expval**2.0
                dN_n.append( np.sqrt( dN_expval ) )
            
            dN_t_n.append( dN_n )

        # turn lists into np.array
        N_t_nj = np.array( N_t_nj ) 
        dN_t_n = np.array( dN_t_n )

        # collect results
        N_t_nj = np.transpose( N_t_nj, (2,0,1) )
        res = {'time': self.t, 'occ_tj': N_t_nj, 'n_states': n_states,
               'n_t': np.sum(N_t_nj, axis=0), 'dn_t': dN_t_n} 
               
        return res  



    def expect_t(self, op, t, rho_t):
        """

        Calculates the expectation value of the given operator at time t, 
        given the state of the system at that time moment.

        Arguments:
        ----------
        op: callable 
            operator 
        t: float
            time moment
        rho_t: qutip.Qobj
            the state of the system at time t (in the density matri formulation)

        Returns:
        --------
        expval: float 
            the expectation value at time t

        """

        op_t = op(t) 
        expval = expect( op_t, rho_t )
        return expval
  

    def for_expect(self, results):
        """

        Evaluation of the time evolution of the expectation values of 
        given operators. 

        Arguments:
        ----------
        results: qutip.Results 
            results of solving the von Neumann equation

        Returns:
        --------
        expval: dict
            the time series of the expectation values  

        """

        op_dict = self.ops_dict
        states = results.states
        t = self.t

        expval = {}
        for name in op_dict.keys():
            expval[name] = np.zeros( (len(op_dict[name]), t.shape[0]) )
        
        for name in op_dict.keys():
            for i, op_i in enumerate( op_dict[name] ):
                # this can be optimized with einsum !! (future update)
                for j, t_j in enumerate(self.t):
                    expval[name][i][j] = self.expect_t(op_i, t_j, states[j]) 
        
        return expval


    def evaluate_int_obs(self, op_expect, dpars=None, 
                         time_evolution=False, q_per_period=False):
        """

        Status: The documentation needs fixing.

        This method is used for integrating an observable over time - applied
        for periodic (in time) systems. 

            
        Arguments:
        ----------
        op_expect: list 
            the time-evolved expectation value of some operator
        dpars: dict  (default: None)
            given parameters that characterize the evolution (will use
            the default ones if not provided)
        time_evolution: bool     (default: False) 
            evaluate the time evolution of the integral of the expectation 
            value 
        all_time : bool     (default: True) 
            evaluates Q = \int_0^{nT} dt' <O>(t'), if True, else: 
            it evaluates Q = 1/T  \int_{n}^{(n+1)T} dt' <O>(t') 

        Returns:
        --------
        t: np.array 
            time domain  if all_time == True 
        npt: np.array 
            time domain in number of periods if all_time == False 

        Q: np.ndarray (when time_evolution == True) or float  (default: False)
            the product of the integration of the expectation value
        
        """
        
        if not isinstance(op_expect, np.ndarray): 
            op_expect = np.asarray( op_expect )

        if dpars == None:   

            # here, n_t is the time slices per period
            n_t, n_periods, dt = self.n_t, self.n_periods, self.dt 
            n_t = n_t * n_periods
        else:

            # here, n_t is the total number of time slices 
            n_t = dpars['n_t_total'] 
            n_periods = dpars['n_periods']
            t = np.linspace( dpars['ti'], dpars['tf'], n_t)
            dt = ( dpars['tf'] - dpars['ti'] ) / n_t
            ni, nf = dpars['ni'], dpars['nf']
        

        if time_evolution:

            op_expect_to_int = np.vstack( [op_expect]*n_t )
            A = np.ones( (n_t, n_t) )
            mask = np.tril( A )
            op_expect_to_int = op_expect_to_int * mask
            Q = trapezoidal( op_expect_to_int, dt )
            return t, Q 

        else:
            
            if q_per_period:   # evaluates q = 1/T int_n^{n+1}T <O>(t)
                
                q_n = []
                n_t_per_period = n_t // n_periods
                for n_i in range(n_periods):

                    n_o = n_i*n_t_per_period
                    op_expect_cut = op_expect[n_o:n_o+n_t_per_period]
                    Q = trapezoidal(op_expect_cut, dt)
                    q_n.append( Q )

                npt = ni + np.array([j for j in range(n_periods)])
                q_n = np.array(q_n) 

                return npt, q_n 

            else:
                intg = trapezoidal(op_expect, dt)
                Q = intg / self.n_periods 
                return Q

    
    def evaluate_int_obs_multiple(self, eop_ls, dpars=None, 
                                  eval_type='per period'):
        """ 

        Status: Under development - not tested 

        Evaluates the integral over time of an expectation value of an 
        operator (can be used for multiple operators).

        Arguments:
        ----------
        eop_ls : list
            a list of expectaction values of operators
        dpars 
        eval_type : str     (default: 'per period')
            options: 'per period', 'all time' 

        Returns:
        --------
        int_ls : list
            a list of the time average of the given expectaction values

        """

        if eval_type == 'per period':
            evol = False
            q_per_period = True
        elif eval_type == 'all time':
            evol = True
            q_per_period = None 
        elif eval_type == 'single q':
            evol = False
            q_per_period = False

        int_ls = []
        for ej in eop_ls:
            t, q = self.evaluate_int_obs(ej, dpars=dpars, time_evolution=evol,
                                         q_per_period=q_per_period)
            int_ls.append( q )
        
        return t, int_ls



    def create_floquet(self, return_floquet_elements=False):
        """

        Status: Complete.

        Find the Floquet operator, the floquet states and the corresponding
        quasienergies. These can be used to evaluate any observable (
        see apply_floquet).

        Arguments:
        ----------
        return_floquet_elements: bool   (default: False)
            returns the quantities, essential for floquet theory, if True

        Returns:
        --------
        floquet_elements: dict
            a dictionary containing the essential floquet quantities, returned
            if return_floquet_elements is True

        """
        

        # load variables from class dictionary (the following are for one period)
        n_t = self.n_t
        tau = self.period
        time = np.linspace( 0.0, tau, n_t) 
    
        # generate the time evolution operator within a period
        h = lambda t, args={}: self.H(t)
        U = propagator(h, time) #, progress_bar=True) 
        
        F_op = Qobj( U[-1] )
        
        #print(U)
        f_states, self.f_energies = floquet_modes(h, tau, U=F_op)

        #states_ar = np.asarray( [np.asarray(f_s.data.to_array()) for f_s in f_states] )
        # construct U (columns are the eigenstates of Ham - in order of 
        # increasing energy)
        #print(states_ar)
        #UF = Qobj( np.squeeze(states_ar).T )
        #UF = Qobj( np.identity( 16) )
        
        self.U, self.UF, self.F_op = np.array(U), np.array(f_states) , F_op
        
        if return_floquet_elements:
            floquet_elements = {'U': np.array(U), 'f_states': f_states, 
                                'f_energies': f_energies, 
                                'F_op': F_op, 'time': time}
            return floquet_elements


    def load_floquet(self, floquet_elements):
        """

        Status: Complete.

        Loads the Floquet elements that were previously calculated. 
        The floquet elements are assumed to have been saved from last time.


        Arguments:
        ----------
        floquet_elements : dict 

        """

        self.U = floquet_elements['U']
        self.UF = floquet_elements['UF']
        self.F_op = floquet_elements['F_op']


    
    def stroboscopic_floquet(self, ni, nf, dn=1):
        """
    
        Newly created function - UNDER DEVELOPMENT

        Evaluates observables at stroboscopic times. 

        Arguments:
        ----------
        dn : int    (default: 1)
            step between adjacent periods  
         

        """
        
        assert nf > ni, 'nf must be larger than ni'
        assert (nf-ni)%dn == 0, 'nf-ni not divisible by dn'

        # load from class dictionary 
        tau = self.period     
 
        dN = nf - ni    # total periods within the interval
        Ntotal = int( dN / dn ) + 1    # total evaluations at stroboscopic times 
        np_ls = np.linspace( ni, nf, Ntotal, dtype=int)
        
        # save parameters that characterize dynamics
        eval_pars = {'period': tau, 'ni': ni, 'nf': nf, 'dn': dn, 
                     'Ntotal': Ntotal}

        # load floquet elements from class dictionary 
        F_op = self.F_op 
        Fshape = F_op.shape[0]  # since it is a square matrix
        
        # load initial state
        rho0 = self.rho0
        rho0_ext = np.array( [rho_f]*Ntotal ) 

        # create a stack of the ith powers of the floquet operator
        temp_ls = [] 
        Fdn = np.linalg.matrix_power( F_op, dn )
        for i in range(ni, nf+1, dn):
            
            if i == ni:
                Fn = F_op
            else:
                Fn = np.dot( Fn, Fdn )
            temp_ls = temp_ls + [Fn] 
        
        # create U(n\tau)
        F_op_ext = np.array( temp_ls ) 
        F_op_ext_dag = np.conjugate( np.transpose( F_op_ext, (0,2,1) ) ) 
        
        # load operators 
        ops_dict = self.ops_dict 

        expval_stb = {nm: [] for nm in ops_dict.keys()}
        for nm in ops_dict.keys():
            for j, op_j in enumerate(ops_dict[nm]):
            
                op_stb = np.array( [op_j( (n_i+1)*tau ) for n_i in np_ls] )

                pA = np.einsum( 'kij,kjm->kim', op_stb, F_op_ext)
                pB = np.einsum( 'kij,kjm->kim', F_op_ext_dag, pA)
                pC = np.einsum( 'kij,kjm->kim', rho0_ext, pB)

                expval_stb[nm].append( np.real( np.einsum( 'kii->k', pC ) ) )

        expval_stb['np_ls'] = np_ls  
        expval_stb['eval_pars'] = eval_pars
 
        return expval_stb


    def q_floquet(self):
        """

        Status: It works but the documentation needs work.

        This function evaluates the pumped charge over infinite number of
        periods. This is achieved when we work in the Floquet basis. 

        Returns:
        --------
        q_inf : list
            list of the (infinite period) averaged current - charge - for 
            each link of the chain

        """
        
        # load variables from the class dictionary  
        n_t = self.n_t      # time steps per period
        tau = self.period    
        Nh=self.UF.size
        ti, tf = 0.0, tau 
        time_evolve = np.linspace( ti, tf, n_t) 

        # save parameters that characterize dynamics
        eval_pars = {'n_t_total': n_t, 'n_periods': 1, 
                     'period': tau, 'ti': ti, 'tf': tf, 'ni': 0, 'nf': 1} 

        #overlap between Floquet states and initial state
        rho0_f = self.floquet_projection()
        #rho0_f = np.array([np.abs(self.rho0.overlap( UF_s )) for UF_s in self.UF])
        
        # FSstate_t is a (n_t,Nh) array that contains the evolved Floquet states
        FState_t = np.tensordot(self.U.reshape([n_t,1]),self.UF.reshape([1,Nh]),axes=[1,0])
        
        op_j = self.ops_dict['current']
                   
        # compute the expectation value of the current on each Floquet state at each time step
        op_t = np.array([expect(op_j,state_f) for state_f in FState_t]).reshape(FState_t.shape)

        # sum over floqeut states weighted by their occupation number
        op_t_weighted = np.tensordot(rho0_f, op_t, axes=[0,1])
        
        #q_inf.append( np.real(q)[0] )
        q_inf= simpson(op_t_weighted,x=time_evolve)
        return q_inf 



    def floquet_projection(self):
        """

        Status: It works - but needs generalization
        
        Projection of the initial state to the floquet basis. 

        """

        if self.rho0.type == 'ket':
            rho0_f = np.array([np.abs(self.rho0.overlap( UF_s )) for UF_s in self.UF])
            f_occ = rho0_f**2 

        else:
            UF = np.squeeze( np.array( [uf_i.full() for uf_i in self.UF ] ) )
            UF_dag = np.squeeze( np.conjugate( np.transpose( UF) ) ) #UF.dag()
            rho0 = self.rho0 
        
            f_occ = np.squeeze( np.diagonal( np.dot( UF_dag, np.dot( rho0, UF)) ) )
            f_occ = np.real( f_occ )

        return f_occ

#------------------------------------------------------------------------------
# end of qobs class
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
