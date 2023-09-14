#------------------------------------------------------------------------------
# open_jja_class.py
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# import necessary libraries
#------------------------------------------------------------------------------

import numpy as np 
from qutip import Qobj, sigmap, sigmam, sigmaz, identity, tensor, ket2dm, expect  

#------------------------------------------------------------------------------
# open_jja class 
#------------------------------------------------------------------------------

class RM_model: 

    def __init__(self):
        """
        
        Class initialization.
        A class for the JJAs system with S.C. leads (open boundary terms
        - and closed as well)

        """

        # set a seed for random number generator (default one) 
        self.set_seed()

        # set values to the lattice parameters (default ones)
        self.set_lattice_pars()

        # set values to the dynamic parameters (default ones)
        self.set_dynamic_pars(omega=1.0)
        
        # initialize disorder (default: no disorder)
        self.random_disorder()

        # initilize the time-dependent parameters of the Hamiltonian
        pars = {'E0': 5.0, 'dE': 2.0, 'D': 5.0, 
                'e0_leads': 1.0, 'dE_leads': 0.0, 'ec0': 0.0, 'ecx': 0.0,
                'phi_L': 0.0, 'phi_R': np.pi/3.0}
        self.initialize_Htpars(pars) 

    
    def set_lattice_pars(self, n_cells=5, bc='open'):
        """
        
        Sets values to the parameters related to the lattice construction.

        Arguments:
        ----------
        n_cells: int    (default: 5)
            number of unit cells
        bc: str     (default: open) 
            type of boundary conditions 

        """        

        self.n_cells = n_cells 
        self.system_size = 2 * n_cells 
        self.bc = bc 


    def set_dynamic_pars(self, omega, n_p=1, n_t=300, return_domain=False):
        """
        
        Sets values to the parameters related to the dynamics.

        Arguments:
        ----------
        omega: float 
            the frequency of the drive
        n_p: int    (default: 1)
            the number of periods for the evolution 
        n_t: int    (default: 300)
            the number of time slices of the time extend of 1 period
        return_domain: bool     (default: False)
            returns the time domain
        
        Returns:
        --------
        self.time_domain: np.array
            returns the time domain (if return_domain is True)
        """

        self.omega = omega 
        self.n_p = n_p 
        self.n_t = n_t      # per period
        self.period = 2.0 * np.pi / omega 
        self.t0 = 0.0       # option to set a different initial time moment
        self.tf = n_p * self.period 
        self.dt = ( self.tf - self.t0 ) / (n_t * n_p) 

        self.time_domain = np.linspace( self.t0, self.tf, self.n_t * n_p) 
        
        if return_domain:
            return self.time_domain


    def set_seed(self, seed=10):
        """

        Set a (custom) seed for the random number generator.

        Arguments:
        ----------
        seed: int   (default: 10)
            the seed

        """
        np.random.seed(seed)


    def random_disorder(self, Wejp=0.0, Wejs=0.0, Wmup=0.0, Wmus=0.0,
                        return_vals=False): 
        """
        
        Sets random disorder. The default values correspond to having no 
        disorder. 

        Arguments:
        ----------
        Wejp : float    (default: 0.0)
            strength of disorder in the form of a phase in the EJs 
        Wejs : float    (default: 0.0)
            strength of disorder in the form of a shift in the EJs
        Wmup : float    (default: 0.0)
            strength of disorder in the form of a phase in the MUs 
        Wmus : float    (default: 0.0)
            strength of disorder in the form of a shift in the MUs
        return_vals : bool  (default: False)
            return values of the disorder (only when is True)

        Returns:
        --------
        disorder_rv: dict 
            a dictionary of the random variable values for the disorder

        """

        self.disorder = {'Wejp': Wejp, 'Wejs': Wejs, 'Wmup': Wmup, 
                         'Wmus': Wmus}

        self.set_disorder = False
        if (Wejp == 0.0 or Wejs == 0.0 or Wmup == 0.0 or Wmus == 0.0):
            self.set_disorder = True

        # random disorder in the josephson energies
        self.ej_phase_rv = np.random.uniform(- Wejp, Wejp, self.system_size)
        self.ej_shift_rv = np.random.uniform(- Wejs, Wejs, self.system_size)

        # random disorder in the chemical potential
        self.mu_phase_rv = np.random.uniform(- Wmup, Wmup, self.system_size)
        self.mu_shift_rv = np.random.uniform(- Wmus, Wmus, self.system_size)

        if return_vals:
            disorder_rv = {'ej_phase': self.ej_phase_rv,
                           'ej_shift': self.ej_shift_rv,
                           'mu_phase': self.mu_phase_rv,
                           'mu_shift': self.mu_shift_rv}
            return disorder_rv 


    def Hpars_coord(self, pars, coords):
        """

        Assigns values to the parameters E0, dE, and Delta of the 
        time-dependent parameters of the Hamiltonian, based on the 
        coordinate system we choose for these parameters.

        Arguments:
        ----------
        pars : dict 
            dictionary setting values to the constant parameters 
        coords: str 
            system of coordinates for the constant parameters

        """

        error_msg = "Invalid coordinate system. Choose one of the available ones."
        assert coords in ["cartesian","spherical"], error_msg 

        if coords == "cartesian":
            self.E0 = pars["E0"] 
            self.dE = pars["dE"] 
            self.delta = pars["D"] 
    
        elif coords == "spherical":
            self.R = pars["R"] 
            self.theta = pars["theta"] 
            self.phi = pars["phi"]
            
            self.E0 = self.R * np.sin(self.theta) * np.cos(self.phi) 
            self.dE = self.R * np.sin(self.theta) * np.sin(self.phi)
            self.delta = self.R * np.cos(self.theta)  


    def initialize_Htpars(self, pars, EJ1_shift=0.0, coords='cartesian',
                          return_pars=False):
        """
        
        Sets up the time-dependent parameters of the Hamiltonian. 
        
        Parameters:
        -----------
        pars : dict 
            dictionary setting values to the constant parameters,
            entries: E0,dE,D (if cartesian), R,theta,phi (if spherical),
            e0_leads, dE_leads, phi_L, phi_R 
        EJ1_shift : float (default: 0.0)
            sets a fixed energy shift in EJ1 -> used to exit from the top. phase
        coords: str     (default: 'cartesian')
            system of coordinates for the constant parameters
        return_pars : bool (default: False)
            option to return parameters (in lambda function form) 

        Returns:
        --------
        t_pars: dict    (if return_pars == True)
            returns the time-dependent parameters as lambda functions

        """ 

        self.Hpars_coord(pars, coords) 
        
        self.EJ1_shift = EJ1_shift 
        self.e0_leads = pars['e0_leads']
        self.dE_leads = pars['dE_leads']

        self.ec0 = pars['ec0']      # self capacitance
        self.ecx = pars['ecx']      # (nearest neighbour) cross capacitance

        self.phi_L = pars['phi_L']
        self.phi_R = pars['phi_R']
        
        # define local variables (short names)
        E0, dE, D, EJ1sh = self.E0, self.dE, self.delta, self.EJ1_shift
        e0_leads, dE_leads = self.e0_leads, self.dE_leads
        ec0, ecx = self.ec0, self.ecx
        om = self.omega

        # add disorder in the Josephson energies
        ej_phase = self.ej_phase_rv
        ej_shift = self.ej_shift_rv

        # ( note: i, j : indices for unit cell/link ) 
        self.EJ1 = lambda t, i=0, j=0 : ( E0 + dE * np.cos(om*t + ej_phase[2*i])\
                                          + ej_shift[2*j] + EJ1sh ) / 2.0
        self.EJ2 = lambda t, i=0, j=0 : ( E0 - dE * np.cos(om*t + ej_phase[2*i+1])\
                                          + ej_shift[2*j+1] ) / 2.0

        # add disorder in the chemical potentials
        mu_shift = self.mu_shift_rv
        mu_phase = self.mu_phase_rv

        # ( note: i, j : indices for unit cell (with i=0 the first unit cell) )
        self.muA = lambda t, i=0, j=0: ( D * np.sin(om*t + mu_phase[2*i])\
                                         + mu_shift[2*j] )  
        self.muB = lambda t, i=0, j=0: ( - D * np.sin(om*t + mu_phase[2*i+1])\
                                         + mu_shift[2*j+1] )  

        # hopping at the leads   # whyI use sin here?? 
        self.EJ_leads = lambda t : ( e0_leads - dE_leads * np.sin(om*t) )/ 2.0


        # added terms for the interaction - induced charge(!)
        if self.ecx != 0.0:
            self.ng = lambda t, j: (1.0 + (-1.)**(j+1)*(D/ec0)*np.sin(om*t) )/2

        # returns the time-dependent parameters if return_pars == True
        if return_pars: 
            t_pars = {'EJ1': self.EJ1, 'EJ2': self.EJ2, 'muA': self.muA, 
                      'muB': self.muB, 'EJ_leads': self.EJ_leads, 
                      'n_cells': self.n_cells}
            return t_pars


    def hamiltonian(self, t, args={}):  # to satisfy the requirements of qutip
        """

        The time-dependent hamiltonian of the system (in the full 
        hilbert space).

        Arguments:
        ----------
        t: int 
            the current time moment
        args: dict      (default: {})
            dictionary for additional arguments (it's there mostly due to
            the fact that qutip.mesolve needs this construction)

        Returns:
        --------
        H : qutip.Qobj
            the hamiltonian at the given time moment t

        """

        N, L = self.n_cells, self.system_size
        H = Qobj()
        sp, sm, sz = sigmap(), sigmam(), sigmaz()
        idd = identity(2)

        # intra-cell hopping
        intra = Qobj()
        for j in range(1, N+1):
            ls = [ idd ] * L  
            j1 = 2 * j - 1 
            j2 = 2 * j 
            ls[j1-1] = sm 
            ls[j2-1] = sp

            term = tensor( ls )
            hop = self.EJ1(t, i=j-1, j=j-1)  
            hop = 0.0 if hop < 0.0 else hop  # !!!!!!
            intra = intra - hop * ( term + term.dag() ) 

        H = H + intra
     
        # inter-cell hopping
        inter = Qobj()
        for j in range(1, N-1+1):
            ls = [ idd ] * L  
            j1 = 2 * j 
            j2 = 2 * j + 1
            ls[j1-1] = sm 
            ls[j2-1] = sp

            term = tensor( ls )
            hop = self.EJ2(t, i=j-1, j=j-1)
            hop = 0.0 if hop < 0.0 else hop #!!!!
            inter = inter - hop * ( term + term.dag() ) 
           
        H = H + inter 
      
        # chemical potential 
        chem = Qobj()
        for j in range(1, 2*N+1):
            ls = [ idd ] * L  
            ls[j-1] = (idd - sz) / 2.0

            term = tensor( ls )
            chpot = ( self.muA(t, i=j//2, j=j//2) if (j-1)%2 == 0 else
                      self.muB(t, i=j//2-1, j=j//2-1) )
            chem = chem + chpot * ( term + term.dag() ) 
           
        H = H + chem

        # boundary terms 
        
        # when bc is 'open' and we want no leads, set e0leads and dEleads to
        # zeros
        if self.bc == 'open':    # with leads (or not) 

            ls = [ idd ] * L 
            left_lead = np.exp( - 1.0j * self.phi_L )
            ls[0] = left_lead * sp  
            term = tensor(ls) 
            H = H - self.EJ_leads(t) * ( term + term.dag() )  
     
            ls = [ idd ] * L 
            right_lead = np.exp( - 1.0j * self.phi_R )
            ls[-1] = right_lead * sp  
            term = tensor(ls) 
            H = H - self.EJ_leads(t) * ( term + term.dag() )  

        elif self.bc == 'closed':
            
            ls = [ idd ] * L  
            ls[0] = sm 
            ls[-1] = sp
            term = tensor( ls )
            hop = self.EJ2(t, i=N-1, j=N-1)
            hop = 0.0 if hop < 0.0 else hop #!!!!
            H = H - hop * ( term + term.dag() ) 
       
        # interaction : with cross capacitance
        if self.ecx != 0.0 and not bool(args):
            num = (idd - sz) / 2.0
            for i in range(L-1):
                ls = [idd]*L
                ls[i] = num 
                ls[i+1] = num
                term = tensor(ls)
                H = H + self.ecx * term 

                ls = [idd]*L
                ls[i] = num
                term = tensor(ls) 
                H = H - self.ecx * self.ng(t, i+1) * term 

                ls = [idd]*L
                ls[i+1] = num 
                term = tensor(ls)
                H = H - self.ecx * self.ng(t, i) * term 
         
            #fix this - make it one with open
            if self.bc == 'closed':
                
                ls = [idd]*L
                ls[-1] = num 
                ls[0] = num
                term = tensor(ls)
                H = H + self.ecx * term 

                ls = [idd]*L
                ls[-1] = num
                term = tensor(ls) 
                H = H - self.ecx * self.ng(t, 0) * term 

                ls = [idd]*L
                ls[0] = num 
                term = tensor(ls)
                H = H - self.ecx * self.ng(t, -1) * term 

        return H 



    def initial_state(self):
        """

        Sets up the initial state 

        Returns:
        --------
        gs: qutip.Qobj
            the ground state of the system

        """
        
        ham = self.hamiltonian(0.0)     #, args={'withU': False})
        _, states = ham.eigenstates()
        
        # pick lowest-energy configuration 
        gs = ket2dm( states[0] )
        return gs

    
    def current_op(self, lat):
        """
        
        The current operator.

        Arguments:
        ----------
        lat: list 
            must be a list of two indeces, s and j where j is the unit cell
            index and s takes values of 0,1 (sublattice index - bulk) and 2 
            (boundary term).
            if bc == 'open', then along with s=2, we need to specify j, 
            which takes values of j = 0 (left lead), and j = 1 (right lead).
            if bc == 'closed', then we pass s=2, with any value of j 
            (it will not have an impact on the result)  
        t: float 
            time moment

        Returns:
        --------
        curr: qutip.Qobj
            the current operator for the given time moment 

        """ 
        
        s, j = lat[0], lat[1]
        sp, sm, idd = sigmap(), sigmam(), identity(2)
        L = self.system_size
        N = L // 2

        def inner_f(t):

            curr = Qobj()
            ls = [idd] * L

            if s == 0 or s == 1:

                j1 = 2 * j - 1 + s  
                j2 = 2 * j + s
                ls[j1-1] = sm 
                ls[j2-1] = sp

                term = tensor( ls )
                hop = ( self.EJ1(t, i=j-1, j=j-1) if s == 0 else 
                        self.EJ2(t, i=j-1, j=j-1) )
                hop = 0.0 if hop < 0.0 else hop #!!!!
                curr = - hop * ( term - term.dag() )    # - term.dag() current!

            elif s == 2:

                if self.bc == 'closed':
                
                    ls[0] = sm 
                    ls[-1] = sp
                    term = tensor( ls )
                    hop = - self.EJ2(t, i=N-1, j=N-1)
                    hop = 0.0 if hop < 0.0 else hop #!!!!
                    curr = hop * ( - term + term.dag() ) 

                elif self.bc == 'open':

                    lead_phase = ( np.exp( - 1.0j * self.phi_L ) if j == 0 else
                                   np.exp( - 1.0j * self.phi_R ) )
                    ls[0 if j == 0 else -1] = lead_phase * sp  
                    term = tensor(ls) 
                    hop = - self.EJ_leads(t)
                    #hop = 0.0 if hop < 0.0 else hop #!!!!
                    curr = (-1.0)**(j) * hop * ( term - term.dag() )  
                    # - term.dag() current!
         
            curr = curr / 1.0j  # for the current to be real 
            return curr

        return inner_f    


    def number_op(self, lat, power2=False):
        """

        The occupation number operator through the magnetization operator.

        Arguments:
        ----------
        lat: list
            provides with s (sublattice), j (unit cell) indices 
            (note: s \in \{0,1\}, j \in \{ 1, 2, 3, ... N\})
        t: float
            time moment

        Returns:
        --------
        occ: qutip.Qobj
            number operator 

        """

        s, j = lat[0], lat[1]
        sz, idd = sigmaz(), identity(2)
        L = self.system_size

        def inner_f(t): 
            occ = Qobj()
            ls = [idd] * L 
            ls[2*j-2+s] = (idd - sz) / 2.0 
            occ = tensor(ls)
            if power2:
                occ = occ * occ
            return occ

        return inner_f


    def get_ops(self):
        """
        
        Prepare a dictionary of operators for which we want to calculate the
        expectation values - related to the problem. 

        For this problem, we are interested in the expectation values of the
        following operators:
            - current
            - occupation number/magnetization
            - hamiltonian (energy) 

        Returns:
        --------
        op_dict: dict
            dictionary of operators that takes as an argument the time moment
            t

        """

        L = self.system_size
        op_dict = {'occupation': [], 'nsq': [], 'current': []}

        # set up number operators
        for j in range(1, L//2+1):
            for s in range(0, 2):
                op_dict['occupation'] = ( op_dict['occupation'] +
                                          [self.number_op([s,j])] )


        # set up number operators squared (for fluctuations)
        for j in range(1, L//2+1):
            for s in range(0, 2):
                op_dict['nsq'] = ( op_dict['nsq'] +
                                 [self.number_op([s,j], power2=True)])
 
        # set up current operators
        for j in range(1, L//2+1):
            for s in range(0, 2):

                if not (j == L//2 and s == 1):
                    op_dict['current'] = ( op_dict['current'] +
                                           [self.current_op([s,j])] )
        
        op_dict['energy'] = [self.hamiltonian]

        s = 2
        if self.bc == 'open':
            op_dict['current'] = op_dict['current'] + [self.current_op([s,0])] 
            op_dict['current'] = op_dict['current'] + [self.current_op([s,1])] 
        elif self.bc == 'closed':
            op_dict['current'] = op_dict['current'] + [self.current_op([s,0])] 

        cur_list = op_dict['current'].copy()
        op_dict['current'] = []
        # average current
        n_cr = len( cur_list )
        aver_curr = lambda t : sum( [j_i(t) for j_i in cur_list] ) / n_cr 
        op_dict['current'] = [aver_curr]

        return op_dict 
 
#------------------------------------------------------------------------------
# end of open_jja class
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
