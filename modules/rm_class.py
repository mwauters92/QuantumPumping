#------------------------------------------------------------------------------
# open_jja_class.py
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# import necessary libraries
#------------------------------------------------------------------------------

import numpy as np 
from qutip import sigmap, sigmam, sigmaz, identity   
from qutip import Qobj, num, qeye, tensor, ket2dm, expect
from modules.custom_operators import *


#------------------------------------------------------------------------------
# open_jja class 
#------------------------------------------------------------------------------

class RM_model: 

    def __init__(self, mp):
        """
        
        Class initialization.
        A class for the JJAs system with S.C. leads (open boundary terms
        - and closed as well)

        """

        # set a seed for random number generator (default one) 
        self.set_seed()
        
        self.set_lattice_pars(L=mp['L'], pbc=mp['pbc'])
        self.set_dynamic_pars(omega=mp['omega'], n_p=mp['n_p'], n_t=mp['n_t'])

        
        # initialize disorder (default: no disorder)
        self.random_disorder()

        # initilize the time-dependent parameters of the Hamiltonian
        #self.set_Htpars(mp) 

        self.model = 'RiceMele'    
    def set_lattice_pars(self, L=6, pbc=False):
        """
        
        Sets values to the parameters related to the lattice construction.

        Arguments:
        ----------
        n_cells: int    (default: 5)
            number of unit cells
        bc: str     (default: open) 
            type of boundary conditions 

        """        

        self.system_size = L
        self.n_cells = int(L/2) 
        self.pbc = pbc 


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




    def set_Htpars(self, model_pars, return_pars=False):
        """
        
        Sets up the time-dependent parameters of the Hamiltonian. 
        
        Parameters:
        -----------
        return_pars : bool (default: False)
            option to return parameters (in lambda function form) 

        Returns:
        --------
        t_pars: dict    (if return_pars == True)
            returns the time-dependent parameters as lambda functions

        """ 

        
        self.E0 = model_pars['Ej0']
        self.dE = model_pars['dE']
        self.delta_n = model_pars['delta_n']
        self.Mcut = model_pars['Mcut']
        L = self.system_size 
        self.EJ1_shift =  0.0 #model_pars['EJ1_shift'] 
        self.El = model_pars['El']

        self.Ec0 = model_pars['Ec0']      # self capacitance
        self.U = model_pars['U']      # (nearest neighbour) cross capacitance

        self.phi_L = model_pars['phase'] #        self.phi_R = pars['phi_R']
        j=np.arange(self.system_size)
        
        noise_Ej = model_pars['noise_Ej'] 
        noise_Ec = model_pars['noise_Ec'] 

        self.Ej = self.E0 + (np.random.rand(L)-0.5)*noise_Ej + 0.5*(1+np.cos(j*np.pi))*self.EJ1_shift
        self.Ec = self.Ec0 + (np.random.rand(L)-0.5)*noise_Ec

        self.Nh = tensor([qeye(self.Mcut) for j in range(self.system_size)]).dims        
        print(f'Hilbert space dimension for {self.system_size} islands with cutoff {self.Mcut}: {self.Nh}')       
        self.Ej_t = lambda ph: self.Ej + self.dE*np.cos(ph + j*np.pi)
        self.ng_t = lambda ph: model_pars['average_ng'] + self.delta_n*np.sin(ph+j*np.pi)


        # returns the time-dependent parameters if return_pars == True
        if return_pars: 
            t_pars = {'EJ_t': self.Ej_t,  'ng_t': self.ng_t, 
                      'EJ_leads': self.El, 
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
        H = Qobj(dims=self.Nh)
        cj, cdagj, numj = mydestroy(self.Mcut), mycreate(self.Mcut), num(self.Mcut)
        idd = qeye(self.Mcut)
        
        # we evaluate the Ej_t and ng_t functions at ph=omega*t
        Ej_t_ev = self.Ej_t(self.omega*t)
        ng_t_ev = self.ng_t(self.omega*t) 

        # hopping
        op_list = [ idd ] * L  
        
        for j in range(L-1):
            op_list[j] = cdagj
            op_list [j+1] = cj
            
            H += - 0.5*Ej_t_ev[j]*tensor(op_list)
            op_list[j] = idd

        op_list = [ idd ] * L

        if self.pbc:
            op_list[-1] = cdagj
            op_list[0] = cj
            H += - 0.5*Ej_t_ev[-1]*tensor(op_list)
            
        else:
            op_list[0] = cj
            H += -0.5*self.El*np.exp(-1j*self.phi_L)*tensor(op_list)
            
            op_list[0] = idd
            op_list[-1] = cj
            H += -0.5*self.El*tensor(op_list)

    
        H += H.dag() 

            
        # onsite energy
        op_list = [ idd ] * L  
        for j in range(L):
            op_list[j] =  self.Ec[j]*(numj - ng_t_ev[j])**2
            H += tensor(op_list)
            op_list[j] = idd 
           
        
        # nearest neighbours interactions

        if self.U > 0.0:
            op_list = [ idd ] * L  

            for j in range(L-1):
                op_list[j] =  ( numj - ng_t_ev[j] )
                op_list[j+1] = ( numj - ng_t_ev[j+1] )             
                H += self.U * tensor(op_list)
                op_list[j] = idd

            if self.pbc:
                op_list[0] = ( numj - ng_t_ev[0] )             
                H += self.U * tensor(op_list) 


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
        gs = states[0] 
        return gs

    
    
    def get_bond_current(self,L,Mcut, Ej,  El, phlead, PBC=False):
        '''
        Defines the current operator for a chain of bosonic sites
        Parameters:
            L (int): number of bosonic sites
            Mcut (int): truncation of the local Hilbert space
            Ej (float): Josephson coupling between neighbouring sites. Can be either a number or an array of size L
            El (float): Josephson coupling between leads and the outer SC islands.
            phlead (fload): the phase difference between the left and right SC leads.
            PBC (bool): default False. If True, the system has periodic boundaries

        Returns:
            Jcurrent: average current operator 
            current_density: list with the current density on each link of the chain

        '''

        op_list=[qeye(Mcut) for j in range(L)]

        Jcurrent=Qobj( dims=self.Nh )
        current_density=[]    
        for j in range(L-1):
            op_list[j]=mycreate(Mcut)
            op_list[j+1]=mydestroy(Mcut)
            op_local_jump = -0.5j*Ej[j]*tensor(op_list)
            Jcurrent+= (op_local_jump+op_local_jump.dag())
            current_density.append(op_local_jump+op_local_jump.dag())
            op_list[j] = qeye(Mcut) 

        if PBC:
            op_list=[qeye(Mcut) for j in range(L)]
            op_list[-1] = mycreate(Mcut)
            op_list[0]=mydestroy(Mcut)
            op_local_jump = -0.5j*Ej[0]*tensor(op_list)
            Jcurrent+= (op_local_jump+op_local_jump.dag())
            current_density.append(op_local_jump+op_local_jump.dag())
            links=L
        else: 
            op_list=[qeye(Mcut) for j in range(L)]
            op_list[0]=mydestroy(Mcut)
            op_jump_left = -0.5j*np.exp(-1j*phlead)*El*tensor(op_list) # CHECK signs!!
            Jcurrent+= (op_jump_left+op_jump_left.dag())
            op_list[0]=qeye(Mcut)
            op_list[-1]=mycreate(Mcut)
            op_jump_right=-0.5j*El*tensor(op_list)
            Jcurrent+=(op_jump_right+op_jump_right.dag())
            current_density.append(op_jump_right+op_jump_right.dag())
            links=L+1
        return current_density, Jcurrent/links


    def total_number(self,L, Mcut, density=False):
        """

        Defines the number operator for a chain of bosonic sites.

        Arguments:
        ----------
            L (int): length of the chain (sites, not unit cells)
            Mcut (int): truncation of the local Hilbert space. Mcut=2 for 
                        hard core bosons
            density (bool): default False. If true it returns the local 
                            density operator instead of the number operator

        Returns:
        --------
            qutip.Qobj(dims=[Mcut ^ L]), either the total number operator 
                    N=sum_j n_j or the space resolved local
            boson number n_j

        """

        op_list=[qeye(Mcut) for j in range(L)]
        num_tot = Qobj( dims=self.Nh )
        d_op=[]
        for j in range(L):
            op_list[j] = num(Mcut)
            num_tot += tensor(op_list)
            d_op.append(tensor(op_list))
            op_list[j] = qeye(Mcut)

        if density:
            return d_op
        else:
            return num_tot 

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
        op_dict = {'tot_num': [], 'nsq': [], 'current': [], 'current_density':[]}
        
        tot_num = self.total_number(L, self.Mcut, density=True)
        op_dict['tot_num'] = tot_num

        current_density = []
        for j in range(L):
            current_density.append(lambda t, j=j: self.get_bond_current(L,self.Mcut, 
                            self.Ej_t(self.omega*t), self.El, self.phi_L, PBC=self.pbc)[j]) 
                           

        op_dict['current_density'] = current_density 
        def tot_current(t):
            return (self.get_bond_current(L,self.Mcut,
                            self.Ej_t(self.omega*t), self.El, self.phi_L, PBC=self.pbc)[-1])
        op_dict['current_op'] = tot_current 
        op_dict['current_ev'] = [lambda t, psi : expect(tot_current(t),psi) ] 

        return op_dict 
 
#------------------------------------------------------------------------------
# end of open_jja class
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
