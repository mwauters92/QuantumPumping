#------------------------------------------------------------------------------
# hh_class.py
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# import necessary libraries
#------------------------------------------------------------------------------

import numpy as np 
from qutip import Qobj, num, qeye, tensor, ket2dm, expect, create, destroy  

#------------------------------------------------------------------------------
# hhm class 
#------------------------------------------------------------------------------

class HH_model: 

    def __init__(self):
        """
        
        Class initialization.

        """

        # set a seed for random number generator (default one) 
        self.set_seed()

        # set values to the lattice parameters (default ones)
        self.set_lattice_pars()

        # set values to the dynamic parameters (default ones)
        self.set_dynamic_pars(omega=0.05)
        

        
    def set_lattice_pars(self, L=6, pbc=False):
        """
        
        Sets values to the parameters related to the lattice construction.

        Arguments:
        ----------
        L: int    (default: 6)
            system size
        pbc: bool     (default: False) 
            True for periodic bc

        """        

        self.system_size = L 
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
        self.t0 = 0.0      
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


    def set_Htpars(self, El, phase, delta_n=None, n_ave=0.5, Ej0=1.0, Ec0=4.0,  
                   noise_Ej=0.0, noise_Ec=0.0, Mcut=2, return_pars=False):


        L = self.system_size 
         

        self.phase, self.El = phase, El
        self.Ec0, self.Ej0 = Ec0, Ej0
        
        if delta_n is None:
            self.delta_n = 0.5*Ej0/Ec0
        else:
            self.delta_n = delta_n
        
        j = np.arange(L)
        self.ng_t = lambda ph: delta_n*np.cos(ph - 2*np.pi/3*j)
        
        self.n_ave=n_ave
        
        self.Mcut = Mcut  

        self.Ej = Ej0 + (np.random.rand(L-1)-0.5)*noise_Ej
        self.Ec = Ec0 + (np.random.rand(L)-0.5)*noise_Ec

        self.noise_Ej = noise_Ej
        self.noise_Ec = noise_Ec
        
        self.Hsc_J = self.Hsc_Josephson(self.system_size, self.Ej, self.El, 
                                        self.phase, self.Mcut, self.pbc)
        
        error_msg = ('Hsc_J is not a Hermitian operator. You may want to \
                     reconsider what you are doing.')
        assert self.Hsc_J.isherm, error_msg


    def hamiltonian(self, t, args={}):
        """
        Defines the time-dependent part of the Hamiltonian, i.e. the on-site energy. 
        It requires Hsc_J to be defined before this function is called.
        The function is written so that it is callable by qutip.mesolve()
        Parameters:
            t (float): time
        
        Returns:
            H+Hsc_J: total time-dependent Hamiltonian
        """
       
        L = self.system_size      
        
        ng = self.n_ave + self.ng_t( self.omega * t ) 
        
        op_list = [qeye(self.Mcut) for j in range(L)]
        
        H = Qobj( dims=tensor(op_list).dims )

        for j in range(L):

            op_list[j] = self.Ec[j] * ( num(self.Mcut) - ng[j] )**2
            H += tensor(op_list)
            op_list[j] = qeye(self.Mcut)
        
        return H + self.Hsc_J



    def initial_state(self):
        """

        Sets up the initial state 

        Returns:
        --------
        gs: qutip.Qobj
            the ground state of the system

        """

        
        
        ng = self.n_ave + self.ng_t(0)
        Hsc = self.hamiltonian(0) 
        #Hsc_chain_static(L, ng, self.Ec, self.Ej, 0, 0, Mcut=Mcut, PBC=PBC) 
        _, states = Hsc.eigenstates()

        # pick lowest-energy configuration 
        gs = states[0]
        return gs #states[0]


    def get_ops(self):
        """
        
        Prepare a dictionary of operators for which we want to calculate the
        expectation values - related to the problem. 

        For this problem, we are interested in the expectation values of the
        following operators:
            - current
            - total number

        Returns:
        --------
        op_dict: dict
            dictionary of operators that takes as an argument the time moment

        """

        L = self.system_size 
                
        tot_current, current_density = self.get_bond_current(L, self.Mcut, self.Ej, self.El, 
                                                        self.phase, PBC=self.pbc)
        tot_num = self.total_number(L, self.Mcut, density=True)

        #op_dict = {'tot_num': tot_num, 'current': tot_current}
        #op_dict = {'tot_num': tot_num, 'current': current_density}
        op_dict = {'tot_num': tot_num, 'current': tot_current, 
                   'current_density': current_density}

        return op_dict 

 

    def total_number(self,L, Mcut, density=False):
        '''
        defines the number operator for a chain of bosonic sites
        Parameters:
            L (int): length of the chain (sites, not unit cells)
            Mcut (int): truncation of the local Hilbert space. Mcut=2 for hard core bosons
            density (bool): default False. If true it returns the local density operator instead of the number operator

        Returns:
            qutip.Qobj(dims=[Mcut ^ L]), either the total number operator N=sum_j n_j or the space resolved local
            boson number n_j

        '''
        op_list=[qeye(Mcut) for j in range(L)]
        num_tot = Qobj(dims=tensor(op_list).dims)
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


    def get_bond_current(self,L,Mcut, Ej, El, phlead, PBC=False):
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

        Jcurrent=Qobj(dims=tensor(op_list).dims)
        current_density=[]    
        Ej=expand_variable(Ej,L-1)

        for j in range(L-1):
            op_list[j]=create(Mcut)
            op_list[j+1]=destroy(Mcut)
            op_local_jump = -0.5j*Ej[j]*tensor(op_list)
            Jcurrent+= (op_local_jump+op_local_jump.dag())
            current_density.append(op_local_jump+op_local_jump.dag())
            op_list[j] = qeye(Mcut) 

        if PBC:
            op_list=[qeye(Mcut) for j in range(L)]
            op_list[-1] = create(Mcut)
            op_list[0]=destroy(Mcut)
            op_local_jump = -0.5j*Ej[0]*tensor(op_list)
            Jcurrent+= (op_local_jump+op_local_jump.dag())
            current_density.append(op_local_jump+op_local_jump.dag())
            links=L
        else: 
            op_list=[qeye(Mcut) for j in range(L)]
            op_list[0]=destroy(Mcut)
            op_jump_left = -0.5j*np.exp(-1j*phlead)*El*tensor(op_list) # CHECK signs!!
            Jcurrent+= (op_jump_left+op_jump_left.dag())
            op_list[0]=qeye(Mcut)
            op_list[-1]=create(Mcut)
            op_jump_right=-0.5j*El*tensor(op_list)
            Jcurrent+=(op_jump_right+op_jump_right.dag())
            links=L+1
        return Jcurrent/links, current_density





    def Hsc_Josephson(self,L,Ej,El,phlead, Mcut=2, PBC=False):
        '''
        Defines the static part of the HH Hamiltonian, i.e. the Josephson couplings.
        Parameters:
            L (int): number of sites
            Ej (float): tunnelling amplitude. Can be either a number or an array of size L
            El (float): Josephson coupling between leads and the outer SC islands.
            phlead (float): the phase difference between the left and right SC leads.
            Mcut (int): default 2, truncation of the local Hilbert space.
            PBC (bool): default False. If True, the system has PBC

        Returns:
            H: hopping Hamiltonian
        '''
        Ej=expand_variable(Ej,L-1)

        op_list=[qeye(Mcut) for j in range(L)]

        H=Qobj(dims=tensor(op_list).dims)

        for j in range(L-1):
            op_list[j]=create(Mcut)
            op_list[j+1]=destroy(Mcut)
            op_local_jump = -0.5*Ej[j]*tensor(op_list)
            H+= (op_local_jump+op_local_jump.dag())
            op_list[j] = qeye(Mcut)

        if PBC:
            op_list=[qeye(Mcut) for j in range(L)]
            op_list[-1] = create(Mcut)
            op_list[0]=destroy(Mcut)
            op_local_jump = -0.5*Ej[j]*tensor(op_list)
            H+= (op_local_jump+op_local_jump.dag())
        else:
            op_list=[qeye(Mcut) for j in range(L)]
            op_list[0]=destroy(Mcut)
            op_jump_left = -0.5*El*np.exp(-1j*phlead)*tensor(op_list)
            H+= (op_jump_left+op_jump_left.dag())
            op_list[0]=qeye(Mcut)
            op_list[-1]=create(Mcut)
            op_jump_right = -0.5*El*tensor(op_list)
            H+=(op_jump_right+op_jump_right.dag())

        return H

#------------------------------------------------------------------------------
# end of hh class
#------------------------------------------------------------------------------
def expand_variable(x,L):
    '''
    utility function: if x is an array or a list it does nothing, if x is a number it creates an array of 
    dimension L with x in each entry.
    '''
    if np.size(x)==1:
        x = np.ones(L)*x
    return x


def Hsc_chain_static(L, ng, Ec, Ej,El, phlead,  Mcut=2, return_dimension_only=False, PBC=False):
    '''
    Defines the Harper-Hofstadter Hamiltonian in the context of JJ arrays. 
    The charge degeneracy points correspond to half integer values of ng. 
    Parameters:
        L (int): number of sites
        ng (float): induced charge. Can be either a number or an array of size L
        Ec (float): charging energy of each site. Can be either a number or an array of size L
        Ej (float): tunnelling amplitude between neighbouring sites. Can be either a number or an array of size L
        El (float): Josephson coupling between leads and the outer SC islands.
        phlead (fload): the phase difference between the left and right SC leads.
        Mcut (int): truncation of the local Hilbert space
        
        return_dimension_only (bool): default False. If True, returns only the dimension of the qutip.Qobj describing the Hamiltonian
        PBC (bool): default False. If True, the Hamiltonian has periodic boundaries.
        
    Returns:
        H: qutip.Qobj(dims=[Mcut^L])
    '''
    ng=expand_variable(ng,L)
    Ec=expand_variable(Ec,L)
    Ej=expand_variable(Ej,L-1)
    
    op_list=[qeye(Mcut) for j in range(L)]
    
    H=Qobj(dims=tensor(op_list).dims)
    
    if return_dimension_only:
        return(H.dims)
    
    # on site energy
    for j in range(L):
        op_list[j]=Ec[j]*(num(Mcut)-ng[j])**2-Ec[j]*ng[j]**2 
        H+= tensor(op_list) 
        op_list[j]=qeye(Mcut)
    
    # interaction
    for j in range(L-1):
        op_list[j]=create(Mcut)
        op_list[j+1]=destroy(Mcut)
        op_local_jump = -0.5*Ej[j]*tensor(op_list) 
        H+= (op_local_jump+op_local_jump.dag())
        op_list[j] = qeye(Mcut)
    
    if PBC:
        op_list=[qeye(Mcut) for j in range(L)]
        op_list[-1] = create(Mcut)
        op_list[0] = destroy(Mcut)
        op_local_jump = -0.5*Ej[0]*tensor(op_list)
        H+= (op_local_jump+op_local_jump.dag())
    else:
        op_list=[qeye(Mcut) for j in range(L)]
        op_list[0]=destroy(Mcut)
        op_jump_left = -0.5*El*np.exp(-1j*phlead)*tensor(op_list)
        H+= (op_jump_left+op_jump_left.dag())
        op_list[0]=qeye(Mcut)
        op_list[-1]=create(Mcut)
        op_jump_right = -0.5*El*tensor(op_list)
        H+=(op_jump_right+op_jump_right.dag())
        
    return H
#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
