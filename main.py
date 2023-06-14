import argparse
from tqdm import tqdm
from modules import *

parser=argparse.ArgumentParser(description='This script calculates ldos and total current in a chain of SC islands connected to a SC lead in each end')

# Definition of constants
parser.add_argument('Nphi', type=int, help='Number of values of the phase difference of left and right lead between 0 and 2pi')
parser.add_argument('NEl', type=int, help='Number of values of the lead-island coupling between 0.1Ej0 and 1.5Ej0')
parser.add_argument('Nt', type=int, help='Number of values of the time between 0 and 2pi') 

parser.add_argument('--L', default=6, type=int, help='Length of the SC chain (sites)')
parser.add_argument('--Mcut', default=2, type=int, help='Truncation of the local Hilbert space. Mcut=2 for hard core bosons')
parser.add_argument('--num_periods', default=1, type=int, help='number of periods used in the time evolution')
parser.add_argument('--PBC', default=False, type=bool, help='Periodic boundary conditions')
parser.add_argument('--noise_Ej', default=0, type=float, help='Noise of the Josephson coupling. Can be either a single variable or a list of length L-1')
parser.add_argument('--noise_Ec', default=0, type=float, help='Noise of the on-site charging energy. Can be either a single variable or a list of length L')
parser.add_argument('--Ej0', default=0.5, type=float, help='Josephson coupling. Can be either a single variable or a list of length L-1')
parser.add_argument('--Ec0', default=1.0, type=float, help='The on-site charging energy. Can be either a single variable or a list of length L')
parser.add_argument('--delta_n', default=0.5, type=float, help='The amplitude of the ng oscillations away from 1/2')
parser.add_argument('--omega', default=0.05, type=float, help='The frequency with which the gate voltage is varied')
parser.add_argument('--El_start', default=0.1, type=float, help='Initial value of the coupling with the leads')
parser.add_argument('--El_end', default=1.5, type=float, help='Final value of the coupling with the leads')

argums = parser.parse_args()

# parameters

NEl=argums.NEl
Nphi=argums.Nphi

El_start=argums.El_start
El_end=argums.El_end
Ej0=argums.Ej0

Eltab = np.linspace(Ej0*El_start, Ej0*El_end, NEl)
phleadtab = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

hh = hhm() 
hh.set_lattice_pars(L=argums.L, pbc=argums.PBC)
hh.set_dynamic_pars(omega=argums.omega, n_p=argums.num_periods, n_t=argums.Nt)

curr = np.zeros( (NEl, Nphi ) ) 

for j, el in tqdm( enumerate(Eltab), desc="El" ):

    for k, ph in tqdm( enumerate(phleadtab), desc="ph" ):
 
        # set values to the Hamiltonian parameters  
        hh.set_Htpars(El=el, phase=ph, delta_n=argums.delta_n, Ej0=argums.Ej0,
                      Ec0=argums.Ec0, noise_Ej=argums.noise_Ej, 
                      noise_Ec=argums.noise_Ec, Mcut=argums.Mcut )

        # initialize qobs class
        hhqob = ojja_qobs(hh)

        # calculates the many-body spectrum and gap
        #_ = hhqob.instantaneous_energies()
        #eg = hhqob.find_gap()
        
        hhqob.create_floquet()
        q = hhqob.q_floquet()

        curr[j][k] = q

print(curr)
print('done')
# Then save it in some form.


#dens_t = res.expect[:L]
#curr_t = res.expect[L:][0]
#densitytab[phlead,El]=dens_t
#currenttab[phlead,El]=curr_t


