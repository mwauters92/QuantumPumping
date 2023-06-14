#------------------------------------------------------------------------------
# main.py
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# import libraries
#------------------------------------------------------------------------------

import sys
sys.path.append('./')
import numpy as np 
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
from modules import * 
import time

#------------------------------------------------------------------------------
# argparse 
#------------------------------------------------------------------------------

parser=ArgumentParser(description='This script calculates ldos and total current in a chain of SC islands connected to a SC lead in each end')

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

parser.add_argument('--unit_cells', default=2, type=int,
                    help='Number of unit cells.')
parser.add_argument('--qinf', default=False, type=bool, 
                    help=('Calculate the pumped charge in the infinite-time \
                    limit.') )
parser.add_argument('--et', default=False, type=bool, 
                    help='Calculate the instantaneous many-body spectrum.')
parser.add_argument('--qephi', default=False, type=bool,
                    help='Calculate the quasi-energy spectrum.') 

args = parser.parse_args()

#------------------------------------------------------------------------------
# simulation type 
#------------------------------------------------------------------------------

def set_sim_type(mp, args):
    
    if args.qinf:
        mp['sim_type'] = "Pumped charge at the infinite time."
    if args.et:
        mp['sim_type'] = "Instantaneous many-body spectrum."
    if args.qephi:
        mp['sim_type'] = "Many-body quasi-energy spectrum."

#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------

if __name__ == "__main__":

    #-------------------------------------------------------

    week = 60 
    num = args.num_file
    to_save = True 

    mp = {'omega': args.omega, 'n_p': args.num_periods, 'n_t': 300, 
          'pbc': False} 

    mp['Ej0'] = 1.0  
    mp['Ec0'] = 4.0 
    mp['delta_n'] = 0.5
    mp['noise_Ej'] = 0.0
    mp['noise_Ec'] = 0.0

    mp['N'] = args.unit_cells
    mp['L'] = 3 * mp['N']
    mp['Mcut'] = 2

    mp['El_start'] = 0.1
    mp['El_end'] = 1.5
    mp['el_num'] = 1 
    mp['phi_num'] = 20 
    
    mp['comments'] = ['']
    mp['filename'] = f'w{week}_n{num}_hhwb_TEST'    

    set_sim_type(mp, args)

    # temporary 
    path_name = "data"
    mp['path_name'] = path_name
    
    #-------------------------------------------------------

    ej0 = mp['Ej0'] 
    #el_tab = np.linspace( ej0 * mp['El_start'], ej0 * mp['El_end'], 
    #                      mp['el_num'])
    el_tab = [1.0]
    phi_tab = np.linspace(0.0, 2.0 * np.pi, mp['phi_num'], endpoint=False)
    
    #-------------------------------------------------------

    data_new = {'el_ls': el_tab, 'phi_ls': phi_tab}

    start = time.perf_counter()

    # ----------------

    hh = hhm() 
    hh.set_lattice_pars(L=mp['L'], pbc=mp['pbc'])
    hh.set_dynamic_pars(omega=mp['omega'], n_p=mp['n_p'], n_t=mp['n_t'])

    curr = np.zeros( (mp['el_num'], mp['phi_num'] ) ) 

    for j, el in tqdm( enumerate(el_tab), desc="El" ):

        for k, ph in tqdm( enumerate(phi_tab), desc="ph" ):

     
            # set values to the Hamiltonian parameters  
            hh.set_Htpars(El=el, phase=ph, delta_n=mp['delta_n'], 
                          Ej0=mp['Ej0'], Ec0=mp['Ec0'], noise_Ej=mp['noise_Ej'], 
                          noise_Ec=mp['noise_Ec'], Mcut=mp['Mcut'] )

            # initialize qobs class
            hhqob = ojja_qobs(hh)

            if args.qinf: 

                hhqob.create_floquet()
                q = hhqob.q_floquet()
                curr[j][k] = q

            if args.et:
            
                res = hhqob.instantaneous_energies()
                eg = hhqob.find_gap()

            if args.qephi:

                continue

    data_new = {'current': curr}

    # ----------------

    end = time.perf_counter()
    dur = end - start
    mp['comments'] += [f'Run time (in secs): {dur:.2f}'] 

    data = {'model_pars': mp, 'data': data_new} 

    #-------------------------------------------------------
     
    # save data 

    if to_save:
        with open(f'{path_name}/{mp["filename"]}.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #-------------------------------------------------------
    
    print("Simulations have been completed!")

    #-------------------------------------------------------

#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
