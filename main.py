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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
from modules import * 
import time

#------------------------------------------------------------------------------
# argparse 
#------------------------------------------------------------------------------

msg = ('This script calculates ldos and total current in a chain of SC\
       islands connected to a SC lead in each end.')
parser = ArgumentParser(description=msg, 
                        formatter_class=ArgumentDefaultsHelpFormatter)

# to be removed?
parser.add_argument('--Mcut', metavar='', default=2, type=int, 
                    help=('Truncation of the local Hilbert space. Mcut=2 for\
                    hard core bosons.') )
parser.add_argument('--num_periods', metavar='', default=1, type=int, 
                    help='number of periods used in the time evolution.')
parser.add_argument('--nt', metavar='', default=301, type=int, 
                    help='Number of values of the time between 0 and 2pi.')
parser.add_argument('--nuc', metavar='', default=2, type=int, 
                    help='Number of unit cells.')
parser.add_argument('--PBC', action='store_true',  
                    help='Periodic boundary conditions')

parser.add_argument('--ej0', metavar='', default=0.2, type=float, 
                    help=('Josephson coupling. Can be either a single variable\
                    or a list of length L-1.') )
parser.add_argument('--ec0', metavar='', default=1.0, type=float, 
                    help=('The on-site charging energy. Can be either a single\
                    variable or a list of length L.'))
parser.add_argument('--delta_n', metavar='', default=None, type=float, 
                    help='The amplitude of the oscillations in the induced charge.')
parser.add_argument('--ave_ng', metavar='', default=0.5, type=float, 
                    help='The average value of the induced charge, sets the chemical potential.')
parser.add_argument('--noise_Ej', metavar='', default=0.0, type=float, 
                    help=('Noise of the Josephson coupling. Can be either a \
                    single variable or a list of length L-1.') )
parser.add_argument('--noise_Ec', metavar='', default=0.0, type=float, 
                    help=('Noise of the on-site charging energy. Can be either\
                   a single variable or a list of length L.') )

parser.add_argument('--omega', metavar='', default=0.05, type=float, 
                    help='The frequency with which the gate voltage is varied')

parser.add_argument('--El_start', metavar='', default=0.1, type=float, 
                    help='Initial value of the coupling with the leads.')
parser.add_argument('--El_end', metavar='', default=1.5, type=float, 
                    help='Final value of the coupling with the leads.')
parser.add_argument('--Nel', metavar='', default=20, type=int, 
                    help=('Number of values of the lead-island coupling') )
parser.add_argument('--Nphi', metavar='', default=20, type=int, 
                    help=('Number of values of the phase difference of left\
                    and right lead between 0 and 2pi.') )

parser.add_argument('--qinf', action='store_true', 
                    help=('Calculate the pumped charge in the infinite-time \
                    limit.') )
parser.add_argument('--et', action='store_true', 
                    help='Calculate the instantaneous many-body spectrum.')
parser.add_argument('--qephi', action='store_true', 
                    help='Calculate the quasi-energy spectrum.') 

parser.add_argument('--path_name', type=str,default='data', 
                    help='path to the folder where to save the results') 

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
    num = 100 #args.num_file
    to_save = True 

    mp = {'omega': args.omega, 'n_p': args.num_periods, 'n_t': args.nt, 
          'pbc': args.PBC} 

    mp['Ej0'] = args.ej0  
    mp['Ec0'] = args.ec0

    if args.delta_n is None:
        mp['delta_n'] = 0.5*args.ej0/args.ec0 
    else: 
        mp['delta_n'] = args.delta_n 
    
    mp['average_ng'] = args.ave_ng
    mp['noise_Ej'] = args.noise_Ej
    mp['noise_Ec'] = args.noise_Ec

    mp['N'] = args.nuc
    mp['L'] = 3 * mp['N']
    mp['Mcut'] = args.Mcut

    mp['El_start'] = args.El_start 
    mp['El_end'] = args.El_end
    mp['el_num'] = args.Nel 
    mp['phi_num'] = args.Nphi
    
    mp['comments'] = ['']
    mp['filename'] = f'w{week}_n{num}_hhwb_TEST'    

    set_sim_type(mp, args)

    path_name = args.path_name
    mp['path_name'] = args.path_name
    
    #-------------------------------------------------------

    el_tab = np.linspace( mp['Ej0'] * mp['El_start'], mp['Ej0'] * mp['El_end'], 
                          mp['el_num'])
    #el_tab = [1.0]
    phi_tab = np.linspace(0.0, 2.0 * np.pi, mp['phi_num'], endpoint=False)
    
    #-------------------------------------------------------

    data_new = {'el_ls': el_tab, 'phi_ls': phi_tab}

    start = time.perf_counter()

    # ----------------

    hh = hhm() 
    hh.set_lattice_pars(L=mp['L'], pbc=mp['pbc'])
    hh.set_dynamic_pars(omega=mp['omega'], n_p=mp['n_p'], n_t=mp['n_t'])

    curr = np.zeros( (mp['el_num'], mp['phi_num'] ) ) 
    etspec = np.zeros( (mp['el_num'], mp['phi_num'] ) ) 

    for j, el in tqdm( enumerate(el_tab), desc="El" ):

        for k, ph in tqdm( enumerate(phi_tab), desc="ph" ):

     
            # set values to the Hamiltonian parameters  
            hh.set_Htpars(El=el, phase=ph, delta_n=mp['delta_n'], n_ave=mp['average_ng'], 
                          Ej0=mp['Ej0'], Ec0=mp['Ec0'], noise_Ej=mp['noise_Ej'], 
                          noise_Ec=mp['noise_Ec'], Mcut=mp['Mcut'] )

            # initialize qobs class
            hhqob = ojja_qobs(hh)

            if args.qinf: 

                hhqob.create_floquet()
                q = hhqob.q_floquet()
                curr[j][k] = q
                print(f'Charge in the infinite time limit :{q}')
            if args.et:
            
                res = hhqob.instantaneous_energies()
                #eg = hhqob.find_gap()
                data_new['time'] = res['time'] 
                etspec[j][k] = res['inst_energies']

            if args.qephi:

                continue

    if args.qinf: 
        data_new['current'] = curr
    if args.et: 
        data_new['energies'] = etspec 
    #if args.qephi:
        #continue

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

    print("--> Data are saved!") 

    #-------------------------------------------------------
    
    print("------------------------------------------------")
    print("--> End of simulations!")
    print(f"Run time (in secs): {dur:.2f}") 
    print("------------------------------------------------")

    #-------------------------------------------------------

#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
