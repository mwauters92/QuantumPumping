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

parser.add_argument('--code', metavar='', required=False, type=int, 
                    help='Set a code name to the file.')

# to be removed?
parser.add_argument('--Mcut', metavar='', default=2, type=int, 
                    help=('Truncation of the local Hilbert space. Mcut=2 for\
                    hard core bosons.') )
parser.add_argument('--num_periods', metavar='', default=1, type=int, 
                    help='number of periods used in the time evolution.')
parser.add_argument('--nt', metavar='', default=301, type=int, 
                    help='Number of values of the time between 0 and 2pi.')
parser.add_argument('--n_sites', metavar='', default=6, type=int, 
                    help='The number of sites.')
parser.add_argument('--PBC', action='store_true',  
                    help='Periodic boundary conditions')

parser.add_argument('--ej0', metavar='', default=0.2, type=float, 
                    help=('Josephson coupling. Can be either a single variable\
                    or a list of length L-1.') )
parser.add_argument('--ec0', metavar='', default=1.0, type=float, 
                    help=('The on-site charging energy. Can be either a single\
                    variable or a list of length L.'))
parser.add_argument('--U', metavar='', default=0.0, type=float, 
                    help=('The nearest neighbour interaction strength.'))

parser.add_argument('--delta_n', metavar='', default=None, type=float, 
                    help=('The amplitude of the oscillations in the induced\
                    charge.') )
parser.add_argument('--ave_ng', metavar='', default=0.5, type=float, 
                    help=('The average value of the induced charge, sets\
                    the chemical potential.') )


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

parser.add_argument('--path_name', metavar='', type=str, default='data', 
                    help='Path to the folder where to save the results.') 

args = parser.parse_args()

#------------------------------------------------------------------------------
# simulation type 
#------------------------------------------------------------------------------

def set_sim_type(mp, args):
    
    if args.qinf:
        mp['sim_type'] = 'Pumped charge at the infinite time.'
        filename_ext = 'qinf'
    if args.et:
        mp['sim_type'] = 'Instantaneous many-body spectrum.'
        filename_ext = 'specEt'
    if args.qephi:
        mp['sim_type'] = 'Many-body quasi-energy spectrum.'
        filename_ext = 'specQEphi'

    return filename_ext

#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------

if __name__ == "__main__":

    #-------------------------------------------------------
    
    code = args.code 
    to_save = True      # whether to save data or not  

    mp = {'omega': args.omega, 'n_p': args.num_periods, 'n_t': args.nt, 
          'pbc': args.PBC} 

    mp['Ej0'] = args.ej0  
    mp['Ec0'] = args.ec0
    mp['U'] = args.U

    if args.delta_n is None:
        mp['delta_n'] = 0.5 * args.ej0 / args.ec0 
    else: 
        mp['delta_n'] = args.delta_n 
    
    mp['average_ng'] = args.ave_ng
    mp['noise_Ej'] = args.noise_Ej
    mp['noise_Ec'] = args.noise_Ec

    mp['L'] = args.n_sites 
    mp['Mcut'] = args.Mcut

    mp['El_start'] = args.El_start 
    mp['El_end'] = args.El_end
    mp['el_num'] = args.Nel 
    mp['phi_num'] = args.Nphi
    
    mp['comments'] = ['']
    filename_ext = set_sim_type(mp, args)

    mp['filename'] = f'hhwb_{filename_ext}_{code}'    

    path_name = args.path_name
    mp['path_name'] = args.path_name
    
    #-------------------------------------------------------

    ej0 = mp['Ej0']

    # if the number of El values is 1, then the el_tab list contains
    # only the [ej0*El_start] 
    if mp['el_num'] == 1:
        el_tab = [ ej0 * mp['El_start'] ]
    else:
        el_tab = np.linspace( ej0 * mp['El_start'], ej0 * mp['El_end'], 
                              mp['el_num'])
    phi_tab = np.linspace(0.0, 2.0 * np.pi, mp['phi_num'], endpoint=False)
    
    #-------------------------------------------------------

    data_new = {'el_ls': el_tab, 'phi_ls': phi_tab}

    start = time.perf_counter()

    # ----------------

    hh = HH_model() 
    hh.set_lattice_pars(L=mp['L'], pbc=mp['pbc'])
    hh.set_dynamic_pars(omega=mp['omega'], n_p=mp['n_p'], n_t=mp['n_t'])

    curr = np.zeros( (mp['el_num'], mp['phi_num'] ) ) 
    etspec = np.zeros( (mp['el_num'], mp['phi_num'] ) ) 
    qespec_fen_ls = [] 
    qespec_focc_ls = [] 

    for j, el in tqdm( enumerate(el_tab), desc="El" ):
    
        qespec_fen_ls.append( [] ) 
        qespec_focc_ls.append( [] ) 
        for k, ph in tqdm( enumerate(phi_tab), desc="ph" ):

            # set values to the Hamiltonian parameters  
            hh.set_Htpars(El=el, phase=ph, delta_n=mp['delta_n'], 
                          n_ave=mp['average_ng'], Ej0=mp['Ej0'], 
                          Ec0=mp['Ec0'], noise_Ej=mp['noise_Ej'], 
                          noise_Ec=mp['noise_Ec'], U=mp['U'],
                          Mcut=mp['Mcut'])

            # initialize the dynamics class
            hhqob = Qdynamics(hh)

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

                felements = hhqob.create_floquet(return_floquet_elements=True)
                f_occ = hhqob.floquet_projection()
                qespec_fen_ls[j].append( felements['f_energies'] )
                qespec_focc_ls[j].append( f_occ )

    if args.qinf: 
        data_new['current'] = curr
    if args.et: 
        data_new['energies'] = etspec 
    if args.qephi:
        data_new['f_energies'] = qespec_fen_ls
        data_new['f_occ'] = qespec_focc_ls

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
