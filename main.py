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

parser.add_argument('--model', metavar='', default='HH', type=str,
                    help=' <HH> for Harper-Hofstadter or <RM> for RiceMele model')
parser.add_argument('--code', metavar='', type=int, 
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

parser.add_argument('--ej0', metavar='', default=1.0, type=float, 
                    help=('Josephson coupling. Can be either a single variable\
                    or a list of length L-1.') )

parser.add_argument('--delta_ej', metavar='', default=1.0, type=float, 
                    help=('Modulation of the Josephson coupling for thr RM model') )

parser.add_argument('--ec0', metavar='', default=4.0, type=float, 
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

parser.add_argument('--qpump_finite', action='store_true',
                    help='Compute time evolution over <num_periods> periods')

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
    
    filename_ext = 'JJA_'
    if args.qpump_finite:
        mp['sim_type'] = 'Dynamics for a finite number of periods'
        filename_ext += 'qpumpFinite_'
    if args.qinf:
        mp['sim_type'] = 'Pumped charge at the infinite time.'
        filename_ext += 'qinf_'
    if args.et:
        mp['sim_type'] = 'Instantaneous many-body spectrum.'
        filename_ext += 'specEt_'
    if args.qephi:
        mp['sim_type'] = 'Many-body quasi-energy spectrum.'
        filename_ext += 'specQEphi_'

    return filename_ext

#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------

if __name__ == "__main__":

    #-------------------------------------------------------
    
    code = args.code 
    to_save = True      # whether to save data or not  
    model_type = args.model

    mp = {'omega': args.omega, 'n_p': args.num_periods, 'n_t': args.nt, 
          'pbc': args.PBC} 

    mp['Ej0'] = args.ej0  
    mp['Ec0'] = args.ec0
    mp['U'] = args.U
    mp['dE'] = args.delta_ej
    if args.delta_n is None:
        # this choice it to reproduce the original HH model
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

    mp['filename'] = f'{filename_ext}_{code}'    

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
    if model_type == 'HH':
        quantum_model = HH_model(mp)
    elif model_type == 'RM': 
        quantum_model = RM_model(mp)
    else:
        raise ValueError(f'Model {model_type} is not implemented. Please choose between HH and RM')

    #quantum_model.set_lattice_pars(L=mp['L'], pbc=mp['pbc'])
    #quantum_model.set_dynamic_pars(omega=mp['omega'], n_p=mp['n_p'], n_t=mp['n_t'])

    curr = np.zeros( (mp['el_num'], mp['phi_num'] ) ) 
    
    if args.qpump_finite:
        local_curr = np.zeros( (mp['el_num'], mp['phi_num'], mp['n_p']*mp['n_t']) ) 
        density = np.zeros( (mp['el_num'], mp['phi_num'],mp['L'], mp['n_p']*mp['n_t']) ) 
    
    etspec = []  
    qespec_fen_ls = [] 
    qespec_focc_ls = [] 

    for j, el in tqdm( enumerate(el_tab), desc="El" ):
    
        qespec_fen_ls.append( [] ) 
        qespec_focc_ls.append( [] ) 
        for k, ph in tqdm( enumerate(phi_tab), desc="ph" ):
            mp['El'] = el
            mp['phase'] = ph
            # set values to the Hamiltonian parameters  
            quantum_model.set_Htpars(mp)
            '''delta_n=mp['delta_n'], 
                          n_ave=mp['average_ng'], Ej0=mp['Ej0'], 
                          Ec0=mp['Ec0'], noise_Ej=mp['noise_Ej'], 
                          noise_Ec=mp['noise_Ec'], U=mp['U'],
                          Mcut=mp['Mcut'], delta_ej=mp['dE'])
            '''
            # initialize the dynamics class
            quantum_model_dynamics = Qdynamics(quantum_model)
            
            if args.qpump_finite:
                density[j,k,:,:], local_curr[j,k,:] = quantum_model_dynamics.evolve()  

            if args.qinf or args.qephi:
                
                quantum_model_dynamics.create_floquet()

            if args.qinf: 

                q = quantum_model_dynamics.q_floquet()
                curr[j][k] = q
                print(f'Charge in the infinite time limit :{q}')

            if args.et:
            
                res = quantum_model_dynamics.instantaneous_energies()
                data_new['time'] = res['time'] 
                etspec.append( res['inst_energies'] )
                spectrum_shape = res['inst_energies'].shape
                print(spectrum_shape)
            if args.qephi:

                f_occ = quantum_model_dynamics.floquet_projection()
                qespec_fen_ls[j].append( quantum_model_dynamics.f_energies )
                qespec_focc_ls[j].append( f_occ )

    if args.qinf: 
        data_new['current'] = curr
    if args.et: 
        data_new['energies'] = np.array(etspec).reshape([args.Nel, args.Nphi, spectrum_shape[0], spectrum_shape[1]])
    if args.qephi:
        data_new['f_energies'] = qespec_fen_ls
        data_new['f_occ'] = qespec_focc_ls
    if args.qpump_finite:
        data_new['finite_time_current'] = local_curr
        data_new['local_density'] = density
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

    print("------------------------------------------------")
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
