#------------------------------------------------------------------------------
# plot_qephi.py
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# import libraries
#------------------------------------------------------------------------------

import sys
import pickle
import numpy as np 

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

#------------------------------------------------------------------------------
# main 
#------------------------------------------------------------------------------

if __name__ == "__main__":
    
    #------------------------------------------------------

    week = 70 
    code, nf = 815, 9 
    file_type = 'osjja_floqenerg_varyphi'

    # load data (work in progress - to be optimized)   
    #data_ls = combine_dat(week, code, nf=nf, file_type=file_type)   
    #x, y, yc = prepare_xy_qe(data_ls)

    # load data (DYI) in the following form: 
    # x : phi range ( times N_phi ) 
    # y : quasienergies
    # yc : overlaps in FLoquet basis
    


    # -----------


    # for the dark color points to appear on top of the bright ones  
    lim = 0.8 
    indsA = np.where( yc < lim) 
    xA, yA, ycA = x[indsA], y[indsA], yc[indsA]
    indsB = np.where( yc > lim )
    xB, yB, ycB = x[indsB], y[indsB], yc[indsB]

    #print(data_ls[0]['model_pars'])
    xt = [0.0, np.pi, 2.0*np.pi]
    xtl = [0, r"$\pi$", r"$2\pi$"]

    plt.rcParams["figure.figsize"] = (6,4)
    fig, ax = plt.subplots() 
    
    #scat = ax.scatter( x, y, c=yc, cmap='OrRd', s=2, vmin=0.0, vmax=1.0)
    scat = ax.scatter( xA, yA, c=ycA, cmap='OrRd', s=2, vmin=0.0, vmax=1.0)
    ax.scatter( xB, yB, c=ycB, cmap='OrRd', s=2, vmin=0.0, vmax=1.0)

    ax.set_xlabel( r'$\phi$', fontsize=23 )
    ax.set_ylabel( r'$\tau \mathcal{E}_{\nu} $', fontsize=23 ) 
    ax.set_xlim( -0.1, 2.0 * np.pi + 0.1 )
    ax.set_xticks( xt )
    ax.set_xticklabels( xtl )
    ax.set_yticks( [-0.025, 0.0, 0.025] )
    ax.set_yticklabels( xtl) 

    ax.tick_params( labelsize=20 )

    cbar = plt.colorbar(scat)
    cbar.set_label( r'$ \mathcal{N}_{\nu} $', rotation=0, labelpad=12, 
                    fontsize=23) 
    cbar.ax.tick_params(labelsize=20)
    #fig.savefig( 'rm_qephi_L4_10dE7.pdf' )  # 800, 9 
    #fig.savefig( 'rm_qephi_L6_10dE7.pdf' )  # 801, 9
    #fig.savefig( 'rm_qephi_L8_10dE7.pdf' )  # 802, 9
    #fig.savefig( 'rm_qephi_L6_10dE12.pdf' )  # 813, 9
    fig.savefig( 'rm_qephi_L6_10dE5.pdf' )  # 815, 9
    fig.savefig( 'test.pdf' )
 
    #------------------------------------------------------


#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
