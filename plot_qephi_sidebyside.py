#------------------------------------------------------------------------------
# plot_qephi_sidebyside.py
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

    file_type = 'osjja_floqenerg_varyphi'
    cons_overlap = True #False

    plt.rcParams["figure.figsize"] = (12,4)
    fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.2,0.24])

    for n, ni in enumerate(codes): 

        # load data 
        #data_ls = combine_dat(week, ni, nf=nf, file_type=file_type)   
        #x, y, yc = prepare_xy_qe(data_ls)


        # load data (DYI) in the following form: 
        # x : phi range ( times N_phi ) 
        # y : quasienergies
        # yc : overlaps in FLoquet basis
        
        # -----------

        # for the dark color points to appear on top of the bright ones  
        if cons_overlap:
            lim = 0.8 
            indsA = np.where( yc < lim) 
            xA, yA, ycA = x[indsA], y[indsA], yc[indsA]
            indsB = np.where( yc > lim )
            xB, yB, ycB = x[indsB], y[indsB], yc[indsB]

        xt = [0.0, np.pi, 2.0*np.pi]
        ytl = [r"$-\pi$", 0, r"$\pi$"]
        xtl = [0, r"$\pi$", r"$2\pi$"]

        if cons_overlap: 
            scat = ax[n].scatter( xA, yA, c=ycA, cmap='OrRd', s=2, vmin=0.0, vmax=1.0)
            ax[n].scatter( xB, yB, c=ycB, cmap='OrRd', s=2, vmin=0.0, vmax=1.0)
        else:
            scat = ax[n].scatter( x, y, c=yc, cmap='OrRd', s=2, vmin=0.0, vmax=1.0)

        ax[n].set_xlabel( r'$\phi$', fontsize=23 )
        ax[n].set_ylabel( r'$\tau \mathcal{E}_{\nu} $', fontsize=23, labelpad=-5) 
        ax[n].set_xlim( -0.1, 2.0 * np.pi + 0.1 )
        ax[n].set_xticks( xt )
        ax[n].set_xticklabels( xtl )
        ax[n].set_yticks( [-0.025, 0.0, 0.025] )
        ax[n].set_yticklabels( ytl )  
        ax[n].tick_params( labelsize=20 )

    ax[0].annotate('(a)', xy=(0.07, 0.87), xycoords='figure fraction', 
                 fontsize=20)

    ax[1].annotate('(b)', xy=(0.46, 0.87), xycoords='figure fraction', 
                 fontsize=20)

    cbar = plt.colorbar(scat)
    cbar.ax.set_title( r'$ \mathcal{N}_{\nu} $', fontsize=23, pad=15) 
    cbar.ax.tick_params( labelsize=20 )
    fig.savefig( 'rm_qephi_double.pdf' )  # 70, 813-815, 9
    fig.savefig( 'test.pdf' )
 
    #------------------------------------------------------


#------------------------------------------------------------------------------
# end of .py file
#------------------------------------------------------------------------------
