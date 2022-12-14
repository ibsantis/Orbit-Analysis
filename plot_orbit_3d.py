#!/usr/bin/env python3

"""

    ======================
    = Plotting 3D orbits =
    ======================

    Make a 2-panel plot showing the 3D orbit
        - X vs Y
        - X vs Z

    Also, plot each panel individually to see only 2 components at a time

    Orbit lines are color-coded based on lookback time!

"""

# Import the necesary packages
from galpy.orbit import Orbit
import orbit_io
import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12b', location='mac')
#sim_data.galaxy = 'Remus'
print('Set paths')


# Read in the data
data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy)


# Set up variables outside of the loop
R200m = data['host.radius'][0]
tlb = data['time.sim'][-1] - np.flip(data['time.sim'])


"""
    Make the two panel plots showing:
        - X vs Y
        - X vs Z
"""
# Loop through each of the satellites
for i in range(0, len(data['indices.z0'][:,0])):
    #
    # Only plot the ones that have fallen into the host
    if data['infall.check'][i]:
        #
        # Set up the figure and axis labels
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        ax[0].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
        ax[0].set_xlabel('X [kpc]', fontsize=28)
        ax[0].set_ylabel('Y [kpc]', fontsize=28)
        ax[1].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
        ax[1].set_xlabel('X [kpc]', fontsize=28)
        ax[1].set_ylabel('Z [kpc]', fontsize=28)
        #
        # Set up the plotting data and lookback time array
        xx = data['d.sim'][i][:,0]
        yy = data['d.sim'][i][:,1]
        zz = data['d.sim'][i][:,2]
        #
        # Only want to plot post infall to make things a bit cleaner
        mask = (tlb[:len(xx)] < data['first.infall.time.lb'][i])
        #
        prop = cm.viridis((tlb-np.min(tlb))/(np.max(tlb)-np.min(tlb)))
        #
        # Loop through the orbit
        for j in np.arange(len(xx[mask])-1):
            ax[0].plot([xx[mask][j], xx[mask][j+1]], [yy[mask][j], yy[mask][j+1]], c=prop[j])
            ax[1].plot([xx[mask][j], xx[mask][j+1]], [zz[mask][j], zz[mask][j+1]], c=prop[j])
        #
        # Set up the tick parameters
        ax[0].tick_params(axis='both', which='both', bottom=True, labelsize=20)
        ax[1].tick_params(axis='both', which='both', bottom=True, labelsize=20)
        #
        # Plot the colorbar and save the figure
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=tlb.min(), vmax=tlb.max()))
        plt.colorbar(sm, label='Lookback time [Gyr]')
        fig.suptitle('{0}: satellite {1}'.format(sim_data.galaxy, i+1))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0)
        plt.savefig(sim_data.home_dir+'/orbit_data/plots/subhalo_integration/3d/'+sim_data.galaxy+'/2_panel/'+sim_data.galaxy+'_sat_'+str(i+1)+'.pdf')
        plt.close()


"""
    Make the X vs Y panel only
"""
# Loop through each of the satellites
for i in range(0, len(data['indices.z0'][:,0])):
    #
    # Only plot the ones that have fallen into the host
    if data['infall.check'][i]:
        #
        # Set up the figure and axis labels
        fig, ax = plt.subplots(1, 1, figsize=(10,8))
        ax.set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
        ax.set_xlabel('X [kpc]', fontsize=28)
        ax.set_ylabel('Y [kpc]', fontsize=28)
        #
        # Set up the plotting data and lookback time array
        xx = data['d.sim'][i][:,0]
        yy = data['d.sim'][i][:,1]
        #
        # Only want to plot post infall to make things a bit cleaner
        mask = (tlb[:len(xx)] < data['first.infall.time.lb'][i])
        #
        prop = cm.viridis((tlb-np.min(tlb))/(np.max(tlb)-np.min(tlb)))
        #
        # Loop through the orbit
        for j in np.arange(len(xx[mask])-1):
            ax.plot([xx[mask][j], xx[mask][j+1]], [yy[mask][j], yy[mask][j+1]], c=prop[j])
        #
        # Set up the tick parameters
        ax.tick_params(axis='both', which='both', bottom=True, labelsize=24)
        plt.title('{0}: satellite {1}, z=0 index = {2}'.format(sim_data.galaxy, i+1, data['indices.z0'][:,0][i]))
        #
        # Plot the colorbar and save the figure
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=tlb.min(), vmax=tlb.max()))
        #plt.colorbar(sm, label='Lookback time [Gyr]')
        #
        cbar_ax = fig.add_axes([0.23, 0.2, 0.3, 0.035]) #left, bottom, width, height
        cb = fig.colorbar(sm, cax=cbar_ax, ticks = [tlb.min(), 2, 4, 6, 8, 10, 12, 13.78],  orientation = 'horizontal')
        cb.ax.minorticks_off()
        cb.ax.set_xticklabels(['0', '2', '4', '6', '8', '10', '12', '13.8'], fontsize=14) # Used to make custom labels on the tick marks themselves
        cb.ax.set_title(r"Lookback time [Gyr]", fontsize=16)
        #
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0)
        plt.savefig(sim_data.home_dir+'/orbit_data/plots/subhalo_integration/3d/'+sim_data.galaxy+'/x_vs_y/'+sim_data.galaxy+'_sat_'+str(i+1)+'.pdf')
        plt.close()


"""
    Make the X vs Y panel only
"""
# Loop through each of the satellites
for i in range(0, len(data['indices.z0'][:,0])):
    #
    # Only plot the ones that have fallen into the host
    if data['infall.check'][i]:
        #
        # Set up the figure and axis labels
        fig, ax = plt.subplots(1, 1, figsize=(10,8))
        ax.set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
        ax.set_xlabel('X [kpc]', fontsize=28)
        ax.set_ylabel('Z [kpc]', fontsize=28)
        #
        # Set up the plotting data and lookback time array
        xx = data['d.sim'][i][:,0]
        zz = data['d.sim'][i][:,2]
        #
        # Only want to plot post infall to make things a bit cleaner
        mask = (tlb[:len(xx)] < data['first.infall.time.lb'][i])
        #
        prop = cm.viridis((tlb-np.min(tlb))/(np.max(tlb)-np.min(tlb)))
        #
        # Loop through the orbit
        for j in np.arange(len(xx[mask])-1):
            ax.plot([xx[mask][j], xx[mask][j+1]], [zz[mask][j], zz[mask][j+1]], c=prop[j])
        #
        # Set up the tick parameters
        ax.tick_params(axis='both', which='both', bottom=True, labelsize=24)
        plt.title('{0}: satellite {1}, z=0 index = {2}'.format(sim_data.galaxy, i+1, data['indices.z0'][:,0][i]))
        #
        # Plot the colorbar and save the figure
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=tlb.min(), vmax=tlb.max()))
        #plt.colorbar(sm, label='Lookback time [Gyr]')
        #
        cbar_ax = fig.add_axes([0.23, 0.2, 0.3, 0.035]) #left, bottom, width, height
        cb = fig.colorbar(sm, cax=cbar_ax, ticks = [tlb.min(), 2, 4, 6, 8, 10, 12, 13.78],  orientation = 'horizontal')
        cb.ax.minorticks_off()
        cb.ax.set_xticklabels(['0', '2', '4', '6', '8', '10', '12', '13.8'], fontsize=14) # Used to make custom labels on the tick marks themselves
        cb.ax.set_title(r"Lookback time [Gyr]", fontsize=16)
        #
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0)
        plt.savefig(sim_data.home_dir+'/orbit_data/plots/subhalo_integration/3d/'+sim_data.galaxy+'/x_vs_z/'+sim_data.galaxy+'_sat_'+str(i+1)+'.pdf')
        plt.close()
