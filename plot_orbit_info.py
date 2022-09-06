#!/usr/bin/env python3

"""

    ========================
    = Integrating subhalos =
    ========================

    Integrate subhalos in custom potential
        - Disk (radial and vertical) model
        - DM halo model

"""

# Import packages
from galpy.orbit import Orbit
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
from astropy import units as u
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='mac')
host = 2
#
if sim_data.num_gal == 1:
    data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy)
elif sim_data.num_gal == 2:
    if host == 1:
        data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_1)
    if host == 2:
        data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_2)

# Loop over the number of subhalos
for i in range(0, len(data['indices.z0'][:,0])):
    # Check to see if the subhalo fell in and experienced a pericenter
    if (data['infall.check'][i]):# & (data['pericenter.check.sim'][i]):
        #
        d_model = data['d.tot.model'][i]
        v_model = data['v.tot.model'][i]
        L_model = np.linalg.norm(data['L.model'][i], axis=1)
        #
        # Set up the distances and times to plot
        d_mask = (data['d.tot.sim'][i] >= 0)
        d_data = data['d.tot.sim'][i][d_mask]
        #
        lookback_time = np.flip(data['time.sim'][-1] - data['time.sim'])
        times = lookback_time[:len(d_data)]
        times_model = lookback_time
        #
        v_data = data['v.tot.sim'][i][:len(times)]
        L_data = data['L.tot.sim'][i][:len(times)]
        #
        # Set up the figure
        plt.rcParams["font.family"] = "serif"
        plt.figure(figsize=(10, 12))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312, sharex=ax1)
        ax3 = plt.subplot(313, sharex=ax2)
        #
        # Plot the distances
        ax1.plot(times, d_data, 'k', label='Simulation')
        ax1.plot(times_model, d_model, label='Model', alpha=0.5)
        ax1.plot(times, data['host.radius'][:len(times)], 'k', alpha=0.3) # NEED TO CHECK IF I SHOULD DIVIDE BY MASS RATIO
        ax1.set_xlim(times[-1], times[0])
        #
        # Check to see if there were infall, pericenter, or apocenter events
        infall = data['infall.check'][i]
        peri = data['pericenter.check.sim'][i]
        #
        # If there were, plot when they occurred
        if infall:
            infall_time = data['first.infall.time.lb'][i]
            ax1.axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
        #
        if peri:
            for j in data['pericenter.time.lb.sim'][i][data['pericenter.time.lb.sim'][i] != -1]:
                ax1.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
        #
        # Set the labels and save the figure
        ax1.set_ylim(top=np.nanmax(d_data)+100)
        ax1.label_outer()
        ax1.set_ylabel('Host Distance [kpc]', fontsize=20)
        ax1.legend(prop={'size': 16})
        #
        # Plot the velocity data
        ax2.plot(times, v_data, 'k')
        ax2.plot(times_model, v_model, alpha=0.5)
        ax2.set_xlim(times[-1], times[0])
        ax2.label_outer()
        if infall == True:
            infall_time = data['first.infall.time.lb'][i]
            ax2.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
        if peri:
            for j in data['pericenter.time.lb.sim'][i][data['pericenter.time.lb.sim'][i] != -1]:
                ax2.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
        #
        ax2.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=20)
        #
        # Plot the velocity data
        ax3.plot(times, L_data/1000, 'k')
        ax3.plot(times_model, L_model/1000, alpha=0.5)
        ax3.set_xlim(times[-1], times[0])
        ax3.set_ylabel('$\\ell$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
        if infall == True:
            infall_time = data['first.infall.time.lb'][i]
            ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
        if peri:
            for j in data['pericenter.time.lb.sim'][i][data['pericenter.time.lb.sim'][i] != -1]:
                ax3.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
        #
        ax3.set_xlabel('Lookback time [Gyr]', fontsize=32)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        #
        if sim_data.num_gal == 1:
            plt.savefig(sim_data.home_dir+'/orbit_data/plots/subhalo_integration/'+sim_data.galaxy+'/'+sim_data.galaxy+'_sub_'+str(i+1)+'.pdf')
        if sim_data.num_gal == 2:
            if host == 1:
                plt.savefig(sim_data.home_dir+'/orbit_data/plots/subhalo_integration/'+sim_data.gal_1+'/'+sim_data.gal_1+'_sub_'+str(i+1)+'.pdf')
            if host == 2:
                plt.savefig(sim_data.home_dir+'/orbit_data/plots/subhalo_integration/'+sim_data.gal_2+'/'+sim_data.gal_2+'_sub_'+str(i+1)+'.pdf')
        plt.close()
