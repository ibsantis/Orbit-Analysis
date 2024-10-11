#!/usr/bin/python3

"""
    =========================================
    = Paper III Orbit History Distributions =
    =========================================

    Create the multi-panel orbit history PDFs for each satellite.

    This will create a figure that shows the differential and cumulative PDFs of:
        - Infall time
        - Apocenter time and distance (recent)
        - Pericenter time, distance, and velocity (recent and minimum)
        - Disance, radial velocity, and tangential velocity
"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import pandas as pd
import satellite_io
import matplotlib
from matplotlib import pyplot as plt
import time
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
sat_analysis = satellite_io.SatelliteAnalysis(gal1='m12i', location=loc)
#
print('Set paths')

# Read in the snapshot dictionary and the entire tree
lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12n']

mw_sats_1Mpc = ['Antlia 2', 'Aquarius 2', 'Bootes 1', 'Bootes 2', 'Bootes 3', \
                'Canes Venatici 1', 'Canes Venatici 2', 'Carina', 'Carina 2', \
                'Carina 3', 'Cetus 2', 'Cetus 3', 'Columba 1', 'Coma Berenices', \
                'Crater 2', 'DES J0225+0304', 'Draco', 'Draco 2', 'Eridanus 2', \
                'Eridanus 3', 'Fornax', 'Grus 1', 'Grus 2', 'Hercules', \
                'Horologium 1', 'Horologium 2', 'Hydra 2', 'Hydrus 1', 'Indus 1', \
                'Indus 2', 'Leo 1', 'Leo 2', 'Leo 4', 'Leo 5', 'Leo A', 'Leo T', \
                'Pegasus 3', 'Phoenix', 'Phoenix 2', 'Pictor 1', 'Pictor 2', \
                'Pisces 2', 'Reticulum 2', 'Reticulum 3', 'Sagittarius 2', \
                'Sculptor', 'Segue 1', 'Segue 2', 'Sextans 1', 'Triangulum 2', \
                'Tucana', 'Tucana 2', 'Tucana 3', 'Tucana 4', 'Tucana 5', \
                'Ursa Major 1', 'Ursa Major 2', 'Ursa Minor', 'Virgo 1', \
                'Willman 1']


#galaxy = 'Sculptor'
for galaxy in mw_sats_1Mpc:
    #
    satellite_name = galaxy.replace(' ', '_')
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path=sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{satellite_name}.txt')
    #
    if len(gal_data['Host']) == 0:
        continue
    #
    orbit_dictionary = dict()
    orbit_dictionary['first.infall.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.num'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.vel'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.vel'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['apocenter.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['apocenter.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['halo.mass.peak'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['distance'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.rad'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.tan'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['L.tot.sim'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['v.tot.sim'] = np.zeros(gal_data.shape[0])
    #
    for sim_name in galaxies:
        if sim_name in np.array(gal_data['Host']):
            # Read in the mini data and snapshot information
            mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=True)
            snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)
            #
            orbit_history = sat_analysis.orbit_property_distribution(sim_name, mini_data, gal_data, snaps)
            mask = np.where(sim_name == gal_data['Host'])[0]
            for key in orbit_history.keys():
                orbit_dictionary[key][mask] = orbit_history[key]
    if len(orbit_history['distance']) == 0:
        continue
    #
    # Plot the orbit history cumulative
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(4, 3, figsize=(16,12))
    #
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.infall', 0.25)
        axs[0,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[0,0].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,0].set_xlabel('Lookback infall time [Gyr]', fontsize=14)
    axs[0,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.apo', 0.25)
        axs[0,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[0,1].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,1].set_xlabel('Lookback apocenter time [Gyr]', fontsize=14)
    axs[0,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.apo', 5)
        axs[0,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[0,2].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,2].set_xlabel('Apocenter distance [kpc]', fontsize=14)
    axs[0,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.peri', 0.25)
        axs[1,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,0].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,0].set_xlabel('Lookback recent pericenter time [Gyr]', fontsize=14)
    axs[1,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
        axs[1,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,1].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,1].set_xlabel('Recent pericenter distance [kpc]', fontsize=14)
    axs[1,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
        axs[1,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,2].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,2].set_xlabel('Recent pericenter velocity [km s$^{-1}$]', fontsize=14)
    axs[1,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.peri', 0.25)
        axs[2,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,0].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,0].set_xlabel('Lookback minimum pericenter time [Gyr]', fontsize=14)
    axs[2,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
        axs[2,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,1].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,1].set_xlabel('Minimum pericenter distance [kpc]', fontsize=14)
    axs[2,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
        axs[2,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,2].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,2].set_xlabel('Minimum pericenter velocity [km s$^{-1}$]', fontsize=14)
    axs[2,2].tick_params(axis='both', which='major', labelsize=12)
    #
    x = orbit_dictionary['distance']
    binss, half_binss = sat_analysis.binning_scheme(x, 'd.z0', 5)
    axs[3,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[3,0].hist(x, binss, density=False, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,0].set_xlabel('Distance [kpc]', fontsize=14)
    axs[3,0].tick_params(axis='both', which='major', labelsize=12)
    #
    x = orbit_dictionary['velocity.rad']
    binss, half_binss = sat_analysis.binning_scheme(x, 'v.rad', 1)
    axs[3,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[3,1].hist(x, binss, density=False, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,1].set_xlabel('Radial velocity [km s$^{-1}$]', fontsize=14)
    axs[3,1].tick_params(axis='both', which='major', labelsize=12)
    #
    x = orbit_dictionary['velocity.tan']
    binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 5)
    axs[3,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[3,2].hist(x, binss, density=True, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,2].set_xlabel('Tangential velocity [km s$^{-1}$]', fontsize=14)
    axs[3,2].tick_params(axis='both', which='major', labelsize=12)
    #
    plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=14)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/cumulative/'+satellite_name+'_history_cumulative.pdf')
    plt.close()
