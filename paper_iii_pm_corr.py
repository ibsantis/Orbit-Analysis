#!/usr/bin/python3

"""
    ========================================
    = Paper III Proper motion correlations =
    ========================================

    Create the multi-panel plots of various orbit properties 
    as a function of proper motions (tangential velocity).

    Includes both the scatter plot of values, as well as the
    median and 68% scatter about the median.
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

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']

mw_sats = ['HIZSS 3(A)', 'HIZSS 3B', 'NGC 55', 'LMC', 'SMC', 'IC 4662', 'IC 5152', 'NGC 6822', 'NGC 3109', 'IC 3104', \
           'Sextans B', 'DDO 190', 'DDO 125', 'Sextans A', 'NGC 4163', 'Sagittarius dSph', 'UGC 8508', 'Fornax', 'UGC 4879', \
           'UGC 9128', 'GR 8', 'Leo A', 'Leo 1', 'Sagittarius dIrr', 'ESO 294-G010', 'DDO 113', 'Sculptor', 'Antlia 2', 'Aquarius (DDO 210)',\
           'Phoenix', 'Leo 2', 'Antlia B', 'Tucana', 'KKR 3', 'Carina', 'Leo P', 'Crater 2', 'Ursa Minor', 'Sextans 1', \
           'Draco', 'Canes Venatici 1', 'Leo T', 'Eridanus 2', 'Bootes 1', 'Hercules', 'Bootes 3', 'Sagittarius 2', \
           'Canes Venatici 2', 'Ursa Major 1', 'Leo 4', 'Hydra 2', 'Hydrus 1', 'Carina 2', 'Ursa Major 2', 'Aquarius 2', \
           'Indus 2', 'Coma Berenices', 'Leo 5', 'Pisces 2', 'Columba 1', 'Tucana 5', 'Pegasus 3', 'Grus 2', 'Tucana 2', \
           'Reticulum 2', 'Horologium 1', 'Pictor 1', 'Tucana 4', 'Indus 1', 'Grus 1', 'Reticulum 3', 'Pictor 2', 'Bootes 2',\
           'Willman 1', 'Phoenix 2', 'Cetus 3', 'Carina 3', 'Eridanus 3', 'Segue 2', 'Triangulum 2', 'Horologium 2', 'Tucana 3',\
           'Segue 1', 'DES J0225+0304', 'Virgo 1', 'Draco 2', 'Cetus 2']

# Proper motion plots
for galaxy in mw_sats:
    #
    gal_data = sat_analysis.read_subhalo_matches(galaxy)
    satellite_name = galaxy.replace(' ', '_')
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
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(4, 3, figsize=(16,16))
    #
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['first.infall.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='t.infall', bins=binss)
        axs[0,0].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[0,0].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[0,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[0,0].set_ylabel('Lookback infall time [Gyr]', fontsize=12)
    axs[0,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['apocenter.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='t.apo', bins=binss)
        axs[0,1].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[0,1].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[0,1].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[0,1].set_ylabel('Lookback apocenter time [Gyr]', fontsize=12)
    axs[0,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['apocenter.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='d.apo', bins=binss)
        axs[0,2].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[0,2].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[0,2].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[0,2].set_ylabel('Apocenter distance [kpc]', fontsize=12)
    axs[0,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['pericenter.rec.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='t.peri.rec', bins=binss)
        axs[1,0].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[1,0].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[1,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[1,0].set_ylabel('Lookback recent pericenter time [Gyr]', fontsize=12)
    axs[1,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['pericenter.rec.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='d.peri.rec', bins=binss)
        axs[1,1].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[1,1].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[1,1].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[1,1].set_ylabel('Recent pericenter distance [kpc]', fontsize=12)
    axs[1,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['pericenter.rec.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='v.peri.rec', bins=binss)
        axs[1,2].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[1,2].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[1,2].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[1,2].set_ylabel('Recent pericenter velocity [km s$^{-1}$]', fontsize=12)
    axs[1,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['pericenter.min.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='t.peri.min', bins=binss)
        axs[2,0].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[2,0].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[2,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[2,0].set_ylabel('Lookback minimum pericenter time [Gyr]', fontsize=12)
    axs[2,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['pericenter.min.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='d.peri.min', bins=binss)
        axs[2,1].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[2,1].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[2,1].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[2,1].set_ylabel('Minimum pericenter distance [kpc]', fontsize=12)
    axs[2,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['pericenter.min.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='v.peri.min', bins=binss)
        axs[2,2].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[2,2].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[2,2].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[2,2].set_ylabel('Minimum pericenter velocity [km s$^{-1}$]', fontsize=12)
    axs[2,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['velocity.tan'][m]
        y = orbit_dictionary['pericenter.num'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 50)
        med, upper, lower, highest, lowest = sat_analysis.median_and_scatter(x, y, xtype='v.tan', ytype='N.peri', bins=binss)
        axs[3,0].fill_between(binss[:-1]+half_binss, upper, lower, color='b', alpha=0.15)
        axs[3,0].plot(binss[:-1]+half_binss, med, color='b', markersize=10, alpha=0.6)
        axs[3,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.3)
    axs[3,0].set_ylabel('Pericenter Number', fontsize=12)
    axs[3,0].tick_params(axis='both', which='major', labelsize=12)
    #
    axs[3,1].axison = False
    axs[3,2].axison = False
    axs[3,0].set_xlabel('Proper Motion [km s$^{-1}$]', fontsize=14)
    axs[2,1].set_xlabel('Proper Motion [km s$^{-1}$]', fontsize=14)
    axs[2,2].set_xlabel('Proper Motion [km s$^{-1}$]', fontsize=14)
    plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=14)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/proper_motion_corr/'+satellite_name+'_vtans.pdf')
    plt.close()
