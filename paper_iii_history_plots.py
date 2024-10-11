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
    # Plot the orbit history histograms
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(4, 3, figsize=(16,12))
    #
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.infall', 0.25)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[0,0].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=0.25, color='k', alpha=0.4, edgecolor=None)
        #axs[0,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[0,0].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[0,0].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[0,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[0,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,0].set_xlabel('Lookback infall time [Gyr]', fontsize=18)
    axs[0,0].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.apo', 0.25)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[0,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=0.25, color='k', alpha=0.4, edgecolor=None)
        #axs[0,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[0,1].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[0,1].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[0,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[0,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,1].set_xlabel('Lookback apocenter time [Gyr]', fontsize=18)
    axs[0,1].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.apo', 5)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[0,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
        #axs[0,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[0,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[0,2].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[0,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[0,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,2].set_xlabel('Apocenter distance [kpc]', fontsize=18)
    axs[0,2].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.peri', 0.25)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[1,0].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=0.25, color='k', alpha=0.4, edgecolor=None)
        #axs[1,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[1,0].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[1,0].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[1,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,0].set_xlabel('Lookback recent pericenter time [Gyr]', fontsize=18)
    axs[1,0].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[1,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
        #axs[1,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[1,1].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[1,1].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[1,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,1].set_xlabel('Recent pericenter distance [kpc]', fontsize=18)
    axs[1,1].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[1,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=10, color='k', alpha=0.4, edgecolor=None)
        #axs[1,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[1,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[1,2].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[1,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,2].set_xlabel('Recent pericenter velocity [km s$^{-1}$]', fontsize=18)
    axs[1,2].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.peri', 0.25)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[2,0].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=0.25, color='k', alpha=0.4, edgecolor=None)
        #axs[2,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[2,0].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[2,0].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[2,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,0].set_xlabel('Lookback minimum pericenter time [Gyr]', fontsize=18)
    axs[2,0].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[2,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
        #axs[2,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[2,1].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[2,1].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[2,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,1].set_xlabel('Minimum pericenter distance [kpc]', fontsize=18)
    axs[2,1].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[2,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=10, color='k', alpha=0.4, edgecolor=None)
        #axs[2,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[2,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[2,2].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[2,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,2].set_xlabel('Minimum pericenter velocity [km s$^{-1}$]', fontsize=18)
    axs[2,2].tick_params(axis='both', which='major', labelsize=14)
    #
    # Blank subpanel
    axs[3,0].axison = False
    #
    m = (orbit_dictionary['pericenter.num'] != -1)
    x = orbit_dictionary['pericenter.num']
    binss, half_binss = sat_analysis.binning_scheme(x, 'N.peri', 1)
    p = np.histogram(x, binss, density=True, weights=gal_data['Weight'])
    axs[3,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=1, color='k', alpha=0.4, edgecolor=None)
    x_mean = np.sum(orbit_dictionary['pericenter.num']*gal_data['Weight'])/np.sum(gal_data['Weight'])
    y_mean = 1.1
    std = np.sqrt(np.sum((orbit_dictionary['pericenter.num']-x_mean)**2*gal_data['Weight'])/np.sum(gal_data['Weight'])/np.sum(gal_data['Weight'][m]))
    axs[3,1].errorbar(x_mean, y_mean, xerr=std, color='k', lw=3.5, capsize=0)
    axs[3,1].scatter(x_mean, y_mean, s=75, marker='s', c='k')
    axs[3,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[3,1].hist(x, binss, density=True, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,1].set_xlabel('Number of pericentric passages', fontsize=18)
    axs[3,1].tick_params(axis='both', which='major', labelsize=14)
    #
    # Blank subpanel
    axs[3,2].axison = False
    #
    plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=20)
    plt.tight_layout()
    #plt.show()
    #plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/both_hist/'+satellite_name+'_history_both.pdf')
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/'+satellite_name+'_history_both.pdf')
    plt.close()

