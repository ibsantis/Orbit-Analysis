#!/usr/bin/python3
# pyright: reportOperatorIssue=false

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

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12n', 'm12q', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']

mw_sats_1Mpc =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
                    'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
                    'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
                    'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus III', 'Eridanus IV', \
                    'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
                    'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
                    'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
                    'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
                    'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sagittarius II', 'Sculptor', 'Segue 1', \
                    'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
                    'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
                    'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']


n_555 = [12, 330, 378, 83, 131, 3, 2, 305, 42, 213, 15, 13, 75, 89, 125, 5, 243, 153, 43, 13, 26, 57, 4, 125, 4, 99, 99, 111, 163, 301, 212, 49, 52, 0, 25, 167, 255, 347, 18, 67, 517, 362, 178, 13, 387, 127, 77, 363, 104, 954, 2, 107, 4, 103, 20, 12, 42, 17, 15, 151, 0, 183, 113, 193, 261, 20, 62, 87, 45, 53]

#galaxy = 'Sculptor'
for sat_idx, galaxy in enumerate(mw_sats_1Mpc):
    #
    satellite_name = galaxy.replace(' ', '_')
    if n_555[sat_idx] < 10:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_10_10_10/weights_{satellite_name}.txt'
    else:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
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
    reionization_distances = np.zeros(gal_data.shape[0])
    for sim_name in galaxies:
        if sim_name in np.array(gal_data['Host']):
            # Read in the mini data and snapshot information
            mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=True)
            snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)
            #
            orbit_history = sat_analysis.orbit_property_distribution(sim_name, mini_data, gal_data, snaps)
            reionDists = sat_analysis.reionization_distance(sim_name, mini_data, gal_data, snaps)
            mask = np.where(sim_name == gal_data['Host'])[0]
            for key in orbit_history.keys():
                orbit_dictionary[key][mask] = orbit_history[key]
            reionization_distances[mask] = reionDists[mask]
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
    axs[0,0].set_xlabel('$t_{\\rm lookback}$ of infall [Gyr]', fontsize=18)
    axs[0,0].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 't.apo', 0.25)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[3,0].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=0.25, color='k', alpha=0.4, edgecolor=None)
        #axs[0,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[3,0].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[3,0].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[3,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[3,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,0].set_xlabel('$t_{\\rm lookback}$ of apocenter [Gyr]', fontsize=18)
    axs[3,0].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.apo', 5)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[2,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
        #axs[0,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[2,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[2,2].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[2,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,2].set_xlabel('Apocenter distance [kpc]', fontsize=18)
    axs[2,2].tick_params(axis='both', which='major', labelsize=14)
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
    axs[1,0].set_xlabel('$t_{\\rm lookback}$ of recent pericenter [Gyr]', fontsize=18)
    axs[1,0].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[0,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
        #axs[1,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[0,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[0,2].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[0,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[0,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,2].set_xlabel('Recent pericenter distance [kpc]', fontsize=18)
    axs[0,2].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[1,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=10, color='k', alpha=0.4, edgecolor=None)
        #axs[1,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[1,1].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[1,1].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[1,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,1].set_xlabel('Recent pericenter velocity [km s$^{-1}$]', fontsize=18)
    axs[1,1].tick_params(axis='both', which='major', labelsize=14)
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
    axs[2,0].set_xlabel('$t_{\\rm lookback}$ of minimum pericenter [Gyr]', fontsize=18)
    axs[2,0].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[1,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
        #axs[2,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[1,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[1,2].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[1,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[1,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[1,2].set_xlabel('Minimum pericenter distance [kpc]', fontsize=18)
    axs[1,2].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[2,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=10, color='k', alpha=0.4, edgecolor=None)
        #axs[2,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[2,1].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[2,1].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[2,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[2,1].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[2,1].set_xlabel('Minimum pericenter velocity [km s$^{-1}$]', fontsize=18)
    axs[2,1].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (reionization_distances >= 0)
    if np.sum(m) != 0:
        x = reionization_distances[m]
        binss, half_binss = sat_analysis.binning_scheme(x, 'z.dist', 80)
        p = np.histogram(x, binss, density=True, weights=gal_data['Weight'][m])
        axs[3,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=80, color='k', alpha=0.4, edgecolor=None)
        #axs[3,0].hist(x, binss, density=True, weights=gal_data['Weight'][m], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        y_med = 1.1
        sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        axs[3,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
        axs[3,2].scatter(x_med, y_med, s=75, marker='s', c='k')
        axs[3,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
        axs[3,2].hist(x, binss, density=True, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,2].set_xlabel('Distance at $z = 7$ [kpc co-moving]', fontsize=18)
    axs[3,2].tick_params(axis='both', which='major', labelsize=14)
    #
    m = (orbit_dictionary['pericenter.num'] != -1)
    x = orbit_dictionary['pericenter.num']
    binss, half_binss = sat_analysis.binning_scheme(x, 'N.peri', 1)
    p = np.histogram(x, binss, density=True, weights=gal_data['Weight'])
    axs[0,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=1, color='k', alpha=0.4, edgecolor=None)
    x_mean = np.sum(orbit_dictionary['pericenter.num']*gal_data['Weight'])/np.sum(gal_data['Weight'])
    y_mean = 1.1
    std = np.sqrt(np.sum((orbit_dictionary['pericenter.num']-x_mean)**2*gal_data['Weight'])/np.sum(gal_data['Weight'])/np.sum(gal_data['Weight'][m]))
    axs[0,1].errorbar(x_mean, y_mean, xerr=std, color='k', lw=3.5, capsize=0)
    axs[0,1].scatter(x_mean, y_mean, s=75, marker='s', c='k')
    axs[0,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[0,1].hist(x, binss, density=True, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[0,1].set_xlabel('Number of pericentric passages', fontsize=18)
    axs[0,1].tick_params(axis='both', which='major', labelsize=14)
    #
    N_analogs = len(gal_data['Weight'])
    galaxy_tex = galaxy.replace(" ", r"\ ")
    info_lines1 = [
        rf"$\mathbf{{{galaxy_tex}}}$: {N_analogs} analogs"
    ]
    sig_d = lg_data[galaxy]['host.distance.total.err']
    sig_vrad = lg_data[galaxy]['host.velocity.rad.err']
    sig_vtan = lg_data[galaxy]['host.velocity.tan.err']
    if N_analogs > 10:
        if sig_d < 5:
            sig_d = 5.0
        if sig_vrad < 5:
            sig_vrad = 5.0
        if sig_vtan < 5:
            sig_vtan = 5.0
    else:
        if sig_d < 10:
            sig_d = 10.0
        if sig_vrad < 10:
            sig_vrad = 10.0
        if sig_vtan < 10:
            sig_vtan = 10.0
    info_lines2 = [
        rf"$d$: {lg_data[galaxy]['host.distance.total']:.1f} $\pm$ {sig_d:.1f} kpc",
        rf"$v_{{\rm rad}}$: {lg_data[galaxy]['host.velocity.rad']:.1f} $\pm$ {sig_vrad:.1f} km/s",
        rf"$v_{{\rm tan}}$: {lg_data[galaxy]['host.velocity.tan']:.1f} $\pm$ {sig_vtan:.1f} km/s",
    ]
    info_text1 = "\n".join(info_lines1)
    info_text2 = "\n".join(info_lines2)

    # --- Turn the panel into a clean text area
    axs[3,1].set_axis_off()
    axs[3,1].text(
        0.01, 0.97, info_text1,
        transform=axs[3,1].transAxes,
        ha="left", va="top",
        fontsize=20,
        linespacing=1.4,
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1.0)
    )
    axs[3,1].text(
        0.01, 0.70, info_text2,
        transform=axs[3,1].transAxes,
        ha="left", va="top",
        fontsize=18,
        linespacing=1.4
    )
    # Blank subpanel
    #axs[3,1].axison = False
    #
    #plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=20)
    plt.tight_layout()
    #plt.show()
    #plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/both_hist/'+satellite_name+'_history_both.pdf')
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/histograms_reorder/'+satellite_name+'_history_both.pdf')
    plt.close()

