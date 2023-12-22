

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


#galaxy = 'Sculptor'
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
    axs[0,0].set_xlabel('Lookback infall time [Gyr]', fontsize=14)
    axs[0,0].tick_params(axis='both', which='major', labelsize=12)
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
    axs[0,1].set_xlabel('Lookback apocenter time [Gyr]', fontsize=14)
    axs[0,1].tick_params(axis='both', which='major', labelsize=12)
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
    axs[0,2].set_xlabel('Apocenter distance [kpc]', fontsize=14)
    axs[0,2].tick_params(axis='both', which='major', labelsize=12)
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
    axs[1,0].set_xlabel('Lookback recent pericenter time [Gyr]', fontsize=14)
    axs[1,0].tick_params(axis='both', which='major', labelsize=12)
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
    axs[1,1].set_xlabel('Recent pericenter distance [kpc]', fontsize=14)
    axs[1,1].tick_params(axis='both', which='major', labelsize=12)
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
    axs[1,2].set_xlabel('Recent pericenter velocity [km s$^{-1}$]', fontsize=14)
    axs[1,2].tick_params(axis='both', which='major', labelsize=12)
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
    axs[2,0].set_xlabel('Lookback minimum pericenter time [Gyr]', fontsize=14)
    axs[2,0].tick_params(axis='both', which='major', labelsize=12)
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
    axs[2,1].set_xlabel('Minimum pericenter distance [kpc]', fontsize=14)
    axs[2,1].tick_params(axis='both', which='major', labelsize=12)
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
    axs[2,2].set_xlabel('Minimum pericenter velocity [km s$^{-1}$]', fontsize=14)
    axs[2,2].tick_params(axis='both', which='major', labelsize=12)
    #
    x = orbit_dictionary['distance']
    binss, half_binss = sat_analysis.binning_scheme(x, 'd.z0', 5)
    p = np.histogram(x, binss, density=True, weights=gal_data['Weight'])
    axs[3,0].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
    #axs[3,0].hist(x, binss, density=True, weights=gal_data['Weight'], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
    x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'])
    y_med = 1.1
    sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'])
    sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'])
    axs[3,0].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
    axs[3,0].scatter(x_med, y_med, s=75, marker='s', c='k')
    axs[3,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[3,0].hist(x, binss, density=True, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,0].set_xlabel('Distance [kpc]', fontsize=14)
    axs[3,0].tick_params(axis='both', which='major', labelsize=12)
    #
    x = orbit_dictionary['velocity.rad']
    binss, half_binss = sat_analysis.binning_scheme(x, 'v.rad', 1)
    p = np.histogram(x, binss, density=True, weights=gal_data['Weight'])
    axs[3,1].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=1, color='k', alpha=0.4, edgecolor=None)
    #axs[3,1].hist(x, binss, density=True, weights=gal_data['Weight'], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
    x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'])
    y_med = 1.1
    sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'])
    sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'])
    axs[3,1].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
    axs[3,1].scatter(x_med, y_med, s=75, marker='s', c='k')
    axs[3,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[3,1].hist(x, binss, density=True, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,1].set_xlabel('Radial velocity [km s$^{-1}$]', fontsize=14)
    axs[3,1].tick_params(axis='both', which='major', labelsize=12)
    #
    x = orbit_dictionary['velocity.tan']
    binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 5)
    p = np.histogram(x, binss, density=True, weights=gal_data['Weight'])
    axs[3,2].bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=5, color='k', alpha=0.4, edgecolor=None)
    #axs[3,2].hist(x, binss, density=True, weights=gal_data['Weight'], linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
    x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'])
    y_med = 1.1
    sigma_one_om = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'])
    sigma_one_op = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'])
    axs[3,2].errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
    axs[3,2].scatter(x_med, y_med, s=75, marker='s', c='k')
    axs[3,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs[3,2].hist(x, binss, density=True, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs[3,2].set_xlabel('Tangential velocity [km s$^{-1}$]', fontsize=14)
    axs[3,2].tick_params(axis='both', which='major', labelsize=12)
    #
    plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=14)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/both_hist/'+satellite_name+'_history_both.pdf')
    plt.close()
    #
    # # Plot the orbit history cumulative
    # plt.rcParams["font.family"] = "serif"
    # f, axs = plt.subplots(4, 3, figsize=(16,12))
    # #
    # m = (orbit_dictionary['first.infall.time.lb'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['first.infall.time.lb'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 't.infall', 0.25)
    #     axs[0,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[0,0].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[0,0].set_xlabel('Lookback infall time [Gyr]', fontsize=14)
    # axs[0,0].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['apocenter.time.lb'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['apocenter.time.lb'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 't.apo', 0.25)
    #     axs[0,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[0,1].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[0,1].set_xlabel('Lookback apocenter time [Gyr]', fontsize=14)
    # axs[0,1].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['apocenter.dist'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['apocenter.dist'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 'd.apo', 5)
    #     axs[0,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[0,2].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[0,2].set_xlabel('Apocenter distance [kpc]', fontsize=14)
    # axs[0,2].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['pericenter.rec.time.lb'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 't.peri', 0.25)
    #     axs[1,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[1,0].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[1,0].set_xlabel('Lookback recent pericenter time [Gyr]', fontsize=14)
    # axs[1,0].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['pericenter.rec.dist'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
    #     axs[1,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[1,1].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[1,1].set_xlabel('Recent pericenter distance [kpc]', fontsize=14)
    # axs[1,1].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['pericenter.rec.vel'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
    #     axs[1,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[1,2].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[1,2].set_xlabel('Recent pericenter velocity [km s$^{-1}$]', fontsize=14)
    # axs[1,2].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['pericenter.min.time.lb'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 't.peri', 0.25)
    #     axs[2,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[2,0].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[2,0].set_xlabel('Lookback minimum pericenter time [Gyr]', fontsize=14)
    # axs[2,0].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['pericenter.min.dist'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['pericenter.min.dist'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 'd.peri', 5)
    #     axs[2,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[2,1].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[2,1].set_xlabel('Minimum pericenter distance [kpc]', fontsize=14)
    # axs[2,1].tick_params(axis='both', which='major', labelsize=12)
    # #
    # m = (orbit_dictionary['pericenter.min.vel'] != -1)
    # if np.sum(m) != 0:
    #     x = orbit_dictionary['pericenter.min.vel'][m]
    #     binss, half_binss = sat_analysis.binning_scheme(x, 'v.peri', 10)
    #     axs[2,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    #     axs[2,2].hist(x, binss, density=False, weights=gal_data['Weight'][m], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[2,2].set_xlabel('Minimum pericenter velocity [km s$^{-1}$]', fontsize=14)
    # axs[2,2].tick_params(axis='both', which='major', labelsize=12)
    # #
    # x = orbit_dictionary['distance']
    # binss, half_binss = sat_analysis.binning_scheme(x, 'd.z0', 5)
    # axs[3,0].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    # axs[3,0].hist(x, binss, density=False, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[3,0].set_xlabel('Distance [kpc]', fontsize=14)
    # axs[3,0].tick_params(axis='both', which='major', labelsize=12)
    # #
    # x = orbit_dictionary['velocity.rad']
    # binss, half_binss = sat_analysis.binning_scheme(x, 'v.rad', 1)
    # axs[3,1].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    # axs[3,1].hist(x, binss, density=False, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[3,1].set_xlabel('Radial velocity [km s$^{-1}$]', fontsize=14)
    # axs[3,1].tick_params(axis='both', which='major', labelsize=12)
    # #
    # x = orbit_dictionary['velocity.tan']
    # binss, half_binss = sat_analysis.binning_scheme(x, 'v.tan', 5)
    # axs[3,2].axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    # axs[3,2].hist(x, binss, density=True, weights=gal_data['Weight'], cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    # axs[3,2].set_xlabel('Tangential velocity [km s$^{-1}$]', fontsize=14)
    # axs[3,2].tick_params(axis='both', which='major', labelsize=12)
    # #
    # plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=14)
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/cumulative/'+satellite_name+'_history_cumulative.pdf')
    # plt.close()
    #
    # Plot the phase-space information
    # plt.rcParams["font.family"] = "serif"
    # f, axs = plt.subplots(3, 3, figsize=(12,12))
    # #
    # axs[0,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
    # axs[1,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['velocity.rad'], s=15, alpha=0.8, c='k')
    # axs[2,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['distance'], s=15, alpha=0.8, c='k')
    # #
    # axs[1,1].scatter(orbit_dictionary['distance'], orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
    # axs[2,1].scatter(orbit_dictionary['distance'], orbit_dictionary['velocity.rad'], s=15, alpha=0.8, c='k')
    # #
    # axs[2,2].scatter(orbit_dictionary['velocity.rad'], orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
    # #
    # axs[0,0].set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=14)
    # axs[1,0].set_ylabel('Radial velocity [km s$^{-1}$]', fontsize=14)
    # axs[2,0].set_ylabel('Distance [kpc]', fontsize=14)
    # axs[1,1].set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=14)
    # axs[2,1].set_ylabel('Radial velocity [km s$^{-1}$]', fontsize=14)
    # axs[2,2].set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=14)
    # axs[2,0].set_xlabel('log $M_{\\rm halo, peak}$', fontsize=14)
    # axs[2,1].set_xlabel('Distance [kpc]', fontsize=14)
    # axs[2,2].set_xlabel('Radial velocity [km s$^{-1}$]', fontsize=14)
    # #
    # plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=14)
    # axs[0,0].tick_params(axis='both', which='major', labelsize=12)
    # axs[1,0].tick_params(axis='both', which='major', labelsize=12)
    # axs[2,0].tick_params(axis='both', which='major', labelsize=12)
    # axs[0,1].tick_params(axis='both', which='major', labelsize=12)
    # axs[1,1].tick_params(axis='both', which='major', labelsize=12)
    # axs[2,1].tick_params(axis='both', which='major', labelsize=12)
    # axs[0,2].tick_params(axis='both', which='major', labelsize=12)
    # axs[1,2].tick_params(axis='both', which='major', labelsize=12)
    # axs[2,2].tick_params(axis='both', which='major', labelsize=12)
    # #
    # axs[0,1].axison = False
    # axs[0,2].axison = False
    # axs[1,2].axison = False
    # #
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/phase_space/'+satellite_name+'_phase_space_properties.pdf')
    # plt.close()










# Work on master plots
sat_mstar = []
sat_dist = []
v_tan = []
#
first_infall = []
nperi = []
tperi_rec = []
dperi_rec = []
vperi_rec = []
tperi_min = []
dperi_min = []
vperi_min = []
tapo_rec = []
dapo_rec = []
mhalo = []
#
for galaxy in mw_sats:
    #
    gal_data = sat_analysis.read_subhalo_matches(galaxy)
    satellite_name = galaxy.replace(' ', '_')
    #
    if len(gal_data['Host']) == 0:
        continue
    #
    sat_mstar.append(lg_data[galaxy]['mass.star'])
    sat_dist.append(lg_data[galaxy]['host.distance.total'])
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
    # Infall times
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        first_infall.append((x_med, x_lower, x_upper))
    else:
        first_infall.append((-1, -1, -1))
    #
    # Pericenter number
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.num'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        nperi.append((x_med, x_lower, x_upper))
    else:
        nperi.append((-1, -1, -1))
    #
    # Recent pericenter time
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        tperi_rec.append((x_med, x_lower, x_upper))
    else:
        tperi_rec.append((-1, -1, -1))
    #
    # Recent pericenter distance
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        dperi_rec.append((x_med, x_lower, x_upper))
    else:
        dperi_rec.append((-1, -1, -1))
    #
    # Recent pericenter velocity
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        vperi_rec.append((x_med, x_lower, x_upper))
    else:
        vperi_rec.append((-1, -1, -1))
    #
    # Minimum pericenter time
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        tperi_min.append((x_med, x_lower, x_upper))
    else:
        tperi_min.append((-1, -1, -1))
    #
    # Minimum pericenter distance
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        dperi_min.append((x_med, x_lower, x_upper))
    else:
        dperi_min.append((-1, -1, -1))
    #
    # Minimum pericenter velocity
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        vperi_min.append((x_med, x_lower, x_upper))
    else:
        vperi_min.append((-1, -1, -1))
    #
    # Recnet apocenter time
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        tapo_rec.append((x_med, x_lower, x_upper))
    else:
        tapo_rec.append((-1, -1, -1))
    #
    # Recent apocenter distance
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.dist'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        dapo_rec.append((x_med, x_lower, x_upper))
    else:
        dapo_rec.append((-1, -1, -1))
    

sat_mstar = np.asarray(sat_mstar)
sat_dist = np.asarray(sat_dist)

"""
    Infall time plots
"""
first_infall = np.asarray(first_infall)
mask = (first_infall[:,0] != -1)
meds = first_infall[:,0]
uppers = first_infall[:,1]
lowers = first_infall[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#c76438', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/infall_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#c76438', alpha=0.7)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/infall_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#c76438')
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/infall_vs_dist_zoom.pdf')
plt.close()



"""
    Pericenter number plots
"""
orbit_prop = np.asarray(nperi)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#2b5b0c', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/nperi_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#2b5b0c', alpha=0.7)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/nperi_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#2b5b0c', alpha=0.7)
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/nperi_vs_dist_zoom.pdf')
plt.close()



"""
    Recent pericenter time plots
"""
orbit_prop = np.asarray(tperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#432471', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#432471', alpha=0.7)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#432471', alpha=0.7)
axs.set_xlim(0,420)
axs.set_ylim(-0.2, 8)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent pericenter distance plots
"""
orbit_prop = np.asarray(dperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#d05151')
axs.set_xscale('log')
axs.set_ylim(0, 140)
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#d05151')
axs.set_ylim(0, 140)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#d05151')
axs.set_ylim(0, 140)
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent pericenter velocity plots
"""
orbit_prop = np.asarray(vperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#1542b0')
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/vperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#1542b0')
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/vperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#1542b0')
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/vperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter time plots
"""
orbit_prop = np.asarray(tperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#780d3f')
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#780d3f')
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#780d3f')
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter distance plots
"""
orbit_prop = np.asarray(dperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#572135')
axs.set_xscale('log')
axs.set_ylim(0, 140)
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#572135')
axs.set_ylim(0, 140)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#572135')
axs.set_xlim(0,420)
axs.set_ylim(0, 140)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter velocity plots
"""
orbit_prop = np.asarray(vperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#f38e00')
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/vperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#f38e00')
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/vperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#f38e00')
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/vperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Recent apocenter time plots
"""
orbit_prop = np.asarray(tapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#6e1d16')
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#6e1d16')
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tapo_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#6e1d16')
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/tapo_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent apocenter distance plots
"""
orbit_prop = np.asarray(dapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
uppers = orbit_prop[:,1]
lowers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#4e2026')
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#4e2026')
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dapo_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#4e2026')
axs.set_xlim(0,420)
axs.set_ylim(0,500)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary/dapo_rec_vs_dist_zoom.pdf')
plt.close()












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
    f, axs = plt.subplots(4, 3, figsize=(16,12))
    #
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[0,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[0,0].set_xlabel('Lookback infall time [Gyr]', fontsize=14)
    axs[0,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[0,1].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[0,1].set_xlabel('Lookback apocenter time [Gyr]', fontsize=14)
    axs[0,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.dist'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[0,2].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[0,2].set_xlabel('Apocenter distance [kpc]', fontsize=14)
    axs[0,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[1,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[1,0].set_xlabel('Lookback recent pericenter time [Gyr]', fontsize=14)
    axs[1,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[1,1].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[1,1].set_xlabel('Recent pericenter distance [kpc]', fontsize=14)
    axs[1,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[1,2].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[1,2].set_xlabel('Recent pericenter velocity [km s$^{-1}$]', fontsize=14)
    axs[1,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[2,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[2,0].set_xlabel('Lookback minimum pericenter time [Gyr]', fontsize=14)
    axs[2,0].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[2,1].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[2,1].set_xlabel('Minimum pericenter distance [kpc]', fontsize=14)
    axs[2,1].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[2,2].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[2,2].set_xlabel('Minimum pericenter velocity [km s$^{-1}$]', fontsize=14)
    axs[2,2].tick_params(axis='both', which='major', labelsize=12)
    #
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.num'][m]
        y = orbit_dictionary['velocity.tan'][m]
        axs[3,0].scatter(x, y, s=20, marker='o', c='k', alpha=0.5)
    axs[3,0].set_xlabel('Pericenter Number', fontsize=14)
    axs[3,0].tick_params(axis='both', which='major', labelsize=12)
    #
    axs[3,1].axison = False
    axs[3,2].axison = False
    axs[0,0].set_ylabel('Proper Motion [km s$^{-1}$]', fontsize=14)
    axs[1,0].set_ylabel('Proper Motion [km s$^{-1}$]', fontsize=14)
    axs[2,0].set_ylabel('Proper Motion [km s$^{-1}$]', fontsize=14)
    axs[3,0].set_ylabel('Proper Motion [km s$^{-1}$]', fontsize=14)
    plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=14)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/proper_motion_corr/'+satellite_name+'_vtans.pdf')
    plt.close()





















# Test on m12m 
galaxy = 'Carina'
gal_data = sat_analysis.read_subhalo_matches(galaxy)
#
sim_name = 'm12m'
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=True)
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)

# get the matches in m12m
mask = (gal_data['Host'] == sim_name)
# get an array of the tree indices that are matches in m12m
subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
# Get indices within the mini data to select properties
mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])

# Need to check if subhalo fell in after match. If so, then don't count it.
mini_data_match_inds = mini_data_match_inds[mini_data['first.infall.snap'][mini_data_match_inds] < gal_data['Snapshot at match'][mask]]


"""
    Save the infall time, relative to match time
"""
# Snapshot at match
time_at_match = snaps['time'][gal_data['Snapshot at match'][mask]]
time_since_infall_and_match = time_at_match - mini_data['first.infall.time'][mini_data_match_inds]

"""
    Pericenter stuff
"""
t_peri_array_lb = (-1)*np.ones(len(mini_data_match_inds)) ## Second property to save - the lookback time to the most recent pericenter.
d_peri_array = (-1)*np.ones(len(mini_data_match_inds))
v_peri_array = (-1)*np.ones(len(mini_data_match_inds))
t_apo_array_lb = (-1)*np.ones(len(mini_data_match_inds)) 
d_apo_array = (-1)*np.ones(len(mini_data_match_inds))
#
for i in range(0, len(mini_data_match_inds)):
    mask_peri = (mini_data['pericenter.time.sim'][mini_data_match_inds][i] != -1)
    time_since_dperi_rec = time_at_match[i] - mini_data['pericenter.time.sim'][mini_data_match_inds][i][mask_peri]
    if (np.sum(time_since_dperi_rec > 0) != 0):
        t_peri_array_lb[i] = time_since_dperi_rec[time_since_dperi_rec > 0][0]
        d_peri_array[i] = mini_data['pericenter.dist.sim'][mini_data_match_inds][i][mask_peri][0]
        v_peri_array[i] = mini_data['pericenter.vel.sim'][mini_data_match_inds][i][mask_peri][0]
    mask_apo = (mini_data['apocenter.time.sim'][mini_data_match_inds][i] != -1)
    time_since_dapo_rec = time_at_match[i] - mini_data['apocenter.time.sim'][mini_data_match_inds][i][mask_apo]
    if (np.sum(time_since_dapo_rec > 0) != 0):
        t_apo_array_lb[i] = time_since_dapo_rec[time_since_dapo_rec > 0][0]
        d_apo_array[i] = mini_data['apocenter.dist.sim'][mini_data_match_inds][i][mask_apo][0]
    #
    vel_rad[i] = mini_data['v.rad.sim'][mini_data_match_inds]


vel_rad = (-1)*np.ones(len(mini_data_match_inds))
for i in range(0, len(mini_data_match_inds)):
    time_ind = snaps['index'][-1] - np.where(np.min(np.abs(time_at_match[i] - snaps['time'])) == np.abs(time_at_match[i] - snaps['time']))[0][0]
    vel_rad[i] = mini_data['v.rad.sim'][mini_data_match_inds][i][time_ind]


vel_rad = (-1)*np.ones(len(mini_data_match_inds))
for i in range(0, len(mini_data_match_inds)):
    time_ind = snaps['index'][-1] - np.asarray(gal_data['Snapshot at match'][mask])[i]
    vel_rad[i] = mini_data['v.rad.sim'][mini_data_match_inds][i][time_ind]

