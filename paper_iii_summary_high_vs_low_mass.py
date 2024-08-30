
#!/usr/bin/python3

"""
    =============================
    = Paper III Summary Figures =
    =============================

    Create plots showing various orbit history properties 
    as a function of either distance or stellar mass. This will
    plot each MW satellite as a point, and the error-bars will
    show the 68% scatter among the subhalo analogs for that
    MW satellite.
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

# Based on HALO mass
# galaxies_high = ['m12b', 'm12c', 'm12f', 'm12m', 'Thelma', 'Romulus', 'm12n']
# galaxies_low = ['m12i', 'm12w', 'Romeo', 'Juliet', 'Louise', 'Remus']

# Based on GALAXY mass
galaxies_high = ['m12b', 'm12f', 'm12i', 'm12m', 'Romeo', 'Thelma', 'Romulus']
galaxies_low = ['m12c', 'm12w', 'Juliet', 'Louise', 'Remus', 'm12n']


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

# Work on master plots
sat_mstar = []
sat_dist = []
v_tan = []
#
first_infall_hi = []
nperi_hi = []
tperi_rec_hi = []
dperi_rec_hi = []
vperi_rec_hi = []
tperi_min_hi = []
dperi_min_hi = []
vperi_min_hi = []
tapo_rec_hi = []
dapo_rec_hi = []
mhalo_hi = []
#
first_infall_lo = []
nperi_lo = []
tperi_rec_lo = []
dperi_rec_lo = []
vperi_rec_lo = []
tperi_min_lo = []
dperi_min_lo = []
vperi_min_lo = []
tapo_rec_lo = []
dapo_rec_lo = []
mhalo_lo = []
#
for galaxy in mw_sats_1Mpc:
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
    orbit_dictionary['first.infall.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.num'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.dist'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.vel'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.min.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.min.dist'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.min.vel'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['apocenter.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['apocenter.dist'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['halo.mass.peak'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['distance'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['velocity.rad'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['velocity.tan'] = (-1)*np.ones(gal_data.shape[0])
    #
    maskHigh = np.array([True if gal_data['Host'][i] in galaxies_high else False for i in range(len(gal_data['Host']))])
    maskLow = np.array([True if gal_data['Host'][i] in galaxies_low else False for i in range(len(gal_data['Host']))])
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
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        first_infall_hi.append((x_med, x_lower, x_upper))
        #
    else:
        first_infall_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        first_infall_lo.append((x_med, x_lower, x_upper))
    else:
        first_infall_lo.append((-1, -1, -1))
    #
    # Pericenter number
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['pericenter.num'][m * maskHigh]
        x_med = np.sum(x*gal_data['Weight'][m * maskHigh])
        x_std = np.sqrt(np.sum((x-x_med)**2*gal_data['Weight'][m * maskHigh])/np.sum(gal_data['Weight'][m * maskHigh])/np.sum(gal_data['Weight'][m * maskHigh]))
        nperi_hi.append((x_med, x_std, -1))
    else:
        nperi_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['pericenter.num'][m * maskLow]
        x_med = np.sum(x*gal_data['Weight'][m * maskLow])
        x_std = np.sqrt(np.sum((x-x_med)**2*gal_data['Weight'][m * maskLow])/np.sum(gal_data['Weight'][m * maskLow])/np.sum(gal_data['Weight'][m * maskLow]))
        nperi_lo.append((x_med, x_std, -1))
    else:
        nperi_lo.append((-1, -1, -1))
    #
    # Recent pericenter time
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        tperi_rec_hi.append((x_med, x_lower, x_upper))
    else:
        tperi_rec_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        tperi_rec_lo.append((x_med, x_lower, x_upper))
    else:
        tperi_rec_lo.append((-1, -1, -1))
    #
    # Recent pericenter distance
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        dperi_rec_hi.append((x_med, x_lower, x_upper))
    else:
        dperi_rec_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        dperi_rec_lo.append((x_med, x_lower, x_upper))
    else:
        dperi_rec_lo.append((-1, -1, -1))
    #
    # Recent pericenter velocity
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        vperi_rec_hi.append((x_med, x_lower, x_upper))
    else:
        vperi_rec_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        vperi_rec_lo.append((x_med, x_lower, x_upper))
    else:
        vperi_rec_lo.append((-1, -1, -1))
    #
    # Minimum pericenter time
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        tperi_min_hi.append((x_med, x_lower, x_upper))
    else:
        tperi_min_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        tperi_min_lo.append((x_med, x_lower, x_upper))
    else:
        tperi_min_lo.append((-1, -1, -1))
    #
    # Minimum pericenter distance
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        dperi_min_hi.append((x_med, x_lower, x_upper))
    else:
        dperi_min_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        dperi_min_lo.append((x_med, x_lower, x_upper))
    else:
        dperi_min_lo.append((-1, -1, -1))
    #
    # Minimum pericenter velocity
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        vperi_min_hi.append((x_med, x_lower, x_upper))
    else:
        vperi_min_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        vperi_min_lo.append((x_med, x_lower, x_upper))
    else:
        vperi_min_lo.append((-1, -1, -1))
    #
    # Recnet apocenter time
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        tapo_rec_hi.append((x_med, x_lower, x_upper))
    else:
        tapo_rec_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        tapo_rec_lo.append((x_med, x_lower, x_upper))
    else:
        tapo_rec_lo.append((-1, -1, -1))
    #
    # Recent apocenter distance
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m * maskHigh) != 0:
        x = orbit_dictionary['apocenter.dist'][m * maskHigh]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskHigh])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskHigh])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskHigh])
        dapo_rec_hi.append((x_med, x_lower, x_upper))
    else:
        dapo_rec_hi.append((-1, -1, -1))
    #
    if np.sum(m * maskLow) != 0:
        x = orbit_dictionary['apocenter.dist'][m * maskLow]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLow])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLow])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLow])
        dapo_rec_lo.append((x_med, x_lower, x_upper))
    else:
        dapo_rec_lo.append((-1, -1, -1))

sat_mstar = np.asarray(sat_mstar)
sat_dist = np.asarray(sat_dist)

"""
    Infall time plots
"""
first_infall_hi = np.asarray(first_infall_hi)
mask_hi = (first_infall_hi[:,0] != -1)
meds_hi = first_infall_hi[:,0]
lowers_hi = first_infall_hi[:,1]
uppers_hi = first_infall_hi[:,2]
#
first_infall_lo = np.asarray(first_infall_lo)
mask_lo = (first_infall_lo[:,0] != -1)
meds_lo = first_infall_lo[:,0]
lowers_lo = first_infall_lo[:,1]
uppers_lo = first_infall_lo[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/infall_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/infall_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
axs.set_xlim(0, 420)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/infall_vs_dist_zoom.pdf')
plt.close()



"""
    Pericenter number plots
"""
orbit_prop = np.asarray(nperi_hi)
mask_hi = (orbit_prop[:,0] != -1)
means_hi = orbit_prop[:,0]
stds_hi = orbit_prop[:,1]
#
orbit_prop = np.asarray(nperi_lo)
mask_lo = (orbit_prop[:,0] != -1)
means_lo = orbit_prop[:,0]
stds_lo = orbit_prop[:,1]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], means_hi[mask_hi][0], yerr=stds_hi[mask_hi][0], color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], means_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], means_lo[mask_lo][0], yerr=stds_lo[mask_lo][0], color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], means_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], means_hi[mask_hi][i], yerr=stds_hi[mask_hi][i], color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], means_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], means_lo[mask_lo][i], yerr=stds_lo[mask_lo][i], color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], means_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(ymin=0)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/nperi_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], means_hi[mask_hi][0], yerr=stds_hi[mask_hi][0], color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], means_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], means_lo[mask_lo][0], yerr=stds_lo[mask_lo][0], color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], means_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], means_hi[mask_hi][i], yerr=stds_hi[mask_hi][i], color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], means_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], means_lo[mask_lo][i], yerr=stds_lo[mask_lo][i], color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], means_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(ymin=0)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/nperi_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], means_hi[mask_hi][0], yerr=stds_hi[mask_hi][0], color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], means_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], means_lo[mask_lo][0], yerr=stds_lo[mask_lo][0], color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], means_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], means_hi[mask_hi][i], yerr=stds_hi[mask_hi][i], color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], means_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], means_lo[mask_lo][i], yerr=stds_lo[mask_lo][i], color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], means_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
axs.set_ylim(ymin=0)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/nperi_vs_dist_zoom.pdf')
plt.close()


"""
    Recent pericenter time plots
"""
orbit_prop = np.asarray(tperi_rec_hi)
mask_hi = (orbit_prop[:,0] != -1)
meds_hi = orbit_prop[:,0]
lowers_hi = orbit_prop[:,1]
uppers_hi = orbit_prop[:,2]
#
orbit_prop = np.asarray(tperi_rec_lo)
mask_lo = (orbit_prop[:,0] != -1)
meds_lo = orbit_prop[:,0]
lowers_lo = orbit_prop[:,1]
uppers_lo = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent pericenter distance plots
"""
orbit_prop = np.asarray(dperi_rec_hi)
mask_hi = (orbit_prop[:,0] != -1)
meds_hi = orbit_prop[:,0]
lowers_hi = orbit_prop[:,1]
uppers_hi = orbit_prop[:,2]
#
orbit_prop = np.asarray(dperi_rec_lo)
mask_lo = (orbit_prop[:,0] != -1)
meds_lo = orbit_prop[:,0]
lowers_lo = orbit_prop[:,1]
uppers_lo = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(0, 150)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(0, 150)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(0, 150)
axs.set_xlim(0, 420)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter time plots
"""
orbit_prop = np.asarray(tperi_min_hi)
mask_hi = (orbit_prop[:,0] != -1)
meds_hi = orbit_prop[:,0]
lowers_hi = orbit_prop[:,1]
uppers_hi = orbit_prop[:,2]
#
orbit_prop = np.asarray(tperi_min_lo)
mask_lo = (orbit_prop[:,0] != -1)
meds_lo = orbit_prop[:,0]
lowers_lo = orbit_prop[:,1]
uppers_lo = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
axs.set_xlim(0, 420)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter distance plots
"""
orbit_prop = np.asarray(dperi_min_hi)
mask_hi = (orbit_prop[:,0] != -1)
meds_hi = orbit_prop[:,0]
lowers_hi = orbit_prop[:,1]
uppers_hi = orbit_prop[:,2]
#
orbit_prop = np.asarray(dperi_min_lo)
mask_lo = (orbit_prop[:,0] != -1)
meds_lo = orbit_prop[:,0]
lowers_lo = orbit_prop[:,1]
uppers_lo = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xscale('log')
axs.set_ylim(0, 150)
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_ylim(0, 150)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dperi_min_vs_dist.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_ylim(0, 150)
axs.set_xlim(0, 420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Recent apocenter time plots
"""
orbit_prop = np.asarray(tapo_rec_hi)
mask_hi = (orbit_prop[:,0] != -1)
meds_hi = orbit_prop[:,0]
lowers_hi = orbit_prop[:,1]
uppers_hi = orbit_prop[:,2]
#
orbit_prop = np.asarray(tapo_rec_lo)
mask_lo = (orbit_prop[:,0] != -1)
meds_lo = orbit_prop[:,0]
lowers_lo = orbit_prop[:,1]
uppers_lo = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tapo_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlim(0, 420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/tapo_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent apocenter distance plots
"""
orbit_prop = np.asarray(dapo_rec_hi)
mask_hi = (orbit_prop[:,0] != -1)
meds_hi = orbit_prop[:,0]
lowers_hi = orbit_prop[:,1]
uppers_hi = orbit_prop[:,2]
#
orbit_prop = np.asarray(dapo_rec_lo)
mask_lo = (orbit_prop[:,0] != -1)
meds_lo = orbit_prop[:,0]
lowers_lo = orbit_prop[:,1]
uppers_lo = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_mstar[mask_hi])):
    axs.errorbar(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_mstar[mask_lo])):
    axs.errorbar(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dapo_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_hi][0], meds_hi[mask_hi][0], yerr=np.array([[meds_hi[mask_hi][0]-lowers_hi[mask_hi][0]],[uppers_hi[mask_hi][0]-meds_hi[mask_hi][0]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_hi][0], meds_hi[mask_hi][0], s=50, c='#c76438', alpha=0.5, label='High-mass hosts')
#
axs.errorbar(sat_dist[mask_lo][0], meds_lo[mask_lo][0], yerr=np.array([[meds_lo[mask_lo][0]-lowers_lo[mask_lo][0]],[uppers_lo[mask_lo][0]-meds_lo[mask_lo][0]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_lo][0], meds_lo[mask_lo][0], s=50, c='#389BC7', alpha=0.5, label='Low-mass hosts')
#
for i in range(1, len(sat_dist[mask_hi])):
    axs.errorbar(sat_dist[mask_hi][i], meds_hi[mask_hi][i], yerr=np.array([[meds_hi[mask_hi][i]-lowers_hi[mask_hi][i]],[uppers_hi[mask_hi][i]-meds_hi[mask_hi][i]]]), color='#c76438', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_hi][i], meds_hi[mask_hi][i], s=50, c='#c76438', alpha=0.5)
#
for i in range(1, len(sat_dist[mask_lo])):
    axs.errorbar(sat_dist[mask_lo][i], meds_lo[mask_lo][i], yerr=np.array([[meds_lo[mask_lo][i]-lowers_lo[mask_lo][i]],[uppers_lo[mask_lo][i]-meds_lo[mask_lo][i]]]), color='#389BC7', alpha=0.5, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_lo][i], meds_lo[mask_lo][i], s=50, c='#389BC7', alpha=0.5)
#
axs.set_xlim(0,420)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/hi_vs_lo/dapo_rec_vs_dist_zoom.pdf')
plt.close()













