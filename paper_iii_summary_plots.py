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

#galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']
galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12n']

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

# mw_sats_1Mpc_old = ['Antlia 2', 'Aquarius 2', 'Bootes 1', 'Bootes 2', 'Bootes 3', \
#                 'Canes Venatici 1', 'Canes Venatici 2', 'Carina', 'Carina 2', \
#                 'Carina 3', 'Cetus 2', 'Cetus 3', 'Columba 1', 'Coma Berenices', \
#                 'Crater 2', 'DES J0225+0304', 'Draco', 'Draco 2', 'Eridanus 2', \
#                 'Eridanus 3', 'Fornax', 'Grus 1', 'Grus 2', 'Hercules', \
#                 'Horologium 1', 'Horologium 2', 'Hydra 2', 'Hydrus 1', 'Indus 1', \
#                 'Indus 2', 'Leo 1', 'Leo 2', 'Leo 4', 'Leo 5', 'Leo A', 'Leo T', \
#                 'Pegasus 3', 'Phoenix', 'Phoenix 2', 'Pictor 1', 'Pictor 2', \
#                 'Pisces 2', 'Reticulum 2', 'Reticulum 3', 'Sagittarius 2', \
#                 'Sculptor', 'Segue 1', 'Segue 2', 'Sextans 1', 'Triangulum 2', \
#                 'Tucana', 'Tucana 2', 'Tucana 3', 'Tucana 4', 'Tucana 5', \
#                 'Ursa Major 1', 'Ursa Major 2', 'Ursa Minor', 'Virgo 1', \
#                 'Willman 1']
                
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

mw_sats_proposal =  ['Horologium 2', 'Pisces 2', 'Hydra 2', 'Eridanus 2', 'Willman 1', 'Reticulum 3', \
                     'Columba 1', 'Ursa Major 1', 'Pictor 1', 'Bootes 2', 'Grus 1', 'Tucana 5', 'Coma Berenices', \
                     'Eridanus 3', 'Tucana 4', 'Triangulum 2', 'Sagittarius 2', 'Segue 1', 'Grus 2', 'Phoenix 2', \
                     'Horologium 1', 'Tucana 2', 'Segue 2', 'Ursa Major 2', 'Reticulum 2', 'Ursa Minor', \
                     'Canes Venatici 1', 'Sextans 1', 'Phoenix', 'Carina 2', 'Tucana 3', 'Carina', 'Fornax', \
                     'Hydrus 1', 'Pegasus 3', 'Cetus 2', 'Virgo 1']

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
elltot = []
ketot = []
mhalo = []
#
for galaxy in mw_sats_1Mpc:
    #
    satellite_name = galaxy.replace(' ', '_')
    file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
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
    orbit_dictionary['L.tot.sim'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['distance'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.rad'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.tan'] = np.zeros(gal_data.shape[0])
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
        x_med = np.sum(x*gal_data['Weight'][m])
        x_std = np.sqrt(np.sum((x-x_med)**2*gal_data['Weight'][m])/np.sum(gal_data['Weight'][m])/np.sum(gal_data['Weight'][m]))
        nperi.append((x_med, x_std, -1))
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
    #
    # Angular momentum at match
    m = (orbit_dictionary['L.tot.sim'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['L.tot.sim'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        elltot.append((x_med, x_lower, x_upper))
    else:
        elltot.append((-1, -1, -1))
    #
    # Specific Kinetic Energy at match
    m = (orbit_dictionary['v.tot.sim'] != -1)
    if np.sum(m) != 0:
        x = 0.5*orbit_dictionary['v.tot.sim'][m]**2
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        ketot.append((x_med, x_lower, x_upper))
    else:
        ketot.append((-1, -1, -1))
    

sat_mstar = np.asarray(sat_mstar)
sat_dist = np.asarray(sat_dist)

"""
    Infall time plots
"""
first_infall = np.asarray(first_infall)
mask = (first_infall[:,0] != -1)
meds = first_infall[:,0]
lowers = first_infall[:,1]
uppers = first_infall[:,2]

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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#c76438', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#c76438', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_vs_dist_zoom.pdf')
plt.close()



"""
    Pericenter number plots
"""
orbit_prop = np.asarray(nperi)
mask = (orbit_prop[:,0] != -1)
means = orbit_prop[:,0]
stds = orbit_prop[:,1]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], means[mask][i], s=50, c='#2b5b0c', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], means[mask][i], s=50, c='#2b5b0c', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], means[mask][i], s=50, c='#2b5b0c', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_vs_dist_zoom.pdf')
plt.close()



"""
    Recent pericenter time plots
"""
orbit_prop = np.asarray(tperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#432471', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#432471', alpha=0.7)
axs.set_xlim(0,425)
#axs.set_ylim(-0.2, 8)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent pericenter distance plots
"""
orbit_prop = np.asarray(dperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#d05151', alpha=0.7)
axs.set_xscale('log')
axs.set_ylim(0, 140)
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), alpha=0.7, color='#d05151', lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#d05151', alpha=0.7)
axs.set_ylim(0, 140)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#d05151', alpha=0.7)
axs.set_ylim(0, 140)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent pericenter velocity plots
"""
orbit_prop = np.asarray(vperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#1542b0', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#1542b0', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#1542b0', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter time plots
"""
orbit_prop = np.asarray(tperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#780d3f', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#780d3f', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#780d3f', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter distance plots
"""
orbit_prop = np.asarray(dperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#572135', alpha=0.7)
axs.set_xscale('log')
axs.set_ylim(0, 140)
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#572135', alpha=0.7)
axs.set_ylim(0, 140)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#572135', alpha=0.7)
axs.set_xlim(0,425)
axs.set_ylim(0, 140)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Minimum pericenter velocity plots
"""
orbit_prop = np.asarray(vperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#f38e00', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#f38e00', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#f38e00', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_min_vs_dist_zoom.pdf')
plt.close()



"""
    Recent apocenter time plots
"""
orbit_prop = np.asarray(tapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#6e1d16', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#6e1d16', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tapo_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#6e1d16', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tapo_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Recent apocenter distance plots
"""
orbit_prop = np.asarray(dapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#4e2026', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#4e2026', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dapo_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#4e2026', alpha=0.7)
axs.set_xlim(0,425)
axs.set_ylim(0,500)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dapo_rec_vs_dist_zoom.pdf')
plt.close()



"""
    Specific Angular momentum plots
"""
orbit_prop = np.asarray(elltot)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i]/1e4, s=50, c='#4e2026', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$\\ell [10^4\ kpc\ km\ s^{-1}]$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ell_vs_mstar.pdf')
plt.close()

# Vs Mstar (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i]/1e4, s=50, c='#4e2026', alpha=0.7)
axs.set_xscale('log')
axs.set_ylim(0, 5)
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$\\ell [10^4\ kpc\ km\ s^{-1}]$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ell_vs_mstar_zoom.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=50, c='#4e2026', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$\\ell [10^4\ kpc\ km\ s^{-1}]$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ell_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=50, c='#4e2026', alpha=0.7)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$\\ell [10^4\ kpc\ km\ s^{-1}]$', fontsize=24)
axs.set_xlim(0, 420)
axs.set_ylim(0, 5)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ell_vs_dist_zoom.pdf')
plt.close()



"""
    Kinetic Energy plots
"""
orbit_prop = np.asarray(ketot)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i]/1e4, s=50, c='#4e2026', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('Specific Kinetic Energy $[10^4\ km^2\ s^{-2}]$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ke_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=50, c='#4e2026', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('Specific Kinetic Energy $[10^4\ km^2\ s^{-2}]$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ke_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=50, c='#4e2026', alpha=0.7)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('Specific Kinetic Energy $[10^4\ km^2\ s^{-2}]$', fontsize=24)
axs.set_xlim(0, 420)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ke_vs_dist_zoom.pdf')
plt.close()









"""
    Infall time plots
"""
first_infall = np.asarray(first_infall)
mask = (first_infall[:,0] != -1)
meds = first_infall[:,0]
lowers = first_infall[:,1]
uppers = first_infall[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(16,6))
#
for i in range(0, len(sat_mstar[mask])):
    axs[0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.7, lw=3.5, capsize=0)
    axs[0].scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#c76438', alpha=0.7)

for i in range(0, len(sat_mstar[mask])):
    axs[1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.7, lw=3.5, capsize=0)
    axs[1].scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#c76438', alpha=0.7)
#
axs[0].set_xlim(0,425)
#
axs[1].set_xscale('log')
#
axs[0].set_xlabel('Distance from MW [kpc]', fontsize=24)
axs[1].set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs[0].set_ylabel('$t_{\\rm lookback, infall}$ [Gyr]', fontsize=24)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_plot.pdf')
plt.close()




"""
    Pericenter number plots
"""
orbit_prop = np.asarray(nperi)
mask = (orbit_prop[:,0] != -1)
means = orbit_prop[:,0]
stds = orbit_prop[:,1]

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(16,6))
#
for i in range(0, len(sat_mstar[mask])):
    axs[0].errorbar(sat_dist[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs[0].scatter(sat_dist[mask][i], means[mask][i], s=50, c='#2b5b0c', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1].errorbar(sat_mstar[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.7, lw=3.5, capsize=0)
    axs[1].scatter(sat_mstar[mask][i], means[mask][i], s=50, c='#2b5b0c', alpha=0.7)
#
axs[0].set_xlim(0,425)
#
axs[1].set_xscale('log')
#
axs[0].set_xlabel('Distance from MW [kpc]', fontsize=24)
axs[1].set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs[0].set_ylabel('$N_{\\rm peri}$', fontsize=24)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_plot.pdf')
plt.close()




"""
    Recent pericenter metric plots
"""
orbit_prop = np.asarray(tperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]
#
## Pericenter time
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 2, figsize=(16,10))
# Vs distance
for i in range(0, len(sat_mstar[mask])):
    axs[0,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.7, lw=3.5, capsize=0)
    axs[0,0].scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#432471', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[0,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.7, lw=3.5, capsize=0)
    axs[0,1].scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#432471', alpha=0.7)

## Pericenter distance
orbit_prop = np.asarray(dperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]
#
# Vs distance (zoom)
for i in range(0, len(sat_mstar[mask])):
    axs[1,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.7, lw=3.5, capsize=0)
    axs[1,0].scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#d05151', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.7, lw=3.5, capsize=0)
    axs[1,1].scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#d05151', alpha=0.7)
#
axs[0,0].set_xlim(0,425)
axs[1,0].set_xlim(0,425)
axs[1,0].set_ylim(0,140)
axs[1,1].set_ylim(0,140)
#
axs[1,1].set_xscale('log')
axs[0,1].set_xscale('log')
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False, labelleft=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
axs[1,0].set_xlabel('Distance from MW [kpc]', fontsize=24)
axs[1,1].set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs[0,0].set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
axs[1,0].set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/recent_peri_both.pdf')
plt.close()




"""
    Minimum pericenter metric plots
"""
orbit_prop = np.asarray(tperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]
#
# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 2, figsize=(16,10))
#
for i in range(0, len(sat_mstar[mask])):
    axs[0,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.7, lw=3.5, capsize=0)
    axs[0,0].scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#780d3f', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[0,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.7, lw=3.5, capsize=0)
    axs[0,1].scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#780d3f', alpha=0.7)
#
## distance trends
orbit_prop = np.asarray(dperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]
#
# Vs distance (zoom)
for i in range(0, len(sat_mstar[mask])):
    axs[1,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.7, lw=3.5, capsize=0)
    axs[1,0].scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#572135', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.7, lw=3.5, capsize=0)
    axs[1,1].scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#572135', alpha=0.7)

axs[0,0].set_xlim(0,425)
axs[1,0].set_xlim(0,425)
axs[1,0].set_ylim(0,140)
axs[1,1].set_ylim(0,140)
#
axs[1,1].set_xscale('log')
axs[0,1].set_xscale('log')
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False, labelleft=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
axs[1,0].set_xlabel('Distance from MW [kpc]', fontsize=24)
axs[1,1].set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs[0,0].set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
axs[1,0].set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/minimum_peri_both.pdf')
plt.close()




"""
    Recent apocenter metric plots
"""
orbit_prop = np.asarray(tapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]
#
# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 2, figsize=(16,10))
#
for i in range(0, len(sat_mstar[mask])):
    axs[0,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.7, lw=3.5, capsize=0)
    axs[0,0].scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#6e1d16', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[0,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.7, lw=3.5, capsize=0)
    axs[0,1].scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#6e1d16', alpha=0.7)
#
# Distance trends
orbit_prop = np.asarray(dapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]
#
# Vs distance (zoom)
for i in range(0, len(sat_mstar[mask])):
    axs[1,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs[1,0].scatter(sat_dist[mask][i], meds[mask][i], s=50, c='#4e2026', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.7, lw=3.5, capsize=0)
    axs[1,1].scatter(sat_mstar[mask][i], meds[mask][i], s=50, c='#4e2026', alpha=0.7)
#
axs[0,0].set_xlim(0,425)
axs[1,0].set_xlim(0,425)
#
axs[0,1].set_xscale('log')
axs[1,1].set_xscale('log')
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False, labelleft=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
axs[1,0].set_xlabel('Distance from MW [kpc]', fontsize=24)
axs[1,1].set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs[0,0].set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
axs[1,0].set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/apocenter_both.pdf')
plt.close()




"""
    Combined pericenter velocity plots
        NOT INCLUDING IN PAPER
"""
orbit_prop = np.asarray(vperi_rec)
mask_rec = (orbit_prop[:,0] != -1)
meds_rec = orbit_prop[:,0]
lowers_rec = orbit_prop[:,1]
uppers_rec = orbit_prop[:,2]

orbit_prop = np.asarray(vperi_min)
mask_min = (orbit_prop[:,0] != -1)
meds_min = orbit_prop[:,0]
lowers_min = orbit_prop[:,1]
uppers_min = orbit_prop[:,2]

width_rec = uppers_rec[mask_rec] - lowers_rec[mask_rec]
width_min = uppers_min[mask_min] - lowers_min[mask_min]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_mstar[mask_rec][0], meds_rec[mask_rec][0], yerr=np.array([[meds_rec[mask_rec][0]-lowers_rec[mask_rec][0]],[uppers_rec[mask_rec][0]-meds_rec[mask_rec][0]]]), color='#1542b0', alpha=0.35, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_rec][0], meds_rec[mask_rec][0], s=50, c='#1542b0', alpha=0.35, label='Recent')
axs.errorbar(sat_mstar[mask_min][0], meds_min[mask_min][0], yerr=np.array([[meds_min[mask_min][0]-lowers_min[mask_min][0]],[uppers_min[mask_min][0]-meds_min[mask_min][0]]]), color='#f38e00', alpha=0.35, lw=3.5, capsize=0)
axs.scatter(sat_mstar[mask_min][0], meds_min[mask_min][0], s=50, c='#f38e00', alpha=0.35, label='Minimum')
for i in range(1, len(sat_mstar[mask_rec])):
    axs.errorbar(sat_mstar[mask_rec][i], meds_rec[mask_rec][i], yerr=np.array([[meds_rec[mask_rec][i]-lowers_rec[mask_rec][i]],[uppers_rec[mask_rec][i]-meds_rec[mask_rec][i]]]), color='#1542b0', alpha=0.35, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_rec][i], meds_rec[mask_rec][i], s=50, c='#1542b0', alpha=0.35)
for i in range(1, len(sat_mstar[mask_min])):
    axs.errorbar(sat_mstar[mask_min][i], meds_min[mask_min][i], yerr=np.array([[meds_min[mask_min][i]-lowers_min[mask_min][i]],[uppers_min[mask_min][i]-meds_min[mask_min][i]]]), color='#f38e00', alpha=0.35, lw=3.5, capsize=0)
    axs.scatter(sat_mstar[mask_min][i], meds_min[mask_min][i], s=50, c='#f38e00', alpha=0.35)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri}$ [km s$^{-1}$]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_both_vs_mstar.pdf')
plt.close()


# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.errorbar(sat_dist[mask_rec][0], meds_rec[mask_rec][0], yerr=np.array([[meds_rec[mask_rec][0]-lowers_rec[mask_rec][0]],[uppers_rec[mask_rec][0]-meds_rec[mask_rec][0]]]), color='#1542b0', alpha=0.35, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_rec][0], meds_rec[mask_rec][0], s=50, c='#1542b0', alpha=0.35, label='Recent')
axs.errorbar(sat_dist[mask_min][0], meds_min[mask_min][0], yerr=np.array([[meds_min[mask_min][0]-lowers_min[mask_min][0]],[uppers_min[mask_min][0]-meds_min[mask_min][0]]]), color='#f38e00', alpha=0.35, lw=3.5, capsize=0)
axs.scatter(sat_dist[mask_min][0], meds_min[mask_min][0], s=50, c='#f38e00', alpha=0.35, label='Minimum')
for i in range(1, len(sat_dist[mask_rec])):
    axs.errorbar(sat_dist[mask_rec][i], meds_rec[mask_rec][i], yerr=np.array([[meds_rec[mask_rec][i]-lowers_rec[mask_rec][i]],[uppers_rec[mask_rec][i]-meds_rec[mask_rec][i]]]), color='#1542b0', alpha=0.35, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_rec][i], meds_rec[mask_rec][i], s=50, c='#1542b0', alpha=0.35)
for i in range(1, len(sat_dist[mask_min])):
    axs.errorbar(sat_dist[mask_min][i], meds_min[mask_min][i], yerr=np.array([[meds_min[mask_min][i]-lowers_min[mask_min][i]],[uppers_min[mask_min][i]-meds_min[mask_min][i]]]), color='#f38e00', alpha=0.35, lw=3.5, capsize=0)
    axs.scatter(sat_dist[mask_min][i], meds_min[mask_min][i], s=50, c='#f38e00', alpha=0.35)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri}$ [km s$^{-1}$]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_both_vs_dist.pdf')
plt.close()


