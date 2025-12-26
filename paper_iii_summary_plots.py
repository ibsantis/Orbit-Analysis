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
from scipy.stats import spearmanr
from scipy.stats import pearsonr
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
galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12n', 'm12q', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']

# mw_sats = ['HIZSS 3(A)', 'HIZSS 3B', 'NGC 55', 'LMC', 'SMC', 'IC 4662', 'IC 5152', 'NGC 6822', 'NGC 3109', 'IC 3104', \
#            'Sextans B', 'DDO 190', 'DDO 125', 'Sextans A', 'NGC 4163', 'Sagittarius dSph', 'UGC 8508', 'Fornax', 'UGC 4879', \
#            'UGC 9128', 'GR 8', 'Leo A', 'Leo 1', 'Sagittarius dIrr', 'ESO 294-G010', 'DDO 113', 'Sculptor', 'Antlia 2', 'Aquarius (DDO 210)',\
#            'Phoenix', 'Leo 2', 'Antlia B', 'Tucana', 'KKR 3', 'Carina', 'Leo P', 'Crater 2', 'Ursa Minor', 'Sextans 1', \
#            'Draco', 'Canes Venatici 1', 'Leo T', 'Eridanus 2', 'Bootes 1', 'Hercules', 'Bootes 3', 'Sagittarius 2', \
#            'Canes Venatici 2', 'Ursa Major 1', 'Leo 4', 'Hydra 2', 'Hydrus 1', 'Carina 2', 'Ursa Major 2', 'Aquarius 2', \
#            'Indus 2', 'Coma Berenices', 'Leo 5', 'Pisces 2', 'Columba 1', 'Tucana 5', 'Pegasus 3', 'Grus 2', 'Tucana 2', \
#            'Reticulum 2', 'Horologium 1', 'Pictor 1', 'Tucana 4', 'Indus 1', 'Grus 1', 'Reticulum 3', 'Pictor 2', 'Bootes 2',\
#            'Willman 1', 'Phoenix 2', 'Cetus 3', 'Carina 3', 'Eridanus 3', 'Segue 2', 'Triangulum 2', 'Horologium 2', 'Tucana 3',\
#            'Segue 1', 'DES J0225+0304', 'Virgo 1', 'Draco 2', 'Cetus 2']

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
                
# mw_sats_1Mpc =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
#                     'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
#                     'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
#                     'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus III', 'Eridanus IV', \
#                     'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
#                     'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
#                     'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
#                     'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
#                     'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sagittarius II', 'Sculptor', 'Segue 1', \
#                     'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
#                     'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
#                     'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']

# mw_sats_proposal =  ['Horologium 2', 'Pisces 2', 'Hydra 2', 'Eridanus 2', 'Willman 1', 'Reticulum 3', \
#                      'Columba 1', 'Ursa Major 1', 'Pictor 1', 'Bootes 2', 'Grus 1', 'Tucana 5', 'Coma Berenices', \
#                      'Eridanus 3', 'Tucana 4', 'Triangulum 2', 'Sagittarius 2', 'Segue 1', 'Grus 2', 'Phoenix 2', \
#                      'Horologium 1', 'Tucana 2', 'Segue 2', 'Ursa Major 2', 'Reticulum 2', 'Ursa Minor', \
#                      'Canes Venatici 1', 'Sextans 1', 'Phoenix', 'Carina 2', 'Tucana 3', 'Carina', 'Fornax', \
#                      'Hydrus 1', 'Pegasus 3', 'Cetus 2', 'Virgo 1']

mw_sats_1Mpc_no_GC =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
                    'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
                    'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
                    'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus IV', \
                    'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
                    'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
                    'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
                    'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
                    'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sculptor', 'Segue 1', \
                    'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
                    'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
                    'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']


# Work on master plots
sat_mstar = []
sat_dist = []
v_tan = []
v_rad = []
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
#

n_555 = [12, 330, 378, 83, 131, 3, 2, 305, 42, 213, 15, 13, 75, 89, 125, 5, 243, 153, 43, 13, 26, 57, 125, 4, 99, 99, 111, 163, 301, 212, 49, 52, 0, 25, 167, 255, 347, 18, 67, 517, 362, 178, 13, 387, 127, 77, 363, 104, 954, 2, 4, 103, 20, 12, 42, 17, 15, 151, 0, 183, 113, 193, 261, 20, 62, 87, 45, 53]

#galaxy = 'Sculptor'
for sat_idx, galaxy in enumerate(mw_sats_1Mpc_no_GC):
    #
    satellite_name = galaxy.replace(' ', '_')
    if n_555[sat_idx] < 10:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_10_10_10/weights_{satellite_name}.txt'
    else:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
    #
    if len(gal_data['Host']) == 0:
        print(satellite_name)
        continue
    #
    sat_mstar.append(lg_data[galaxy]['mass.star'])
    sat_dist.append(lg_data[galaxy]['host.distance.total'])
    v_tan.append(lg_data[galaxy]['host.velocity.tan'])
    v_rad.append(lg_data[galaxy]['host.velocity.rad'])
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
v_tan = np.asarray(v_tan)
v_rad = np.asarray(v_rad)


def correlations(property1, property2, corr_type='spearman'):
    if corr_type == 'spearman':
        x1, x2 = spearmanr(property1, property2)
    else:
        x1, x2 = pearsonr(property1, property2)
    return x1, x2


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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#c76438', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#c76438', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#c76438', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#c76438', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#c76438', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], means[mask][i], s=75, c='#2b5b0c', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], means[mask][i], s=75, c='#2b5b0c', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], means[mask][i], s=75, c='#2b5b0c', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], means[mask][i], s=75, c='#2b5b0c', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], means[mask][i], s=75, c='#2b5b0c', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], means[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], means[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], means[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], means[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], means[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], means[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], means[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], means[mask], corr_type='pearson')}")


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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#432471', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#432471', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#432471', alpha=0.7)
axs.set_xlim(0,425)
#axs.set_ylim(-0.2, 8)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#432471', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#432471', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_rec_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")




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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#d05151', alpha=0.7)
axs.set_xscale('log')
axs.set_ylim(0, 160)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), alpha=0.5, color='#d05151', lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#d05151', alpha=0.7)
axs.set_ylim(0, 160)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#d05151', alpha=0.7)
axs.set_ylim(0, 160)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#d05151', alpha=0.7)
axs.set_ylim(0, 160)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#d05151', alpha=0.7)
axs.set_ylim(0, 160)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_rec_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#1542b0', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#1542b0', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#1542b0', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#1542b0', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#1542b0', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#1542b0', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, rec}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_rec_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#780d3f', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#780d3f', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#780d3f', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_min_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#780d3f', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_min_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#780d3f', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_min_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#572135', alpha=0.7)
axs.set_xscale('log')
axs.set_ylim(0, 160)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#572135', alpha=0.7)
axs.set_ylim(0, 160)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#572135', alpha=0.7)
axs.set_xlim(0,425)
axs.set_ylim(0, 160)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_min_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#572135', alpha=0.7)
axs.set_ylim(0, 160)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_min_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#572135', alpha=0.7)
axs.set_ylim(0, 160)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dperi_min_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#f38e00', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#f38e00', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#f38e00', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_min_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#f38e00', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_min_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#f38e00', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#f38e00', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri, min}$ [km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_min_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#6e1d16', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#6e1d16', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#6e1d16', alpha=0.7)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tapo_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#6e1d16', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tapo_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#6e1d16', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tapo_rec_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xlim(0,425)
axs.set_ylim(0,500)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dapo_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dapo_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dapo_rec_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")



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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i]/1e4, s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i]/1e4, s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_mstar[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i]/1e4, s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=75, c='#4e2026', alpha=0.7)
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
    axs.errorbar(sat_dist[mask][i], meds[mask][i]/1e4, yerr=np.array([[meds[mask][i]/1e4-lowers[mask][i]/1e4],[uppers[mask][i]/1e4-meds[mask][i]/1e4]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i]/1e4, s=75, c='#4e2026', alpha=0.7)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('Specific Kinetic Energy $[10^4\ km^2\ s^{-2}]$', fontsize=24)
axs.set_xlim(0, 420)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/ke_vs_dist_zoom.pdf')
plt.close()




dreion = np.array([[1349.4, 1331.1, 1933.0],
                [1433.7, 1022.1, 1999.6],
                [1627.9, 1168.4, 2267.1],
                [1707.8, 1067.4, 1914.7],
                [1468.8, 1054.6, 2064.1],
                [1196.0, 788.4, 1686.8],
                [2380.0, 1918.5, 2841.5],
                [1137.4, 735.2, 1727.6],
                [1499.7, 1240.0, 2167.9],
                [1564.8, 1141.3, 2292.7],
                [1506.6, 1248.7, 1918.4],
                [1740.4, 1492.2, 1858.9],
                [1447.6, 609.8, 2109.6],
                [1312.5, 973.5, 1615.9],
                [1451.1, 717.8, 1942.3],
                [3521.4, 2093.9, 3583.8],
                [1839.7, 1409.2, 2370.4],
                [1415.6, 879.9, 2070.8],
                [1414.0, 1233.2, 1723.5],
                [1509.4, 1480.3, 2136.7],
                [1287.6, 671.7, 1489.0],
                [2472.4, 1653.7, 2872.8],
                [1290.0, 794.3, 2398.0],
                [2233.3, 1907.0, 2378.0],
                [1367.1, 963.7, 1892.9],
                [1041.7, 844.8, 1619.7],
                [1716.3, 1457.0, 2357.7],
                [1227.1, 868.3, 1896.5],
                [1582.9, 1137.6, 2395.1],
                [1760.4, 1215.5, 2193.8],
                [1974.7, 1047.5, 2531.6],
                [1579.1, 1167.3, 2120.3],
                [2545.0, 2545.0, 2545.0],
                [1615.8, 1045.1, 1972.8],
                [1581.1, 1170.5, 2379.0],
                [1653.8, 1266.3, 2081.9],
                [1630.2, 1125.7, 2179.7],
                [2342.3, 1524.9, 2424.9],
                [2010.8, 1577.4, 3029.5],
                [1570.3, 1017.5, 2148.5],
                [1829.7, 1285.3, 2347.6],
                [1286.9, 846.6, 2071.4],
                [2871.8, 2108.8, 3011.3],
                [1682.1, 1165.6, 2233.8],
                [1828.5, 1322.4, 2403.0],
                [1467.7, 944.5, 2116.5],
                [1823.5, 1327.4, 2337.6],
                [972.8, 813.1, 1441.1],
                [1464.7, 988.4, 2105.5],
                [1489.5, 1120.6, 1489.5],
                [1606.9, 1238.8, 2169.9],
                [1114.8, 875.0, 1735.1],
                [1055.4, 735.7, 1418.1],
                [1794.5, 1622.7, 2027.7],
                [1685.5, 1360.8, 2874.2],
                [1324.6, 822.9, 1759.6],
                [3121.5, 3052.1, 3191.0],
                [1400.6, 906.1, 1781.8],
                [803.8, 803.8, 803.8],
                [1074.2, 830.9, 1602.2],
                [1124.4, 805.9, 1558.2],
                [1529.3, 1016.5, 2023.5],
                [1304.5, 822.5, 2064.0],
                [1972.6, 1450.6, 1980.8],
                [1623.0, 1073.7, 2512.3],
                [1347.2, 1074.1, 2049.8],
                [1983.2, 1600.1, 2502.8],
                [886.2, 499.0, 1105.4]])

"""
    Recent apocenter distance plots
"""
orbit_prop = np.asarray(dreion)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]/1000
lowers = orbit_prop[:,1]/1000
uppers = orbit_prop[:,2]/1000

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('Distance at $z=7$ [Mpc co-moving]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dreion_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xlim(xmin=0)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('Distance at $z=7$ [Mpc co-moving]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dreion_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(sat_mstar[mask])):
    axs.errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xlim(0,425)
#axs.set_ylim(0,500)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('Distance at $z=7$ [Mpc co-moving]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dreion_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_rad[mask])):
    axs.errorbar(v_rad[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('Distance at $z=7$ [Mpc co-moving]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dreion_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
for i in range(0, len(v_tan[mask])):
    axs.errorbar(v_tan[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('Distance at $z=7$ [Mpc co-moving]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dreion_rec_vs_vtan.pdf')
plt.close()

print("# Spearman correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='spearman')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='spearman')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='spearman')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='spearman')}")

print("# Pearson correlations with: ")
print(f"mass: {correlations(sat_mstar[mask], meds[mask], corr_type='pearson')}")
print(f"dist: {correlations(sat_dist[mask], meds[mask], corr_type='pearson')}")
print(f"vrad: {correlations(v_rad[mask], meds[mask], corr_type='pearson')}")
print(f"vtan: {correlations(v_tan[mask], meds[mask], corr_type='pearson')}")








LMC_idxs = [11, 12, 27, 30, 43, 47]


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
    axs[0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs[0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#c76438', alpha=0.7)

for i in range(0, len(sat_mstar[mask])):
    axs[1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs[1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#c76438', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/infall_plot_LMC_marked.pdf')
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
    axs[0].errorbar(sat_dist[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.5, lw=1.5, capsize=0)
    axs[0].scatter(sat_dist[mask][i], means[mask][i], s=75, c='#2b5b0c', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1].errorbar(sat_mstar[mask][i], means[mask][i], yerr=stds[mask][i], color='#2b5b0c', alpha=0.5, lw=1.5, capsize=0)
    axs[1].scatter(sat_mstar[mask][i], means[mask][i], s=75, c='#2b5b0c', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = means[i]
    yerr = stds[i]
    #
    axs[0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/nperi_plot_LMC_marked.pdf')
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
    axs[0,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.5, lw=1.5, capsize=0)
    axs[0,0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#432471', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[0,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#432471', alpha=0.5, lw=1.5, capsize=0)
    axs[0,1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#432471', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[mask][i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[0,0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0,0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[0,1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0,1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
#
## Pericenter distance
orbit_prop = np.asarray(dperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]
#
# Vs distance (zoom)
for i in range(0, len(sat_mstar[mask])):
    axs[1,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.5, lw=1.5, capsize=0)
    axs[1,0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#d05151', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#d05151', alpha=0.5, lw=1.5, capsize=0)
    axs[1,1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#d05151', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[1,0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1,0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[1,1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1,1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
#
axs[0,0].set_xlim(0,425)
axs[1,0].set_xlim(0,425)
axs[1,0].set_ylim(0,160)
axs[1,1].set_ylim(0,160)
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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/recent_peri_both_LMC_marked.pdf')
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
    axs[0,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.5, lw=1.5, capsize=0)
    axs[0,0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#780d3f', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[0,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#780d3f', alpha=0.5, lw=1.5, capsize=0)
    axs[0,1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#780d3f', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[0,0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0,0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[0,1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0,1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
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
    axs[1,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.5, lw=1.5, capsize=0)
    axs[1,0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#572135', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#572135', alpha=0.5, lw=1.5, capsize=0)
    axs[1,1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#572135', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[1,0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1,0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[1,1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1,1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
#
axs[0,0].set_xlim(0,425)
axs[1,0].set_xlim(0,425)
axs[1,0].set_ylim(0,160)
axs[1,1].set_ylim(0,160)
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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/minimum_peri_both_LMC_marked.pdf')
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
    axs[0,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.5, lw=1.5, capsize=0)
    axs[0,0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#6e1d16', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[0,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#6e1d16', alpha=0.5, lw=1.5, capsize=0)
    axs[0,1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#6e1d16', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[0,0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0,0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[0,1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0,1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
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
    axs[1,0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs[1,0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
#
# Vs Mstar
for i in range(0, len(sat_mstar[mask])):
    axs[1,1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#4e2026', alpha=0.5, lw=1.5, capsize=0)
    axs[1,1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#4e2026', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[1,0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1,0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[1,1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1,1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/apocenter_both_LMC_marked.pdf')
plt.close()


"""
    Reionization plots
"""
orbit_prop = np.asarray(dreion)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]/1000
lowers = orbit_prop[:,1]/1000
uppers = orbit_prop[:,2]/1000

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(16,6))
#
for i in range(0, len(sat_mstar[mask])):
    axs[0].errorbar(sat_dist[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#3E3A7A', alpha=0.5, lw=1.5, capsize=0)
    axs[0].scatter(sat_dist[mask][i], meds[mask][i], s=75, c='#3E3A7A', alpha=0.7)

for i in range(0, len(sat_mstar[mask])):
    axs[1].errorbar(sat_mstar[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color='#3E3A7A', alpha=0.5, lw=1.5, capsize=0)
    axs[1].scatter(sat_mstar[mask][i], meds[mask][i], s=75, c='#3E3A7A', alpha=0.7)
#
for i in LMC_idxs:
    #
    x_dist = sat_dist[i]
    x_mstar = sat_mstar[i]
    y = meds[i]
    yerr = np.array([[y - lowers[i]], [uppers[i] - y]])
    #
    axs[0].errorbar(x_dist, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[0].scatter(x_dist, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)

    # --- Stellar mass panel (right) ---
    axs[1].errorbar(x_mstar, y, yerr=yerr, color='k', lw=1.5, capsize=0, zorder=6, alpha=0.7)
    axs[1].scatter(x_mstar, y, s=75, marker='*', color='k', edgecolor='k', zorder=7, alpha=0.7)
#
axs[0].set_xlim(0,425)
#
axs[1].set_xscale('log')
#
axs[0].set_xlabel('Distance from MW [kpc]', fontsize=24)
axs[1].set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs[0].set_ylabel('Distance at $z=7$ [Mpc co-mov.]', fontsize=22)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/dreion_plot_LMC_marked.pdf')
plt.close()



"""
    tperi vs dperi
"""
from matplotlib.colors import LogNorm, Normalize
orbit_prop_1 = np.asarray(dperi_rec)
orbit_prop_2 = np.asarray(tperi_rec)
mask = (orbit_prop_1[:,0] != -1)&(orbit_prop_2[:,0] != -1)
meds_1 = orbit_prop_1[:,0]
lowers_1 = orbit_prop_1[:,1]
uppers_1 = orbit_prop_1[:,2]
meds_2 = orbit_prop_2[:,0]
lowers_2 = orbit_prop_2[:,1]
uppers_2 = orbit_prop_2[:,2]

LMC_idxs = [11, 12, 27, 30, 43, 47]

norm_d = Normalize(vmin=np.min(sat_dist[mask]),  vmax=np.max(sat_dist[mask]))
norm_vr = Normalize(vmin=np.min(v_rad[mask]),  vmax=np.max(v_rad[mask]))
norm_vt = Normalize(vmin=np.min(v_tan[mask]),  vmax=np.max(v_tan[mask]))
norm_m = LogNorm(vmin=np.min(sat_mstar[mask]), vmax=np.max(sat_mstar[mask]))
lmc_mask = np.zeros_like(mask, dtype=bool)
lmc_mask[LMC_idxs] = True
valid_lmc_mask     = mask & lmc_mask
valid_non_lmc_mask = mask & (~lmc_mask)

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 2, figsize=(16,12))
#
sc0 = axs[0,0].scatter(meds_2[valid_non_lmc_mask], meds_1[valid_non_lmc_mask], s=75, c=sat_dist[valid_non_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, norm=norm_d)
plt.colorbar(sc0, ax=axs[0,0], label=r'$d_{\rm sat}$ [kpc]')
axs[0,0].scatter( meds_2[valid_lmc_mask], meds_1[valid_lmc_mask], s=100, marker='*', c=sat_dist[valid_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, edgecolor=None, zorder=7, norm=norm_d)
#
sc1 = axs[0,1].scatter(meds_2[valid_non_lmc_mask], meds_1[valid_non_lmc_mask], s=75, c=sat_mstar[valid_non_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, norm=norm_m)
plt.colorbar(sc1, ax=axs[0,1], label=r'$M_{\rm star}$ [$M_{\odot}$]]')
axs[0,1].scatter( meds_2[valid_lmc_mask], meds_1[valid_lmc_mask], s=100, marker='*', c=sat_mstar[valid_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, edgecolor=None, zorder=7, norm=norm_m)
#
sc2 = axs[1,0].scatter(meds_2[valid_non_lmc_mask], meds_1[valid_non_lmc_mask], s=75, c=v_rad[valid_non_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, norm=norm_vr)
plt.colorbar(sc2, ax=axs[1,0], label=r'$v_{\rm rad}$ [km/s]')
axs[1,0].scatter( meds_2[valid_lmc_mask], meds_1[valid_lmc_mask], s=100, marker='*', c=v_rad[valid_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, edgecolor=None, zorder=7, norm=norm_vr)
#
sc3 = axs[1,1].scatter(meds_2[valid_non_lmc_mask], meds_1[valid_non_lmc_mask], s=75, c=v_tan[valid_non_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, norm=norm_vt)
plt.colorbar(sc3, ax=axs[1,1], label=r'$v_{\rm tan}$ [km/s]')
axs[1,1].scatter( meds_2[valid_lmc_mask], meds_1[valid_lmc_mask], s=100, marker='*', c=v_tan[valid_lmc_mask], cmap=plt.cm.plasma, alpha=0.7, edgecolor=None, zorder=7, norm=norm_vt)
#
axs[1,0].set_xlabel('$t_{\\rm peri,rec}$ [Gyr]', fontsize=24)
axs[1,1].set_xlabel('$t_{\\rm peri,rec}$ [Gyr]', fontsize=24)
axs[0,0].set_ylabel('$d_{\\rm peri,rec}$ [kpc]', fontsize=24)
axs[1,0].set_ylabel('$d_{\\rm peri,rec}$ [kpc]', fontsize=24)
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False, labelleft=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/tperi_vs_dperi.pdf')
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
axs.errorbar(sat_mstar[mask_rec][0], meds_rec[mask_rec][0], yerr=np.array([[meds_rec[mask_rec][0]-lowers_rec[mask_rec][0]],[uppers_rec[mask_rec][0]-meds_rec[mask_rec][0]]]), color='#1542b0', alpha=0.35, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_rec][0], meds_rec[mask_rec][0], s=75, c='#1542b0', alpha=0.35, label='Recent')
axs.errorbar(sat_mstar[mask_min][0], meds_min[mask_min][0], yerr=np.array([[meds_min[mask_min][0]-lowers_min[mask_min][0]],[uppers_min[mask_min][0]-meds_min[mask_min][0]]]), color='#f38e00', alpha=0.35, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_min][0], meds_min[mask_min][0], s=75, c='#f38e00', alpha=0.35, label='Minimum')
for i in range(1, len(sat_mstar[mask_rec])):
    axs.errorbar(sat_mstar[mask_rec][i], meds_rec[mask_rec][i], yerr=np.array([[meds_rec[mask_rec][i]-lowers_rec[mask_rec][i]],[uppers_rec[mask_rec][i]-meds_rec[mask_rec][i]]]), color='#1542b0', alpha=0.35, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_rec][i], meds_rec[mask_rec][i], s=75, c='#1542b0', alpha=0.35)
for i in range(1, len(sat_mstar[mask_min])):
    axs.errorbar(sat_mstar[mask_min][i], meds_min[mask_min][i], yerr=np.array([[meds_min[mask_min][i]-lowers_min[mask_min][i]],[uppers_min[mask_min][i]-meds_min[mask_min][i]]]), color='#f38e00', alpha=0.35, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_min][i], meds_min[mask_min][i], s=75, c='#f38e00', alpha=0.35)
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
axs.errorbar(sat_dist[mask_rec][0], meds_rec[mask_rec][0], yerr=np.array([[meds_rec[mask_rec][0]-lowers_rec[mask_rec][0]],[uppers_rec[mask_rec][0]-meds_rec[mask_rec][0]]]), color='#1542b0', alpha=0.35, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_rec][0], meds_rec[mask_rec][0], s=75, c='#1542b0', alpha=0.35, label='Recent')
axs.errorbar(sat_dist[mask_min][0], meds_min[mask_min][0], yerr=np.array([[meds_min[mask_min][0]-lowers_min[mask_min][0]],[uppers_min[mask_min][0]-meds_min[mask_min][0]]]), color='#f38e00', alpha=0.35, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_min][0], meds_min[mask_min][0], s=75, c='#f38e00', alpha=0.35, label='Minimum')
for i in range(1, len(sat_dist[mask_rec])):
    axs.errorbar(sat_dist[mask_rec][i], meds_rec[mask_rec][i], yerr=np.array([[meds_rec[mask_rec][i]-lowers_rec[mask_rec][i]],[uppers_rec[mask_rec][i]-meds_rec[mask_rec][i]]]), color='#1542b0', alpha=0.35, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_rec][i], meds_rec[mask_rec][i], s=75, c='#1542b0', alpha=0.35)
for i in range(1, len(sat_dist[mask_min])):
    axs.errorbar(sat_dist[mask_min][i], meds_min[mask_min][i], yerr=np.array([[meds_min[mask_min][i]-lowers_min[mask_min][i]],[uppers_min[mask_min][i]-meds_min[mask_min][i]]]), color='#f38e00', alpha=0.35, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_min][i], meds_min[mask_min][i], s=75, c='#f38e00', alpha=0.35)
axs.set_xlim(0,425)
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$v_{\\rm peri}$ [km s$^{-1}$]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/summary_1Mpc/vperi_both_vs_dist.pdf')
plt.close()


