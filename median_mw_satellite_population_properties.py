#!/usr/bin/python3

"""
    =============================================
    = Median MW satellite population properties =
    =============================================

    This will find the median and 68 percentile for each orbit property
    across the entire population. One value for the whole population.
    
    This is to test how the different selection criteria affect
    summary statistics.

    If I instead were to try taking the median across the population in the
    "simulation/orbit_data/paper_III/floor_tests_headers/physical/<property>.csv"
    then I would be taking a median of a bunch of medians.

    Instead, in this script, I get all of the orbit property values for
    each satellite analog in one giant array (e.g. all 14358 galaxies
    for the 10/10/10 selection) and then take the weighted median afterward, which
    is better than a median of medians.

"""

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

# Set up iterables and empty arrays
floor_list = [(1, 1, 1),
              (1, 3, 3),
              (1, 3, 5),
              (1, 5, 5),
              (1, 7, 7),
              (1, 10, 10),
              (3, 1, 1),
              (3, 3, 3),
              (3, 3, 5),
              (3, 5, 5),
              (3, 7, 7),
              (3, 10, 10),
              (5, 1, 1),
              (5, 3, 3),
              (5, 3, 5),
              (5, 5, 5),
              (5, 7, 7),
              (5, 10, 10),
              (7, 1, 1),
              (7, 3, 3),
              (7, 3, 5),
              (7, 5, 5),
              (7, 7, 7),
              (7, 10, 10),
              (10, 1, 1),
              (10, 3, 3),
              (10, 3, 5),
              (10, 5, 5),
              (10, 7, 7),
              (10, 10, 10)]
#
first_infall = (-1)*np.ones(len(floor_list))
nperi = (-1)*np.ones(len(floor_list))
tperi_rec = (-1)*np.ones(len(floor_list))
dperi_rec = (-1)*np.ones(len(floor_list))
vperi_rec = (-1)*np.ones(len(floor_list))
tperi_min = (-1)*np.ones(len(floor_list))
dperi_min = (-1)*np.ones(len(floor_list))
vperi_min = (-1)*np.ones(len(floor_list))
tapo_rec = (-1)*np.ones(len(floor_list))
dapo_rec = (-1)*np.ones(len(floor_list))
elltot = (-1)*np.ones(len(floor_list))
ketot = (-1)*np.ones(len(floor_list))
#
first_infall_68 = (-1)*np.ones(len(floor_list))
nperi_68 = (-1)*np.ones(len(floor_list))
tperi_rec_68 = (-1)*np.ones(len(floor_list))
dperi_rec_68 = (-1)*np.ones(len(floor_list))
vperi_rec_68 = (-1)*np.ones(len(floor_list))
tperi_min_68 = (-1)*np.ones(len(floor_list))
dperi_min_68 = (-1)*np.ones(len(floor_list))
vperi_min_68 = (-1)*np.ones(len(floor_list))
tapo_rec_68 = (-1)*np.ones(len(floor_list))
dapo_rec_68 = (-1)*np.ones(len(floor_list))
elltot_68 = (-1)*np.ones(len(floor_list))
ketot_68 = (-1)*np.ones(len(floor_list))
#
######
loop_start = time.time()

for i, (floor_d, floor_vr, floor_vt) in enumerate(floor_list):
    #
    floor_d_str = str(floor_d)
    floor_vr_str = str(floor_vr)
    floor_vt_str = str(floor_vt)
    #
    temp_infall = []
    temp_nperi = []
    temp_tperi_rec = []
    temp_dperi_rec = []
    temp_vperi_rec = []
    temp_tperi_min = []
    temp_dperi_min = []
    temp_vperi_min = []
    temp_tapo = []
    temp_dapo = []
    temp_ell = []
    temp_ke = []
    #
    weight_infall = []
    weight_nperi = []
    weight_tperi_rec = []
    weight_dperi_rec = []
    weight_vperi_rec = []
    weight_tperi_min = []
    weight_dperi_min = []
    weight_vperi_min = []
    weight_tapo = []
    weight_dapo = []
    weight_ell = []
    weight_ke = []
    #
    for j, galaxy in enumerate(mw_sats_1Mpc):
        #
        satellite_name = galaxy.replace(' ', '_')
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_{floor_d_str}_{floor_vr_str}_{floor_vt_str}/weights_{satellite_name}.txt'
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
        # Infall time
        m = (orbit_dictionary['first.infall.time.lb'] != -1)
        if np.sum(m) != 0:
            temp_infall.append(orbit_dictionary['first.infall.time.lb'][m])
            weight_infall.append(gal_data['Weight'][m])
        #
        # Pericenter number
        m = (orbit_dictionary['pericenter.num'] != -1)
        if np.sum(m) != 0:
            temp_nperi.append(orbit_dictionary['pericenter.num'][m])
            weight_nperi.append(gal_data['Weight'][m]) 
        #
        # Recent pericenter time
        m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
        if np.sum(m) != 0:
            temp_tperi_rec.append(orbit_dictionary['pericenter.rec.time.lb'][m])
            weight_tperi_rec.append(gal_data['Weight'][m])
        #
        # Recent pericenter distance
        m = (orbit_dictionary['pericenter.rec.dist'] != -1)
        if np.sum(m) != 0:
            temp_dperi_rec.append(orbit_dictionary['pericenter.rec.dist'][m])
            weight_dperi_rec.append(gal_data['Weight'][m])
        #
        # Recent pericenter velocity
        m = (orbit_dictionary['pericenter.rec.vel'] != -1)
        if np.sum(m) != 0:
            temp_vperi_rec.append(orbit_dictionary['pericenter.rec.vel'][m])
            weight_vperi_rec.append(gal_data['Weight'][m])
        #
        # Minimum pericenter time
        m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
        if np.sum(m) != 0:
            temp_tperi_min.append(orbit_dictionary['pericenter.min.time.lb'][m])
            weight_tperi_min.append(gal_data['Weight'][m])
        #
        # Minimum pericenter distance
        m = (orbit_dictionary['pericenter.min.dist'] != -1)
        if np.sum(m) != 0:
            temp_dperi_min.append(orbit_dictionary['pericenter.min.dist'][m])
            weight_dperi_min.append(gal_data['Weight'][m])
        #
        # Minimum pericenter velocity
        m = (orbit_dictionary['pericenter.min.vel'] != -1)
        if np.sum(m) != 0:
            temp_vperi_min.append(orbit_dictionary['pericenter.min.vel'][m])
            weight_vperi_min.append(gal_data['Weight'][m])
        #
        # Apocenter time
        m = (orbit_dictionary['apocenter.time.lb'] != -1)
        if np.sum(m) != 0:
            temp_tapo.append(orbit_dictionary['apocenter.time.lb'][m])
            weight_tapo.append(gal_data['Weight'][m])
        #
        # Apocenter distance
        m = (orbit_dictionary['apocenter.dist'] != -1)
        if np.sum(m) != 0:
            temp_dapo.append(orbit_dictionary['apocenter.dist'][m])
            weight_dapo.append(gal_data['Weight'][m])
        #
        # Kinetic energy (specific KE at match)
        m = (orbit_dictionary['v.tot.sim'] != -1)
        if np.sum(m) != 0:
            temp_ke.append(0.5*orbit_dictionary['v.tot.sim'][m]**2)
            weight_ke.append(gal_data['Weight'][m])
        #
        # Specific angular momentum
        m = (orbit_dictionary['L.tot.sim'] != -1)
        if np.sum(m) != 0:
            temp_ell.append(orbit_dictionary['L.tot.sim'][m])
            weight_ell.append(gal_data['Weight'][m])
    # Infall time
    temp_infall = np.hstack(temp_infall)
    weight_infall = np.hstack(weight_infall)
    first_infall[i] = ut.math.percentile_weighted(temp_infall, 50, weight_infall)
    x_lower = ut.math.percentile_weighted(temp_infall, 15.87, weight_infall)
    x_upper = ut.math.percentile_weighted(temp_infall, 84.13, weight_infall)
    first_infall_68[i] = x_upper - x_lower
    #
    # Pericenter number
    temp_nperi = np.hstack(temp_nperi)
    weight_nperi = np.hstack(weight_nperi)
    nperi[i] = np.sum(temp_nperi*weight_nperi)
    nperi_68[i] = np.sqrt(np.sum((temp_nperi-nperi[i])**2*weight_nperi)/np.sum(weight_nperi)/np.sum(weight_nperi))
    #
    # Recent pericenter time
    temp_tperi_rec = np.hstack(temp_tperi_rec)
    weight_tperi_rec = np.hstack(weight_tperi_rec)
    tperi_rec[i] = ut.math.percentile_weighted(temp_tperi_rec, 50, weight_tperi_rec)
    x_lower = ut.math.percentile_weighted(temp_tperi_rec, 15.87, weight_tperi_rec)
    x_upper = ut.math.percentile_weighted(temp_tperi_rec, 84.13, weight_tperi_rec)
    tperi_rec_68[i] = x_upper - x_lower
    #
    # Recent pericenter distance
    temp_dperi_rec = np.hstack(temp_dperi_rec)
    weight_dperi_rec = np.hstack(weight_dperi_rec)
    dperi_rec[i] = ut.math.percentile_weighted(temp_dperi_rec, 50, weight_dperi_rec)
    x_lower = ut.math.percentile_weighted(temp_dperi_rec, 15.87, weight_dperi_rec)
    x_upper = ut.math.percentile_weighted(temp_dperi_rec, 84.13, weight_dperi_rec)
    dperi_rec_68[i] = x_upper - x_lower
    #
    # Recent pericenter velocity
    temp_vperi_rec = np.hstack(temp_vperi_rec)
    weight_vperi_rec = np.hstack(weight_vperi_rec)
    vperi_rec[i] = ut.math.percentile_weighted(temp_vperi_rec, 50, weight_vperi_rec)
    x_lower = ut.math.percentile_weighted(temp_vperi_rec, 15.87, weight_vperi_rec)
    x_upper = ut.math.percentile_weighted(temp_vperi_rec, 84.13, weight_vperi_rec)
    vperi_rec_68[i] = x_upper - x_lower
    #
    # Minimum pericenter time
    temp_tperi_min = np.hstack(temp_tperi_min)
    weight_tperi_min = np.hstack(weight_tperi_min)
    tperi_min[i] = ut.math.percentile_weighted(temp_tperi_min, 50, weight_tperi_min)
    x_lower = ut.math.percentile_weighted(temp_tperi_min, 15.87, weight_tperi_min)
    x_upper = ut.math.percentile_weighted(temp_tperi_min, 84.13, weight_tperi_min)
    tperi_min_68[i] = x_upper - x_lower
    #
    # Minimum pericenter distance
    temp_dperi_min = np.hstack(temp_dperi_min)
    weight_dperi_min = np.hstack(weight_dperi_min)
    dperi_min[i] = ut.math.percentile_weighted(temp_dperi_min, 50, weight_dperi_min)
    x_lower = ut.math.percentile_weighted(temp_dperi_min, 15.87, weight_dperi_min)
    x_upper = ut.math.percentile_weighted(temp_dperi_min, 84.13, weight_dperi_min)
    dperi_min_68[i] = x_upper - x_lower
    #
    # Minimum pericenter velocity
    temp_vperi_min = np.hstack(temp_vperi_min)
    weight_vperi_min = np.hstack(weight_vperi_min)
    vperi_min[i] = ut.math.percentile_weighted(temp_vperi_min, 50, weight_vperi_min)
    x_lower = ut.math.percentile_weighted(temp_vperi_min, 15.87, weight_vperi_min)
    x_upper = ut.math.percentile_weighted(temp_vperi_min, 84.13, weight_vperi_min)
    vperi_min_68[i] = x_upper - x_lower
    #
    # Apocenter time
    temp_tapo = np.hstack(temp_tapo)
    weight_tapo = np.hstack(weight_tapo)
    tapo_rec[i] = ut.math.percentile_weighted(temp_tapo, 50, weight_tapo)
    x_lower = ut.math.percentile_weighted(temp_tapo, 15.87, weight_tapo)
    x_upper = ut.math.percentile_weighted(temp_tapo, 84.13, weight_tapo)
    tapo_rec_68[i] = x_upper - x_lower
    #
    # Apocenter distance
    temp_dapo = np.hstack(temp_dapo)
    weight_dapo = np.hstack(weight_dapo)
    dapo_rec[i] = ut.math.percentile_weighted(temp_dapo, 50, weight_dapo)
    x_lower = ut.math.percentile_weighted(temp_dapo, 15.87, weight_dapo)
    x_upper = ut.math.percentile_weighted(temp_dapo, 84.13, weight_dapo)
    dapo_rec_68[i] = x_upper - x_lower
    #
    # Kinetic Energy
    temp_ke = np.hstack(temp_ke)
    weight_ke = np.hstack(weight_ke)
    ketot[i] = ut.math.percentile_weighted(temp_ke, 50, weight_ke)
    x_lower = ut.math.percentile_weighted(temp_ke, 15.87, weight_ke)
    x_upper = ut.math.percentile_weighted(temp_ke, 84.13, weight_ke)
    ketot_68[i] = x_upper - x_lower
    #
    # Specific angular momentum
    temp_ell = np.hstack(temp_ell)
    weight_ell = np.hstack(weight_ell)
    elltot[i] = ut.math.percentile_weighted(temp_ell, 50, weight_ell)
    x_lower = ut.math.percentile_weighted(temp_ell, 15.87, weight_ell)
    x_upper = ut.math.percentile_weighted(temp_ell, 84.13, weight_ell)
    elltot_68[i] = x_upper - x_lower

loop_end = time.time()
print(f'Loop ended in {loop_end - loop_start} seconds!')


# Create the fiducial array
temp_infall = []
temp_nperi = []
temp_tperi_rec = []
temp_dperi_rec = []
temp_vperi_rec = []
temp_tperi_min = []
temp_dperi_min = []
temp_vperi_min = []
temp_tapo = []
temp_dapo = []
temp_ell = []
temp_ke = []
#
weight_infall = []
weight_nperi = []
weight_tperi_rec = []
weight_dperi_rec = []
weight_vperi_rec = []
weight_tperi_min = []
weight_dperi_min = []
weight_vperi_min = []
weight_tapo = []
weight_dapo = []
weight_ell = []
weight_ke = []
#
for j, galaxy in enumerate(mw_sats_1Mpc):
    #
    satellite_name = galaxy.replace(' ', '_')
    file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{satellite_name}.txt'
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
    # Infall time
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m) != 0:
        temp_infall.append(orbit_dictionary['first.infall.time.lb'][m])
        weight_infall.append(gal_data['Weight'][m])
    #
    # Pericenter number
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m) != 0:
        temp_nperi.append(orbit_dictionary['pericenter.num'][m])
        weight_nperi.append(gal_data['Weight'][m]) 
    #
    # Recent pericenter time
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        temp_tperi_rec.append(orbit_dictionary['pericenter.rec.time.lb'][m])
        weight_tperi_rec.append(gal_data['Weight'][m])
    #
    # Recent pericenter distance
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        temp_dperi_rec.append(orbit_dictionary['pericenter.rec.dist'][m])
        weight_dperi_rec.append(gal_data['Weight'][m])
    #
    # Recent pericenter velocity
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        temp_vperi_rec.append(orbit_dictionary['pericenter.rec.vel'][m])
        weight_vperi_rec.append(gal_data['Weight'][m])
    #
    # Minimum pericenter time
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        temp_tperi_min.append(orbit_dictionary['pericenter.min.time.lb'][m])
        weight_tperi_min.append(gal_data['Weight'][m])
    #
    # Minimum pericenter distance
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        temp_dperi_min.append(orbit_dictionary['pericenter.min.dist'][m])
        weight_dperi_min.append(gal_data['Weight'][m])
    #
    # Minimum pericenter velocity
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        temp_vperi_min.append(orbit_dictionary['pericenter.min.vel'][m])
        weight_vperi_min.append(gal_data['Weight'][m])
    #
    # Apocenter time
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        temp_tapo.append(orbit_dictionary['apocenter.time.lb'][m])
        weight_tapo.append(gal_data['Weight'][m])
    #
    # Apocenter distance
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        temp_dapo.append(orbit_dictionary['apocenter.dist'][m])
        weight_dapo.append(gal_data['Weight'][m])
    #
    # Kinetic energy (specific KE at match)
    m = (orbit_dictionary['v.tot.sim'] != -1)
    if np.sum(m) != 0:
        temp_ke.append(0.5*orbit_dictionary['v.tot.sim'][m]**2)
        weight_ke.append(gal_data['Weight'][m])
    #
    # Specific angular momentum
    m = (orbit_dictionary['L.tot.sim'] != -1)
    if np.sum(m) != 0:
        temp_ell.append(orbit_dictionary['L.tot.sim'][m])
        weight_ell.append(gal_data['Weight'][m])
# Infall time
temp_infall = np.hstack(temp_infall)
weight_infall = np.hstack(weight_infall)
first_infall_fid = ut.math.percentile_weighted(temp_infall, 50, weight_infall)
x_lower = ut.math.percentile_weighted(temp_infall, 15.87, weight_infall)
x_upper = ut.math.percentile_weighted(temp_infall, 84.13, weight_infall)
first_infall_68_fid = x_upper - x_lower
#
# Pericenter number
temp_nperi = np.hstack(temp_nperi)
weight_nperi = np.hstack(weight_nperi)
nperi_fid = np.sum(temp_nperi*weight_nperi)
nperi_68_fid = np.sqrt(np.sum((temp_nperi-nperi_fid)**2*weight_nperi)/np.sum(weight_nperi)/np.sum(weight_nperi))
#
# Recent pericenter time
temp_tperi_rec = np.hstack(temp_tperi_rec)
weight_tperi_rec = np.hstack(weight_tperi_rec)
tperi_rec_fid = ut.math.percentile_weighted(temp_tperi_rec, 50, weight_tperi_rec)
x_lower = ut.math.percentile_weighted(temp_tperi_rec, 15.87, weight_tperi_rec)
x_upper = ut.math.percentile_weighted(temp_tperi_rec, 84.13, weight_tperi_rec)
tperi_rec_68_fid = x_upper - x_lower
#
# Recent pericenter distance
temp_dperi_rec = np.hstack(temp_dperi_rec)
weight_dperi_rec = np.hstack(weight_dperi_rec)
dperi_rec_fid = ut.math.percentile_weighted(temp_dperi_rec, 50, weight_dperi_rec)
x_lower = ut.math.percentile_weighted(temp_dperi_rec, 15.87, weight_dperi_rec)
x_upper = ut.math.percentile_weighted(temp_dperi_rec, 84.13, weight_dperi_rec)
dperi_rec_68_fid = x_upper - x_lower
#
# Recent pericenter velocity
temp_vperi_rec = np.hstack(temp_vperi_rec)
weight_vperi_rec = np.hstack(weight_vperi_rec)
vperi_rec_fid = ut.math.percentile_weighted(temp_vperi_rec, 50, weight_vperi_rec)
x_lower = ut.math.percentile_weighted(temp_vperi_rec, 15.87, weight_vperi_rec)
x_upper = ut.math.percentile_weighted(temp_vperi_rec, 84.13, weight_vperi_rec)
vperi_rec_68_fid = x_upper - x_lower
#
# Minimum pericenter time
temp_tperi_min = np.hstack(temp_tperi_min)
weight_tperi_min = np.hstack(weight_tperi_min)
tperi_min_fid = ut.math.percentile_weighted(temp_tperi_min, 50, weight_tperi_min)
x_lower = ut.math.percentile_weighted(temp_tperi_min, 15.87, weight_tperi_min)
x_upper = ut.math.percentile_weighted(temp_tperi_min, 84.13, weight_tperi_min)
tperi_min_68_fid = x_upper - x_lower
#
# Minimum pericenter distance
temp_dperi_min = np.hstack(temp_dperi_min)
weight_dperi_min = np.hstack(weight_dperi_min)
dperi_min_fid = ut.math.percentile_weighted(temp_dperi_min, 50, weight_dperi_min)
x_lower = ut.math.percentile_weighted(temp_dperi_min, 15.87, weight_dperi_min)
x_upper = ut.math.percentile_weighted(temp_dperi_min, 84.13, weight_dperi_min)
dperi_min_68_fid = x_upper - x_lower
#
# Minimum pericenter velocity
temp_vperi_min = np.hstack(temp_vperi_min)
weight_vperi_min = np.hstack(weight_vperi_min)
vperi_min_fid = ut.math.percentile_weighted(temp_vperi_min, 50, weight_vperi_min)
x_lower = ut.math.percentile_weighted(temp_vperi_min, 15.87, weight_vperi_min)
x_upper = ut.math.percentile_weighted(temp_vperi_min, 84.13, weight_vperi_min)
vperi_min_68_fid = x_upper - x_lower
#
# Apocenter time
temp_tapo = np.hstack(temp_tapo)
weight_tapo = np.hstack(weight_tapo)
tapo_rec_fid = ut.math.percentile_weighted(temp_tapo, 50, weight_tapo)
x_lower = ut.math.percentile_weighted(temp_tapo, 15.87, weight_tapo)
x_upper = ut.math.percentile_weighted(temp_tapo, 84.13, weight_tapo)
tapo_rec_68_fid = x_upper - x_lower
#
# Apocenter distance
temp_dapo = np.hstack(temp_dapo)
weight_dapo = np.hstack(weight_dapo)
dapo_rec_fid = ut.math.percentile_weighted(temp_dapo, 50, weight_dapo)
x_lower = ut.math.percentile_weighted(temp_dapo, 15.87, weight_dapo)
x_upper = ut.math.percentile_weighted(temp_dapo, 84.13, weight_dapo)
dapo_rec_68_fid = x_upper - x_lower
#
# Kinetic Energy
temp_ke = np.hstack(temp_ke)
weight_ke = np.hstack(weight_ke)
ketot_fid = ut.math.percentile_weighted(temp_ke, 50, weight_ke)
x_lower = ut.math.percentile_weighted(temp_ke, 15.87, weight_ke)
x_upper = ut.math.percentile_weighted(temp_ke, 84.13, weight_ke)
ketot_68_fid = x_upper - x_lower
#
# Specific angular momentum
temp_ell = np.hstack(temp_ell)
weight_ell = np.hstack(weight_ell)
elltot_fid = ut.math.percentile_weighted(temp_ell, 50, weight_ell)
x_lower = ut.math.percentile_weighted(temp_ell, 15.87, weight_ell)
x_upper = ut.math.percentile_weighted(temp_ell, 84.13, weight_ell)
elltot_68_fid = x_upper - x_lower


# Add in the fiducial values to the beginning of the arrays since I didn't do that in the first place...
first_infall = np.insert(first_infall, 0, first_infall_fid)
first_infall_68 = np.insert(first_infall_68, 0, first_infall_68_fid)
nperi = np.insert(nperi, 0, nperi_fid)
nperi_68 = np.insert(nperi_68, 0, nperi_68_fid)
tperi_rec = np.insert(tperi_rec, 0, tperi_rec_fid)
tperi_rec_68 = np.insert(tperi_rec_68, 0, tperi_rec_68_fid)
dperi_rec = np.insert(dperi_rec, 0, dperi_rec_fid)
dperi_rec_68 = np.insert(dperi_rec_68, 0, dperi_rec_68_fid)
vperi_rec = np.insert(vperi_rec, 0, vperi_rec_fid)
vperi_rec_68 = np.insert(vperi_rec_68, 0, vperi_rec_68_fid)
tperi_min = np.insert(tperi_min, 0, tperi_min_fid)
tperi_min_68 = np.insert(tperi_min_68, 0, tperi_min_68_fid)
dperi_min = np.insert(dperi_min, 0, dperi_min_fid)
dperi_min_68 = np.insert(dperi_min_68, 0, dperi_min_68_fid)
vperi_min = np.insert(vperi_min, 0, vperi_min_fid)
vperi_min_68 = np.insert(vperi_min_68, 0, vperi_min_68_fid)
tapo_rec = np.insert(tapo_rec, 0, tapo_rec_fid)
tapo_rec_68 = np.insert(tapo_rec_68, 0, tapo_rec_68_fid)
dapo_rec = np.insert(dapo_rec, 0, dapo_rec_fid)
dapo_rec_68 = np.insert(dapo_rec_68, 0, dapo_rec_68_fid)
ketot = np.insert(ketot, 0, ketot_fid)
ketot_68 = np.insert(ketot_68, 0, ketot_68_fid)
elltot = np.insert(elltot, 0, elltot_fid)
elltot_68 = np.insert(elltot_68, 0, elltot_68_fid)

# Now put everything into a single array for pandas
data = np.array([first_infall, 
                 first_infall_68,
                 nperi,
                 nperi_68,
                 tperi_rec,
                 tperi_rec_68,
                 dperi_rec,
                 dperi_rec_68,
                 vperi_rec,
                 vperi_rec_68,
                 tperi_min,
                 tperi_min_68,
                 dperi_min,
                 dperi_min_68,
                 vperi_min,
                 vperi_min_68,
                 tapo_rec,
                 tapo_rec_68,
                 dapo_rec,
                 dapo_rec_68,
                 ketot,
                 ketot_68,
                 elltot,
                 elltot_68])



row_labels = ['infall', 'infall.68', 
              'n.peri', 'n.peri.68', 
              't.peri.rec', 't.peri.rec.68', 
              'd.peri.rec', 'd.peri.rec.68', 
              'v.peri.rec', 'v.peri.rec.68', 
              't.peri.min', 't.peri.min.68', 
              'd.peri.min', 'd.peri.min.68', 
              'v.peri.min', 'v.peri.min.68', 
              't.apo', 't.apo.68', 
              'd.apo', 'd.apo.68', 
              'ke', 'ke.68', 'ell', 'ell.68']

column_labels = ['Fiducial',
              '1, 1, 1',
              '1, 3, 3',
              '1, 3, 5',
              '1, 5, 5',
              '1, 7, 7',
              '1, 10, 10',
              '3, 1, 1',
              '3, 3, 3',
              '3, 3, 5',
              '3, 5, 5',
              '3, 7, 7',
              '3, 10, 10',
              '5, 1, 1',
              '5, 3, 3',
              '5, 3, 5',
              '5, 5, 5',
              '5, 7, 7',
              '5, 10, 10',
              '7, 1, 1',
              '7, 3, 3',
              '7, 3, 5',
              '7, 5, 5',
              '7, 7, 7',
              '7, 10, 10',
              '10, 1, 1',
              '10, 3, 3',
              '10, 3, 5',
              '10, 5, 5',
              '10, 7, 7',
              '10, 10, 10']


# Now put everything in the pandas dataframe and save to a file
df = pd.DataFrame(data, index=row_labels, columns=column_labels)
df.to_csv(sim_data.home_dir+'/orbit_data/paper_III/mw_population_floor_test_medians.csv')