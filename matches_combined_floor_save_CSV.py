#!/usr/bin/python3

"""
    =============================
    = Comparing combined tweaks =
    =============================

    ...
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
first_infall = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
nperi = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
tperi_rec = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
dperi_rec = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
vperi_rec = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
tperi_min = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
dperi_min = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
vperi_min = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
tapo_rec = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
dapo_rec = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
elltot = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
ketot = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
#
first_infall_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
nperi_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
tperi_rec_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
dperi_rec_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
vperi_rec_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
tperi_min_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
dperi_min_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
vperi_min_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
tapo_rec_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
dapo_rec_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
elltot_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
ketot_68 = (-1)*np.ones((len(mw_sats_1Mpc), len(floor_list)))
#
######
for i, (floor_d, floor_vr, floor_vt) in enumerate(floor_list):
    if floor_d == False:
        floor_d_str = "F"
    else:
        floor_d_str = str(floor_d)
    if floor_vr == False:
        floor_vr_str = "F"
    else:
        floor_vr_str = str(floor_vr)
    if floor_vt == False:
        floor_vt_str = "F"
    else:
        floor_vt_str = str(floor_vt)
    for j, galaxy in enumerate(mw_sats_1Mpc):
        #
        satellite_name = galaxy.replace(' ', '_')
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_{floor_d_str}_{floor_vr_str}_{floor_vt_str}/weights_{satellite_name}.txt'
        gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
        #
        if len(gal_data['Host']) == 0:
            continue
        #
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
            first_infall[j, i] = x_med
            first_infall_68[j, i] = x_upper - x_lower
        else:
            first_infall_68[j, i] = -1
            first_infall_68[j, i] = -1
        #
        # Pericenter number
        m = (orbit_dictionary['pericenter.num'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['pericenter.num'][m]
            x_med = np.sum(x*gal_data['Weight'][m])
            x_std = np.sqrt(np.sum((x-x_med)**2*gal_data['Weight'][m])/np.sum(gal_data['Weight'][m])/np.sum(gal_data['Weight'][m]))
            nperi[j, i] = x_med
            nperi_68[j, i] = x_std
        else:
            nperi[j, i] = -1
            nperi_68[j, i] = -1
        #
        # Recent pericenter time
        m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['pericenter.rec.time.lb'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            tperi_rec[j, i] = x_med
            tperi_rec_68[j, i] = x_upper - x_lower
        else:
            tperi_rec[j, i] = -1
            tperi_rec_68[j, i] = -1
        #
        # Recent pericenter distance
        m = (orbit_dictionary['pericenter.rec.dist'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['pericenter.rec.dist'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            dperi_rec[j, i] = x_med
            dperi_rec_68[j, i] = x_upper - x_lower
        else:
            dperi_rec[j, i] = -1
            dperi_rec_68[j, i] = -1
        #
        # Recent pericenter velocity
        m = (orbit_dictionary['pericenter.rec.vel'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['pericenter.rec.vel'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            vperi_rec[j, i] = x_med
            vperi_rec_68[j, i] = x_upper - x_lower
        else:
            vperi_rec[j, i] = -1
            vperi_rec_68[j, i] = -1
        #
        # Minimum pericenter time
        m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['pericenter.min.time.lb'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            tperi_min[j, i] = x_med
            tperi_min_68[j, i] = x_upper - x_lower
        else:
            tperi_min[j, i] = -1
            tperi_min_68[j, i] = -1
        #
        # Minimum pericenter distance
        m = (orbit_dictionary['pericenter.min.dist'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['pericenter.min.dist'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            dperi_min[j, i] = x_med
            dperi_min_68[j, i] = x_upper - x_lower
        else:
            dperi_min[j, i] = -1
            dperi_min_68[j, i] = -1
        #
        # Minimum pericenter velocity
        m = (orbit_dictionary['pericenter.min.vel'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['pericenter.min.vel'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            vperi_min[j, i] = x_med
            vperi_min_68[j, i] = x_upper - x_lower
        else:
            vperi_min[j, i] = -1
            vperi_min_68[j, i] = -1
        #
        # Recnet apocenter time
        m = (orbit_dictionary['apocenter.time.lb'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['apocenter.time.lb'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            tapo_rec[j, i] = x_med
            tapo_rec_68[j, i] = x_upper - x_lower
        else:
            tapo_rec[j, i] = -1
            tapo_rec_68[j, i] = -1
        #
        # Recent apocenter distance
        m = (orbit_dictionary['apocenter.dist'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['apocenter.dist'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            dapo_rec[j, i] = x_med
            dapo_rec_68[j, i] = x_upper - x_lower
        else:
            dapo_rec[j, i] = -1
            dapo_rec_68[j, i] = -1
        #
        # Angular momentum at match
        m = (orbit_dictionary['L.tot.sim'] != -1)
        if np.sum(m) != 0:
            x = orbit_dictionary['L.tot.sim'][m]
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            elltot[j, i] = x_med
            elltot_68[j, i] = x_upper - x_lower
        else:
            elltot[j, i] = -1
            elltot_68[j, i] = -1
        #
        # Specific Kinetic Energy at match
        m = (orbit_dictionary['v.tot.sim'] != -1)
        if np.sum(m) != 0:
            x = 0.5*orbit_dictionary['v.tot.sim'][m]**2
            x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
            ketot[j, i] = x_med
            ketot_68[j, i] = x_upper - x_lower
        else:
            ketot[j, i] = -1
            ketot_68[j, i] = -1




np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/infall_median.csv", first_infall, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/nperi_median.csv", nperi, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/tperi_rec_median.csv", tperi_rec, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/dperi_rec_median.csv", dperi_rec, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/vperi_rec_median.csv", vperi_rec, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/tperi_min_median.csv", tperi_min, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/dperi_min_median.csv", dperi_min, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/vperi_min_median.csv", vperi_min, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/tapo_median.csv", tapo_rec, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/dapo_median.csv", dapo_rec, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/ell_median.csv", elltot, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/ke_median.csv", ketot, delimiter=',', fmt='%.6f')

np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/infall_width.csv", first_infall_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/nperi_width.csv", nperi_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/tperi_rec_width.csv", tperi_rec_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/dperi_rec_width.csv", dperi_rec_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/vperi_rec_width.csv", vperi_rec_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/tperi_min_width.csv", tperi_min_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/dperi_min_width.csv", dperi_min_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/vperi_min_width.csv", vperi_min_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/tapo_width.csv", tapo_rec_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/dapo_width.csv", dapo_rec_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/ell_width.csv", elltot_68, delimiter=',', fmt='%.6f')
np.savetxt("simulation/orbit_data/paper_III/floor_testing_physical/ke_width.csv", ketot_68, delimiter=',', fmt='%.6f')
