#!/usr/bin/python3

"""
    =======================================
    = Count analogs from floor selections =
    =======================================

    This will count the number of analogs we select based
    on the various selection criteria we implemented.

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
            temp_infall.append(np.sum(m))
        #
        # Pericenter number
        m = (orbit_dictionary['pericenter.num'] != -1)
        if np.sum(m) != 0:
            temp_nperi.append(np.sum(m))
        #
        # Recent pericenter time
        m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
        if np.sum(m) != 0:
            temp_tperi_rec.append(np.sum(m))
        #
        # Recent pericenter distance
        m = (orbit_dictionary['pericenter.rec.dist'] != -1)
        if np.sum(m) != 0:
            temp_dperi_rec.append(np.sum(m))
        #
        # Recent pericenter velocity
        m = (orbit_dictionary['pericenter.rec.vel'] != -1)
        if np.sum(m) != 0:
            temp_vperi_rec.append(np.sum(m))
        #
        # Minimum pericenter time
        m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
        if np.sum(m) != 0:
            temp_tperi_min.append(np.sum(m))
        #
        # Minimum pericenter distance
        m = (orbit_dictionary['pericenter.min.dist'] != -1)
        if np.sum(m) != 0:
            temp_dperi_min.append(np.sum(m))
        #
        # Minimum pericenter velocity
        m = (orbit_dictionary['pericenter.min.vel'] != -1)
        if np.sum(m) != 0:
            temp_vperi_min.append(np.sum(m))
        #
        # Apocenter time
        m = (orbit_dictionary['apocenter.time.lb'] != -1)
        if np.sum(m) != 0:
            temp_tapo.append(np.sum(m))
        #
        # Apocenter distance
        m = (orbit_dictionary['apocenter.dist'] != -1)
        if np.sum(m) != 0:
            temp_dapo.append(np.sum(m))
        #
        # Kinetic energy (specific KE at match)
        m = (orbit_dictionary['v.tot.sim'] != -1)
        if np.sum(m) != 0:
            temp_ke.append(np.sum(m))
        #
        # Specific angular momentum
        m = (orbit_dictionary['L.tot.sim'] != -1)
        if np.sum(m) != 0:
            temp_ell.append(np.sum(m))
    # Infall time
    temp_infall = np.hstack(temp_infall)
    first_infall[i] = np.sum(temp_infall)
    #
    # Pericenter number
    temp_nperi = np.hstack(temp_nperi)
    nperi[i] = np.sum(temp_nperi)
    #
    # Recent pericenter time
    temp_tperi_rec = np.hstack(temp_tperi_rec)
    tperi_rec[i] = np.sum(temp_tperi_rec)
    #
    # Recent pericenter distance
    temp_dperi_rec = np.hstack(temp_dperi_rec)
    dperi_rec[i] = np.sum(temp_dperi_rec)
    #
    # Recent pericenter velocity
    temp_vperi_rec = np.hstack(temp_vperi_rec)
    vperi_rec[i] = np.sum(temp_vperi_rec)
    #
    # Minimum pericenter time
    temp_tperi_min = np.hstack(temp_tperi_min)
    tperi_min[i] = np.sum(temp_tperi_min)
    #
    # Minimum pericenter distance
    temp_dperi_min = np.hstack(temp_dperi_min)
    dperi_min[i] = np.sum(temp_dperi_min)
    #
    # Minimum pericenter velocity
    temp_vperi_min = np.hstack(temp_vperi_min)
    vperi_min[i] = np.sum(temp_vperi_min)
    #
    # Apocenter time
    temp_tapo = np.hstack(temp_tapo)
    tapo_rec[i] = np.sum(temp_tapo)
    #
    # Apocenter distance
    temp_dapo = np.hstack(temp_dapo)
    dapo_rec[i] = np.sum(temp_dapo)
    #
    # Kinetic Energy
    temp_ke = np.hstack(temp_ke)
    ketot[i] = np.sum(temp_ke)
    #
    # Specific angular momentum
    temp_ell = np.hstack(temp_ell)
    elltot[i] = np.sum(temp_ell)

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
        temp_infall.append(np.sum(m))
    #
    # Pericenter number
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m) != 0:
        temp_nperi.append(np.sum(m))
    #
    # Recent pericenter time
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        temp_tperi_rec.append(np.sum(m))
    #
    # Recent pericenter distance
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        temp_dperi_rec.append(np.sum(m))
    #
    # Recent pericenter velocity
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        temp_vperi_rec.append(np.sum(m))
    #
    # Minimum pericenter time
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        temp_tperi_min.append(np.sum(m))
    #
    # Minimum pericenter distance
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        temp_dperi_min.append(np.sum(m))
    #
    # Minimum pericenter velocity
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        temp_vperi_min.append(np.sum(m))
    #
    # Apocenter time
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        temp_tapo.append(np.sum(m))
    #
    # Apocenter distance
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        temp_dapo.append(np.sum(m))
    #
    # Kinetic energy (specific KE at match)
    m = (orbit_dictionary['v.tot.sim'] != -1)
    if np.sum(m) != 0:
        temp_ke.append(np.sum(m))
    #
    # Specific angular momentum
    m = (orbit_dictionary['L.tot.sim'] != -1)
    if np.sum(m) != 0:
        temp_ell.append(np.sum(m))
# Infall time
temp_infall = np.hstack(temp_infall)
first_infall_fid = np.sum(temp_infall)
#
# Pericenter number
temp_nperi = np.hstack(temp_nperi)
nperi_fid = np.sum(temp_nperi)
#
# Recent pericenter time
temp_tperi_rec = np.hstack(temp_tperi_rec)
tperi_rec_fid = np.sum(temp_tperi_rec)
#
# Recent pericenter distance
temp_dperi_rec = np.hstack(temp_dperi_rec)
dperi_rec_fid = np.sum(temp_dperi_rec)
#
# Recent pericenter velocity
temp_vperi_rec = np.hstack(temp_vperi_rec)
vperi_rec_fid = np.sum(temp_vperi_rec)
#
# Minimum pericenter time
temp_tperi_min = np.hstack(temp_tperi_min)
tperi_min_fid = np.sum(temp_tperi_min)
#
# Minimum pericenter distance
temp_dperi_min = np.hstack(temp_dperi_min)
dperi_min_fid = np.sum(temp_dperi_min)
#
# Minimum pericenter velocity
temp_vperi_min = np.hstack(temp_vperi_min)
vperi_min_fid = np.sum(temp_vperi_min)
#
# Apocenter time
temp_tapo = np.hstack(temp_tapo)
tapo_rec_fid = np.sum(temp_tapo)
#
# Apocenter distance
temp_dapo = np.hstack(temp_dapo)
dapo_rec_fid = np.sum(temp_dapo)
#
# Kinetic Energy
temp_ke = np.hstack(temp_ke)
ketot_fid = np.sum(temp_ke)
#
# Specific angular momentum
temp_ell = np.hstack(temp_ell)
elltot_fid = np.sum(temp_ell)


# Add in the fiducial values to the beginning of the arrays since I didn't do that in the first place...
first_infall = np.insert(first_infall, 0, first_infall_fid)
nperi = np.insert(nperi, 0, nperi_fid)
tperi_rec = np.insert(tperi_rec, 0, tperi_rec_fid)
dperi_rec = np.insert(dperi_rec, 0, dperi_rec_fid)
vperi_rec = np.insert(vperi_rec, 0, vperi_rec_fid)
tperi_min = np.insert(tperi_min, 0, tperi_min_fid)
dperi_min = np.insert(dperi_min, 0, dperi_min_fid)
vperi_min = np.insert(vperi_min, 0, vperi_min_fid)
tapo_rec = np.insert(tapo_rec, 0, tapo_rec_fid)
dapo_rec = np.insert(dapo_rec, 0, dapo_rec_fid)
ketot = np.insert(ketot, 0, ketot_fid)
elltot = np.insert(elltot, 0, elltot_fid)


# Now put everything into a single array for pandas
data = np.array([first_infall, 
                 nperi,
                 tperi_rec,
                 dperi_rec,
                 vperi_rec,
                 tperi_min,
                 dperi_min,
                 vperi_min,
                 tapo_rec,
                 dapo_rec,
                 ketot,
                 elltot])



row_labels = ['infall', 
              'n.peri', 
              't.peri.rec',
              'd.peri.rec',
              'v.peri.rec',
              't.peri.min',
              'd.peri.min', 
              'v.peri.min', 
              't.apo',
              'd.apo',
              'ke', 'ell']

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
df.to_csv(sim_data.home_dir+'/orbit_data/paper_III/mw_population_floor_test_counts.csv')
