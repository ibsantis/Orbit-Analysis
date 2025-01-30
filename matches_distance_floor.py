
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
import multiprocessing
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
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

halo_mass_dex_error = 0.35
sigma_phase_space = 3
percent_nd_gaussian = 99 # 3 sigma
#
floor = 0.1
floor_type = 'distance'
######
file_save_path = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/{floor_type}_tweaks/floor_{floor}/'

# Distance tweaks
for sat_name in mw_sats_1Mpc:
    tree_index = []
    mass_array = []
    weight = []
    sigma_dif = []
    snapshot = []
    hosts = []
    #
    for name in galaxies:
        snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)

        mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=False)

        # Get the indices of the satellites that are above a given minimum halo mass (1e8 for now)
        sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1=name, location=loc)

        # Get a match for a given LG satellite
        match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=sat_name, mass_err=halo_mass_dex_error)
        if np.isnan(match['mass.star']):
            continue

        # Get the phase-space coordinates of these satellites across all snapshots
        subhalo_dict = sat_match.subhalo_data(tree=None, mini=mini_data, snapshot_data=snaps)

        satellite_match = sat_match.subhalo_match(sat_match.sub_inds, subhalos=subhalo_dict, 
                                                  satellite=match, snapshot_data=snaps, lookback_window=1, 
                                                  max_sigma=sigma_phase_space, probability_max=percent_nd_gaussian, 
                                                  dist_floor=floor)

        mask = (satellite_match['mass.index'] != -1)
        for i in range(0, len(satellite_match['mass.index'][mask])):
            hosts.append(name)
            tree_index.append(satellite_match['tree.index'][mask][i])
            mass_array.append(subhalo_dict['mass.peak'][mask][i])
            #
            mask_w = (satellite_match['weight'][mask][i] > 0)
            ind_w = np.where(np.max(satellite_match['weight'][mask][i][mask_w]) == satellite_match['weight'][mask][i][mask_w])[0][0]
            weight.append(satellite_match['weight'][mask][i][mask_w][ind_w])
            snapshot.append(satellite_match['snapshot'][mask][i][mask_w][ind_w])
            sigma_dif.append(satellite_match['sigma.dif'][mask][i][mask_w][ind_w])
    #
    ws = sat_match.mass_weighting(weight, mass_array, match['mass.peak'], SMHM_slope=0.44)
    #
    param_list = [halo_mass_dex_error, sigma_phase_space, percent_nd_gaussian]
    #
    sat_match.write_subhalo_matches(sat_name, hosts, tree_index, ws, snapshot, param_list, file_path=file_save_path)

