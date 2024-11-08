
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

halo_mass_dex_error = 0.35
sigma_phase_space = 3
percent_nd_gaussian = 99 # 3 sigma
#
error_tweak = 1.99
err_name = 'mass.peak.err'  # host.distance.total.err, host.velocity
file_save_path = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/mass_tweaks/err_tweak_{error_tweak}/'

#sat_name = 'Carina'
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

        mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)

        # Get the indices of the satellites that are above a given minimum halo mass (1e8 for now)
        sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1=name, location=loc)

        # Get a match for a given LG satellite
        match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=sat_name, mass_err=halo_mass_dex_error, err_tweak=error_tweak, specific_err_key=err_name)
        if np.isnan(match['mass.star']):
            continue

        # Get the phase-space coordinates of these satellites across all snapshots
        subhalo_dict = sat_match.subhalo_data(tree=None, mini=mini_data, snapshot_data=snaps)

        satellite_match = sat_match.subhalo_match(sat_match.sub_inds, subhalos=subhalo_dict, satellite=match, snapshot_data=snaps, lookback_window=1, max_sigma=sigma_phase_space, probability_max=percent_nd_gaussian)

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

