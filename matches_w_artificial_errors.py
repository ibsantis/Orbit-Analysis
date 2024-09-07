
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

mw_sats = ['NGC 55', 'LMC', 'SMC', 'IC 4662', 'IC 5152', 'NGC 6822', 'NGC 3109', 'IC 3104', \
           'Sextans B', 'DDO 190', 'DDO 125', 'Sextans A', 'NGC 4163', 'Sagittarius dSph', 'UGC 8508', 'Fornax', 'UGC 4879', \
           'UGC 9128', 'GR 8', 'Leo A', 'Leo 1', 'Sagittarius dIrr', 'ESO 294-G010', 'DDO 113', 'Sculptor', 'Antlia 2', 'Aquarius (DDO 210)',\
           'Phoenix', 'Leo 2', 'Antlia B', 'Tucana', 'KKR 3', 'Carina', 'Leo P', 'Crater 2', 'Ursa Minor', 'Sextans 1', \
           'Draco', 'Canes Venatici 1', 'Leo T', 'Eridanus 2', 'Bootes 1', 'Hercules', 'Bootes 3', 'Sagittarius 2', \
           'Canes Venatici 2', 'Ursa Major 1', 'Leo 4', 'Hydra 2', 'Hydrus 1', 'Carina 2', 'Ursa Major 2', 'Aquarius 2', \
           'Indus 2', 'Coma Berenices', 'Leo 5', 'Pisces 2', 'Columba 1', 'Tucana 5', 'Pegasus 3', 'Grus 2', 'Tucana 2', \
           'Reticulum 2', 'Horologium 1', 'Pictor 1', 'Tucana 4', 'Indus 1', 'Grus 1', 'Reticulum 3', 'Pictor 2', 'Bootes 2',\
           'Willman 1', 'Phoenix 2', 'Cetus 3', 'Carina 3', 'Eridanus 3', 'Segue 2', 'Triangulum 2', 'Horologium 2', 'Tucana 3',\
           'Segue 1', 'DES J0225+0304', 'Virgo 1', 'Draco 2', 'Cetus 2']

final_dict = dict()
halo_mass_dex_error = 0.35
sigma_phase_space = 3
percent_nd_gaussian = 99 # 3 sigma
alpha=0.58
#
start_time = time.time()

sat_name = 'Carina'
tree_index = []
mass_array = []
weight = []
sigma_dif = []
snapshot = []
hosts = []
error_tweak = 0.8
#
for name in galaxies:
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)

    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)

    # Get the indices of the satellites that are above a given minimum halo mass (1e8 for now)
    sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1=name, location=loc)

    # Get a match for a given LG satellite
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=sat_name, mass_err=halo_mass_dex_error, err_tweak=error_tweak)
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
sat_match.write_subhalo_matches(sat_name, hosts, tree_index, ws, snapshot, param_list, err_tweak=error_tweak)
