

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import pandas as pd
import satellite_io
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
aligned = True
#
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100') # Saves snapshots, redshifts, lookback times, etc. to an array
#halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal, assign_hosts_rotation=aligned, catalog_hdf5_directory='catalog_hdf5')
lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']

final_dict = dict()

tree_index = []
mass_array = []
weight = []
sigma_dif = []
snapshot = []
#
hosts = []

for name in galaxies:
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)

    # Get the indices of the satellites that are above a given minimum halo mass (1e8 for now)
    sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1=name, location=loc)

    # Get a match for a given LG satellite
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name='Sculptor', mass_err=0.25)

    # Get the phase-space coordinates of these satellites across all snapshots
    subhalo_dict = sat_match.subhalo_data(tree=None, mini=mini_data, snapshot_data=snaps)

    satellite_match = sat_match.subhalo_match(sat_match.sub_inds, subhalos=subhalo_dict, satellite=match, snapshot_data=snaps)

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








#### Add this stuff to the subhalo_match function when I can!
np.log10(subhalo_dict['mass.peak'][satellite_match['mass.index'][585]]) # There's only the one satellite that matches
np.log10(match['mass.peak'])
#
weights_m = 10 ** (0.44 * (np.log10(subhalo_dict['mass.peak'][satellite_match['mass.index'][585]]) - np.log10(match['mass.peak'])))
weights_m /= weights_m.sum()  # normalize

temp = np.max(satellite_match['weight'][satellite_match['mass.index'][585]]) 
temp *= weights_m # save this as the final weight for the subhalo!
temp /= temp.sum() # This step just turns the weight into one because there is only one match.....













########################
"""
    This block will be its own function
"""
indices = sat_match.sub_inds
n_snapshots = 30
satellite = match
probability_max = 95
subhalos = subhalo_dict
max_sigma = 3
snapshot_data = snaps
#
sub_match = {}
sub_match['mass.index'] = (-1)*np.ones(indices.shape[0], int)
sub_match['tree.index'] = (-1)*np.ones(indices.shape[0], int)
sub_match['weight'] = (-1)*np.ones((indices.shape[0], n_snapshots))
sub_match['sigma.dif'] = (-1)*np.ones((indices.shape[0], n_snapshots))
sub_match['snapshot'] = (-1)*np.ones((indices.shape[0], n_snapshots), int)
#
properties = [prop_name for prop_name in sorted(satellite.keys()) if '.star' not in prop_name and '.err' not in prop_name]
#
dof_number = int(len(properties))
if dof_number == 1:
    sigma_dif_68, sigma_dif_95 = 1.0, 2.0
elif dof_number == 2:
    sigma_dif_68, sigma_dif_95 = 1.36, 2.27 # These come from integrating an n-d gaussian to these limits to return 0.68 and 0.95
elif dof_number == 3:
    sigma_dif_68, sigma_dif_95 = 1.56, 2.42
elif dof_number == 4:
    sigma_dif_68, sigma_dif_95 = 1.69, 2.52
#
if probability_max == 68: # What is this for?
    sigma_dif_max = sigma_dif_68
elif probability_max == 95:
    sigma_dif_max = sigma_dif_95
else:
    sigma_dif_max = sigma_dif_95
#
# Get subhalos within +/- N sigma * 0.25 dex of M_halo,peak 
mass_kind = 'mass.peak'
mass_halo_log = np.log10(satellite[mass_kind])
mass_inds = ut.array.get_indices(subhalos[mass_kind], [10**(mass_halo_log - max_sigma*satellite[mass_kind+'.err']), 10**(mass_halo_log + max_sigma*satellite[mass_kind+'.err'])])
#
coord_names = [prop_name for prop_name in sorted(subhalos.keys()) if prop_name != 'mass.peak' and prop_name != 'snapshot']
properties = [prop_name for prop_name in sorted(subhalos.keys()) if prop_name != 'snapshot']
#
for snap_ind in range(0, n_snapshots):
    match_inds = mass_inds
    for prop_name in coord_names:
        prop_limits = ut.binning.get_bin_limits([satellite[prop_name], max_sigma*satellite[prop_name+'.err']], 'error')
        prop_values = subhalos[prop_name][:,snap_ind]
        match_inds = ut.array.get_indices(prop_values, prop_limits, match_inds)
        #print(snap_ind, match_inds)
    if len(match_inds) != 0:
        #
        sigma_difs_z = np.zeros(len(match_inds))
        #
        for prop_name in properties:
            if 'mass' in prop_name:
                prop_values = np.log10(subhalos[prop_name][match_inds])
                match_prop = np.log10(satellite[prop_name])
                sigma_difs_z += (
                (prop_values - match_prop) / satellite[prop_name+'.err']
                ) **2
            else:
                prop_values = subhalos[prop_name][match_inds, snap_ind]
                sigma_difs_z += (
                    (prop_values - satellite[prop_name]) / satellite[prop_name+'.err']
                    ) **2
        sigma_difs_z = np.sqrt(sigma_difs_z)
        #masks = ut.array.get_indices(sigma_difs_z, [0, sigma_dif_max]) # this is what's fucking things up
        #sigma_difs_z = sigma_difs_z[masks]
        if len(sigma_difs_z) != 0:
            weights_z = ut.math.Function.gaussian_normalized(sigma_difs_z)
            sub_match['mass.index'][match_inds] = match_inds
            sub_match['tree.index'][match_inds] = sat_match.sub_inds[:,0][match_inds]
            sub_match['snapshot'][match_inds, snap_ind] = np.flip(snapshot_data['index'])[:n_snapshots][snap_ind]
            sub_match['weight'][match_inds, snap_ind] = weights_z
            sub_match['sigma.dif'][match_inds, snap_ind] = sigma_difs_z
            print('* Satellite(s) {0} are a match at snapshot {1}'.format(match_inds, np.flip(snapshot_data['index'])[:n_snapshots][snap_ind]))
            print('* {0}, {1} within 68 percent, 95 percent limits'.format(np.sum(sigma_difs_z < sigma_dif_68), np.sum(sigma_difs_z < sigma_dif_95)))
        else:
            print('No matches.')
# weight by mass to ensure that centroid remains near input mass
# slope of weight corresponds to mass function:
# dn/dlog(m_sub) ~ log(m_sub) ^ -0.85 -> dn/dlog(m_star) ~ log(m_star) ^ -0.44
weights_m = np.zeros(len(sub_match['mass.index']))
for i in range(0, len(weights_m)):
    if sub_match['mass.index'][i] != -1:
        weights_m[i] = 10 ** (0.44 * (np.log10(subhalo_dict[mass_kind][i]) - np.log10(match[mass_kind])))
mask = (weights_m != 0)
weights_m[mask] /= weights_m[mask].sum()  # normalize
#
#
########## PICK UP HERE NEXT TIME
# normalize overall weights
for i in range(0, len(sub_match['weight'])):
    mask = (sub_match['weight'][i] != -1)
    if np.sum(mask) != 0:
        sub_match['weight'][i][mask] *= weights_m[i]
        sub_match['weight'][i][mask] /= sub_match['weight'].sum()  # normalize