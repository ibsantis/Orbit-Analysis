
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import pandas as pd
import satellite_io
import matplotlib
from matplotlib import pyplot as plt

print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
aligned = True
#
print('Set paths')

# Read in the snapshot dictionary and the entire tree
#snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100') # Saves snapshots, redshifts, lookback times, etc. to an array
#halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal, assign_hosts_rotation=aligned, catalog_hdf5_directory='catalog_hdf5')
lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)
#tlb = snaps['time'][-1] - np.flip(snaps['time'])

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


mass_dex = 0.35
probability_max = 99
max_sigma = 3.0
lookback_window = 1
min_m = 1e8
sat_name = 'Sculptor'

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']

for name in galaxies:
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name) # Saves snapshots, redshifts, lookback times, etc. to an array
    tlb = snaps['time'][-1] - np.flip(snaps['time'])
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=False)

    # Get the indices of the satellites that are above a given minimum halo mass (1e8 for now)
    sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1=name, location=loc, minimum_mass=min_m)

    # Get a match for a given LG satellite
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=sat, mass_err=mass_dex)
    satellite = match

    # Get the phase-space coordinates of these satellites across all snapshots
    subhalo_dict = sat_match.subhalo_data(tree=None, mini=mini_data, snapshot_data=snaps)
    max_time_window = snaps['time'][-1] - lookback_window
    n_snapshots = snaps['index'][-1] - np.where(np.min(np.abs(max_time_window - snaps['time'])) == np.abs(max_time_window - snaps['time']))[0][0]
    print('* Host: {0}, N_snap: {1}, mass_dex_error: {2}, max_sigma: {3}'.format(name, n_snapshots, mass_dex, max_sigma))
    #print('* Going back {0} snapshots to a lookback time of {1} Gyr'.format(n_snapshots, np.around(tlb[n_snapshots],2)))
    subhalos = subhalo_dict

    indices = sat_match.sub_inds
    sub_match = {}
    sub_match['mass.index.all'] = (-1)*np.ones(indices.shape[0], int)
    sub_match['mass.index.cull'] = (-1)*np.ones(indices.shape[0], int)
    sub_match['tree.index'] = (-1)*np.ones(indices.shape[0], int)
    sub_match['weight'] = (-1)*np.ones((indices.shape[0], n_snapshots))
    sub_match['sigma.dif'] = (-1)*np.ones((indices.shape[0], n_snapshots))
    sub_match['snapshot'] = (-1)*np.ones((indices.shape[0], n_snapshots), int)
    #
    properties = [prop_name for prop_name in sorted(satellite.keys()) if '.star' not in prop_name and '.err' not in prop_name and ~np.isnan(satellite[prop_name])]
    #
    print(len(properties))
    #dof_number = int(len(properties))
    dof_number = len([i for i in range(0, len(properties)) if ~np.isnan(satellite[properties[i]])])
    print(dof_number)
    #
    for prop_name in properties:
        if ~np.isnan(satellite[prop_name]) and np.isnan(satellite[prop_name+'.err']):
            if 'dist' in prop_name:
                satellite[prop_name+'.err'] = 5
            if 'vel' in prop_name:
                satellite[prop_name+'.err'] = 5
    #
    if dof_number == 1:
        sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.0, 2.0, 3.0
    elif dof_number == 2:
        sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.36, 2.27, 3.206
    elif dof_number == 3:
        sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.56, 2.42, 3.32
    elif dof_number == 4:
        sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.69, 2.52, 3.4 #results in 99.7%
    #
    if probability_max == 68:
        sigma_dif_max = sigma_dif_68
    elif probability_max == 95:
        sigma_dif_max = sigma_dif_95
    elif probability_max == 99:
        sigma_dif_max = sigma_dif_99
    else:
        sigma_dif_max = sigma_dif_95
    #
    # Get subhalos within +/- N sigma * 0.25 dex of M_halo,peak 
    mass_kind = 'mass.peak'
    mass_halo_log = np.log10(satellite[mass_kind])
    mass_inds = ut.array.get_indices(subhalos[mass_kind], [10**(mass_halo_log - max_sigma*satellite[mass_kind+'.err']), 10**(mass_halo_log + max_sigma*satellite[mass_kind+'.err'])])
    #print('* The number of satellites within +/- {0}*{1} dex: {2}'.format(max_sigma, satellite[mass_kind+'.err'], len(mass_inds)))
    #
    # Create a list of coordinate names and property names to loop through
    coord_names = [prop_name for prop_name in properties if prop_name != 'mass.peak']
    #
    # Loop through snapshots
    for snap_ind in range(0, n_snapshots):
        #
        # Use satellites already selected by mass
        match_inds = mass_inds
        # Loop through the 6D coordinates
        for prop_name in coord_names:
            # Get the bin limits for a given property based on the observed satellite values and max error
            prop_limits = ut.binning.get_bin_limits([satellite[prop_name], max_sigma*satellite[prop_name+'.err']], 'error')
            # Set up another 2D array for the subhalo coordinates at a given snapshot
            prop_values = subhalos[prop_name][:,snap_ind]
            # Get the indices of the subhalos that are within the bin limits
            match_inds = ut.array.get_indices(prop_values, prop_limits, match_inds)
        # If there are matches, continue!
        if len(match_inds) != 0:
            # Set up null array to save to
            sigma_difs_z = np.zeros(len(match_inds))
            # Loop through the 6D + mass properties
            for prop_name in properties:
                # If mass is the property, take the log of the values and calculate sigma_dif
                if 'mass' in prop_name:
                    prop_values = np.log10(subhalos[prop_name][match_inds])
                    match_prop = np.log10(satellite[prop_name])
                    sigma_difs_z += (
                    (prop_values - match_prop) / satellite[prop_name+'.err']
                    ) **2
                # If 6D coords, do the same thing without the log
                else:
                    prop_values = subhalos[prop_name][match_inds, snap_ind]
                    sigma_difs_z += (
                        (prop_values - satellite[prop_name]) / satellite[prop_name+'.err']
                        ) **2
            #
            # Finally take the square root of sigmas
            sigma_difs_z = np.sqrt(sigma_difs_z)
            # Only keep cases that are within our max allowed error
            masks1 = ut.array.get_indices(sigma_difs_z, [0, sigma_dif_max])
            masks2 = ut.array.get_indices(sigma_difs_z, [0, np.inf])
            sigma_difs_z = sigma_difs_z[masks1]
            match_inds1 = match_inds[masks1]
            match_inds2 = match_inds[masks2]
            #
            # If there are are still matches, continue!
            if len(sigma_difs_z) != 0:
                # calculate the weights from the gaussian arguments (sigma_difs)
                weights_z = ut.math.Function.gaussian_normalized(sigma_difs_z)
                # Save all of the data to the arrays
                sub_match['mass.index.cull'][match_inds1] = match_inds1
                sub_match['mass.index.all'][match_inds2] = match_inds2
                sub_match['tree.index'][match_inds1] = sat_match.sub_inds[:,0][match_inds1]
                sub_match['snapshot'][match_inds1, snap_ind] = np.flip(snaps['index'])[:n_snapshots][snap_ind]
                sub_match['weight'][match_inds1, snap_ind] = weights_z
                sub_match['sigma.dif'][match_inds1, snap_ind] = sigma_difs_z
    #print('* Number that are within infinite error of {0}: {1}'.format(np.inf, np.sum(sub_match['mass.index.all'] != -1)))
    #print('* Number that are within max allowed error of {0}: {1}'.format(sigma_dif_max, np.sum(sub_match['mass.index.cull'] != -1)))

###################################################################################################################


mass_dex = 0.35
probability_max = 99
max_sigma = 3.0
lookback_window = 1
min_m = 1e8
sat_name = 'Sculptor'

tree_index = []
mass_array = []
weight = []
weight_norm = []
dperi = []
sigma_dif = []
snapshot = []
#
hosts = []
#
for name in galaxies:
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name) # Saves snapshots, redshifts, lookback times, etc. to an array

    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)

    # Get the indices of the satellites that are above a given minimum halo mass (1e8 for now)
    sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1=name, location=loc)

    # Get a match for a given LG satellite
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=sat_name, mass_err=0.35)

    # Get the phase-space coordinates of these satellites across all snapshots
    subhalo_dict = sat_match.subhalo_data(tree=None, mini=mini_data, snapshot_data=snaps)

    satellite_match = sat_match.subhalo_match(sat_match.sub_inds, subhalos=subhalo_dict, satellite=match, snapshot_data=snaps, lookback_window=1, max_sigma=3, probability_max=99)

    mask = (satellite_match['mass.index'] != -1)
    for i in range(0, len(satellite_match['mass.index'][mask])):
        hosts.append(name)
        tree_index.append(satellite_match['tree.index'][mask][i])
        mass_array.append(satellite_match['mass.peak'][mask][i])
        dperi.append(mini_data['pericenter.dist.sim'][:,0][mask][i])
        #
        mask_w = (satellite_match['weight'][mask][i] > 0)
        ind_w = np.where(np.max(satellite_match['weight'][mask][i][mask_w]) == satellite_match['weight'][mask][i][mask_w])[0][0]
        weight.append(satellite_match['weight'][mask][i][mask_w][ind_w])
        snapshot.append(satellite_match['snapshot'][mask][i][mask_w][ind_w])
        sigma_dif.append(satellite_match['sigma.dif'][mask][i][mask_w][ind_w])


# Plot the most recent pericenter distances
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(20,8))
#
binsize = 0.01
mass_log_sub = np.log10(mass_array)
mass_log_sat = np.log10(match['mass.peak'])
x = mass_log_sub
#
minn = binsize*np.floor(np.nanmin(x)/binsize)
maxx = binsize*np.ceil(np.nanmax(x)/binsize)
if minn < 0:
    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
else:
    bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
binss = np.linspace(minn, maxx, bin_num)
half_bin = (binss[1]-binss[0])/2
axs[0].hist(x, binss, density=False, weights=weight, linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
axs[0].set_xlim(minn, maxx)
axs[0].set_xlabel('log $M_{\\rm halo,peak}/M_{\\odot}$', fontsize=24)
axs[0].tick_params(axis='both', which='major', labelsize=28)
#
binsize = 1
x = dperi
#
minn = binsize*np.floor(np.nanmin(x)/binsize)
maxx = binsize*np.ceil(np.nanmax(x)/binsize)
if minn < 0:
    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
else:
    bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
binss = np.linspace(minn, maxx, bin_num)
half_bin = (binss[1]-binss[0])/2
axs[1].hist(x, binss, density=False, weights=weight, linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
#
axs[1].set_xlim(minn, maxx)
axs[1].set_xlabel('Pericenter distance [kpc]', fontsize=24)
axs[0].set_ylabel('Probability', fontsize=24)
plt.suptitle('Sculptor, 3 sigma, Nsat = {0}, log Mpeak = 9.71'.format(len(weight)), fontsize=24)
axs[1].tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/sculptor_fiducial.pdf')
plt.close()





ws = sat_match.mass_weighting(weight, mass_array, match['mass.peak'])

# Plot the most recent pericenter distances
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(20,8))
#
binsize = 0.01
mass_log_sub = np.log10(mass_array)
mass_log_sat = np.log10(match['mass.peak'])
x = mass_log_sub
#
minn = binsize*np.floor(np.nanmin(x)/binsize)
maxx = binsize*np.ceil(np.nanmax(x)/binsize)
if minn < 0:
    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
else:
    bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
binss = np.linspace(minn, maxx, bin_num)
half_bin = (binss[1]-binss[0])/2
axs[0].hist(x, binss, density=False, weights=ws, linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
axs[0].set_xlim(minn, maxx)
axs[0].set_xlabel('log $M_{\\rm halo,peak}/M_{\\odot}$', fontsize=24)
axs[0].tick_params(axis='both', which='major', labelsize=28)
#
binsize = 1
x = dperi
#
minn = binsize*np.floor(np.nanmin(x)/binsize)
maxx = binsize*np.ceil(np.nanmax(x)/binsize)
if minn < 0:
    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
else:
    bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
binss = np.linspace(minn, maxx, bin_num)
half_bin = (binss[1]-binss[0])/2
axs[1].hist(x, binss, density=False, weights=ws, linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4)
#
axs[1].set_xlim(minn, maxx)
axs[1].set_xlabel('Pericenter distance [kpc]', fontsize=24)
axs[0].set_ylabel('Probability', fontsize=24)
plt.suptitle('Sculptor, 3 sigma, Nsat = {0}, log Mpeak = 9.71'.format(len(weight)), fontsize=24)
axs[1].tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/sculptor_fiducial_mass_weighted.pdf')
plt.close()

















"""
max_sigma = 3
95% selection
"""
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(8,8))
#
nsnaps = np.array([30,50,100,200])
n1 = np.array([5, 8, 16, 29])
n2 = np.array([13, 22, 40, 70])
#
axs.plot(nsnaps, n1, marker='o', markersize=10, label='95% selection')
axs.plot(nsnaps, n2, marker='o', markersize=10, label='99% selection')
axs.legend(prop={'size': 24}, loc='best')
axs.tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
#
axs.set_xlabel('Snapshot window size', fontsize=24)
axs.set_ylabel('Number of Satellites', fontsize=24)
plt.title('Sculptor', fontsize=26)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/nsat_vs_snap_window.pdf')
plt.close()

"""
plotting different max sigma selections
"""
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 2, figsize=(12,12))
#
nsnaps = np.array([30,50,100])
#
n_25_pre = np.array([5, 9, 16])
n_25_post = np.array([5, 8, 16])
#
n_30_pre = np.array([6, 12, 21])
n_30_post = np.array([5, 8, 16])
#
n_35_pre = np.array([10, 17, 25])
n_35_post = np.array([5, 8, 16])
#
n_40_pre = np.array([15, 22, 37])
n_40_post = np.array([5, 8, 16])
#
axs[0,0].plot(nsnaps, n_25_pre, marker='o', color='b', markersize=10, linestyle=':', label='Pre-2.52 max sig') # type: ignore
axs[0,0].plot(nsnaps, n_25_post, marker='x', color='r', markersize=10, linestyle='--', label='Post-2.52 max sig') # type: ignore
#
axs[0,1].plot(nsnaps, n_30_pre, marker='o', color='b', markersize=10, linestyle=':')
axs[0,1].plot(nsnaps, n_30_post, marker='x', color='r', markersize=10, linestyle='--')
#
axs[1,0].plot(nsnaps, n_35_pre, marker='o', color='b', markersize=10, linestyle=':')
axs[1,0].plot(nsnaps, n_35_post, marker='x', color='r', markersize=10, linestyle='--')
#
axs[1,1].plot(nsnaps, n_40_pre, marker='o', color='b', markersize=10, linestyle=':')
axs[1,1].plot(nsnaps, n_40_post, marker='x', color='r', markersize=10, linestyle='--')
#
axs[0,0].legend(prop={'size': 16}, loc='best')
axs[0,0].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
axs[0,1].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
axs[1,0].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
axs[1,1].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
#
axs[1,0].set_xlabel('Snapshot window size', fontsize=24)
axs[1,1].set_xlabel('Snapshot window size', fontsize=24)
axs[0,0].set_ylabel('Number of Satellites', fontsize=24)
axs[1,0].set_ylabel('Number of Satellites', fontsize=24)
#
axs[0,0].set_title('2.5 sigma', fontsize=22)
axs[0,1].set_title('3.0 sigma', fontsize=22)
axs[1,0].set_title('3.5 sigma', fontsize=22)
axs[1,1].set_title('4.0 sigma', fontsize=22)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/nsat_vs_snap_window_sigmas.pdf')
plt.close()



"""
 Plotting different selections for each satellite example
"""
x_array = np.array([0.25, 0.30, 0.35, 0.40])
#
# Sculptor
scu_sig30_95 = np.array([10, 12, 18, 20])
scu_sig30_99 = np.array([23, 31, 44, 66])
scu_sig35_95 = np.array([10, 12, 18, 20])
scu_sig35_99 = np.array([26, 35, 49, 67])
scu_sig40_95 = np.array([10, 12, 18, 20])
scu_sig40_99 = np.array([26, 35, 49, 67])
#
# Leo 2
leo_sig30_95 = np.array([15, 20, 24, 27])
leo_sig30_99 = np.array([36, 53, 70, 96])
leo_sig35_95 = np.array([15, 20, 24, 27])
leo_sig35_99 = np.array([37, 54, 76, 102])
leo_sig40_95 = np.array([15, 20, 24, 27])
leo_sig40_99 = np.array([37, 54, 76, 102])
#
# Carina
car_sig30_95 = np.array([19, 24, 39, 49])
car_sig30_99 = np.array([61, 91, 124, 165])
car_sig35_95 = np.array([19, 24, 39, 49])
car_sig35_99 = np.array([65, 97, 137, 178])
car_sig40_95 = np.array([19, 24, 39, 49])
car_sig40_99 = np.array([65, 97, 137, 178])
#
# Draco
dra_sig30_95 = np.array([7, 10, 13, 18])
dra_sig30_99 = np.array([21, 28, 42, 54])
dra_sig35_95 = np.array([7, 10, 13, 18])
dra_sig35_99 = np.array([23, 32, 46, 55])
dra_sig40_95 = np.array([7, 10, 13, 18])
dra_sig40_99 = np.array([23, 32, 46, 55])
#
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 2, figsize=(12,12))
#
axs[0,0].plot(x_array, scu_sig30_95, marker='o', color='b', markersize=10, alpha=0.6, linestyle=':', label='$3\\sigma\\times$ phase-space error')
axs[0,0].plot(x_array, scu_sig30_99, marker='o', color='b', markersize=10, alpha=0.6, linestyle='--')
axs[0,0].plot(x_array, scu_sig35_95, marker='x', color='r', markersize=10, alpha=0.6, linestyle=':', label='$3.5\\sigma\\times$ phase-space error')
axs[0,0].plot(x_array, scu_sig35_99, marker='x', color='r', markersize=10, alpha=0.6, linestyle='--')
axs[0,0].plot(x_array, scu_sig40_95, marker='s', color='g', markersize=10, alpha=0.6, linestyle=':', label='$4\\sigma\\times$ phase-space error')
axs[0,0].plot(x_array, scu_sig40_99, marker='s', color='g', markersize=10, alpha=0.6, linestyle='--')
#
axs[0,1].plot(x_array, leo_sig30_95+1000, color='k', alpha=0.8, linestyle=':', label='95% selection')
axs[0,1].plot(x_array, leo_sig30_99+1000, color='k', alpha=0.8, linestyle='--', label='99% selection')
axs[0,1].plot(x_array, leo_sig30_95, marker='o', color='b', markersize=10, alpha=0.6, linestyle=':')
axs[0,1].plot(x_array, leo_sig30_99, marker='o', color='b', markersize=10, alpha=0.6, linestyle='--')
axs[0,1].plot(x_array, leo_sig35_95, marker='x', color='r', markersize=10, alpha=0.6, linestyle=':')
axs[0,1].plot(x_array, leo_sig35_99, marker='x', color='r', markersize=10, alpha=0.6, linestyle='--')
axs[0,1].plot(x_array, leo_sig40_95, marker='s', color='g', markersize=10, alpha=0.6, linestyle=':')
axs[0,1].plot(x_array, leo_sig40_99, marker='s', color='g', markersize=10, alpha=0.6, linestyle='--')
#
axs[1,0].plot(x_array, car_sig30_95, marker='o', color='b', markersize=10, alpha=0.6, linestyle=':')
axs[1,0].plot(x_array, car_sig30_99, marker='o', color='b', markersize=10, alpha=0.6, linestyle='--')
axs[1,0].plot(x_array, car_sig35_95, marker='x', color='r', markersize=10, alpha=0.6, linestyle=':')
axs[1,0].plot(x_array, car_sig35_99, marker='x', color='r', markersize=10, alpha=0.6, linestyle='--')
axs[1,0].plot(x_array, car_sig40_95, marker='s', color='g', markersize=10, alpha=0.6, linestyle=':')
axs[1,0].plot(x_array, car_sig40_99, marker='s', color='g', markersize=10, alpha=0.6, linestyle='--')
#
axs[1,1].plot(x_array, dra_sig30_95, marker='o', color='b', markersize=10, alpha=0.6, linestyle=':')
axs[1,1].plot(x_array, dra_sig30_99, marker='o', color='b', markersize=10, alpha=0.6, linestyle='--')
axs[1,1].plot(x_array, dra_sig35_95, marker='x', color='r', markersize=10, alpha=0.6, linestyle=':')
axs[1,1].plot(x_array, dra_sig35_99, marker='x', color='r', markersize=10, alpha=0.6, linestyle='--')
axs[1,1].plot(x_array, dra_sig40_95, marker='s', color='g', markersize=10, alpha=0.6, linestyle=':')
axs[1,1].plot(x_array, dra_sig40_99, marker='s', color='g', markersize=10, alpha=0.6, linestyle='--')
#
axs[0,0].legend(prop={'size': 16}, loc='upper left')
axs[0,1].legend(prop={'size': 16}, loc='best')
#
axs[0,0].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
axs[0,1].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
axs[1,0].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
axs[1,1].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=22)
#
axs[0,0].set_ylim([-1, 70])
axs[0,1].set_ylim([-1, 105])
#
axs[0,0].set_xlabel('$\\Delta M$ selection [dex]', fontsize=24)
axs[0,1].set_xlabel('$\\Delta M$ selection [dex]', fontsize=24)
axs[1,0].set_xlabel('$\\Delta M$ selection [dex]', fontsize=24)
axs[1,1].set_xlabel('$\\Delta M$ selection [dex]', fontsize=24)
axs[0,0].set_ylabel('Number of Satellites', fontsize=24)
axs[1,0].set_ylabel('Number of Satellites', fontsize=24)
#
axs[0,0].set_title('Sculptor: $M_* = 1.8\\times10^6 M_{\\odot}$', fontsize=22)
axs[0,1].set_title('Leo 2: $M_* = 6.7\\times10^5 M_{\\odot}$', fontsize=22)
axs[1,0].set_title('Carina: $M_* = 5.2\\times10^5 M_{\\odot}$', fontsize=22)
axs[1,1].set_title('Draco: $M_* = 3\\times10^5 M_{\\odot}$', fontsize=22)
#
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/nsat_vs_mass_vs_sigma_4sats.pdf')
plt.close()
