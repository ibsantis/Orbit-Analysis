

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

nsnap = 30

for name in galaxies:
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)

    # Get the indices of the satellites that are above a given minimum halo mass (1e8 for now)
    sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1=name, location=loc)

    # Get a match for a given LG satellite
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name='Sculptor', mass_err=0.25)

    # Get the phase-space coordinates of these satellites across all snapshots
    subhalo_dict = sat_match.subhalo_data(tree=None, mini=mini_data, snapshot_data=snaps)

    satellite_match = sat_match.subhalo_match(sat_match.sub_inds, subhalos=subhalo_dict, satellite=match, snapshot_data=snaps, n_snapshots=nsnap)

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

ws = sat_match.mass_weighting(weight, mass_array, match['mass.peak'])

sat_match.write_subhalo_matches('Sculptor', hosts, tree_index, ws, snapshot)




# Save all of the most recent pericenter distances in an empty array
### Change weight to ws to return the same plots as before.
dperi = np.zeros(len(weight))
for i in range(0, len(dperi)):
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+hosts[i]+'_all_subhalos', verbose=True)
    ind = np.where(tree_index[i] == mini_data['indices.z0'][:,0])[0][0]
    dperi[i] = mini_data['pericenter.dist.sim'][:,0][ind]


# Plot the most recent pericenter distances
nsat = len(dperi)
print('Number of matches: {0}'.format(nsat))

#plt.figure(figsize=(10, 8))
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
plt.suptitle('Sculptor, 3 sigma, {0} snapshots, Nsat = {1}, log Mpeak = 9.71'.format(nsnap, nsat), fontsize=24)
axs[1].tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/sculptor_3sig_n'+str(nsnap)+'_and_mass.pdf')
plt.close()


# How to integrate the 1D gaussian
f = lambda x: (1 / (np.sqrt(2*np.pi)*1)) * np.exp(-x**2/(2*1**2))
#
# How to integrate the 2D gaussian
f = lambda x,y: (1 / (np.sqrt(2*np.pi)*1)) * np.exp(-x**2/(2*1**2))*(1 / (np.sqrt(2*np.pi)*1)) * np.exp(-y**2/(2*1**2))
#
# How to integrate the 3D gaussian
f = lambda x,y,z: (1 / (np.sqrt(2*np.pi)*1)) * np.exp(-x**2/(2*1**2))*(1 / (np.sqrt(2*np.pi)*1)) * np.exp(-y**2/(2*1**2))*(1 / (np.sqrt(2*np.pi)*1)) * np.exp(-z**2/(2*1**2))
#
# How to integrate the 4D gaussian
f = lambda x,y,z,w: (1 / (np.sqrt(2*np.pi)*1)) * np.exp(-x**2/(2*1**2))*(1 / (np.sqrt(2*np.pi)*1)) * np.exp(-y**2/(2*1**2))*(1 / (np.sqrt(2*np.pi)*1)) * np.exp(-z**2/(2*1**2))*(1 / (np.sqrt(2*np.pi)*1)) * np.exp(-w**2/(2*1**2))

from scipy.integrate import nquad
n = 1.68
nquad(f, [[-n,n],[-n,n],[-n,n],[-n,n]]) # This should return 68%