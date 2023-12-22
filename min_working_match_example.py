

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
#snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100') # Saves snapshots, redshifts, lookback times, etc. to an array
#halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal, assign_hosts_rotation=aligned, catalog_hdf5_directory='catalog_hdf5')
lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']

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

final_dict = dict()
#
for sat_name in mw_sats:
    tree_index = []
    mass_array = []
    weight = []
    sigma_dif = []
    snapshot = []
    #
    hosts = []
    count = 0
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
            #
            mask_w = (satellite_match['weight'][mask][i] > 0)
            ind_w = np.where(np.max(satellite_match['weight'][mask][i][mask_w]) == satellite_match['weight'][mask][i][mask_w])[0][0]
            weight.append(satellite_match['weight'][mask][i][mask_w][ind_w])
            snapshot.append(satellite_match['snapshot'][mask][i][mask_w][ind_w])
            sigma_dif.append(satellite_match['sigma.dif'][mask][i][mask_w][ind_w])
    #count += len(weight)
    #
    final_dict[sat_name] = len(weight)







#ws = sat_match.mass_weighting(weight, mass_array, match['mass.peak'])

#sat_match.write_subhalo_matches('Sculptor', hosts, tree_index, ws, snapshot)




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