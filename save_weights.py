
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

#galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']
galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12n']

mw_sats_old = ['NGC 55', 'LMC', 'SMC', 'IC 4662', 'IC 5152', 'NGC 6822', 'NGC 3109', 'IC 3104', \
           'Sextans B', 'DDO 190', 'DDO 125', 'Sextans A', 'NGC 4163', 'Sagittarius dSph', 'UGC 8508', 'Fornax', 'UGC 4879', \
           'UGC 9128', 'GR 8', 'Leo A', 'Leo 1', 'Sagittarius dIrr', 'ESO 294-G010', 'DDO 113', 'Sculptor', 'Antlia 2', 'Aquarius (DDO 210)',\
           'Phoenix', 'Leo 2', 'Antlia B', 'Tucana', 'KKR 3', 'Carina', 'Leo P', 'Crater 2', 'Ursa Minor', 'Sextans 1', \
           'Draco', 'Canes Venatici 1', 'Leo T', 'Eridanus 2', 'Bootes 1', 'Hercules', 'Bootes 3', 'Sagittarius 2', \
           'Canes Venatici 2', 'Ursa Major 1', 'Leo 4', 'Hydra 2', 'Hydrus 1', 'Carina 2', 'Ursa Major 2', 'Aquarius 2', \
           'Indus 2', 'Coma Berenices', 'Leo 5', 'Pisces 2', 'Columba 1', 'Tucana 5', 'Pegasus 3', 'Grus 2', 'Tucana 2', \
           'Reticulum 2', 'Horologium 1', 'Pictor 1', 'Tucana 4', 'Indus 1', 'Grus 1', 'Reticulum 3', 'Pictor 2', 'Bootes 2',\
           'Willman 1', 'Phoenix 2', 'Cetus 3', 'Carina 3', 'Eridanus 3', 'Segue 2', 'Triangulum 2', 'Horologium 2', 'Tucana 3',\
           'Segue 1', 'DES J0225+0304', 'Virgo 1', 'Draco 2', 'Cetus 2']

mw_sats =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
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

file_save_path = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/'

###### TWEAK THE CODE SO THAT IT ONLY WORKS ON ONE SATELLITE
final_dict = dict()
halo_mass_dex_error = 0.35
sigma_phase_space = 3
percent_nd_gaussian = 99 # 3 sigma
#alpha=0.58
#
start_time = time.time()
for sat_name in mw_sats:
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
    #######final_dict[sat_name] = len(weight)
    #
    ws = sat_match.mass_weighting(weight, mass_array, match['mass.peak'], SMHM_slope=0.44)
    #
    param_list = [halo_mass_dex_error, sigma_phase_space, percent_nd_gaussian]
    #
    # if np.all(np.isfinite(ws)) and len(ws) != 0:
    #     sat_mass.append(match['mass.peak'])
    #     med_reg.append(np.median(mass_array))
    #     med_w.append(ut.math.percentile_weighted(mass_array, 50, ws))
    #
    sat_match.write_subhalo_matches(sat_name, hosts, tree_index, ws, snapshot, param_list, file_save_path)

# Checking how the mass weighting affects the mass distribution
# sat_mass = np.log10(np.asarray(sat_mass))
# med_reg = np.log10(np.asarray(med_reg))
# med_w = np.log10(np.asarray(med_w)) ########

# frac_diff_reg = (sat_mass - med_reg)/med_reg
# frac_diff_w = (sat_mass - med_w)/med_w
# #
# # Plot the mass vs medians
# plt.rcParams["font.family"] = "serif"
# f, axs = plt.subplots(2, 1, figsize=(12,16))
# axs[0].scatter(sat_mass, frac_diff_reg, s=50, c='k')
# axs[0].set_ylabel('$(M_{\\rm peak, sat} - M_{\\rm med})/M_{\\rm med}$', fontsize=24)
# axs[0].tick_params(axis='both', which='major', labelsize=28)
# #
# axs[1].scatter(sat_mass, frac_diff_w, s=50, c='k')
# axs[1].set_ylabel('$(M_{\\rm peak, sat} - M_{\\rm med,weighted})/M_{\\rm med, weighted}$', fontsize=24)
# axs[1].tick_params(axis='both', which='major', labelsize=28)
# #
# axs[1].set_xlabel('log $M_{\\rm peak, sat}$', fontsize=24)
# axs[1].tick_params(axis='both', which='major', labelsize=28)
# plt.suptitle('$\\alpha = {0}$'.format(alpha), fontsize=24)
# plt.tight_layout()
# #plt.show()
# plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/mass_weight_testing/frac_diffs_alpha_{0}.pdf'.format(alpha))
# plt.close()




# # Testing reading in a file
# # WORKS!
# galaxy = 'Sculptor'

# gal_data = sat_match.read_subhalo_matches(galaxy)




## Compare making pericenter distributions for a few satellites with and without mass weighting
sat_name = 'Sculptor'
#
med_reg = (-1)*np.ones(len(mw_sats))
med_wei = (-1)*np.ones(len(mw_sats))
#
for sat_name in mw_sats:
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
        match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=sat_name, mass_err=halo_mass_dex_error)

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

    weight = np.asarray(weight)
    mass_array = np.asarray(mass_array)
    #
    if np.isfinite(match['mass.star']):
        weight_mass = sat_match.mass_weighting(weight, mass_array, match['mass.peak'])
        #
        med_reg[i] = np.median(mass_array)
        med_wei[i] = ut.math.percentile_weighted(mass_array, 50, weight_mass)


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

plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(20,8))
axs[0].hist(x, binss, density=False, weights=weight, linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4, label='No mass weighting')
axs[0].set_xlim(minn, maxx)
axs[0].set_xlabel('log $M_{\\rm halo,peak}/M_{\\odot}$', fontsize=24)
axs[0].tick_params(axis='both', which='major', labelsize=28)
axs[0].legend(prop={'size': 24}, loc='best')
#
axs[1].hist(x, binss, density=False, weights=weight_mass, linestyle='solid', linewidth=2, histtype='stepfilled', color='k', alpha=0.4, label='With mass weighting')
axs[1].set_xlim(minn, maxx)
axs[1].set_xlabel('log $M_{\\rm halo,peak}/M_{\\odot}$', fontsize=24)
axs[0].set_ylabel('Probability', fontsize=24)
axs[1].legend(prop={'size': 24}, loc='best')
plt.suptitle('{0}, Nsat = {1}, log Mpeak = {2}'.format(sat_name, len(weight), np.around(mass_log_sat, 2)), fontsize=24)
axs[1].tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/mass_weight_testing/'+sat_name+'.pdf')
plt.close()




## Some useful information for stats
stat_info = ut.math.StatisticClass(mass_log_sub, limits=[minn,maxx], bin_width=binsize, bin_number=bin_num, weights=weight)
stat_info.stat # information about medians, percentiles, etc.
stat_info.distr # information about the histogram stuff from Andrew's functions





## compare the regular median to the mass weighted median
sat_name = 'Sculptor'
#
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
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=sat_name, mass_err=halo_mass_dex_error)

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

weight = np.asarray(weight)
mass_array = np.asarray(mass_array)
#
weight_mass = sat_match.mass_weighting(weight, mass_array, match['mass.peak'])

