#!/usr/bin/python3

"""
    ====================
    = Paper III ORBITS =
    ====================

    Plot the actual orbits of subhalo analogs of satellites.

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
import matplotlib.patches as mpatches
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

#galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']
galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12n', 'm12q', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']

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

n_555 = [12, 330, 378, 83, 131, 3, 2, 305, 42, 213, 15, 13, 75, 89, 125, 5, 243, 153, 43, 13, 26, 57, 4, 125, 4, 99, 99, 111, 163, 301, 212, 49, 52, 0, 25, 167, 255, 347, 18, 67, 517, 362, 178, 13, 387, 127, 77, 363, 104, 954, 2, 107, 4, 103, 20, 12, 42, 17, 15, 151, 0, 183, 113, 193, 261, 20, 62, 87, 45, 53]

# Loop through all of the satellites and create a figure for each of them
for sat_idx, galaxy in enumerate(mw_sats_1Mpc):
    satellite_name = galaxy.replace(' ', '_')
    #file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{satellite_name}.txt'
    if n_555[sat_idx] < 10:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_10_10_10/weights_{satellite_name}.txt'
    else:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
    sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
    #
    plot_data = dict()
    plot_data['Hosts'] = []
    #
    for name in galaxies:
        mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)
        snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)
        #
        # get the matches for a simulation host
        mask = (gal_data['Host'] == name)
        # get an array of the tree indices that are matches in m12m
        subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
        # Get indices within the mini data to select properties
        mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
        #
        if len(mini_data_match_inds) != 0:
            plot_data['Hosts'].append(name)
            # Find the infall time
            # Snapshot at match
            snap_at_match = snaps['index'][gal_data['Snapshot at match'][mask]]
            #
            plot_data['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
            #
            for i in range(0, len(mini_data_match_inds)):
                #
                # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
                snap_ind = snaps['index'][-1] - snap_at_match[i]
                plot_data['orbit.distance.'+name][i][snap_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][snap_ind:]
            #
            plot_data['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]
    #
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    #
    for name in galaxies:
        if name in plot_data['Hosts']:
            for i in range(0, plot_data['orbit.distance.'+name].shape[0]):
                mask = (plot_data['orbit.distance.'+name][i] != -1)
                axs.plot(plot_data['orbit.time.lb.'+name][mask]-plot_data['orbit.time.lb.'+name][mask][0], plot_data['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.1)
    #
    axs.set_xlabel('Lookback Time [Gyr]', fontsize=24)
    axs.set_ylabel('Distance [kpc]', fontsize=24)
    axs.set_xlim(0,13.78)
    axs.set_ylim(-5,400)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/orbits/'+satellite_name+'_orbits_all.pdf')
    plt.close()


### The four case study plots for the paper
# Bootes V
galaxy = 'Bootes V'
satellite_name = galaxy.replace(' ', '_')
file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=False)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
plot_dat_bootes_v = dict()
plot_dat_bootes_v['Hosts'] = []
#
galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12n', 'm12q', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']
for name in galaxies:
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=False)
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)
    #
    # get the matches for a simulation host
    mask = (gal_data['Host'] == name)
    # get an array of the tree indices that are matches in m12m
    subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
    # Get indices within the mini data to select properties
    mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
    #
    if len(mini_data_match_inds) != 0:
        plot_dat_bootes_v['Hosts'].append(name)
        # Find the infall time
        # Snapshot at match
        snap_at_match = snaps['index'][gal_data['Snapshot at match'][mask]]
        #
        plot_dat_bootes_v['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
        plot_dat_bootes_v['orbit.v.rad.'+name] = (-1)*np.ones(mini_data['v.rad.sim'][mini_data_match_inds].shape)
        plot_dat_bootes_v['orbit.v.tan.'+name] = (-1)*np.ones(mini_data['v.tan.sim'][mini_data_match_inds].shape)
        #
        for i in range(0, len(mini_data_match_inds)):
            #
            # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
            snap_ind = snaps['index'][-1] - snap_at_match[i]
            plot_dat_bootes_v['orbit.distance.'+name][i][snap_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][snap_ind:]
            plot_dat_bootes_v['orbit.v.rad.'+name][i][snap_ind:] = mini_data['v.rad.sim'][mini_data_match_inds][i][snap_ind:]
            plot_dat_bootes_v['orbit.v.tan.'+name][i][snap_ind:] = mini_data['v.tan.sim'][mini_data_match_inds][i][snap_ind:]
        #
        plot_dat_bootes_v['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]

# Canes Venatici II
galaxy = 'Canes Venatici II'
satellite_name = galaxy.replace(' ', '_')
file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
plot_dat_canven_ii = dict()
plot_dat_canven_ii['Hosts'] = []
#
for name in galaxies:
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)
    #
    # get the matches for a simulation host
    mask = (gal_data['Host'] == name)
    # get an array of the tree indices that are matches in m12m
    subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
    # Get indices within the mini data to select properties
    mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
    #
    if len(mini_data_match_inds) != 0:
        plot_dat_canven_ii['Hosts'].append(name)
        # Find the infall time
        # Snapshot at match
        snap_at_match = snaps['index'][gal_data['Snapshot at match'][mask]]
        #
        plot_dat_canven_ii['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
        plot_dat_canven_ii['orbit.v.rad.'+name] = (-1)*np.ones(mini_data['v.rad.sim'][mini_data_match_inds].shape)
        plot_dat_canven_ii['orbit.v.tan.'+name] = (-1)*np.ones(mini_data['v.tan.sim'][mini_data_match_inds].shape)
        #
        for i in range(0, len(mini_data_match_inds)):
            #
            # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
            snap_ind = snaps['index'][-1] - snap_at_match[i]
            plot_dat_canven_ii['orbit.distance.'+name][i][snap_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][snap_ind:]
            plot_dat_canven_ii['orbit.v.rad.'+name][i][snap_ind:] = mini_data['v.rad.sim'][mini_data_match_inds][i][snap_ind:]
            plot_dat_canven_ii['orbit.v.tan.'+name][i][snap_ind:] = mini_data['v.tan.sim'][mini_data_match_inds][i][snap_ind:]
        #
        plot_dat_canven_ii['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]


# Aquarius III
galaxy = 'Aquarius III'
satellite_name = galaxy.replace(' ', '_')
file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
plot_data_aqu_iii = dict()
plot_data_aqu_iii['Hosts'] = []
#
for name in galaxies:
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)
    #
    # get the matches for a simulation host
    mask = (gal_data['Host'] == name)
    # get an array of the tree indices that are matches in m12m
    subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
    # Get indices within the mini data to select properties
    mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
    #
    if len(mini_data_match_inds) != 0:
        plot_data_aqu_iii['Hosts'].append(name)
        # Find the infall time
        # Snapshot at match
        snap_at_match = snaps['index'][gal_data['Snapshot at match'][mask]]
        #
        plot_data_aqu_iii['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
        plot_data_aqu_iii['orbit.v.rad.'+name] = (-1)*np.ones(mini_data['v.rad.sim'][mini_data_match_inds].shape)
        plot_data_aqu_iii['orbit.v.tan.'+name] = (-1)*np.ones(mini_data['v.tan.sim'][mini_data_match_inds].shape)
        #
        for i in range(0, len(mini_data_match_inds)):
            #
            # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
            snap_ind = snaps['index'][-1] - snap_at_match[i]
            plot_data_aqu_iii['orbit.distance.'+name][i][snap_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][snap_ind:]
            plot_data_aqu_iii['orbit.v.rad.'+name][i][snap_ind:] = mini_data['v.rad.sim'][mini_data_match_inds][i][snap_ind:]
            plot_data_aqu_iii['orbit.v.tan.'+name][i][snap_ind:] = mini_data['v.tan.sim'][mini_data_match_inds][i][snap_ind:]
        #
        plot_data_aqu_iii['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]

# Pegasus III
galaxy = 'Pegasus III'
satellite_name = galaxy.replace(' ', '_')
file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
plot_dat_pegasus_iii = dict()
plot_dat_pegasus_iii['Hosts'] = []
#
for name in galaxies:
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)
    #
    # get the matches for a simulation host
    mask = (gal_data['Host'] == name)
    # get an array of the tree indices that are matches in m12m
    subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
    # Get indices within the mini data to select properties
    mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
    #
    if len(mini_data_match_inds) != 0:
        plot_dat_pegasus_iii['Hosts'].append(name)
        # Find the infall time
        # Snapshot at match
        snap_at_match = snaps['index'][gal_data['Snapshot at match'][mask]]
        #
        plot_dat_pegasus_iii['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
        plot_dat_pegasus_iii['orbit.v.rad.'+name] = (-1)*np.ones(mini_data['v.rad.sim'][mini_data_match_inds].shape)
        plot_dat_pegasus_iii['orbit.v.tan.'+name] = (-1)*np.ones(mini_data['v.tan.sim'][mini_data_match_inds].shape)
        #
        for i in range(0, len(mini_data_match_inds)):
            #
            # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
            snap_ind = snaps['index'][-1] - snap_at_match[i]
            plot_dat_pegasus_iii['orbit.distance.'+name][i][snap_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][snap_ind:]
            plot_dat_pegasus_iii['orbit.v.rad.'+name][i][snap_ind:] = mini_data['v.rad.sim'][mini_data_match_inds][i][snap_ind:]
            plot_dat_pegasus_iii['orbit.v.tan.'+name][i][snap_ind:] = mini_data['v.tan.sim'][mini_data_match_inds][i][snap_ind:]
        #
        plot_dat_pegasus_iii['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]

# Plot the two case studies
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(4, 1, figsize=(16,12))
#
for name in galaxies:
    if name in plot_dat_bootes_v['Hosts']:
        for i in range(0, plot_dat_bootes_v['orbit.distance.'+name].shape[0]):
            mask = (plot_dat_bootes_v['orbit.distance.'+name][i] != -1)
            axs[0].plot(plot_dat_bootes_v['orbit.time.lb.'+name][mask]-plot_dat_bootes_v['orbit.time.lb.'+name][mask][0], plot_dat_bootes_v['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.3)
for name in galaxies:
    if name in plot_dat_canven_ii['Hosts']:
        for i in range(0, plot_dat_canven_ii['orbit.distance.'+name].shape[0]):
            mask = (plot_dat_canven_ii['orbit.distance.'+name][i] != -1)
            axs[1].plot(plot_dat_canven_ii['orbit.time.lb.'+name][mask]-plot_dat_canven_ii['orbit.time.lb.'+name][mask][0], plot_dat_canven_ii['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.3)
for name in galaxies:
    if name in plot_data_aqu_iii['Hosts']:
        for i in range(0, plot_data_aqu_iii['orbit.distance.'+name].shape[0]):
            mask = (plot_data_aqu_iii['orbit.distance.'+name][i] != -1)
            axs[2].plot(plot_data_aqu_iii['orbit.time.lb.'+name][mask]-plot_data_aqu_iii['orbit.time.lb.'+name][mask][0], plot_data_aqu_iii['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.3)
for name in galaxies:
    if name in plot_dat_pegasus_iii['Hosts']:
        for i in range(0, plot_dat_pegasus_iii['orbit.distance.'+name].shape[0]):
            mask = (plot_dat_pegasus_iii['orbit.distance.'+name][i] != -1)
            axs[3].plot(plot_dat_pegasus_iii['orbit.time.lb.'+name][mask]-plot_dat_pegasus_iii['orbit.time.lb.'+name][mask][0], plot_dat_pegasus_iii['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.3)
#
r1 = mpatches.Rectangle(xy=(0.25,275),width=1.65,height=90, facecolor='#D3D3D3', alpha=1, zorder=10)
r2 = mpatches.Rectangle(xy=(0.25,275),width=3.00,height=90, facecolor='#D3D3D3', alpha=1, zorder=10)
r3 = mpatches.Rectangle(xy=(0.25,275),width=2.15,height=90, facecolor='#D3D3D3', alpha=1, zorder=10)
r4 = mpatches.Rectangle(xy=(0.25,75),width=2.05,height=90, facecolor='#D3D3D3', alpha=1, zorder=10)
axs[0].text(0.35,295,'Bootes V',fontsize=24, zorder=11)
axs[1].text(0.35,295,'Canes Venatici II',fontsize=24, zorder=11)
axs[2].text(0.35,295,'Aquarius III',fontsize=24, zorder=11)
axs[3].text(0.35,95,'Pegasus III',fontsize=24, zorder=11)
axs[0].add_patch(r1)
axs[1].add_patch(r2)
axs[2].add_patch(r3)
axs[3].add_patch(r4)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[3].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
#
#axs[0].set_xlabel('Lookback Time [Gyr]', fontsize=24)
axs[0].set_xlim(0, 13.8)
axs[1].set_xlim(0, 13.8)
axs[2].set_xlim(0, 13.8)
axs[3].set_xlim(0, 13.8)
axs[0].set_ylim(0, 400)
axs[1].set_ylim(0, 400)
axs[2].set_ylim(0, 400)
axs[3].set_ylim(0, 400)
axs[3].set_xlabel('Lookback Time [Gyr]', fontsize=26)
axs[0].set_ylabel('Distance [kpc]', fontsize=24)
axs[1].set_ylabel('Distance [kpc]', fontsize=24)
axs[2].set_ylabel('Distance [kpc]', fontsize=24)
axs[3].set_ylabel('Distance [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/case_study_orbits_all.pdf')
plt.close()
