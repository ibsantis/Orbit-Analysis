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

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']

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

# Loop through all of the satellites and create a figure for each of them
plot_data = dict()
plot_data['Hosts'] = []
#
for galaxy in mw_sats_1Mpc:
    gal_data = sat_analysis.read_subhalo_matches(galaxy)
    satellite_name = galaxy.replace(' ', '_')
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
        if len(mini_data_match_inds != 0):
            plot_data['Hosts'].append(name)
            # Find the infall time
            # Snapshot at match
            time_at_match = snaps['time'][gal_data['Snapshot at match'][mask]]
            #
            plot_data['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
            #
            for i in range(0, len(mini_data_match_inds)):
                #
                # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
                time_ind = snaps['index'][-1] - np.where(np.min(np.abs(time_at_match[i] - snaps['time'])) == np.abs(time_at_match[i] - snaps['time']))[0][0]
                plot_data['orbit.distance.'+name][i][time_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][time_ind:]
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
                axs.plot(plot_data['orbit.time.lb.'+name][mask]-plot_data['orbit.time.lb.'+name][mask][0], plot_data['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.2)
    #
    axs.set_xlabel('Lookback Time [Gyr]', fontsize=24)
    axs.set_ylabel('Distance [kpc]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/orbits/'+galaxy+'_orbits_all.pdf')
    plt.close()


### The two case study plots for the paper
# Creating data for Bootes 3
galaxy = 'Bootes 3'
gal_data = sat_analysis.read_subhalo_matches(galaxy)
satellite_name = galaxy.replace(' ', '_')
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
plot_data_boo = dict()
plot_data_boo['Hosts'] = []
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
    if len(mini_data_match_inds != 0):
        plot_data_boo['Hosts'].append(name)
        # Find the infall time
        # Snapshot at match
        time_at_match = snaps['time'][gal_data['Snapshot at match'][mask]]
        #
        plot_data_boo['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
        #
        for i in range(0, len(mini_data_match_inds)):
            #
            # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
            time_ind = snaps['index'][-1] - np.where(np.min(np.abs(time_at_match[i] - snaps['time'])) == np.abs(time_at_match[i] - snaps['time']))[0][0]
            plot_data_boo['orbit.distance.'+name][i][time_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][time_ind:]
        #
        plot_data_boo['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]

# Creating data for Carina
galaxy = 'Carina'
gal_data = sat_analysis.read_subhalo_matches(galaxy)
satellite_name = galaxy.replace(' ', '_')
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
plot_data_car = dict()
plot_data_car['Hosts'] = []
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
    if len(mini_data_match_inds != 0):
        plot_data_car['Hosts'].append(name)
        # Find the infall time
        # Snapshot at match
        time_at_match = snaps['time'][gal_data['Snapshot at match'][mask]]
        #
        plot_data_car['orbit.distance.'+name] = (-1)*np.ones(mini_data['d.tot.sim'][mini_data_match_inds].shape)
        #
        for i in range(0, len(mini_data_match_inds)):
            #
            # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
            time_ind = snaps['index'][-1] - np.where(np.min(np.abs(time_at_match[i] - snaps['time'])) == np.abs(time_at_match[i] - snaps['time']))[0][0]
            plot_data_car['orbit.distance.'+name][i][time_ind:] = mini_data['d.tot.sim'][mini_data_match_inds][i][time_ind:]
        #
        plot_data_car['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]


# Plot the two case studies
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 1, figsize=(16,12))
#
for name in galaxies:
    if name in plot_data_boo['Hosts']:
        for i in range(0, plot_data_boo['orbit.distance.'+name].shape[0]):
            mask = (plot_data_boo['orbit.distance.'+name][i] != -1)
            axs[0].plot(plot_data_boo['orbit.time.lb.'+name][mask]-plot_data_boo['orbit.time.lb.'+name][mask][0], plot_data_boo['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.2)
for name in galaxies:
    if name in plot_data_car['Hosts']:
        for i in range(0, plot_data_car['orbit.distance.'+name].shape[0]):
            mask = (plot_data_car['orbit.distance.'+name][i] != -1)
            axs[1].plot(plot_data_car['orbit.time.lb.'+name][mask]-plot_data_car['orbit.time.lb.'+name][mask][0], plot_data_car['orbit.distance.'+name][i][mask], color='k', linewidth=0.5, alpha=0.2)
#
r1 = mpatches.Rectangle(xy=(0.5,800),width=1.70,height=90, facecolor='#D3D3D3', alpha=1, zorder=10)
r2 = mpatches.Rectangle(xy=(0.5,600),width=1.38,height=68, facecolor='#D3D3D3', alpha=1, zorder=10)
axs[0].text(0.6,820,'Bootes 3',fontsize=24, zorder=11)
axs[1].text(0.6,615,'Carina',fontsize=24, zorder=11)
axs[0].add_patch(r1)
axs[1].add_patch(r2)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
#
#axs[0].set_xlabel('Lookback Time [Gyr]', fontsize=24)
axs[0].set_xlim(0, 13.8)
axs[1].set_xlim(0, 13.8)
axs[1].set_xlabel('Lookback Time [Gyr]', fontsize=26)
axs[0].set_ylabel('Distance [kpc]', fontsize=26)
axs[1].set_ylabel('Distance [kpc]', fontsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/orbits/case_study_orbits_all.pdf')
plt.close()
