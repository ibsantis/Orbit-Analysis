#!/usr/bin/python3

"""

    =======================================
    = Pericenter tree vs particle compare =
    =======================================

    Compare the pericenter dictionary when derived from the host position in the
    halo tree (as derived from the center of the DM halo) vs the host position
    in the snapshot data (as derived from the center of the galaxy, i.e.,
    star particles)

    10/01/21: Ran on Stampede

"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import spatial
import time
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='peloton')
print('Set paths')

if sim_data.num_gal == 1:
    #
    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
    halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
    part = gizmo.io.Read.read_snapshots('star', 'redshift', 0, properties=['position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    #
    # Set up the halo inds and KDTree
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1, dmo=False)
    #
    # Now get the distances from the particle catalog and from the halo tree
    host_positions = np.flip(part.hostz['position'][:,0,:], axis=0)
    host_velocities = np.flip(part.hostz['velocity'][:,0,:], axis=0)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
    #
    # These are the distances that I'm currently using
    halt_dists = orbits.halo_distances(tree=halt)
    halt_vels = orbits.halo_velocities(halt)
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    #
    halo_pos = (-1)*np.ones(orbits.shape)
    halo_vel = (-1)*np.ones(orbits.shape)
    #
    # Loop over each halo
    for i in range(0, orbits.shape[0]):
        hi = 600-np.flip(snaps['index'])[np.isnan(host_positions[:,0])][0]
        si = len(orbits.sub_inds[i][orbits.sub_inds[i] >= 0])
        min_ind = np.min([hi,si])
        #
        halo_pos[i][:min_ind] = np.linalg.norm(halt['position'][orbits.sub_inds[i][:min_ind]] - host_positions[:min_ind], axis=1)*np.flip(snaps['scalefactor'])[:min_ind]
        halo_vel[i][:min_ind] = np.linalg.norm(halt['velocity'][orbits.sub_inds[i][:min_ind]] - host_velocities[:min_ind], axis=1)*np.flip(snaps['scalefactor'])[:min_ind]
    #
    halo_pos_norm = (-1)*np.ones(orbits.shape)
    for i in range(0, len(halo_pos_norm)):
        hi = len(host_radii)
        si = len(halo_pos[i][halo_pos[i] != -1])
        min_ind = np.min([hi,si])
        #
        halo_pos_norm[i][:min_ind] = halo_pos[i][:min_ind]/host_radii[:min_ind]
    #
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
    peris_new = orbits.pericenter_interp(distances=halo_pos, velocities=halo_vel, virial_radii=host_radii, time_array=snaps)
    #
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_part = orbits.infall_times(halo_pos_norm, snaps)
    #
    data_dict = dict()
    for i in peris.keys():
        data_dict[i+'.halt'] = peris[i]
        data_dict[i+'.part'] = peris_new[i]
    for i in infall_info.keys():
        data_dict[i+'.halt'] = infall_info[i]
        data_dict[i+'.part'] = infall_info_part[i]
    data_dict['dtot.halt'] = halt_dists
    data_dict['dtot.part'] = halo_pos
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/peri_check/data_'+sim_data.galaxy, dict_or_array_to_write=data_dict, verbose=True)

if sim_data.num_gal == 2:
    #
    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
    halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
    part = gizmo.io.Read.read_snapshots('star', 'redshift', 0, properties=['position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    #
    ### GALAXY 1
    # Set up the halo inds and KDTree
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1, dmo=False)
    #
    # Now get the distances from the particle catalog and from the halo tree
    host_positions = np.flip(part.hostz['position'][:,0,:], axis=0)
    host_velocities = np.flip(part.hostz['velocity'][:,0,:], axis=0)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
    #
    # These are the distances that I'm currently using
    halt_dists = orbits.halo_distances(tree=halt)
    halt_vels = orbits.halo_velocities(halt)
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    #
    halo_pos = (-1)*np.ones(orbits.shape)
    halo_vel = (-1)*np.ones(orbits.shape)
    #
    # Loop over each halo
    for i in range(0, orbits.shape[0]):
        hi = 600-np.flip(snaps['index'])[np.isnan(host_positions[:,0])][0]
        si = len(orbits.sub_inds[i][orbits.sub_inds[i] >= 0])
        min_ind = np.min([hi,si])
        #
        halo_pos[i][:min_ind] = np.linalg.norm(halt['position'][orbits.sub_inds[i][:min_ind]] - host_positions[:min_ind], axis=1)*np.flip(snaps['scalefactor'])[:min_ind]
        halo_vel[i][:min_ind] = np.linalg.norm(halt['velocity'][orbits.sub_inds[i][:min_ind]] - host_velocities[:min_ind], axis=1)*np.flip(snaps['scalefactor'])[:min_ind]
    #
    halo_pos_norm = (-1)*np.ones(orbits.shape)
    for i in range(0, len(halo_pos_norm)):
        hi = len(host_radii)
        si = len(halo_pos[i][halo_pos[i] != -1])
        min_ind = np.min([hi,si])
        #
        halo_pos_norm[i][:min_ind] = halo_pos[i][:min_ind]/host_radii[:min_ind]
    #
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
    peris_new = orbits.pericenter_interp(distances=halo_pos, velocities=halo_vel, virial_radii=host_radii, time_array=snaps)
    #
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_part = orbits.infall_times(halo_pos_norm, snaps)
    #
    data_dict = dict()
    for i in peris.keys():
        data_dict[i+'.halt'] = peris[i]
        data_dict[i+'.part'] = peris_new[i]
    for i in infall_info.keys():
        data_dict[i+'.halt'] = infall_info[i]
        data_dict[i+'.part'] = infall_info_part[i]
    data_dict['dtot.halt'] = halt_dists
    data_dict['dtot.part'] = halo_pos
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/peri_check/data_'+sim_data.gal_1, dict_or_array_to_write=data_dict, verbose=True)
    #
    ### GALAXY 2
    # Set up the halo inds and KDTree
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=2, dmo=False)
    #
    # Now get the distances from the particle catalog and from the halo tree
    host_positions = np.flip(part.hostz['position'][:,1,:], axis=0)
    host_velocities = np.flip(part.hostz['velocity'][:,1,:], axis=0)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host2.index'][0])]
    #
    # These are the distances that I'm currently using
    halt_dists = orbits.halo_distances(tree=halt, host=2)
    halt_vels = orbits.halo_velocities(halt, host=2)
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    #
    halo_pos = (-1)*np.ones(orbits.shape)
    halo_vel = (-1)*np.ones(orbits.shape)
    #
    # Loop over each halo
    for i in range(0, orbits.shape[0]):
        hi = 600-np.flip(snaps['index'])[np.isnan(host_positions[:,0])][0]
        si = len(orbits.sub_inds[i][orbits.sub_inds[i] >= 0])
        min_ind = np.min([si,hi])
        #
        halo_pos[i][:min_ind] = np.linalg.norm(halt['position'][orbits.sub_inds[i][:min_ind]] - host_positions[:min_ind], axis=1)*np.flip(snaps['scalefactor'])[:min_ind]
        halo_vel[i][:min_ind] = np.linalg.norm(halt['velocity'][orbits.sub_inds[i][:min_ind]] - host_velocities[:min_ind], axis=1)*np.flip(snaps['scalefactor'])[:min_ind]
    #
    halo_pos_norm = (-1)*np.ones(orbits.shape)
    for i in range(0, len(halo_pos_norm)):
        hi = len(host_radii)
        si = len(halo_pos[i][halo_pos[i] != -1])
        min_ind = np.min([si,hi])
        #
        halo_pos_norm[i][:min_ind] = halo_pos[i][:min_ind]/host_radii[:min_ind]
    #
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
    peris_new = orbits.pericenter_interp(distances=halo_pos, velocities=halo_vel, virial_radii=host_radii, time_array=snaps)
    #
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_part = orbits.infall_times(halo_pos_norm, snaps)
    #
    data_dict = dict()
    for i in peris.keys():
        data_dict[i+'.halt'] = peris[i]
        data_dict[i+'.part'] = peris_new[i]
    for i in infall_info.keys():
        data_dict[i+'.halt'] = infall_info[i]
        data_dict[i+'.part'] = infall_info_part[i]
    data_dict['dtot.halt'] = halt_dists
    data_dict['dtot.part'] = halo_pos
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/peri_check/data_'+sim_data.gal_2, dict_or_array_to_write=data_dict, verbose=True)
