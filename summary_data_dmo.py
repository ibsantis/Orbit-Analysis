#!/usr/bin/python3

"""

    ====================
    = Summary Data DMO =
    ====================

    Select halos in the DMO runs and save a their properties
    without integrating them in Galpy.

    Couldn't run on m12z
"""

import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='peloton', dmo=True)
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, host_number=sim_data.num_gal)

if sim_data.num_gal == 1:
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1, dmo=True)
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
    halt_vels = orbits.halo_velocities(halt)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_any = orbits.first_infall_any(halt, snaps)
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
    apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
    angs = orbits.angular_momentum(tree=halt)
    #
    # Save the data to a dictionary
    data_dict = dict()
    #
    # z = 0 indices
    data_dict['indices.z0'] = orbits.sub_inds
    #
    # Stellar mass of the subhalos at z = 0 and peak stellar mass
    data_dict['M.halo.z0'] = halt['mass'][orbits.sub_inds[:,0]]*(1-orbits.baryon_frac)
    data_dict['M.halo.peak'] = halt.prop('mass.peak', orbits.sub_inds[:,0])*(1-orbits.baryon_frac)
    #
    # Infall information
    data_dict['infall.check'] = infall_info['check']
    data_dict['first.infall.snap'] = infall_info['first.infall.snap']
    data_dict['first.infall.time'] = infall_info['first.infall.time']
    data_dict['first.infall.time.lb'] = infall_info['first.infall.time.lb']
    data_dict['all.infall.snap'] = infall_info['all.infall.snap']
    data_dict['all.infall.time'] = infall_info['all.infall.time']
    data_dict['all.infall.time.lb'] = infall_info['all.infall.time.lb']
    #
    data_dict['infall.check.any'] = infall_info_any['infall.check.any']
    data_dict['first.infall.snap.any'] = infall_info_any['first.infall.snap.any']
    data_dict['first.infall.time.any'] = infall_info_any['first.infall.time.any']
    data_dict['first.infall.time.lb.any'] = infall_info_any['first.infall.time.lb.any']
    #
    # Pericenter checks and numbers
    data_dict['pericenter.check.sim'] = peris['pericenter.check']
    data_dict['N.peri.sim'] = peris['pericenter.num']
    #
    # Pericenter distances
    data_dict['pericenter.dist.sim'] = peris['pericenter.dist']
    #
    # Pericenter velocities
    data_dict['pericenter.vel.sim'] = peris['pericenter.vel']
    #
    # Pericenter times
    data_dict['pericenter.time.sim'] = peris['pericenter.time']
    data_dict['pericenter.time.lb.sim'] = peris['pericenter.time.lb']
    #
    # Apocenter checks
    data_dict['apocenter.check.sim'] = apos['apocenter.check']
    #
    # Apocenter distances
    data_dict['apocenter.dist.sim'] = apos['apocenter.dist']
    #
    # Apocenter velocities
    data_dict['apocenter.vel.sim'] = apos['apocenter.vel']
    #
    # Apocenter times
    data_dict['apocenter.time.sim'] = apos['apocenter.time']
    data_dict['apocenter.time.lb.sim'] = apos['apocenter.time.lb']
    #
    # Maximum distances and times
    data_dict['max.dist.sim'] = apos['max.dist']
    data_dict['max.dist.time.sim'] = apos['max.dist.time']
    data_dict['max.dist.time.lb.sim'] = apos['max.dist.time.lb']
    #
    # distance, velocity, Lz vs time
    data_dict['d.tot.sim'] = halt_dists
    data_dict['v.tot.sim'] = halt_vels
    data_dict['L.sim'] = angs['ang.mom.vector']
    data_dict['L.tot.sim'] = angs['ang.mom.total']
    data_dict['L.z.sim'] = angs['ang.mom.vector'][:,:,2]
    data_dict['v.tan.z0'] = halt.prop('host.velocity.tan', orbits.sub_inds[:,0])
    data_dict['v.rad.z0'] = halt.prop('host.velocity.rad', orbits.sub_inds[:,0])
    data_dict['time.sim'] = snaps['time']
    #
    # Save the host radius
    data_dict['host.radius'] = host_radii

    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy+'_dmo', dict_or_array_to_write=data_dict, verbose=True)

if sim_data.num_gal == 2:
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1, dmo=True)
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
    halt_vels = orbits.halo_velocities(halt)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_any = orbits.first_infall_any(halt, snaps)
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
    apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
    angs = orbits.angular_momentum(tree=halt)
    #
    # Save the data to a dictionary
    data_dict = dict()
    #
    # z = 0 indices
    data_dict['indices.z0'] = orbits.sub_inds
    #
    # Stellar mass of the subhalos at z = 0 and peak stellar mass
    data_dict['M.halo.z0'] = halt['mass'][orbits.sub_inds[:,0]]*(1-orbits.baryon_frac)
    data_dict['M.halo.peak'] = halt.prop('mass.peak', orbits.sub_inds[:,0])*(1-orbits.baryon_frac)
    #
    # Infall information
    data_dict['infall.check'] = infall_info['check']
    data_dict['first.infall.snap'] = infall_info['first.infall.snap']
    data_dict['first.infall.time'] = infall_info['first.infall.time']
    data_dict['first.infall.time.lb'] = infall_info['first.infall.time.lb']
    data_dict['all.infall.snap'] = infall_info['all.infall.snap']
    data_dict['all.infall.time'] = infall_info['all.infall.time']
    data_dict['all.infall.time.lb'] = infall_info['all.infall.time.lb']
    #
    data_dict['infall.check.any'] = infall_info_any['infall.check.any']
    data_dict['first.infall.snap.any'] = infall_info_any['first.infall.snap.any']
    data_dict['first.infall.time.any'] = infall_info_any['first.infall.time.any']
    data_dict['first.infall.time.lb.any'] = infall_info_any['first.infall.time.lb.any']
    #
    # Pericenter checks and numbers
    data_dict['pericenter.check.sim'] = peris['pericenter.check']
    data_dict['N.peri.sim'] = peris['pericenter.num']
    #
    # Pericenter distances
    data_dict['pericenter.dist.sim'] = peris['pericenter.dist']
    #
    # Pericenter velocities
    data_dict['pericenter.vel.sim'] = peris['pericenter.vel']
    #
    # Pericenter times
    data_dict['pericenter.time.sim'] = peris['pericenter.time']
    data_dict['pericenter.time.lb.sim'] = peris['pericenter.time.lb']
    #
    # Apocenter checks
    data_dict['apocenter.check.sim'] = apos['apocenter.check']
    #
    # Apocenter distances
    data_dict['apocenter.dist.sim'] = apos['apocenter.dist']
    #
    # Apocenter velocities
    data_dict['apocenter.vel.sim'] = apos['apocenter.vel']
    #
    # Apocenter times
    data_dict['apocenter.time.sim'] = apos['apocenter.time']
    data_dict['apocenter.time.lb.sim'] = apos['apocenter.time.lb']
    #
    # Maximum distances and times
    data_dict['max.dist.sim'] = apos['max.dist']
    data_dict['max.dist.time.sim'] = apos['max.dist.time']
    data_dict['max.dist.time.lb.sim'] = apos['max.dist.time.lb']
    #
    # distance, velocity, Lz vs time
    data_dict['d.tot.sim'] = halt_dists
    data_dict['v.tot.sim'] = halt_vels
    data_dict['L.sim'] = angs['ang.mom.vector']
    data_dict['L.tot.sim'] = angs['ang.mom.total']
    data_dict['L.z.sim'] = angs['ang.mom.vector'][:,:,2]
    data_dict['v.tan.z0'] = halt.prop('host.velocity.tan', orbits.sub_inds[:,0])
    data_dict['v.rad.z0'] = halt.prop('host.velocity.rad', orbits.sub_inds[:,0])
    data_dict['time.sim'] = snaps['time']
    #
    # Save the host radius
    data_dict['host.radius'] = host_radii
    np.sum(peris['pericenter.check'])
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_1+'_dmo', dict_or_array_to_write=data_dict, verbose=True)
    #
    #
    ### GALAXY 2
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=2, dmo=True)
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt, host=2) # set host=1 for the first host, host=2 for the other
    halt_vels = orbits.halo_velocities(halt, host=2)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host2.index'][0])]
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_any = orbits.first_infall_any(halt, snaps)
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
    apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
    angs = orbits.angular_momentum(tree=halt, host=2)
    #
    # Save the data to a dictionary
    data_dict = dict()
    #
    # z = 0 indices
    data_dict['indices.z0'] = orbits.sub_inds
    #
    # Stellar mass of the subhalos at z = 0 and peak stellar mass
    data_dict['M.halo.z0'] = halt['mass'][orbits.sub_inds[:,0]]*(1-orbits.baryon_frac)
    data_dict['M.halo.peak'] = halt.prop('mass.peak', orbits.sub_inds[:,0])*(1-orbits.baryon_frac)
    #
    # Infall information
    data_dict['infall.check'] = infall_info['check']
    data_dict['first.infall.snap'] = infall_info['first.infall.snap']
    data_dict['first.infall.time'] = infall_info['first.infall.time']
    data_dict['first.infall.time.lb'] = infall_info['first.infall.time.lb']
    data_dict['all.infall.snap'] = infall_info['all.infall.snap']
    data_dict['all.infall.time'] = infall_info['all.infall.time']
    data_dict['all.infall.time.lb'] = infall_info['all.infall.time.lb']
    #
    data_dict['infall.check.any'] = infall_info_any['infall.check.any']
    data_dict['first.infall.snap.any'] = infall_info_any['first.infall.snap.any']
    data_dict['first.infall.time.any'] = infall_info_any['first.infall.time.any']
    data_dict['first.infall.time.lb.any'] = infall_info_any['first.infall.time.lb.any']
    #
    # Pericenter checks and numbers
    data_dict['pericenter.check.sim'] = peris['pericenter.check']
    data_dict['N.peri.sim'] = peris['pericenter.num']
    #
    # Pericenter distances
    data_dict['pericenter.dist.sim'] = peris['pericenter.dist']
    #
    # Pericenter velocities
    data_dict['pericenter.vel.sim'] = peris['pericenter.vel']
    #
    # Pericenter times
    data_dict['pericenter.time.sim'] = peris['pericenter.time']
    data_dict['pericenter.time.lb.sim'] = peris['pericenter.time.lb']
    #
    # Apocenter checks
    data_dict['apocenter.check.sim'] = apos['apocenter.check']
    #
    # Apocenter distances
    data_dict['apocenter.dist.sim'] = apos['apocenter.dist']
    #
    # Apocenter velocities
    data_dict['apocenter.vel.sim'] = apos['apocenter.vel']
    #
    # Apocenter times
    data_dict['apocenter.time.sim'] = apos['apocenter.time']
    data_dict['apocenter.time.lb.sim'] = apos['apocenter.time.lb']
    #
    # Maximum distances and times
    data_dict['max.dist.sim'] = apos['max.dist']
    data_dict['max.dist.time.sim'] = apos['max.dist.time']
    data_dict['max.dist.time.lb.sim'] = apos['max.dist.time.lb']
    #
    # distance, velocity, Lz vs time
    data_dict['d.tot.sim'] = halt_dists
    data_dict['v.tot.sim'] = halt_vels
    data_dict['L.sim'] = angs['ang.mom.vector']
    data_dict['L.tot.sim'] = angs['ang.mom.total']
    data_dict['L.z.sim'] = angs['ang.mom.vector'][:,:,2]
    data_dict['v.tan.z0'] = halt.prop('host2.velocity.tan', orbits.sub_inds[:,0])
    data_dict['v.rad.z0'] = halt.prop('host2.velocity.rad', orbits.sub_inds[:,0])
    data_dict['time.sim'] = snaps['time']
    #
    # Save the host radius
    data_dict['host.radius'] = host_radii
    np.sum(peris['pericenter.check'])

    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_2+'_dmo', dict_or_array_to_write=data_dict, verbose=True)
