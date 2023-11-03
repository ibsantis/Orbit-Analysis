#!/usr/bin/env python3

"""

    ========================
    = Integrating subhalos =
    ========================

    Save the mini-data files that I use...

"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
from scipy.interpolate import interp1d
from astropy import units as u
import pandas as pd
import sys
print('Read in the tools')

### Set path and initial parameters
loc = 'stampede'
sim_data = orbit_io.OrbitRead(gal1='m12j', location=loc, fire3=True)
plotting = False
aligned = True
point_mass = False
rotate = False
rot_axis, angle = 0, 180
#
if rotate and point_mass:
    raise AssertionError('Do not rotate point mass model!')
#
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal, assign_hosts_rotation=aligned, catalog_hdf5_directory='catalog_hdf5')
part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=aligned)

# Find the mass ratio to multiply the host radius
mass_ratio = ut.particle.get_halo_properties(part)['mass']/halt['mass'][halt['host.index'][0]]
#
# This initializes the classes and makes sure they inherit from the OrbitRead class
orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, selection='halo', fire3=True)
#
# Run the pipeline on the simulation data
halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
halt_dists_3d = orbits.halo_distances(tree=halt, dist_type='3d')
halt_vels = orbits.halo_velocities(halt, vel_type='total')
halt_rad_vels = orbits.halo_velocities(halt, vel_type='rad')
halt_tan_vels = orbits.halo_velocities(halt, vel_type='tan')
#
host_mhalo = halt['mass'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
#
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii*mass_ratio)
infall_info = orbits.infall_times(halt_dists_norm, snaps)
infall_info_any = orbits.first_infall_any(halt, snaps)
peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
angs = orbits.angular_momentum(tree=halt)
periods = orbits.orbit_period(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
eccs = orbits.eccentricity(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)

# Save the data to a dictionary
data_dict = dict()
#
# z = 0 indices
data_dict['indices.z0'] = orbits.sub_inds
data_dict['id'] = np.arange(len(orbits.sub_inds[:,0]))+1
#
# Stellar mass of the subhalos at z = 0 and peak stellar mass
data_dict['M.star.z0'] = halt['star.mass'][orbits.sub_inds[:,0]]
data_dict['M.star.peak'] = halt.prop('star.mass.peak', orbits.sub_inds[:,0])
data_dict['M.halo.z0'] = halt['mass'][orbits.sub_inds[:,0]]
data_dict['M.halo.peak'] = halt.prop('mass.peak', orbits.sub_inds[:,0])
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
data_dict['d.sim'] = halt_dists_3d
data_dict['v.tot.sim'] = halt_vels
data_dict['v.tan.sim'] = halt_tan_vels
data_dict['v.rad.sim'] = halt_rad_vels
data_dict['L.sim'] = angs['ang.mom.vector']
data_dict['L.tot.sim'] = angs['ang.mom.total']
data_dict['L.z.sim'] = angs['ang.mom.vector'][:,:,2]
#
# Find the angular momentum at each pericenter event
angs_at_peri = (-1)*np.ones(peris['pericenter.dist'].shape)
times = snaps['time'][-1] - np.flip(snaps['time'])
for i in range(0, len(peris['pericenter.check'])):
    if (peris['pericenter.check'][i]):
        mask = (peris['pericenter.time.lb'][i] >= 0)
        for j in range(0, len(peris['pericenter.time.lb'][i][mask])):
            t = peris['pericenter.time.lb'][i][mask][j]
            angs_at_peri[i][j] = angs['ang.mom.total'][i][np.where(np.min(np.abs(times - t)) == np.abs(times - t))[0][0]]
data_dict['L.at.peri'] = angs_at_peri
#
data_dict['time.sim'] = snaps['time']
#
data_dict['orbit.period.peri.sim'] = periods['pericenter.orbit.period']
data_dict['orbit.period.apo.sim'] = periods['apocenter.orbit.period']
data_dict['eccentricity.sim'] = eccs
#
# Save the host radius
data_dict['host.radius'] = host_radii*mass_ratio
data_dict['host.mass'] = host_mhalo*mass_ratio
data_dict['host.mass.ratio'] = mass_ratio
#
print('The host mass at z=0 without the mass ratio is:', halt['mass'][halt['host.index'][0]])
print('The host radius at z=0 without the mass ratio is:', halt['radius'][halt['host.index'][0]])
print('The host mass at z=0 with the mass ratio is:', halt['mass'][halt['host.index'][0]]*mass_ratio)
print('The host radius at z=0 with the mass ratio is:', halt['radius'][halt['host.index'][0]]*mass_ratio)
print('The mass ratio is:', mass_ratio)

ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy, dict_or_array_to_write=data_dict, verbose=True)
