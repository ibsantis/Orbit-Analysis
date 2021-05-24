#!/usr/bin/python3

"""

    ========================
    = Integrating subhalos =
    ========================

    Integrate subhalos in custom potential
        - Disk (radial and vertical) model
        - DM halo model

"""

# Import packages
from galpy.orbit import Orbit
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
from astropy import units as u
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12m', location='peloton')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)

# This initializes the classes and makes sure they inherit from the OrbitRead class
orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton')
orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.galaxy, location='peloton')
orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.galaxy, location='peloton')
#
# Run the pipeline on the simulation data
halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
halt_vels = orbits.halo_velocities(halt)
host_radii = halt['radius'][orbits.sub_inds[0][orbits.sub_inds[0] >= 0]] # Want to divide the other distances by this distance
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
infall_info = orbits.infall_times(halt_dists_norm, snaps)
peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
angs = orbits.angular_momentum(tree=halt)
#
# Initialize the orbits in Galpy
galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt)

# Read in the fitting parameters
fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)

# Import the potentials and combine them for our model
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
potential_two_power = disk_inner+disk_outer+halo_2p

# Integrate all of the orbits in both potentials
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
galpy_orbits.integrate(ts, potential_two_power, method='odeint')
print('Done integrating in potential model')

# Check to see if any of them are close to a pole
poles = orbit_gal.galpy_pole_check(galpy_orbits, ts)
print(poles)
print(np.sum(poles))

galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vR(ts), galpy_orbits.vT(ts))
tts = (-1)*np.linspace(0.0, -13.78, 1378)
peris_galpy = orbit_gal.galpy_pericenter_interp(galpy_orbits.r(ts), galpy_vels, tts)
apos_galpy = orbit_gal.galpy_apocenter_interp(galpy_orbits.r(ts), galpy_vels, tts)

# Save the data to a dictionary
data_dict = dict()
#
# z = 0 indices
data_dict['indices.z0'] = orbits.sub_inds
#
# Stellar mass of the subhalos at z = 0
data_dict['Mstar.z0'] = halt['star.mass'][orbits.sub_inds[:,0]]
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
# Pericenter checks and numbers
data_dict['pericenter.check.sim'] = peris['pericenter.num']
data_dict['N.peri.sim'] = peris['pericenter.num']
data_dict['pericenter.check.galpy'] = peris_galpy['pericenter.num']
data_dict['N.peri.galpy'] = peris_galpy['pericenter.num']
#
# Pericenter distances
data_dict['pericenter.dist.sim'] = peris['pericenter.dist']
data_dict['pericenter.dist.galpy'] = peris_galpy['pericenter.dist']
#
# Pericenter velocities
data_dict['pericenter.vel.sim'] = peris['pericenter.vel']
data_dict['pericenter.vel.galpy'] = peris_galpy['pericenter.vel']
#
# Pericenter times
data_dict['pericenter.time.sim'] = peris['pericenter.time']
data_dict['pericenter.time.galpy'] = peris_galpy['pericenter.time']
data_dict['pericenter.time.lb.sim'] = peris['pericenter.time.lb']
data_dict['pericenter.time.lb.galpy'] = peris_galpy['pericenter.time.lb']
#
# Apocenter checks
data_dict['apocenter.check.sim'] = apos['apocenter.check']
data_dict['apocenter.check.galpy'] = apos_galpy['apocenter.check']
#
# Apocenter distances
data_dict['apocenter.dist.sim'] = apos['apocenter.dist']
data_dict['apocenter.dist.galpy'] = apos_galpy['apocenter.dist']
#
# Apocenter velocities
data_dict['apocenter.vel.sim'] = apos['apocenter.vel']
data_dict['apocenter.vel.galpy'] = apos_galpy['apocenter.vel']
#
# Apocenter times
data_dict['apocenter.time.sim'] = apos['apocenter.time']
data_dict['apocenter.time.galpy'] = apos_galpy['apocenter.time']
data_dict['apocenter.time.lb.sim'] = apos['apocenter.time.lb']
data_dict['apocenter.time.lb.galpy'] = apos_galpy['apocenter.time.lb']
#
# Maximum distances and times
data_dict['max.dist.sim'] = apos['max.dist']
data_dict['max.dist.time.sim'] = apos['max.dist.time']
data_dict['max.dist.time.lb.sim'] = apos['max.dist.time.lb']
#
# distance, velocity, Lz vs time
data_dict['dtot.sim'] = halt_dists
data_dict['vtot.sim'] = halt_vels
#data_dict['Ltot.sim'] = angs['ang.mom.total']
data_dict['Lz.sim'] = angs['ang.mom.vector'][:,:,2]
###### NEED TANGENTIAL VELOCITIES
data_dict['time.sim'] = snaps['time']
#
data_dict['dtot.galpy'] = galpy_orbits.r(ts)
data_dict['vtot.galpy'] = galpy_vels
data_dict['Lz.galpy'] = galpy_orbits.Lz(ts)
data_dict['time.galpy'] = ts

ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/data_'+sim_data.galaxy, dict_or_array_to_write=data_dict, verbose=True)
