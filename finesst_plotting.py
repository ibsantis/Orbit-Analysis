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
gal1 = 'm12i'
loc = 'peloton'
#
if gal1 == 'Romeo':
    gal2 = 'Juliet'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res3500'
    num_gal = 2
elif gal1 == 'Thelma':
    gal2 = 'Louise'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res4000'
    num_gal = 2
elif gal1 == 'Romulus':
    gal2 = 'Remus'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res4000'
    num_gal = 2
else:
    galaxy = gal1
    resolution = '_res7100'
    num_gal = 1
#
if loc == 'mac':
    home_dir = '/Users/isaiahsantistevan/simulation'
elif loc == 'peloton' and num_gal == 1:
    home_dir = '/home/ibsantis/scripts'
    simulation_dir = '/home/awetzel/scratch/'+galaxy+'/'+galaxy+resolution
elif loc == 'peloton' and num_gal == 2:
    home_dir = '/home/ibsantis/scripts'
    simulation_dir = '/home/awetzel/scratch/m12_elvis/'+galaxy+resolution
else:
    home_dir = '/home1/05400/ibsantis/scripts'
    simulation_dir = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'+galaxy+resolution
print('Set paths')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)

orbits = orbit_io.OrbitAnalysis(tree=halt, gal1='m12i', location='peloton')
orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1='m12i', location='peloton')
#
halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
halt_vels = orbits.halo_velocities(halt)
host_radii = halt['radius'][orbits.sub_inds[0][orbits.sub_inds[0] >= 0]] # Want to divide the other distances by this distance
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
infall_info = orbits.first_infall_times(halt_dists_norm, snaps)
peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
angs = orbits.angular_momentum(tree=halt)

# Read in the fitting parameters
fitting_data = pd.read_csv(home_dir+'/orbit_data/fitting_params.csv', index_col=0)

# Import the potentials and create custom ones
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][gal1]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][gal1]*u.kpc, hz=fitting_data['h_z'][gal1]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][gal1]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][gal1]*u.kpc, hz=fitting_data['h_z'][gal1]*u.kpc)
halo = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][gal1]*u.solMass, a=fitting_data['a_halo'][gal1]*u.kpc, alpha=fitting_data['alpha'][gal1], beta=fitting_data['beta'][gal1])
potential_total_best = disk_inner+disk_outer+halo

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][gal1]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][gal1]*u.kpc, hz=fitting_data['h_z'][gal1]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][gal1]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][gal1]*u.kpc, hz=fitting_data['h_z'][gal1]*u.kpc)
nfw = NFWPotential(amp=5.41e11*u.solMass, a=18.73*u.kpc)
potential_total_nfw_good_disk = disk_inner+disk_outer+nfw

disk_outer = DoubleExponentialDiskPotential(amp=1.17e9*2.24*u.solMass/u.kpc**3, hr=2.76*u.kpc, hz=0.41*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=1.59e10*2.24*u.solMass/u.kpc**3, hr=0.61*u.kpc, hz=0.41*u.kpc)
nfw = NFWPotential(amp=5.41e11*u.solMass, a=18.73*u.kpc)
potential_total_nfw_old_disk = disk_inner+disk_outer+nfw

# Get halos that fell into the host; these are IDs at z = 0
halo_1 = 3257469
halo_2 = 3502033
halo_3 = 6719222

# Get the halo properties necessary for orbit initialization
print(halt.prop('host.distance.principal', halo_1))
print(halt.prop('host.distance.principal.cylidnrical', halo_1))
print(halt.prop('host.velocity.principal.cylindrical', halo_1))
print(halt.prop('host.velocity.tan', halo_1))
#
print(halt.prop('host.distance.principal', halo_2))
print(halt.prop('host.distance.principal.cylidnrical', halo_2))
print(halt.prop('host.velocity.principal.cylindrical', halo_2))
print(halt.prop('host.velocity.tan', halo_2))
#
print(halt.prop('host.distance.principal', halo_3))
print(halt.prop('host.distance.principal.cylidnrical', halo_3))
print(halt.prop('host.velocity.principal.cylindrical', halo_3))
print(halt.prop('host.velocity.tan', halo_3))

# Initialize the orbits (R, vR, vT, z, vz, phi)
# Set up time array (negative because integrating backward)
orb_3 = Orbit([231.64*u.kpc, 12.99*u.km/u.s, 86.77*u.km/u.s, 19.84*u.kpc, -39.72*u.km/u.s, 37.89*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_3.integrate(ts, potential_total_best, method='odeint')
d_model = orb_3._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[3] >= 0)
ds = halt_dists[3][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(ds))
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
ax1.legend(prop={'size': 24})
#
v_model = orb_3._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[3] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[3][orbits.sub_inds[3]>=0])[:,0]
ax2.plot(times, vs[:len(times)], label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(v_model), np.nanmax(vs))
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_3_finesst.pdf')
plt.close()

orb_3.integrate(ts, potential_total_nfw_good_disk, method='odeint')
d_model = orb_3._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[3] >= 0)
ds = halt_dists[3][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(ds))
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
ax1.legend(prop={'size': 24})
#
v_model = orb_3._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[3] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[3][orbits.sub_inds[3]>=0])[:,0]
ax2.plot(times, vs[:len(times)], label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(v_model), np.nanmax(vs))
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_3_finesst_nfw_good_disk.pdf')
plt.close()

orb_3.integrate(ts, potential_total_nfw_old_disk, method='odeint')
d_model = orb_3._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[3] >= 0)
ds = halt_dists[3][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(ds))
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
ax1.legend(prop={'size': 24})
#
v_model = orb_3._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[3] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[3][orbits.sub_inds[3]>=0])[:,0]
ax2.plot(times, vs[:len(times)], label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(v_model), np.nanmax(vs))
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_3_finesst_nfw_old_disk.pdf')
plt.close()



## HALO 9
orb_9 = Orbit([271.18*u.kpc, 36.82*u.km/u.s, 35.85*u.km/u.s, 220.78*u.kpc, 24.67*u.km/u.s, 45.41*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_9.integrate(ts, potential_total_best, method='odeint')
d_model = orb_9._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[9] >= 0)
ds = halt_dists[9][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(ds))
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
ax1.legend(prop={'size': 24})
#
v_model = orb_9._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[9] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[9][orbits.sub_inds[9]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(v_model), np.nanmax(vs))
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_9_finesst.pdf')
plt.close()

ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_9.integrate(ts, potential_total_nfw_good_disk, method='odeint')
d_model = orb_9._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[9] >= 0)
ds = halt_dists[9][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(ds))
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
ax1.legend(prop={'size': 24})
#
v_model = orb_9._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[9] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[9][orbits.sub_inds[9]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(v_model), np.nanmax(vs))
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_9_finesst_nfw_good_disk.pdf')
plt.close()

ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_9.integrate(ts, potential_total_nfw_old_disk, method='odeint')
d_model = orb_9._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[9] >= 0)
ds = halt_dists[9][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(ds))
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
ax1.legend(prop={'size': 24})
#
v_model = orb_9._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[9] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[9][orbits.sub_inds[9]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(v_model), np.nanmax(vs))
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_9_finesst_nfw_old_disk.pdf')
plt.close()



orb_22 = Orbit([92.24*u.kpc, 62.08*u.km/u.s, 127.85*u.km/u.s, 106.65*u.kpc, -105.09*u.km/u.s, -82.18*u.deg])
#
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_22.integrate(ts, potential_total_best, method='odeint')
d_model = orb_22._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[22] >= 0)
ds = halt_dists[22][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(d_model)+5)
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
#ax1.legend(prop={'size': 24})
#
v_model = orb_22._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[22] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[22][orbits.sub_inds[22]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(vs)-5, np.nanmax(vs)+5)
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_22_finesst.pdf')
plt.close()

#
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_22.integrate(ts, potential_total_nfw_good_disk, method='odeint')
d_model = orb_22._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[22] >= 0)
ds = halt_dists[22][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(d_model)+5)
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
#ax1.legend(prop={'size': 24})
#
v_model = orb_22._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[22] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[22][orbits.sub_inds[22]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(vs)-5, np.nanmax(vs)+5)
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_22_finesst_nfw_good_disk.pdf')
plt.close()

#
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_22.integrate(ts, potential_total_nfw_old_disk, method='odeint')
d_model = orb_22._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[22] >= 0)
ds = halt_dists[22][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(d_model)+5)
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
#ax1.legend(prop={'size': 24})
#
v_model = orb_22._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[22] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[22][orbits.sub_inds[22]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(vs)-5, np.nanmax(vs)+5)
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_22_finesst_nfw_old_disk.pdf')
plt.close()




orb_32 = Orbit([80.01*u.kpc, -4.90*u.km/u.s, 102.53*u.km/u.s, -60.53*u.kpc, -61.92*u.km/u.s, 44.53*u.deg])
#
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_32.integrate(ts, potential_total_best, method='odeint')
d_model = orb_32._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[32] >= 0)
ds = halt_dists[32][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(d_model)+5)
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
#ax1.legend(prop={'size': 24})
#
v_model = orb_32._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[32] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[32][orbits.sub_inds[32]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(vs)-5, np.nanmax(vs)+5)
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_32_finesst.pdf')
plt.close()

#
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_32.integrate(ts, potential_total_nfw_good_disk, method='odeint')
d_model = orb_32._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[32] >= 0)
ds = halt_dists[32][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(d_model)+5)
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
#ax1.legend(prop={'size': 24})
#
v_model = orb_32._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[22] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[32][orbits.sub_inds[32]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(vs)-5, np.nanmax(vs)+5)
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_32_finesst_nfw_good_disk.pdf')
plt.close()

#
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_32.integrate(ts, potential_total_nfw_old_disk, method='odeint')
d_model = orb_32._parse_plot_quantity(quant='r')
#
d_mask = (halt_dists[32] >= 0)
ds = halt_dists[32][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
ax1.plot(times, ds, label='simulation')
ax1.plot(-1*ts, d_model, label='galpy')
ax1.set_xlim(times[-1], times[0])
ax1.set_ylim(0, np.nanmax(d_model)+5)
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=32)
#ax1.legend(prop={'size': 24})
#
v_model = orb_32._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[32] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[32][orbits.sub_inds[32]>=0])[:,0]
ax2.plot(times, vs, label='simulation')
ax2.plot(-1*ts, v_model, label='galpy')
ax2.set_xlim(times[-1], times[0])
ax2.set_ylim(np.nanmin(vs)-5, np.nanmax(vs)+5)
ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_32_finesst_nfw_old_disk.pdf')
plt.close()
