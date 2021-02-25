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
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)

# This initializes the classes and makes sure they inherit from the OrbitRead class
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
galpy_orbits_best = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_nfw = orbits.galpy_orbit_init(tree=halt)

# Read in the fitting parameters
fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params.csv', index_col=0)

# Import the potentials and create custom ones
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
halo = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
potential_total_best = disk_inner+disk_outer+halo

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
nfw = NFWPotential(amp=5.41e11*u.solMass, a=18.73*u.kpc)
potential_total_nfw = disk_inner+disk_outer+nfw

# Integrate all of the orbits in both potentials
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
galpy_orbits_best.integrate(ts, potential_total_best, method='odeint')
galpy_orbits_nfw.integrate(ts, potential_total_nfw, method='odeint')

for i in range(1, orbits.shape[0]):
    # Integrate the subhalo orbit in each potential
    d_model_best = galpy_orbits_best[i]._parse_plot_quantity(quant='r')
    v_model_best = galpy_orbits_best[i]._parse_plot_quantity(quant='vR')
    #
    d_model_nfw = galpy_orbits_nfw[i]._parse_plot_quantity(quant='r')
    v_model_nfw = galpy_orbits_nfw[i]._parse_plot_quantity(quant='vR')
    #
    # Set up the distances and times to plot
    d_mask = (halt_dists[i] >= 0)
    d_data = halt_dists[i][d_mask]
    lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
    times = lookback_time[:len(d_data)]
    #
    # Set up the figure
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
    #ax3 = plt.subplot(313, sharex=ax1)
    #
    # Plot the distances
    ax1.plot(times, d_data, label='simulation')
    ax1.plot(-1*ts, d_model_best, label='galpy (two-power halo)', alpha=0.8)
    ax1.plot(-1*ts, d_model_nfw, label='galpy (nfw halo)', alpha=0.8)
    ax1.set_xlim(times[-1], times[0])
    #ax1.set_ylim(0, np.nanmax(d_data))
    ax1.get_yaxis().set_label_coords(-0.08, 0.5)
    ax1.label_outer()
    ax1.set_ylabel('r [kpc]', fontsize=32)
    ax1.legend(prop={'size': 24})
    #
    # Set up velocity data
    #v_mask = (halt_vels[i] >= 0)
    v_data = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[i][orbits.sub_inds[i]>=0])[:,0][:len(times)]
    #
    # Plot the velocity data
    ax2.plot(times, v_data)
    ax2.plot(-1*ts, v_model_best)
    ax2.plot(-1*ts, v_model_nfw)
    ax2.set_xlim(times[-1], times[0])
    #ax2.set_ylim(np.nanmin(v_model_best), np.nanmax(v_data))
    #ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
    ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
    ax2.tick_params(axis='both', which='major', labelsize=26)
    #
    # ADD THE L PLOTS HERE
    #
    ax2.set_xlabel('lookback time [Gyr]', fontsize=32)
    plt.tight_layout()
    plt.savefig(orbits.home_dir+'/orbit_data/plots/galpy_plot_checks/sub_'+str(i)+'_data.pdf')
    plt.close()















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
