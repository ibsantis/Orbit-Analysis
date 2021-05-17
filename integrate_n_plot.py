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
halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
halt_vels = orbits.halo_velocities(halt)
host_radii = halt['radius'][orbits.sub_inds[0][orbits.sub_inds[0] >= 0]] # Want to divide the other distances by this distance
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
infall_info = orbits.first_infall_times(halt_dists_norm, snaps)
peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
angs = orbits.angular_momentum(tree=halt)
#
galpy_orbits_2p_all = orbit_gal.galpy_orbit_init(tree=halt)
galpy_orbits_2p_gasdm = orbit_gal.galpy_orbit_init(tree=halt)
galpy_orbits_nfw_all = orbit_gal.galpy_orbit_init(tree=halt)
galpy_orbits_nfw_gasdm = orbit_gal.galpy_orbit_init(tree=halt)

# Read in the fitting parameters
fitting_data_2p_all = pd.read_csv(sim_data.home_dir+'/orbit_data/param_2p_all.csv', index_col=0)
fitting_data_2p_gasdm = pd.read_csv(sim_data.home_dir+'/orbit_data/param_2p_gasdm.csv', index_col=0)
fitting_data_nfw_all = pd.read_csv(sim_data.home_dir+'/orbit_data/param_nfw_all.csv', index_col=0)
fitting_data_nfw_gasdm = pd.read_csv(sim_data.home_dir+'/orbit_data/param_nfw_gasdm.csv', index_col=0)

# Import the potentials and create custom ones
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data_2p_all['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p_all['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p_all['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data_2p_all['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p_all['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p_all['h_z'][sim_data.galaxy]*u.kpc)
#
halo_2p_all = TwoPowerSphericalPotential(amp=fitting_data_2p_all['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p_all['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p_all['alpha'][sim_data.galaxy], beta=fitting_data_2p_all['beta'][sim_data.galaxy])
potential_two_power_all = disk_inner+disk_outer+halo_2p_all
#
halo_2p_gasdm = TwoPowerSphericalPotential(amp=fitting_data_2p_gasdm['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p_gasdm['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p_gasdm['alpha'][sim_data.galaxy], beta=fitting_data_2p_gasdm['beta'][sim_data.galaxy])
potential_two_power_gasdm = disk_inner+disk_outer+halo_2p_gasdm
#
nfw_all = NFWPotential(amp=fitting_data_nfw_all['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw_all['a_halo'][sim_data.galaxy]*u.kpc)
potential_nfw_all = disk_inner+disk_outer+nfw_all
#
nfw_gasdm = NFWPotential(amp=fitting_data_nfw_gasdm['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw_gasdm['a_halo'][sim_data.galaxy]*u.kpc)
potential_nfw_gasdm = disk_inner+disk_outer+nfw_gasdm


# Integrate all of the orbits in both potentials
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
galpy_orbits_2p_all.integrate(ts, potential_two_power_all, method='odeint')
print('Done integrating 2P model')
galpy_orbits_2p_gasdm.integrate(ts, potential_two_power_gasdm, method='odeint')
print('Done integrating 2P model (gas+dm only)')
galpy_orbits_nfw_all.integrate(ts, potential_nfw_all, method='odeint')
print('Done integrating NFW model')
galpy_orbits_nfw_gasdm.integrate(ts, potential_nfw_gasdm, method='odeint')
print('Done integrating NFW (gas+dm only)')

for i in range(1, orbits.shape[0]):
    # Integrate the subhalo orbit in each potential
    d_model_2p_all = galpy_orbits_2p_all[i]._parse_plot_quantity(quant='r')
    v_model_2p_all = galpy_orbits_2p_all[i]._parse_plot_quantity(quant='vR')
    Lz_model_2p_all = galpy_orbits_2p_all[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_2p_gasdm = galpy_orbits_2p_gasdm[i]._parse_plot_quantity(quant='r')
    v_model_2p_gasdm = galpy_orbits_2p_gasdm[i]._parse_plot_quantity(quant='vR')
    Lz_model_2p_gasdm = galpy_orbits_2p_gasdm[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_nfw_all = galpy_orbits_nfw_all[i]._parse_plot_quantity(quant='r')
    v_model_nfw_all = galpy_orbits_nfw_all[i]._parse_plot_quantity(quant='vR')
    Lz_model_nfw_all = galpy_orbits_nfw_all[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_nfw_gasdm = galpy_orbits_nfw_gasdm[i]._parse_plot_quantity(quant='r')
    v_model_nfw_gasdm = galpy_orbits_nfw_gasdm[i]._parse_plot_quantity(quant='vR')
    Lz_model_nfw_gasdm = galpy_orbits_nfw_gasdm[i]._parse_plot_quantity(quant='Lz')
    #
    # Set up the distances and times to plot
    d_mask = (halt_dists[i] >= 0)
    d_data = halt_dists[i][d_mask]
    lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
    times = lookback_time[:len(d_data)]
    v_data = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[i][orbits.sub_inds[i]>=0])[:,0][:len(times)]
    Lz_data = angs['ang.mom.vector'][i][:,2][:len(times)]
    #
    # Set up the figure
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(10, 12))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax2)
    #
    # Plot the distances
    ax1.plot(times, d_data, 'k', label='simulation')
    ax1.plot(-1*ts, d_model_2p_all, label='2-Power (all)', alpha=0.5)
    ax1.plot(-1*ts, d_model_2p_gasdm, label='2-Power (G+DM)', alpha=0.5)
    ax1.plot(-1*ts, d_model_nfw_all, label='NFW (all)', alpha=0.5)
    ax1.plot(-1*ts, d_model_nfw_gasdm, label='NFW (G+DM)', alpha=0.5)
    ax1.set_xlim(times[-1], times[0])
    #
    # Check to see if there were infall, pericenter, or apocenter events
    infall = infall_info['check'][i]
    #
    # If there were, plot when they occurred
    if infall == True:
        infall_time = infall_info['time.lb'][i]
        ax1.axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    # Set the labels and save the figure
    ax1.set_ylim(top=np.nanmax(d_data)+100)
    ax1.label_outer()
    ax1.set_ylabel('r [kpc]', fontsize=32)
    ax1.legend(prop={'size': 16})
    #
    # Plot the velocity data
    ax2.plot(times, v_data, 'k')
    ax2.plot(-1*ts, v_model_2p_all, alpha=0.5)
    ax2.plot(-1*ts, v_model_2p_gasdm, alpha=0.5)
    ax2.plot(-1*ts, v_model_nfw_all, alpha=0.5)
    ax2.plot(-1*ts, v_model_nfw_gasdm, alpha=0.5)
    ax2.set_xlim(times[-1], times[0])
    ax2.label_outer()
    if infall == True:
        infall_time = infall_info['time.lb'][i]
        ax2.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    ax2.set_ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
    #
    # Plot the velocity data
    ax3.plot(times, Lz_data/1000, 'k')
    ax3.plot(-1*ts, Lz_model_2p_all/1000, alpha=0.5)
    ax3.plot(-1*ts, Lz_model_2p_gasdm/1000, alpha=0.5)
    ax3.plot(-1*ts, Lz_model_nfw_all/1000, alpha=0.5)
    ax3.plot(-1*ts, Lz_model_nfw_gasdm/1000, alpha=0.5)
    ax3.set_xlim(times[-1], times[0])
    ax3.set_ylabel('$L_{\\rm z}$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
    if infall == True:
        infall_time = infall_info['time.lb'][i]
        ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    ax3.set_xlabel('lookback time [Gyr]', fontsize=32)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(orbits.home_dir+'/orbit_data/plots/subhalo_integration/'+sim_data.galaxy+'/'+sim_data.galaxy+'_sub_'+str(i)+'.pdf')
    plt.close()
