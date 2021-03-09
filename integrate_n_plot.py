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
orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton')
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
galpy_orbits_best = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_nfw = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_best_nfw = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_nfw_2p = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_2p_nfwA_2pa = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_nfw_2pA_nfwa = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_2p_2pA_nfwa = orbits.galpy_orbit_init(tree=halt)
galpy_orbits_nfw_nfwA_2pa = orbits.galpy_orbit_init(tree=halt)

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
potential_two_power = disk_inner+disk_outer+halo

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
nfw = NFWPotential(amp=5.41e11*u.solMass, a=18.73*u.kpc)
potential_nfw = disk_inner+disk_outer+nfw

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
halo2 = TwoPowerSphericalPotential(amp=5.41e11*u.solMass, a=18.73*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
potential_two_power_nfw = disk_inner+disk_outer+halo2

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
nfw2 = NFWPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc)
potential_nfw_2p = disk_inner+disk_outer+nfw2
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
halo3 = TwoPowerSphericalPotential(amp=5.41e11*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
potential_two_power_nfwA_2pa = disk_inner+disk_outer+halo3

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
nfw3 = NFWPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=18.73*u.kpc)
potential_nfw_2pA_nfwa = disk_inner+disk_outer+nfw3

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
halo4 = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=18.73*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
potential_two_power_2pA_nfwa = disk_inner+disk_outer+halo4

disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
nfw4 = NFWPotential(amp=5.41e11*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc)
potential_nfw_nfwA_2pa = disk_inner+disk_outer+nfw4

# Integrate all of the orbits in both potentials
#ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
ts = snaps['time']*(-1)*u.Gyr
galpy_orbits_best.integrate(ts, potential_two_power, method='odeint')
galpy_orbits_nfw.integrate(ts, potential_nfw, method='odeint')
galpy_orbits_best_nfw.integrate(ts, potential_two_power_nfw, method='odeint')
galpy_orbits_nfw_2p.integrate(ts, potential_nfw_2p, method='odeint')
#
galpy_orbits_2p_2pA_nfwa.integrate(ts, potential_two_power_2pA_nfwa, method='odeint')
galpy_orbits_nfw_2pA_nfwa.integrate(ts, potential_nfw_2pA_nfwa, method='odeint')
galpy_orbits_2p_nfwA_2pa.integrate(ts, potential_two_power_nfwA_2pa, method='odeint')
galpy_orbits_nfw_nfwA_2pa.integrate(ts, potential_nfw_nfwA_2pa, method='odeint')

for i in range(1, orbits.shape[0]):
    # Integrate the subhalo orbit in each potential
    d_model_best = galpy_orbits_best[i]._parse_plot_quantity(quant='r')
    v_model_best = galpy_orbits_best[i]._parse_plot_quantity(quant='vR')
    Lz_model_best = galpy_orbits_best[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_nfw = galpy_orbits_nfw[i]._parse_plot_quantity(quant='r')
    v_model_nfw = galpy_orbits_nfw[i]._parse_plot_quantity(quant='vR')
    Lz_model_nfw = galpy_orbits_nfw[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_best_nfw = galpy_orbits_best_nfw[i]._parse_plot_quantity(quant='r')
    v_model_best_nfw = galpy_orbits_best_nfw[i]._parse_plot_quantity(quant='vR')
    Lz_model_best_nfw = galpy_orbits_best_nfw[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_nfw_2p = galpy_orbits_nfw_2p[i]._parse_plot_quantity(quant='r')
    v_model_nfw_2p = galpy_orbits_nfw_2p[i]._parse_plot_quantity(quant='vR')
    Lz_model_nfw_2p = galpy_orbits_nfw_2p[i]._parse_plot_quantity(quant='Lz')
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
    ax1.plot(-1*ts, d_model_best, label='2-Power', alpha=0.5)
    ax1.plot(-1*ts, d_model_nfw, label='NFW', alpha=0.5)
    ax1.plot(-1*ts, d_model_best_nfw, ':', label='2-Power w/NFW', alpha=0.5)
    ax1.plot(-1*ts, d_model_nfw_2p, '--', label='NFW w/2-Power', alpha=0.5)
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
    ax2.plot(-1*ts, v_model_best, alpha=0.5)
    ax2.plot(-1*ts, v_model_nfw, alpha=0.5)
    ax2.plot(-1*ts, v_model_best_nfw, ':', alpha=0.5)
    ax2.plot(-1*ts, v_model_nfw_2p, '--', alpha=0.5)
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
    ax3.plot(-1*ts, Lz_model_best/1000, alpha=0.5)
    ax3.plot(-1*ts, Lz_model_nfw/1000, alpha=0.5)
    ax3.plot(-1*ts, Lz_model_best_nfw/1000, ':', alpha=0.5)
    ax3.plot(-1*ts, Lz_model_nfw_2p/1000, '--', alpha=0.5)
    ax3.set_xlim(times[-1], times[0])
    ax3.set_ylabel('$L_{\\rm z}$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
    if infall == True:
        infall_time = infall_info['time.lb'][i]
        ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    ax3.set_xlabel('lookback time [Gyr]', fontsize=32)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(orbits.home_dir+'/orbit_data/plots/galpy_plot_checks/'+sim_data.galaxy+'/sub_'+str(i)+'_data_tsim.pdf')
    plt.close()


for i in range(1, orbits.shape[0]):
    # Integrate the subhalo orbit in each potential
    d_model_2p_2pA_nfwa = galpy_orbits_2p_2pA_nfwa[i]._parse_plot_quantity(quant='r')
    v_model_2p_2pA_nfwa = galpy_orbits_2p_2pA_nfwa[i]._parse_plot_quantity(quant='vR')
    Lz_model_2p_2pA_nfwa = galpy_orbits_2p_2pA_nfwa[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_nfw_2pA_nfwa = galpy_orbits_nfw_2pA_nfwa[i]._parse_plot_quantity(quant='r')
    v_model_nfw_2pA_nfwa = galpy_orbits_nfw_2pA_nfwa[i]._parse_plot_quantity(quant='vR')
    Lz_model_nfw_2pA_nfwa = galpy_orbits_nfw_2pA_nfwa[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_2p_nfwA_2pa = galpy_orbits_2p_nfwA_2pa[i]._parse_plot_quantity(quant='r')
    v_model_2p_nfwA_2pa = galpy_orbits_2p_nfwA_2pa[i]._parse_plot_quantity(quant='vR')
    Lz_model_2p_nfwA_2pa = galpy_orbits_2p_nfwA_2pa[i]._parse_plot_quantity(quant='Lz')
    #
    d_model_nfw_nfwA_2pa = galpy_orbits_nfw_nfwA_2pa[i]._parse_plot_quantity(quant='r')
    v_model_nfw_nfwA_2pa = galpy_orbits_nfw_nfwA_2pa[i]._parse_plot_quantity(quant='vR')
    Lz_model_nfw_nfwA_2pa = galpy_orbits_nfw_nfwA_2pa[i]._parse_plot_quantity(quant='Lz')
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
    ax1.plot(-1*ts, d_model_2p_2pA_nfwa, label='2P w/2P A, NFW a', alpha=0.5)
    ax1.plot(-1*ts, d_model_nfw_2pA_nfwa, label='NFW w/2P A, NFW a', alpha=0.5)
    ax1.plot(-1*ts, d_model_2p_nfwA_2pa, ':', label='2P w/ NFW A, 2P a', alpha=0.5)
    ax1.plot(-1*ts, d_model_nfw_nfwA_2pa, '--', label='NFW w/NFW A, 2P a', alpha=0.5)
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
    ax2.plot(-1*ts, v_model_2p_2pA_nfwa, alpha=0.5)
    ax2.plot(-1*ts, v_model_nfw_2pA_nfwa, alpha=0.5)
    ax2.plot(-1*ts, v_model_2p_nfwA_2pa, ':', alpha=0.5)
    ax2.plot(-1*ts, v_model_nfw_nfwA_2pa, '--', alpha=0.5)
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
    ax3.plot(-1*ts, Lz_model_2p_2pA_nfwa/1000, alpha=0.5)
    ax3.plot(-1*ts, Lz_model_nfw_2pA_nfwa/1000, alpha=0.5)
    ax3.plot(-1*ts, Lz_model_2p_nfwA_2pa/1000, ':', alpha=0.5)
    ax3.plot(-1*ts, Lz_model_nfw_nfwA_2pa/1000, '--', alpha=0.5)
    ax3.set_xlim(times[-1], times[0])
    ax3.set_ylabel('$L_{\\rm z}$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
    if infall == True:
        infall_time = infall_info['time.lb'][i]
        ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    ax3.set_xlabel('lookback time [Gyr]', fontsize=32)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(orbits.home_dir+'/orbit_data/plots/galpy_plot_checks/'+sim_data.galaxy+'/sub_'+str(i)+'_data_tsim_v2.pdf')
    plt.close()
