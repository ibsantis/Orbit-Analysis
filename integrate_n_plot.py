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

if sim_data.num_gal == 1:

    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1)
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1)
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
    halt_vels = orbits.halo_velocities(halt)
    halt_rad_vels = orbits.halo_velocities(halt, vel_type='rad')
    halt_tan_vels = orbits.halo_velocities(halt, vel_type='tan')
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])] # Want to divide the other distances by this distance
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
    ts = np.flip(snaps['time'] - snaps['time'][-1])*u.Gyr
    galpy_orbits.integrate(ts, potential_two_power, method='odeint')
    print('Done integrating in potential model')

    # Check to see if any of them are close to a pole
    poles = orbit_gal.galpy_pole_check(galpy_orbits, ts)
    print(poles)
    print(np.sum(poles))

    #galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vR(ts), galpy_orbits.R(ts)*galpy_orbits.vphi(ts))
    galpy_vels = np.sqrt(galpy_orbits.vx(ts)**2+galpy_orbits.vy(ts)**2+galpy_orbits.vz(ts)**2)
    galpy_angs = np.linalg.norm(galpy_orbits.L(ts), axis=2)


    for i in range(0, orbits.shape[0]):
        if (infall_info['check'][i]) & (peris['pericenter.check'][i]):
            # Integrate the subhalo orbit in each potential
            d_model = galpy_orbits[i]._parse_plot_quantity(quant='r')
            #v_model = galpy_orbits[i]._parse_plot_quantity(quant='vR')
            #Lz_model = galpy_orbits[i]._parse_plot_quantity(quant='Lz')
            v_model = galpy_vels[i]
            #v_model = galpy_orbits.vT(ts)[i]
            L_model = galpy_angs[i]
            #
            # Set up the distances and times to plot
            d_mask = (halt_dists[i] >= 0)
            d_data = halt_dists[i][d_mask]
            lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
            times = lookback_time[:len(d_data)]
            #v_data = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[i][orbits.sub_inds[i]>=0])[:,0][:len(times)]
            v_data = halt_vels[i][:len(times)]
            #v_data = halt_tan_vels[i][:len(times)]
            #Lz_data = angs['ang.mom.vector'][i][:,2][:len(times)]
            L_data = angs['ang.mom.total'][i][:len(times)]
            #
            # Set up the figure
            plt.rcParams["font.family"] = "serif"
            plt.figure(figsize=(10, 12))
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312, sharex=ax1)
            ax3 = plt.subplot(313, sharex=ax2)
            #plt.figure(figsize=(10, 8))
            #ax1 = plt.subplot(211)
            #ax2 = plt.subplot(312, sharex=ax1)
            #ax3 = plt.subplot(212, sharex=ax1)
            #
            # Plot the distances
            ax1.plot(times, d_data, 'k', label='Simulation')
            ax1.plot(-1*ts, d_model, label='Model', alpha=0.5)
            ax1.plot(times, host_radii[:len(times)], 'k', alpha=0.3)
            ax1.set_xlim(times[-1], times[0])
            #
            # Check to see if there were infall, pericenter, or apocenter events
            infall = infall_info['check'][i]
            peri = peris['pericenter.check'][i]
            #
            # If there were, plot when they occurred
            if infall:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax1.axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            #
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax1.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
            # Set the labels and save the figure
            ax1.set_ylim(top=np.nanmax(d_data)+100)
            ax1.label_outer()
            ax1.set_ylabel('Host Distance [kpc]', fontsize=20)
            ax1.legend(prop={'size': 16})
            #
            # Plot the velocity data
            ax2.plot(times, v_data, 'k')
            ax2.plot(-1*ts, v_model, alpha=0.5)
            ax2.set_xlim(times[-1], times[0])
            ax2.label_outer()
            if infall == True:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax2.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax2.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')

            ax2.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=20)
            #
            # Plot the velocity data
            ax3.plot(times, L_data/1000, 'k')
            ax3.plot(-1*ts, L_model/1000, alpha=0.5)
            ax3.set_xlim(times[-1], times[0])
            ax3.set_ylabel('$\\ell$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
            if infall == True:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax3.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
            #
            ax3.set_xlabel('lookback time [Gyr]', fontsize=32)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(orbits.home_dir+'/orbit_data/plots/subhalo_integration/'+sim_data.galaxy+'/'+sim_data.galaxy+'_sub_'+str(i+1)+'.pdf')
            plt.close()

if sim_data.num_gal == 2:
    #
    ### GALAXY 1
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1)
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1)
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt, host=1) # set host=1 for the first host, host=2 for the other
    halt_vels = orbits.halo_velocities(halt, host=1)
    halt_rad_vels = orbits.halo_velocities(halt, vel_type='rad', host=1)
    halt_tan_vels = orbits.halo_velocities(halt, vel_type='tan', host=1)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])] # Want to divide the other distances by this distance
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
    apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
    angs = orbits.angular_momentum(tree=halt, host=1)
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt)

    # Read in the fitting parameters
    fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)

    # Import the potentials and combine them for our model
    from galpy.potential import DoubleExponentialDiskPotential # For disks
    from galpy.potential import TwoPowerSphericalPotential # For DM halos
    #
    disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.gal_1]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.gal_1]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_1]*u.kpc)
    disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.gal_1]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.gal_1]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_1]*u.kpc)
    halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.gal_1]*u.solMass, a=fitting_data['a_halo'][sim_data.gal_1]*u.kpc, alpha=fitting_data['alpha'][sim_data.gal_1], beta=fitting_data['beta'][sim_data.gal_1])
    potential_two_power = disk_inner+disk_outer+halo_2p

    # Integrate all of the orbits in both potentials
    ts = np.flip(snaps['time'] - snaps['time'][-1])*u.Gyr
    galpy_orbits.integrate(ts, potential_two_power, method='odeint')
    print('Done integrating in potential model')

    # Check to see if any of them are close to a pole
    poles = orbit_gal.galpy_pole_check(galpy_orbits, ts)
    print(poles)
    print(np.sum(poles))

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vR(ts), galpy_orbits.vT(ts))
    galpy_angs = np.linalg.norm(galpy_orbits.L(ts), axis=2)


    for i in range(1, orbits.shape[0]):
        if (infall_info['check'][i]) & (peris['pericenter.check'][i]):
            # Integrate the subhalo orbit in each potential
            d_model = galpy_orbits[i]._parse_plot_quantity(quant='r')
            #v_model = galpy_vels[i]
            v_model = galpy_orbits.vT(ts)[i]
            L_model = galpy_angs[i]
            #v_model = galpy_orbits[i]._parse_plot_quantity(quant='vR')
            #Lz_model = galpy_orbits[i]._parse_plot_quantity(quant='Lz')
            #
            # Set up the distances and times to plot
            d_mask = (halt_dists[i] >= 0)
            d_data = halt_dists[i][d_mask]
            lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
            times = lookback_time[:len(d_data)]
            #v_data = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[i][orbits.sub_inds[i]>=0])[:,0][:len(times)]
            #v_data = halt_vels[i][:len(times)]
            v_data = halt_tan_vels[i][:len(times)]
            #Lz_data = angs['ang.mom.vector'][i][:,2][:len(times)]
            L_data = angs['ang.mom.total'][i][:len(times)]
            #
            # Set up the figure
            plt.rcParams["font.family"] = "serif"
            plt.figure(figsize=(10, 12))
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312, sharex=ax1)
            ax3 = plt.subplot(313, sharex=ax2)
            #plt.figure(figsize=(10, 8))
            #ax1 = plt.subplot(211)
            #ax2 = plt.subplot(312, sharex=ax1)
            #ax3 = plt.subplot(212, sharex=ax1)
            #
            # Plot the distances
            ax1.plot(times, d_data, 'k', label='Simulation')
            ax1.plot(-1*ts, d_model, label='Model', alpha=0.5)
            ax1.plot(times, host_radii[:len(times)], 'k', alpha=0.3)
            ax1.set_xlim(times[-1], times[0])
            #
            # Check to see if there were infall, pericenter, or apocenter events
            infall = infall_info['check'][i]
            peri = peris['pericenter.check'][i]
            #
            # If there were, plot when they occurred
            if infall:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax1.axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            #
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax1.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
            # Set the labels and save the figure
            ax1.set_ylim(top=np.nanmax(d_data)+100)
            ax1.label_outer()
            ax1.set_ylabel('Host Distance [kpc]', fontsize=20)
            ax1.legend(prop={'size': 16})
            #
            # Plot the velocity data
            ax2.plot(times, v_data, 'k')
            ax2.plot(-1*ts, v_model, alpha=0.5)
            ax2.set_xlim(times[-1], times[0])
            ax2.label_outer()
            if infall == True:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax2.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax2.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')

            ax2.set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=20)
            #
            # Plot the velocity data
            ax3.plot(times, L_data/1000, 'k')
            ax3.plot(-1*ts, L_model/1000, alpha=0.5)
            ax3.set_xlim(times[-1], times[0])
            ax3.set_ylabel('$\\ell$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
            if infall == True:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax3.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
            #
            ax3.set_xlabel('lookback time [Gyr]', fontsize=32)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(orbits.home_dir+'/orbit_data/plots/subhalo_integration/'+sim_data.gal_1+'/'+sim_data.gal_1+'_sub_'+str(i+1)+'.pdf')
            plt.close()

    ### GALAXY 2
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=2)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location='peloton', host=2)
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.gal_1, location='peloton', host=2)
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt, host=2) # set host=1 for the first host, host=2 for the other
    halt_vels = orbits.halo_velocities(halt, host=2)
    halt_rad_vels = orbits.halo_velocities(halt, vel_type='rad', host=2)
    halt_tan_vels = orbits.halo_velocities(halt, vel_type='tan', host=2)
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host2.index'][0])] # Want to divide the other distances by this distance
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
    apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
    angs = orbits.angular_momentum(tree=halt, host=2)
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt, host=2)

    # Read in the fitting parameters
    fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)

    # Import the potentials and combine them for our model
    from galpy.potential import DoubleExponentialDiskPotential # For disks
    from galpy.potential import TwoPowerSphericalPotential # For DM halos
    #
    disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.gal_2]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.gal_2]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_2]*u.kpc)
    disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.gal_2]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.gal_2]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_2]*u.kpc)
    halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.gal_2]*u.solMass, a=fitting_data['a_halo'][sim_data.gal_2]*u.kpc, alpha=fitting_data['alpha'][sim_data.gal_2], beta=fitting_data['beta'][sim_data.gal_2])
    potential_two_power = disk_inner+disk_outer+halo_2p

    # Integrate all of the orbits in both potentials
    ts = np.flip(snaps['time'] - snaps['time'][-1])*u.Gyr
    galpy_orbits.integrate(ts, potential_two_power, method='odeint')
    print('Done integrating in potential model')

    # Check to see if any of them are close to a pole
    poles = orbit_gal.galpy_pole_check(galpy_orbits, ts)
    print(poles)
    print(np.sum(poles))

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vR(ts), galpy_orbits.vT(ts))
    galpy_angs = np.linalg.norm(galpy_orbits.L(ts), axis=2)

    for i in range(1, orbits.shape[0]):
        if (infall_info['check'][i]) & (peris['pericenter.check'][i]):
            # Integrate the subhalo orbit in each potential
            d_model = galpy_orbits[i]._parse_plot_quantity(quant='r')
            #v_model = galpy_vels[i]
            v_model = galpy_orbits.vT(ts)[i]
            L_model = galpy_angs[i]
            #v_model = galpy_orbits[i]._parse_plot_quantity(quant='vR')
            #Lz_model = galpy_orbits[i]._parse_plot_quantity(quant='Lz')
            #
            # Set up the distances and times to plot
            d_mask = (halt_dists[i] >= 0)
            d_data = halt_dists[i][d_mask]
            lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
            times = lookback_time[:len(d_data)]
            #v_data = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[i][orbits.sub_inds[i]>=0])[:,0][:len(times)]
            #v_data = halt_vels[i][:len(times)]
            v_data = halt_tan_vels[i][:len(times)]
            #Lz_data = angs['ang.mom.vector'][i][:,2][:len(times)]
            L_data = angs['ang.mom.total'][i][:len(times)]
            #
            # Set up the figure
            plt.rcParams["font.family"] = "serif"
            plt.figure(figsize=(10, 12))
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312, sharex=ax1)
            ax3 = plt.subplot(313, sharex=ax2)
            #plt.figure(figsize=(10, 8))
            #ax1 = plt.subplot(211)
            #ax2 = plt.subplot(312, sharex=ax1)
            #ax3 = plt.subplot(212, sharex=ax1)
            #
            # Plot the distances
            ax1.plot(times, d_data, 'k', label='Simulation')
            ax1.plot(-1*ts, d_model, label='Model', alpha=0.5)
            ax1.plot(times, host_radii[:len(times)], 'k', alpha=0.3)
            ax1.set_xlim(times[-1], times[0])
            #
            # Check to see if there were infall, pericenter, or apocenter events
            infall = infall_info['check'][i]
            peri = peris['pericenter.check'][i]
            #
            # If there were, plot when they occurred
            if infall:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax1.axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            #
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax1.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
            # Set the labels and save the figure
            ax1.set_ylim(top=np.nanmax(d_data)+100)
            ax1.label_outer()
            ax1.set_ylabel('Host Distance [kpc]', fontsize=20)
            ax1.legend(prop={'size': 16})
            #
            # Plot the velocity data
            ax2.plot(times, v_data, 'k')
            ax2.plot(-1*ts, v_model, alpha=0.5)
            ax2.set_xlim(times[-1], times[0])
            ax2.label_outer()
            if infall == True:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax2.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax2.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')

            ax2.set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=20)
            #
            # Plot the velocity data
            ax3.plot(times, L_data/1000, 'k')
            ax3.plot(-1*ts, L_model/1000, alpha=0.5)
            ax3.set_xlim(times[-1], times[0])
            ax3.set_ylabel('$\\ell$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
            if infall == True:
                infall_time = infall_info['first.infall.time.lb'][i]
                ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
            if peri:
                for j in peris['pericenter.time.lb'][i][peris['pericenter.time.lb'][i] != -1]:
                    ax3.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
            #
            ax3.set_xlabel('lookback time [Gyr]', fontsize=32)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(orbits.home_dir+'/orbit_data/plots/subhalo_integration/'+sim_data.gal_2+'/'+sim_data.gal_2+'_sub_'+str(i+1)+'.pdf')
            plt.close()
