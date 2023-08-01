#!/usr/bin/env python3
#SBATCH --job-name=summary_data_all_halos
##SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --partition=high2    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=200G
##SBATCH --mem=480G
#SBATCH --nodes=1
#SBATCH --ntasks=1    # processes total
#SBATCH --time=04:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/summary/summary_data_all_halos_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

"""

    ========================
    = Integrating subhalos =
    ========================

    Integrate subhalos in custom potential
        - Disk (radial and vertical) model
        - DM halo model

    11/07/22:
        Had a seperate script that calculated orbits that were aligned with
        the disk, but now that there's a new parameter in halo_analysis that
        does this already, I altered this script to account for that. So, I
        deleted summary_data_aligned.py 

"""

# Import packages
from galpy.orbit import Orbit
import orbit_io
import model_io
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
import sys
print('Read in the tools')

### Set path and initial parameters
loc = 'peloton'
sim_data = orbit_io.OrbitRead(gal1=str(sys.argv[1]), location=loc)
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
part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', 600, simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=aligned)

#sim_data.galaxy = 'm12i_lr'

if sim_data.num_gal == 1:
    # Find the mass ratio to multiply the host radius
    mass_ratio = ut.particle.get_halo_properties(part)['mass']/halt['mass'][halt['host.index'][0]]
    #
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, selection='halo')
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, selection='halo')
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, selection='halo')
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
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt, rotate=rotate, rotation=(rot_axis, angle))

    # Read in the fitting parameters
    fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)

    if point_mass:
        # Import the potentials and combine them for our model
        from galpy.potential import KeplerPotential # For disks
        from galpy.potential import TwoPowerSphericalPotential # For DM halos
        #
        mods = model_io.Profiles('/home/ibsantis/scripts')
        disk_mass = mods.disk_radial_mass(30, sim_data.galaxy)*u.solMass # Chose 30 kpc because that's where the enclosed mass converges (for most)
        #
        # disk_mass = (fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3)*np.pi*(fitting_data['r_out'][sim_data.galaxy]*u.kpc)**2*(fitting_data['h_z'][sim_data.galaxy]*u.kpc)
        disk_point_pot = KeplerPotential(amp=disk_mass)
        halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
        #
        potential_two_power = disk_point_pot + halo_2p
    #
    else:
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

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vx(ts), galpy_orbits.vy(ts), galpy_orbits.vz(ts))
    peris_galpy = orbit_gal.galpy_pericenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, virial_radii=host_radii)
    apos_galpy = orbit_gal.galpy_apocenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, infall_array=infall_info)
    eccs_galpy_pot = galpy_orbits.e(pot=potential_two_power)
    eccs_galpy_apsis = orbits.eccentricity(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)

    galpy_orbits_norm = galpy_orbits.r(ts)[:,:len(host_radii*mass_ratio)]/(host_radii*mass_ratio)
    infall_info_galpy = orbits.infall_times(galpy_orbits_norm, snaps)
    infall_info_galpy_static_R200m = orbit_gal.galpy_infall_times(galpy_orbits.r(ts), snaps, distance_threshold=host_radii[0])
    periods_galpy = orbits.orbit_period(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info_galpy)

    galpy_dist_3d = np.ones((orbits.shape[0],len(ts),3))
    galpy_dist_3d[:,:,0] = (-1)*galpy_orbits.x(ts) # x and y have values negative from the sims for some reason...
    galpy_dist_3d[:,:,1] = (-1)*galpy_orbits.y(ts)
    galpy_dist_3d[:,:,2] = galpy_orbits.z(ts)

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
    # Check for poles
    data_dict['pole.check'] = poles
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
    data_dict['infall.check.model'] = infall_info_galpy['check']
    data_dict['first.infall.snap.model'] = infall_info_galpy['first.infall.snap']
    data_dict['first.infall.time.model'] = infall_info_galpy['first.infall.time']
    data_dict['first.infall.time.lb.model'] = infall_info_galpy['first.infall.time.lb']
    data_dict['all.infall.snap.model'] = infall_info_galpy['all.infall.snap']
    data_dict['all.infall.time.model'] = infall_info_galpy['all.infall.time']
    data_dict['all.infall.time.lb.model'] = infall_info_galpy['all.infall.time.lb']
    #
    data_dict['infall.snap.model.R200m'] = infall_info_galpy_static_R200m['infall.snap']
    data_dict['infall.time.model.R200m'] = infall_info_galpy_static_R200m['infall.time']
    data_dict['infall.time.lb.model.R200m'] = infall_info_galpy_static_R200m['infall.time.lb']
    #
    # Pericenter checks and numbers
    data_dict['pericenter.check.sim'] = peris['pericenter.check']
    data_dict['N.peri.sim'] = peris['pericenter.num']
    data_dict['pericenter.check.model'] = peris_galpy['pericenter.check']
    data_dict['N.peri.model'] = peris_galpy['pericenter.num']
    #
    # Pericenter distances
    data_dict['pericenter.dist.sim'] = peris['pericenter.dist']
    data_dict['pericenter.dist.model'] = peris_galpy['pericenter.dist']
    #
    # Pericenter velocities
    data_dict['pericenter.vel.sim'] = peris['pericenter.vel']
    data_dict['pericenter.vel.model'] = peris_galpy['pericenter.vel']
    #
    # Pericenter times
    data_dict['pericenter.time.sim'] = peris['pericenter.time']
    data_dict['pericenter.time.model'] = peris_galpy['pericenter.time']
    data_dict['pericenter.time.lb.sim'] = peris['pericenter.time.lb']
    data_dict['pericenter.time.lb.model'] = peris_galpy['pericenter.time.lb']
    #
    # Apocenter checks
    data_dict['apocenter.check.sim'] = apos['apocenter.check']
    data_dict['apocenter.check.model'] = apos_galpy['apocenter.check']
    #
    # Apocenter distances
    data_dict['apocenter.dist.sim'] = apos['apocenter.dist']
    data_dict['apocenter.dist.model'] = apos_galpy['apocenter.dist']
    #
    # Apocenter velocities
    data_dict['apocenter.vel.sim'] = apos['apocenter.vel']
    data_dict['apocenter.vel.model'] = apos_galpy['apocenter.vel']
    #
    # Apocenter times
    data_dict['apocenter.time.sim'] = apos['apocenter.time']
    data_dict['apocenter.time.model'] = apos_galpy['apocenter.time']
    data_dict['apocenter.time.lb.sim'] = apos['apocenter.time.lb']
    data_dict['apocenter.time.lb.model'] = apos_galpy['apocenter.time.lb']
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
    data_dict['orbit.period.peri.model'] = periods_galpy['pericenter.orbit.period']
    data_dict['orbit.period.apo.model'] = periods_galpy['apocenter.orbit.period']
    data_dict['eccentricity.model.pot'] = eccs_galpy_pot
    data_dict['eccentricity.model.apsis'] = eccs_galpy_apsis
    #
    data_dict['d.tot.model'] = galpy_orbits.r(ts)
    data_dict['d.model'] = galpy_dist_3d
    data_dict['v.tot.model'] = galpy_vels
    data_dict['L.model'] = galpy_orbits.L(ts)
    data_dict['L.z.model'] = galpy_orbits.Lz(ts)
    data_dict['time.model'] = ts
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

    if point_mass:
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy+'_point_mass', dict_or_array_to_write=data_dict, verbose=True)
    elif rotate:
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy+'_'+str(angle)+'deg_axis'+str(rot_axis), dict_or_array_to_write=data_dict, verbose=True)
    else:
        #ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy, dict_or_array_to_write=data_dict, verbose=True)
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy+'_all_subhalos', dict_or_array_to_write=data_dict, verbose=True)
        #ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.galaxy+'_dmo_selection', dict_or_array_to_write=data_dict, verbose=True)

    if plotting:
        orbit_plot.multi_plot(host_rads=host_radii*mass_ratio, infall_dict=infall_info, peri_dict=peris, time_dict=snaps, sim_dist=halt_dists, sim_vel=halt_vels, sim_ell=angs, model_orbits=galpy_orbits, model_times=ts)

if sim_data.num_gal == 2:
    #
    # Find the mass ratio to multiply the host radius
    mass_ratio = ut.particle.get_halo_properties(part)['mass']/halt['mass'][halt['host.index'][0]]
    #
    ### GALAXY 1
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, selection='halo')
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, selection='halo')
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, selection='halo')
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
    halt_dists_3d = orbits.halo_distances(tree=halt, dist_type='3d')
    halt_vels = orbits.halo_velocities(halt)
    halt_rad_vels = orbits.halo_velocities(halt, vel_type='rad')
    halt_tan_vels = orbits.halo_velocities(halt, vel_type='tan')
    #
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
    host_mhalo = halt['mass'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
    #
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii*mass_ratio)
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_any = orbits.first_infall_any(halt, snaps, host=1)
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
    apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
    angs = orbits.angular_momentum(tree=halt)
    periods = orbits.orbit_period(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
    eccs = orbits.eccentricity(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt, host=1, rotate=rotate, rotation=(rot_axis, angle))

    # Read in the fitting parameters
    fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)

    if point_mass:
        # Import the potentials and combine them for our model
        from galpy.potential import KeplerPotential # For disks
        from galpy.potential import TwoPowerSphericalPotential # For DM halos
        #
        mods = model_io.Profiles('/home/ibsantis/scripts')
        disk_mass = mods.disk_radial_mass(30, sim_data.gal_1)*u.solMass # Chose 30 kpc because that's where the enclosed mass converges (for most)
        #
        # disk_mass = (fitting_data['A_disk_out'][sim_data.gal_1]*u.solMass/u.kpc**3)*np.pi*(fitting_data['r_out'][sim_data.gal_1]*u.kpc)**2*(fitting_data['h_z'][sim_data.gal_1]*u.kpc) # Used for the "point mass same" files
        disk_point_pot = KeplerPotential(amp=disk_mass)
        halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.gal_1]*u.solMass, a=fitting_data['a_halo'][sim_data.gal_1]*u.kpc, alpha=fitting_data['alpha'][sim_data.gal_1], beta=fitting_data['beta'][sim_data.gal_1])
        #
        potential_two_power = disk_point_pot + halo_2p
    #
    else:
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

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vx(ts), galpy_orbits.vy(ts), galpy_orbits.vz(ts))
    peris_galpy = orbit_gal.galpy_pericenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, virial_radii=host_radii)
    apos_galpy = orbit_gal.galpy_apocenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, infall_array=infall_info)
    eccs_galpy_pot = galpy_orbits.e(pot=potential_two_power)
    eccs_galpy_apsis = orbits.eccentricity(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)

    galpy_orbits_norm = galpy_orbits.r(ts)[:,:len(host_radii*mass_ratio)]/(host_radii*mass_ratio)
    infall_info_galpy = orbits.infall_times(galpy_orbits_norm, snaps)
    infall_info_galpy_static_R200m = orbit_gal.galpy_infall_times(galpy_orbits.r(ts), snaps, distance_threshold=host_radii[0])
    periods_galpy = orbits.orbit_period(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info_galpy)

    galpy_dist_3d = np.ones((orbits.shape[0],len(ts),3))
    galpy_dist_3d[:,:,0] = (-1)*galpy_orbits.x(ts) # x and y have values negative from the sims for some reason...
    galpy_dist_3d[:,:,1] = (-1)*galpy_orbits.y(ts)
    galpy_dist_3d[:,:,2] = galpy_orbits.z(ts)

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
    # Check for poles
    data_dict['pole.check'] = poles
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
    data_dict['infall.check.model'] = infall_info_galpy['check']
    data_dict['first.infall.snap.model'] = infall_info_galpy['first.infall.snap']
    data_dict['first.infall.time.model'] = infall_info_galpy['first.infall.time']
    data_dict['first.infall.time.lb.model'] = infall_info_galpy['first.infall.time.lb']
    data_dict['all.infall.snap.model'] = infall_info_galpy['all.infall.snap']
    data_dict['all.infall.time.model'] = infall_info_galpy['all.infall.time']
    data_dict['all.infall.time.lb.model'] = infall_info_galpy['all.infall.time.lb']
    #
    data_dict['infall.snap.model.R200m'] = infall_info_galpy_static_R200m['infall.snap']
    data_dict['infall.time.model.R200m'] = infall_info_galpy_static_R200m['infall.time']
    data_dict['infall.time.lb.model.R200m'] = infall_info_galpy_static_R200m['infall.time.lb']
    #
    # Pericenter checks and numbers
    data_dict['pericenter.check.sim'] = peris['pericenter.check']
    data_dict['N.peri.sim'] = peris['pericenter.num']
    data_dict['pericenter.check.model'] = peris_galpy['pericenter.check']
    data_dict['N.peri.model'] = peris_galpy['pericenter.num']
    #
    # Pericenter distances
    data_dict['pericenter.dist.sim'] = peris['pericenter.dist']
    data_dict['pericenter.dist.model'] = peris_galpy['pericenter.dist']
    #
    # Pericenter velocities
    data_dict['pericenter.vel.sim'] = peris['pericenter.vel']
    data_dict['pericenter.vel.model'] = peris_galpy['pericenter.vel']
    #
    # Pericenter times
    data_dict['pericenter.time.sim'] = peris['pericenter.time']
    data_dict['pericenter.time.model'] = peris_galpy['pericenter.time']
    data_dict['pericenter.time.lb.sim'] = peris['pericenter.time.lb']
    data_dict['pericenter.time.lb.model'] = peris_galpy['pericenter.time.lb']
    #
    # Apocenter checks
    data_dict['apocenter.check.sim'] = apos['apocenter.check']
    data_dict['apocenter.check.model'] = apos_galpy['apocenter.check']
    #
    # Apocenter distances
    data_dict['apocenter.dist.sim'] = apos['apocenter.dist']
    data_dict['apocenter.dist.model'] = apos_galpy['apocenter.dist']
    #
    # Apocenter velocities
    data_dict['apocenter.vel.sim'] = apos['apocenter.vel']
    data_dict['apocenter.vel.model'] = apos_galpy['apocenter.vel']
    #
    # Apocenter times
    data_dict['apocenter.time.sim'] = apos['apocenter.time']
    data_dict['apocenter.time.model'] = apos_galpy['apocenter.time']
    data_dict['apocenter.time.lb.sim'] = apos['apocenter.time.lb']
    data_dict['apocenter.time.lb.model'] = apos_galpy['apocenter.time.lb']
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
    data_dict['orbit.period.peri.model'] = periods_galpy['pericenter.orbit.period']
    data_dict['orbit.period.apo.model'] = periods_galpy['apocenter.orbit.period']
    data_dict['eccentricity.model.pot'] = eccs_galpy_pot
    data_dict['eccentricity.model.apsis'] = eccs_galpy_apsis
    #
    data_dict['d.tot.model'] = galpy_orbits.r(ts)
    data_dict['d.model'] = galpy_dist_3d
    data_dict['v.tot.model'] = galpy_vels
    data_dict['L.model'] = galpy_orbits.L(ts)
    data_dict['L.z.model'] = galpy_orbits.Lz(ts)
    data_dict['time.model'] = ts
    #
    # Save the host radius and Mstar
    data_dict['host.radius'] = host_radii*mass_ratio
    data_dict['host.mass'] = host_mhalo*mass_ratio
    data_dict['host.mass.ratio'] = mass_ratio
    #
    print('The host mass at z=0 without the mass ratio is:', halt['mass'][halt['host.index'][0]])
    print('The host radius at z=0 without the mass ratio is:', halt['radius'][halt['host.index'][0]])
    print('The host mass at z=0 with the mass ratio is:', halt['mass'][halt['host.index'][0]]*mass_ratio)
    print('The host radius at z=0 with the mass ratio is:', halt['radius'][halt['host.index'][0]]*mass_ratio)
    print('The mass ratio is:', mass_ratio)

    if point_mass:
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_1+'_point_mass', dict_or_array_to_write=data_dict, verbose=True)
    elif rotate:
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_1+'_'+str(angle)+'deg_axis'+str(rot_axis), dict_or_array_to_write=data_dict, verbose=True)
    else:
        #ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_1, dict_or_array_to_write=data_dict, verbose=True)
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_1+'_all_subhalos', dict_or_array_to_write=data_dict, verbose=True)
        #ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_1+'_dmo_selection', dict_or_array_to_write=data_dict, verbose=True)
    #
    if plotting:
        orbit_plot.multi_plot(host_rads=host_radii, infall_dict=infall_info, peri_dict=peris, time_dict=snaps, sim_dist=halt_dists, sim_vel=halt_vels, sim_ell=angs, model_orbits=galpy_orbits, model_times=ts, host=1)
    #
    ### GALAXY 2
    # Find the mass ratio to multiply the host radius
    mass_ratio = ut.particle.get_halo_properties(part, host_index=1)['mass']/halt['mass'][halt['host2.index'][0]]
    #
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, selection='halo')
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, selection='halo')
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, selection='halo')
    #
    # Run the pipeline on the simulation data
    halt_dists = orbits.halo_distances(tree=halt, host=2) # set host=1 for the first host, host=2 for the other
    halt_dists_3d = orbits.halo_distances(tree=halt, host=2, dist_type='3d')
    halt_vels = orbits.halo_velocities(halt, host=2)
    halt_rad_vels = orbits.halo_velocities(halt, host=2, vel_type='rad')
    halt_tan_vels = orbits.halo_velocities(halt, host=2, vel_type='tan')
    #
    host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host2.index'][0])]
    host_mhalo = halt['mass'][halt.prop('progenitor.main.indices', halt['host2.index'][0])]
    #
    halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii*mass_ratio)
    infall_info = orbits.infall_times(halt_dists_norm, snaps)
    infall_info_any = orbits.first_infall_any(halt, snaps, host=2)
    peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
    apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
    angs = orbits.angular_momentum(tree=halt, host=2)
    periods = orbits.orbit_period(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
    eccs = orbits.eccentricity(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt, host=2, rotate=rotate, rotation=(rot_axis, angle))

    # Read in the fitting parameters
    fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)

    if point_mass:
        # Import the potentials and combine them for our model
        from galpy.potential import KeplerPotential # For disks
        from galpy.potential import TwoPowerSphericalPotential # For DM halos
        #
        mods = model_io.Profiles('/home/ibsantis/scripts')
        disk_mass = mods.disk_radial_mass(30, sim_data.gal_2)*u.solMass # Chose 30 kpc because that's where the enclosed mass converges (for most)
        #
        # disk_mass = (fitting_data['A_disk_out'][sim_data.gal_2]*u.solMass/u.kpc**3)*np.pi*(fitting_data['r_out'][sim_data.gal_2]*u.kpc)**2*(fitting_data['h_z'][sim_data.gal_2]*u.kpc)
        disk_point_pot = KeplerPotential(amp=disk_mass)
        halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.gal_2]*u.solMass, a=fitting_data['a_halo'][sim_data.gal_2]*u.kpc, alpha=fitting_data['alpha'][sim_data.gal_2], beta=fitting_data['beta'][sim_data.gal_2])
        #
        potential_two_power = disk_point_pot + halo_2p
    #
    else:
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

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vx(ts), galpy_orbits.vy(ts), galpy_orbits.vz(ts))
    peris_galpy = orbit_gal.galpy_pericenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, virial_radii=host_radii)
    apos_galpy = orbit_gal.galpy_apocenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, infall_array=infall_info)
    eccs_galpy_pot = galpy_orbits.e(pot=potential_two_power)
    eccs_galpy_apsis = orbits.eccentricity(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)

    galpy_orbits_norm = galpy_orbits.r(ts)[:,:len(host_radii*mass_ratio)]/(host_radii*mass_ratio)
    infall_info_galpy = orbits.infall_times(galpy_orbits_norm, snaps)
    infall_info_galpy_static_R200m = orbit_gal.galpy_infall_times(galpy_orbits.r(ts), snaps, distance_threshold=host_radii[0])
    periods_galpy = orbits.orbit_period(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info_galpy)

    galpy_dist_3d = np.ones((orbits.shape[0],len(ts),3))
    galpy_dist_3d[:,:,0] = (-1)*galpy_orbits.x(ts) # x and y have values negative from the sims for some reason...
    galpy_dist_3d[:,:,1] = (-1)*galpy_orbits.y(ts)
    galpy_dist_3d[:,:,2] = galpy_orbits.z(ts)

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
    # Check for poles
    data_dict['pole.check'] = poles
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
    data_dict['infall.check.model'] = infall_info_galpy['check']
    data_dict['first.infall.snap.model'] = infall_info_galpy['first.infall.snap']
    data_dict['first.infall.time.model'] = infall_info_galpy['first.infall.time']
    data_dict['first.infall.time.lb.model'] = infall_info_galpy['first.infall.time.lb']
    data_dict['all.infall.snap.model'] = infall_info_galpy['all.infall.snap']
    data_dict['all.infall.time.model'] = infall_info_galpy['all.infall.time']
    data_dict['all.infall.time.lb.model'] = infall_info_galpy['all.infall.time.lb']
    #
    data_dict['infall.snap.model.R200m'] = infall_info_galpy_static_R200m['infall.snap']
    data_dict['infall.time.model.R200m'] = infall_info_galpy_static_R200m['infall.time']
    data_dict['infall.time.lb.model.R200m'] = infall_info_galpy_static_R200m['infall.time.lb']
    #
    # Pericenter checks and numbers
    data_dict['pericenter.check.sim'] = peris['pericenter.check']
    data_dict['N.peri.sim'] = peris['pericenter.num']
    data_dict['pericenter.check.model'] = peris_galpy['pericenter.check']
    data_dict['N.peri.model'] = peris_galpy['pericenter.num']
    #
    # Pericenter distances
    data_dict['pericenter.dist.sim'] = peris['pericenter.dist']
    data_dict['pericenter.dist.model'] = peris_galpy['pericenter.dist']
    #
    # Pericenter velocities
    data_dict['pericenter.vel.sim'] = peris['pericenter.vel']
    data_dict['pericenter.vel.model'] = peris_galpy['pericenter.vel']
    #
    # Pericenter times
    data_dict['pericenter.time.sim'] = peris['pericenter.time']
    data_dict['pericenter.time.model'] = peris_galpy['pericenter.time']
    data_dict['pericenter.time.lb.sim'] = peris['pericenter.time.lb']
    data_dict['pericenter.time.lb.model'] = peris_galpy['pericenter.time.lb']
    #
    # Apocenter checks
    data_dict['apocenter.check.sim'] = apos['apocenter.check']
    data_dict['apocenter.check.model'] = apos_galpy['apocenter.check']
    #
    # Apocenter distances
    data_dict['apocenter.dist.sim'] = apos['apocenter.dist']
    data_dict['apocenter.dist.model'] = apos_galpy['apocenter.dist']
    #
    # Apocenter velocities
    data_dict['apocenter.vel.sim'] = apos['apocenter.vel']
    data_dict['apocenter.vel.model'] = apos_galpy['apocenter.vel']
    #
    # Apocenter times
    data_dict['apocenter.time.sim'] = apos['apocenter.time']
    data_dict['apocenter.time.model'] = apos_galpy['apocenter.time']
    data_dict['apocenter.time.lb.sim'] = apos['apocenter.time.lb']
    data_dict['apocenter.time.lb.model'] = apos_galpy['apocenter.time.lb']
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
    data_dict['orbit.period.peri.model'] = periods_galpy['pericenter.orbit.period']
    data_dict['orbit.period.apo.model'] = periods_galpy['apocenter.orbit.period']
    data_dict['eccentricity.model.pot'] = eccs_galpy_pot
    data_dict['eccentricity.model.apsis'] = eccs_galpy_apsis
    #
    data_dict['d.tot.model'] = galpy_orbits.r(ts)
    data_dict['d.model'] = galpy_dist_3d
    data_dict['v.tot.model'] = galpy_vels
    data_dict['L.model'] = galpy_orbits.L(ts)
    data_dict['L.z.model'] = galpy_orbits.Lz(ts)
    data_dict['time.model'] = ts
    #
    # Save the host radius
    data_dict['host.radius'] = host_radii*mass_ratio
    data_dict['host.mass'] = host_mhalo*mass_ratio
    data_dict['host.mass.ratio'] = mass_ratio
    #
    print('The host mass at z=0 without the mass ratio is:', halt['mass'][halt['host2.index'][0]])
    print('The host radius at z=0 without the mass ratio is:', halt['radius'][halt['host2.index'][0]])
    print('The host mass at z=0 with the mass ratio is:', halt['mass'][halt['host2.index'][0]]*mass_ratio)
    print('The host radius at z=0 with the mass ratio is:', halt['radius'][halt['host2.index'][0]]*mass_ratio)
    print('The mass ratio is:', mass_ratio)


    if point_mass:
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_2+'_point_mass', dict_or_array_to_write=data_dict, verbose=True)
    elif rotate:
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_2+'_'+str(angle)+'deg_axis'+str(rot_axis), dict_or_array_to_write=data_dict, verbose=True)
    else:
        #ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_2, dict_or_array_to_write=data_dict, verbose=True)
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_2+'_all_subhalos', dict_or_array_to_write=data_dict, verbose=True)
        #ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_data.gal_2+'_dmo_selection', dict_or_array_to_write=data_dict, verbose=True)

    if plotting:
        orbit_plot.multi_plot(host_rads=host_radii, infall_dict=infall_info, peri_dict=peris, time_dict=snaps, sim_dist=halt_dists, sim_vel=halt_vels, sim_ell=angs, model_orbits=galpy_orbits, model_times=ts, host=2)
