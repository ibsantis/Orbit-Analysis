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
loc = 'peloton'
sim_data = orbit_io.OrbitRead(gal1='Romulus', location=loc)
plotting = False
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', 600, simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)

if sim_data.num_gal == 1:
    # Find the mass ratio to multiply the host radius
    mass_ratio = ut.particle.get_halo_properties(part)['mass']/halt['mass'][halt['host.index'][0]]
    #
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, dmo=False)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, dmo=False)
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, dmo=False)
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
    sub_pos = halt.prop('host.distance', orbits.sub_inds[:,0])
    sub_vel = halt.prop('host.velocity', orbits.sub_inds[:,0])
    #
    sub_pos_rot = ut.coordinate.get_coordinates_rotated(sub_pos, part.host['rotation'][0])
    sub_vel_rot = ut.coordinate.get_coordinates_rotated(sub_vel, part.host['rotation'][0])
    #
    sub_pos_rot_cyl = ut.coordinate.get_positions_in_coordinate_system(sub_pos_rot, system_from='cartesian', system_to='cylindrical')
    sub_vel_rot_cyl = ut.coordinate.get_velocities_in_coordinate_system(sub_vel_rot, sub_pos_rot, system_from='cartesian', system_to='cylindrical')
    #
    sub_orbits = []
    for i in range(0, len(orbits.sub_inds)):
        R = sub_pos_rot_cyl[i,0]
        vR = sub_vel_rot_cyl[i,0]
        vT = sub_vel_rot_cyl[i,1]
        z = sub_pos_rot_cyl[i,2]
        vz = sub_vel_rot_cyl[i,2]
        phi = np.rad2deg(np.arctan(sub_pos_rot[i,1]/sub_pos_rot[i,0]))
        #
        sub_orbits.append(Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.deg]))
    #
    # Convert the cartesian coordinates to cylindrical
    galpy_orbits = Orbit(sub_orbits)

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

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vx(ts), galpy_orbits.vy(ts), galpy_orbits.vz(ts))
    peris_galpy = orbit_gal.galpy_pericenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, virial_radii=host_radii)
    apos_galpy = orbit_gal.galpy_apocenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, infall_array=infall_info)
    eccs_galpy_pot = galpy_orbits.e(pot=potential_two_power)
    eccs_galpy_apsis = orbits.eccentricity(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)

    galpy_orbits_norm = galpy_orbits.r(ts)[:,:len(host_radii*mass_ratio)]/host_radii*mass_ratio
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
    data_dict['time.sim'] = snaps['time']
    #
    data_dict['d.tot.model'] = galpy_orbits.r(ts)
    data_dict['d.model'] = galpy_dist_3d
    data_dict['v.tot.model'] = galpy_vels
    data_dict['L.model'] = galpy_orbits.L(ts)
    data_dict['L.z.model'] = galpy_orbits.Lz(ts)
    data_dict['time.model'] = ts
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/aligned/data_'+sim_data.galaxy+'_aligned', dict_or_array_to_write=data_dict, verbose=True)

    if plotting:
        orbit_plot.multi_plot(host_rads=host_radii*mass_ratio, infall_dict=infall_info, peri_dict=peris, time_dict=snaps, sim_dist=halt_dists, sim_vel=halt_vels, sim_ell=angs, model_orbits=galpy_orbits, model_times=ts)

if sim_data.num_gal == 2:
    #
    # Find the mass ratio to multiply the host radius
    mass_ratio = ut.particle.get_halo_properties(part)['mass']/halt['mass'][halt['host.index'][0]]
    #
    ### GALAXY 1
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, dmo=False)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, dmo=False)
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, dmo=False)
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
    sub_pos = halt.prop('host.distance', orbits.sub_inds[:,0])
    sub_vel = halt.prop('host.velocity', orbits.sub_inds[:,0])
    #
    sub_pos_rot = ut.coordinate.get_coordinates_rotated(sub_pos, part.host['rotation'][0])
    sub_vel_rot = ut.coordinate.get_coordinates_rotated(sub_vel, part.host['rotation'][0])
    #
    sub_pos_rot_cyl = ut.coordinate.get_positions_in_coordinate_system(sub_pos_rot, system_from='cartesian', system_to='cylindrical')
    sub_vel_rot_cyl = ut.coordinate.get_velocities_in_coordinate_system(sub_vel_rot, sub_pos_rot, system_from='cartesian', system_to='cylindrical')
    #
    sub_orbits = []
    for i in range(0, len(orbits.sub_inds)):
        R = sub_pos_rot_cyl[i,0]
        vR = sub_vel_rot_cyl[i,0]
        vT = sub_vel_rot_cyl[i,1]
        z = sub_pos_rot_cyl[i,2]
        vz = sub_vel_rot_cyl[i,2]
        phi = np.rad2deg(np.arctan(sub_pos_rot[i,1]/sub_pos_rot[i,0]))
        #
        sub_orbits.append(Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.deg]))
    #
    # Convert the cartesian coordinates to cylindrical
    galpy_orbits = Orbit(sub_orbits)

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

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vx(ts), galpy_orbits.vy(ts), galpy_orbits.vz(ts))
    peris_galpy = orbit_gal.galpy_pericenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, virial_radii=host_radii)
    apos_galpy = orbit_gal.galpy_apocenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, infall_array=infall_info)
    eccs_galpy_pot = galpy_orbits.e(pot=potential_two_power)
    eccs_galpy_apsis = orbits.eccentricity(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)

    galpy_orbits_norm = galpy_orbits.r(ts)[:,:len(host_radii*mass_ratio)]/host_radii*mass_ratio
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
    data_dict['time.sim'] = snaps['time']
    #
    data_dict['d.tot.model'] = galpy_orbits.r(ts)
    data_dict['d.model'] = galpy_dist_3d
    data_dict['v.tot.model'] = galpy_vels
    data_dict['L.model'] = galpy_orbits.L(ts)
    data_dict['L.z.model'] = galpy_orbits.Lz(ts)
    data_dict['time.model'] = ts
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/aligned/data_'+sim_data.gal_1+'_aligned', dict_or_array_to_write=data_dict, verbose=True)
    #

    #
    ### GALAXY 2
    # Find the mass ratio to multiply the host radius
    mass_ratio = ut.particle.get_halo_properties(part, host_index=1)['mass']/halt['mass'][halt['host2.index'][0]]
    #
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, dmo=False)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, dmo=False)
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, dmo=False)
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
    sub_pos = halt.prop('host2.distance', orbits.sub_inds[:,0])
    sub_vel = halt.prop('host2.velocity', orbits.sub_inds[:,0])
    #
    sub_pos_rot = ut.coordinate.get_coordinates_rotated(sub_pos, part.host['rotation'][1])
    sub_vel_rot = ut.coordinate.get_coordinates_rotated(sub_vel, part.host['rotation'][1])
    #
    sub_pos_rot_cyl = ut.coordinate.get_positions_in_coordinate_system(sub_pos_rot, system_from='cartesian', system_to='cylindrical')
    sub_vel_rot_cyl = ut.coordinate.get_velocities_in_coordinate_system(sub_vel_rot, sub_pos_rot, system_from='cartesian', system_to='cylindrical')
    #
    sub_orbits = []
    for i in range(0, len(orbits.sub_inds)):
        R = sub_pos_rot_cyl[i,0]
        vR = sub_vel_rot_cyl[i,0]
        vT = sub_vel_rot_cyl[i,1]
        z = sub_pos_rot_cyl[i,2]
        vz = sub_vel_rot_cyl[i,2]
        phi = np.rad2deg(np.arctan(sub_pos_rot[i,1]/sub_pos_rot[i,0]))
        #
        sub_orbits.append(Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.deg]))
    #
    # Convert the cartesian coordinates to cylindrical
    galpy_orbits = Orbit(sub_orbits)

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

    galpy_vels = orbit_gal.galpy_velocities(galpy_orbits.vx(ts), galpy_orbits.vy(ts), galpy_orbits.vz(ts))
    peris_galpy = orbit_gal.galpy_pericenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, virial_radii=host_radii)
    apos_galpy = orbit_gal.galpy_apocenter_interp(distances=galpy_orbits.r(ts), velocities=galpy_vels, time_array=snaps, infall_array=infall_info)
    eccs_galpy_pot = galpy_orbits.e(pot=potential_two_power)
    eccs_galpy_apsis = orbits.eccentricity(distances=galpy_orbits.r(ts), velocities=galpy_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)

    galpy_orbits_norm = galpy_orbits.r(ts)[:,:len(host_radii*mass_ratio)]/host_radii*mass_ratio
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
    data_dict['time.sim'] = snaps['time']
    #
    data_dict['d.tot.model'] = galpy_orbits.r(ts)
    data_dict['d.model'] = galpy_dist_3d
    data_dict['v.tot.model'] = galpy_vels
    data_dict['L.model'] = galpy_orbits.L(ts)
    data_dict['L.z.model'] = galpy_orbits.Lz(ts)
    data_dict['time.model'] = ts
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/aligned/data_'+sim_data.gal_2+'_aligned', dict_or_array_to_write=data_dict, verbose=True)
