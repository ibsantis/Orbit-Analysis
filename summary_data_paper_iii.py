






# Import packages
import orbit_io
import halo_analysis as halo
import utilities as ut
print('Read in the tools')

### Set path and initial parameters
loc = 'peloton'
sim_data = orbit_io.OrbitRead(gal1='m12i', location=loc)
aligned = True
#
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal, assign_hosts_rotation=aligned, catalog_hdf5_directory='catalog_hdf5')

# Get the satellite galaxies of interest
orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, dmo=False)

data_dict = dict()
data_dict['ids'] = orbits.sub_inds
data_dict['distance'] = orbits.halo_distances(tree=halt)
data_dict['velocity.rad'] = orbits.halo_velocities(halt, vel_type='rad')
data_dict['velocity.tan'] = orbits.halo_velocities(halt, vel_type='tan')
data_dict['M.star.z0'] = halt['star.mass'][orbits.sub_inds[:,0]]
data_dict['M.halo.z0'] = halt['mass'][orbits.sub_inds[:,0]]
data_dict['M.halo.peak'] = halt.prop('mass.peak', orbits.sub_inds[:,0])
data_dict['snapshot'] = snaps['index']
data_dict['time'] = snaps['time']
data_dict['time.lb'] = snaps['time.lookback']

ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/paper_iii/data_'+sim_data.galaxy, dict_or_array_to_write=data_dict, verbose=True)

