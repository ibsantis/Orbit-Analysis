#!/usr/bin/python3

"""
    ===============================
    = Enclosed mass and Potential =
    ===============================

    Calculating what the enclosed mass and potentials are at various
    distances from the host galaxy at z = 0.

    This is to check if they are consistent with one another, or if we need
    to remove the h^2 correction that Andrew added based on my previous tests.
"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import time
import sys
print('Read in the tools')

### Set path and initial parameters
loc = 'peloton'
host = 'Thelma'
sim_data = orbit_io.OrbitRead(gal1=host, location=loc)
print('Set paths')

# Read in snapshot dictionary and the halo tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
print('Read in halo tree and set up subhalo indices')

# Define the function which calculates the potential for all snapshots
# Also calculates other stuff like: U(R200m), KE_c, U(100kpc,z)
if sim_data.num_gal == 1:
    #
    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    part = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600, properties=['mass', 'position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    #
    # Set up the distance array of interest and empty arrays to save to
    ds = np.array([100, 110, 200, 210, 300, 310, 500, 510])
    data_dict = dict()
    data_dict['mass.enclosed'] = np.zeros(len(ds))
    data_dict['potential'] = np.zeros(len(ds))
    #
    for i in range(0, len(ds)):
        # First calculate the enclosed mass of all particles within the distances
        star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [0, ds[i]])
        gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [0, ds[i]])
        dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [0, ds[i]])
        data_dict['mass.enclosed'][i] = np.sum(part['star']['mass'][star_inds]) + np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
        #
        # Select particles within ds[i] +/- 5 kpc and calculate the mean potential
        dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [ds[i]-5, ds[i]+5])
        data_dict['potential'][i] = np.mean(part['dark']['potential'][dark_inds])
        #
        print('Done with {0} kpc calculations'.format(ds[i]))
    #
    # Save the data to a file
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/all_snapshots/'+sim_data.galaxy+'_host_potential_check', dict_or_array_to_write=data_dict, verbose=True)
    print('All done!')

if sim_data.num_gal == 2:
    #
    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    part = gizmo.io.Read.read_snapshots(['star', 'gas', 'dark'], 'index', 600, properties=['mass', 'position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    #
    # Set up the distance array of interest and empty arrays to save to
    ds = np.array([100, 110, 200, 210, 300, 310, 500, 510])
    #
    data_dict_1 = dict()
    data_dict_1['mass.enclosed'] = np.zeros(len(ds))
    data_dict_1['potential'] = np.zeros(len(ds))
    #
    data_dict_2 = dict()
    data_dict_2['mass.enclosed'] = np.zeros(len(ds))
    data_dict_2['potential'] = np.zeros(len(ds))
    #
    for i in range(0, len(ds)):
        # First calculate the enclosed mass of all particles within the distances
        star_inds_1 = ut.array.get_indices(part['star'].prop('host.distance.total'), [0, ds[i]])
        gas_inds_1 = ut.array.get_indices(part['gas'].prop('host.distance.total'), [0, ds[i]])
        dark_inds_1 = ut.array.get_indices(part['dark'].prop('host.distance.total'), [0, ds[i]])
        data_dict_1['mass.enclosed'][i] = np.sum(part['star']['mass'][star_inds_1]) + np.sum(part['gas']['mass'][gas_inds_1]) + np.sum(part['dark']['mass'][dark_inds_1])
        #
        # Select particles within ds[i] +/- 5 kpc and calculate the mean potential
        dark_inds_1 = ut.array.get_indices(part['dark'].prop('host.distance.total'), [ds[i]-5, ds[i]+5])
        data_dict_1['potential'][i] = np.mean(part['dark']['potential'][dark_inds_1])
        #
        # First calculate the enclosed mass of all particles within the distances
        star_inds_2 = ut.array.get_indices(part['star'].prop('host2.distance.total'), [0, ds[i]])
        gas_inds_2 = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [0, ds[i]])
        dark_inds_2 = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [0, ds[i]])
        data_dict_2['mass.enclosed'][i] = np.sum(part['star']['mass'][star_inds_2]) + np.sum(part['gas']['mass'][gas_inds_2]) + np.sum(part['dark']['mass'][dark_inds_2])
        #
        # Select particles within ds[i] +/- 5 kpc and calculate the mean potential
        dark_inds_2 = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [ds[i]-5, ds[i]+5])
        data_dict_2['potential'][i] = np.mean(part['dark']['potential'][dark_inds_2])
        #
        print('Done with {0} kpc calculations'.format(ds[i]))
    #
    # Save the data to a file
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/all_snapshots/'+sim_data.gal_1+'_host_potential_check', dict_or_array_to_write=data_dict_1, verbose=True)
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/all_snapshots/'+sim_data.gal_2+'_host_potential_check', dict_or_array_to_write=data_dict_2, verbose=True)
    print('All done!')
