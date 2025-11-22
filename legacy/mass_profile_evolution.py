#!/usr/bin/python3

"""

  ==========================
  = Mass Profile Evolution =
  ==========================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  Calculate what the mass profile is accounting for ALL particles
  within bins of spherical r, out to 500 kpc

  Do this every 100 Myr going back to 1 Gyr (11 snapshots total)

  NOTE : This script generates data for Figure 3 in the paper 

"""
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
from orbit_analysis import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12b', location='stampede')
print('Set paths')

# Set up snapshot array to loop through
snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,543,538,534,530,525,521,517,513,509,504,484,463,443,423,404,385,365,346,327,308,289,270,250,231,211,190,169,147,124,99,72,42])
times = np.array([13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.,11.9,11.8,11.3,10.8,10.3,9.8,9.3,8.8,8.3,7.8,7.3,6.8,6.3,5.8,5.3,4.8,4.3,3.8,3.3,2.8,2.3,1.8,1.3,0.8])
rs = np.logspace(np.log10(0.1), np.log10(500), 100)

if sim_data.num_gal == 1:
    # Set up dictionary to save the data
    mass_dict = dict()
    mass_dict['snapshot'] = snaps
    mass_dict['time'] = times
    mass_dict['mass.profile'] = np.zeros((len(snaps), len(rs)))

    # Loop through the snapshots
    for i in range(0, len(snaps)):
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', snaps[i], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        print('Particles at snapshot {0} read in.'.format(snaps[i]))
        #
        # Find the enclosed mass of all particles within 0.1 < R < 500 kpc
        for j in range(0, len(rs)-1):
            star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [rs[j], rs[j+1]])
            gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[j], rs[j+1]])
            dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[j], rs[j+1]])
            mass_dict['mass.profile'][i,j] = np.sum(part['star']['mass'][star_inds]) + np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
            print('Done with step', j)
        #
    # Save this data to a file
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'_mass_profile_evolution', dict_or_array_to_write=mass_dict, verbose=True)

if sim_data.num_gal == 2:
    # Set up dictionary to save the data
    mass_dict_1 = dict()
    mass_dict_1['snapshot'] = snaps
    mass_dict_1['time'] = times
    mass_dict_1['mass.profile'] = np.zeros((len(snaps), len(rs)))
    #
    mass_dict_2 = dict()
    mass_dict_2['snapshot'] = snaps
    mass_dict_2['time'] = times
    mass_dict_2['mass.profile'] = np.zeros((len(snaps), len(rs)))
    #
    # Loop through the snapshots
    for i in range(0, len(snaps)):
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', snaps[i], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        print('Particles at snapshot {0} read in.'.format(snaps[i]))
        #
        # Find the enclosed mass of all particles within 0.1 < R < 500 kpc
        for j in range(0, len(rs)-1):
            star_inds_1 = ut.array.get_indices(part['star'].prop('host1.distance.total'), [rs[j], rs[j+1]])
            gas_inds_1 = ut.array.get_indices(part['gas'].prop('host1.distance.total'), [rs[j], rs[j+1]])
            dark_inds_1 = ut.array.get_indices(part['dark'].prop('host1.distance.total'), [rs[j], rs[j+1]])
            mass_dict_1['mass.profile'][i,j] = np.sum(part['star']['mass'][star_inds_1]) + np.sum(part['gas']['mass'][gas_inds_1]) + np.sum(part['dark']['mass'][dark_inds_1])
            #
            star_inds_2 = ut.array.get_indices(part['star'].prop('host2.distance.total'), [rs[j], rs[j+1]])
            gas_inds_2 = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [rs[j], rs[j+1]])
            dark_inds_2 = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [rs[j], rs[j+1]])
            mass_dict_2['mass.profile'][i,j] = np.sum(part['star']['mass'][star_inds_2]) + np.sum(part['gas']['mass'][gas_inds_2]) + np.sum(part['dark']['mass'][dark_inds_2])
            print('Done with step', j)
        #
    # Save these data to a file
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.gal_1+'_mass_profile_evolution', dict_or_array_to_write=mass_dict_1, verbose=True)
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.gal_2+'_mass_profile_evolution', dict_or_array_to_write=mass_dict_2, verbose=True)


print('All done!')
