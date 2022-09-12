#!/usr/bin/python3

"""
    ==============================
    = Mass profile combine files =
    ==============================

    Combine all of the files into one file for each host galaxy

"""

import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import orbit_io
import sys
print('Read in the tools')


## FOR ISOLATED HOSTS
#
### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1=str(sys.argv[1]), location='peloton')
print('Set paths')

# Set up snapshot array to loop through
#snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,544,539,534,530,525,521,517,513,509,504,484,463,443,424,404,385,365,346,327,308,289,270,250,231,211,190,169,147,124,99,72,42])
#times = np.array([13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8,11.3,10.8,10.3,9.8,9.3,8.8,8.3,7.8,7.3,6.8,6.3,5.8,5.3,4.8,4.3,3.8,3.3,2.8,2.3,1.8,1.3,0.8])
#rs = np.logspace(np.log10(0.1), np.log10(500), 100)
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir)
times = snaps['time'][2:]
snaps = snaps['index'][2:]
rs = np.logspace(np.log10(5), np.log10(500), 25)


if sim_data.num_gal == 1:
    mass_array = np.zeros((len(snaps),len(rs)-1))

    for i in range(0, len(snaps)):
        data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/all_snapshots/'+sim_data.galaxy+'/'+sim_data.galaxy+'_mass_profile_'+str(snaps[i]), verbose=True)
        mass_array[i] = data['array']

    data_dict = dict()
    data_dict['mass.profile'] = mass_array
    data_dict['snapshot'] = snaps
    data_dict['time'] = times
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/all_snapshots/'+sim_data.galaxy+'_mass_profile_all', dict_or_array_to_write=data_dict, verbose=True)



## FOR LG PAIRS
#
### Set path and initial parameters
# Would have to do this on Stampede...
#
#
#
#
if sim_data.num_gal == 2:
    sim_data.galaxy = str(sys.argv[2])
    print('Set paths')

    # Only saving 544 to 543 and 539 to 538 for the LG-pairs...
    #snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,544,539,534,530,525,521,517,513,509,504,484,463,443,424,404,385,365,346,327,308,289,270,250,231,211,190,169,147,124,99,72,42])
    #times = np.array([13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8,11.3,10.8,10.3,9.8,9.3,8.8,8.3,7.8,7.3,6.8,6.3,5.8,5.3,4.8,4.3,3.8,3.3,2.8,2.3,1.8,1.3,0.8])
    #rs = np.logspace(np.log10(0.1), np.log10(500), 100)

    mass_array = np.zeros((len(snaps),len(rs)-1))

    for i in range(0, len(snaps)):
        data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/all_snapshots/'+sim_data.galaxy+'/'+sim_data.galaxy+'_mass_profile_'+str(snaps[i]), verbose=True)
        mass_array[i] = data['array']

    data_dict = dict()
    data_dict['mass.profile'] = mass_array
    data_dict['snapshot'] = snaps
    data_dict['time'] = times
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/all_snapshots/'+sim_data.galaxy+'_mass_profile_all', dict_or_array_to_write=data_dict, verbose=True)
