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
print('Read in the tools')


## FOR ISOLATED HOSTS

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')

# Set up snapshot array to loop through
snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,543,538,534,530,525,521,517,513,509,504,486,446,412,382,356,332,312,294,277,262,248,236,225,214,204,195,187,179,172,165,159,153,148,142,137,133,128,124])
times = np.array([13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8, 11.36, 10.37,9.51,8.73,8.06,7.43,6.90,6.43,6.00,5.60,5.24,4.94,4.66,4.38,4.14,3.92,3.73,3.54,3.37,3.21,3.07,2.93,2.82,2.69,2.58,2.49,2.38,2.29])
rs = np.logspace(np.log10(0.1), np.log10(500), 100)

mass_array = np.zeros((len(snaps),len(rs)-1))

for i in range(0, len(snaps)):
    fff = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'/'+sim_data.galaxy+'_mass_profile_evolution_'+str(snaps[i]), verbose=True)
    mass_array[i] = fff['array']

d = dict()
d['mass_array'] = mass_array
d['snapshot'] = snaps
d['time'] = times
ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'/'+sim_data.galaxy+'_full_mass_profile', dict_or_array_to_write=d, verbose=True)








## FOR LG PAIRS


### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romeo', location='mac')
print('Set paths')

# Set up snapshot array to loop through
snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,544,539,534,530,525,521,517,513,509,504,486,446,412,382,356,332,312,294,277,262,248,236,225,214,204,195,187,179,172,165,159,153,148,142,137,133,128,124])

# Only saving 544 to 543 and 539 to 538 for the LG-pairs...
snap_save = np.array([600,587,582,578,573,569,564,560,556,551,547,543,538,534,530,525,521,517,513,509,504,486,446,412,382,356,332,312,294,277,262,248,236,225,214,204,195,187,179,172,165,159,153,148,142,137,133,128,124])
times = np.array([13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8, 11.36, 10.37,9.51,8.73,8.06,7.43,6.90,6.43,6.00,5.60,5.24,4.94,4.66,4.38,4.14,3.92,3.73,3.54,3.37,3.21,3.07,2.93,2.82,2.69,2.58,2.49,2.38,2.29])

d = dict()
for i in range(0, len(snaps)):
    fff = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'/'+sim_data.galaxy+'_mass_profile_evolution_'+str(snaps[i]), verbose=True)
    d[str(snap_save[i])] = fff['array']

ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'/'+sim_data.galaxy+'_full_mass_profile', dict_or_array_to_write=d, verbose=True)
