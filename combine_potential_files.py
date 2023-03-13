#!/usr/bin/python3

"""
    ===========================
    = Potential combine files =
    ===========================

    Combine all of the files into one file for each host galaxy

"""

import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import orbit_io
import sys
print('Read in the tools')


### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romeo', location='mac')
sim_data.galaxy = 'Juliet'
print('Set paths')

# Set up snapshot array to loop through
#snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir)
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/R_J')
snaps = snaps['index'][3:]

# Read in the z = 0 data so that I can get the length of the arrays
data_z0 = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/potentials_new/all_snapshots/'+sim_data.galaxy+'/'+sim_data.galaxy+'_potentials_600', verbose=True)
#
data_dict = dict()
data_dict['snapshot'] = snaps
data_dict['KE.at.Rvir'] = data_z0['KE.at.Rvir']
data_dict['host.pot.R200m'] = data_z0['host.potential.R200m']
data_dict['host.pot.500kpc'] = np.zeros(len(snaps))
data_dict['subhalo.inds'] = np.zeros((len(data_z0['halo.inds']), len(snaps)))
data_dict['subhalo.pot'] = np.zeros((len(data_z0['halo.inds']), len(snaps)))
#
for i in range(0, len(snaps)):
    data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/potentials_new/all_snapshots/'+sim_data.galaxy+'/'+sim_data.galaxy+'_potentials_'+str(snaps[i]), verbose=True)
    data_dict['host.pot.500kpc'][i] = data['host.potential.500kpc']
    data_dict['subhalo.inds'][:,i] = data['halo.inds']
    data_dict['subhalo.pot'][:,i] = data['subhalo.potential']
#
ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials_new/all_snapshots/'+sim_data.galaxy+'_potentials_all', dict_or_array_to_write=data_dict, verbose=True)
