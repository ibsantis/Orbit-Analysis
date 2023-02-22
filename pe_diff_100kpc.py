#!/usr/bin/env python3
#SBATCH --job-name=checking_pe_diff
#SBATCH --partition=high2
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks=1    # processes total
#SBATCH --time=04:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/checking_pe_diff_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

"""
    Checking how the difference in U at 100 kpc between different times changes with time, and how it compares to GM/R at different times
"""
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import orbit_io
from numba import jit
from scipy import spatial
import matplotlib
from matplotlib import pyplot as plt
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
times = np.arange(13.8, 2.55, -0.25)
indices = np.zeros(len(times), int)
for i in range(0, len(times)):
    indices[i] = np.where(np.min(np.abs(snaps['time']-t[i])) == np.abs(snaps['time']-t[i]))[0][0]
print('Set paths')

# Set up an empty array to save to
data_dict = dict()
data_dict['times'] = times
data_dict['snaps'] = indices
data_dict['pe.diff'] = (-1)*np.ones(len(indices)-1)
data_dict['GMR.diff'] = (-1)*np.ones(len(indices)-1)
#
for i in range(0, len(indices)-1):
    # Read in the halo tree, snapshot dictionary, and orbit class so that I can access the subhalo indices easily
    part1 = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', indices[i], properties=['position', 'potential', 'mass'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    part2 = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', indices[i+1], properties=['position', 'potential', 'mass'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    #
    # find the dm particles at 100 kpc to calculate the potential at 100 kpc
    inds1 = ut.array.get_indices(part1['dark'].prop('host.distance.total'), [98, 102])
    inds2 = ut.array.get_indices(part2['dark'].prop('host.distance.total'), [98, 102])
    #
    # Calculate the potential and save the difference
    pe1 = np.mean(part1['dark']['potential'][inds1])
    pe2 = np.mean(part2['dark']['potential'][inds2])
    #
    data_dict['pe.diff'][i] = (pe1-pe2)
    #
    # Get all the particles within 100 kpc
    dark_inds1 = ut.array.get_indices(part1['dark'].prop('host.distance.total'), [0, 100])
    dark_inds2 = ut.array.get_indices(part2['dark'].prop('host.distance.total'), [0, 100])
    gas_inds1 = ut.array.get_indices(part1['gas'].prop('host.distance.total'), [0, 100])
    gas_inds2 = ut.array.get_indices(part2['gas'].prop('host.distance.total'), [0, 100])
    star_inds1 = ut.array.get_indices(part1['star'].prop('host.distance.total'), [0, 100])
    star_inds2 = ut.array.get_indices(part2['star'].prop('host.distance.total'), [0, 100])
    #
    # Calculate the enclosed mass
    menc_1 = np.sum(part1['star']['mass'][star_inds1])+np.sum(part1['gas']['mass'][gas_inds1])+np.sum(part1['dark']['mass'][dark_inds1])
    menc_2 = np.sum(part2['star']['mass'][star_inds2])+np.sum(part2['gas']['mass'][gas_inds2])+np.sum(part2['dark']['mass'][dark_inds2])
    #
    # Calculate GM/R and save the difference
    gmr1 = ((6.67*10**(-11)*2*10**(30))/(10**3*3.086*10**(16)*1000**2))*menc_1/100
    gmr2 = ((6.67*10**(-11)*2*10**(30))/(10**3*3.086*10**(16)*1000**2))*menc_2/100
    #
    data_dict['GMR.diff'][i] = (gmr1-gmr2)

print('Done with loop')
ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/'+sim_data.galaxy+'_checking_delta_u_and_gmr', dict_or_array_to_write=data_dict, verbose=True)
print('All done.')
