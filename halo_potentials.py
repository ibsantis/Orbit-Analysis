#!/usr/bin/python3

"""
 ==========================================
 = Progenitor halo star particle tracking =
 ==========================================

 Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2020

 GOAL:
    1. Select luminous halos .
    2. For each halo calculate the mean and median gravitational potential
       using their member star particles.
    3. Save the data to a file.
"""

# Import all of the tools for analysis
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import orbit_times as ot
import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12i'
loc = 'stampede'

if gal1 == 'Romeo':
    gal2 = 'Juliet'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res3500'
    num_gal = 2
elif gal1 == 'Thelma':
    gal2 = 'Louise'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res4000'
    num_gal = 2
elif gal1 == 'Romulus':
    gal2 = 'Remus'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res4000'
    num_gal = 2
else:
    galaxy = gal1
    resolution = '_res7100'
    num_gal = 1

if loc == 'mac':
    home_dir = '/Users/isaiahsantistevan/simulation'
else:
    home_dir = '/home1/05400/ibsantis/scripts'
simulation_dir = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'+galaxy+resolution
print('Set paths')

# Read in the entire tree
snaps = ut.simulation.read_snapshot_times(directory=simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory='/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12i_res7100')

# Set up array of snapshot values starting from 600 to 0 (redshift 0 to 99)
ss = np.flip(np.arange(601))
# Get Luminous halo indices
lum_ind = ut.array.get_indices(halt['star.mass'], [1e-6, np.inf])
# Get list of halo inds for each snapshot
# This has length 601 (from 600 to 0)
halo_inds = [ut.array.get_indices(halt['snapshot'], ss[i], lum_ind) for i in range(0, len(ss))]

"""
Method:
    0. Set up empty lists to save the data to.
       Each element of this list will contain an array for each snapshot (up to snapshot = 1)
    1. Read in the particle data at the snapshot.
    2. For each halo, calculate the mean and median gravitational potential using
       only the halo's star particle members.
    3. Save the data to a dictionary.
"""
# Set up empty lists
halo_potentials_mean = []
halo_potentials_median = []

start = time.time()
# Loop over snapshots
# Only go until snapshot = 2 because there are no stars at snapshot = 1
for i in range(0, 599):
    part_at_z = gizmo.io.Read.read_snapshots('star', 'snapshot', ss[i], simulation_directory=simulation_dir, properties=['position', 'potential'], assign_host_coordinates=False)
    halo_potentials_mean.append(np.asarray([np.mean(part_at_z['star']['potential'][halt['star.indices'][halo_inds[i][j]]]) for j in range(0, len(halo_inds[i]))]))
    halo_potentials_median.append(np.asarray([np.median(part_at_z['star']['potential'][halt['star.indices'][halo_inds[i][j]]]) for j in range(0, len(halo_inds[i]))]))
end = time.time()
print('Finished calculating the halo potentials in', end-start, 'seconds')

# Throw the data in a dictionary and save it to a file
data_dict = dict()
# Halo indices will be arrays of indices at each snapshot. (has length = 601)
data_dict['halo.indices'] = halo_inds
data_dict['halo.potential.mean'] = pot_mean_all
data_dict['halo.potential.median'] = pot_med_all
#
file_save = open(home_dir+'/orbit_data/pickles/'+galaxy+'_halo_potentials.p','wb')
pickle.dump(data_dict, file_save)
file_save.close()
print('All done?')
