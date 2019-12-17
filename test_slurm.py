#!/usr/bin/python3

"""
    =======================
    = Compare Pericenters =
    =======================

    Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Fall Quarter 2019

    Read in the tree data and compare the pericenters for normalized and
    un-normalized distances.
    Make plots for both for all subhalos .
    Save the orbit information to a pickle file.

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
# Need to initialize the class
orbits = ot.OrbitAnalysis()
#################################################################################################
# Select the luminous subhalos first
subhalo_inds = orbits.get_luminous_halos(halt)
#################################################################################################

# Get the distances from the host for each halo for all snapshots
# This goes from z = 0 to z'
halt_dists = orbits.halo_distances(halt, subhalo_inds)

# Figure out what the virial radius is for the host and make another plot
# These go from z = 0 to z'
host_radii = halt['radius'][subhalo_inds[0]] # Want to divide the other distances by this distance

# Need to use the mask for each halo on these radii so that the lengths are equal
# Some of the halos existed longer than the "host", that's okay?
# Goes from z = 0 to z'
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii) # Originally had the function written out here

#################################################################################################
"""
Infall times
"""

# Calculate what the infall times were (i.e. when are the normalized distances < 1?)
# Goes from z = 0 to z'
infall_info = orbits.first_infall_times(halt_dists_norm, snaps)
#infall_info['snapshot']
#infall_info['time']

# Save these values to a file or something...?
#################################################################################################

orbit_info = orbits.pericenter_interp(halt_dists, halt_dists_norm, host_radii, snaps)

angs = orbits.angular_momentum(halt, subhalo_inds)

"""
# Testing something out
ls_1 = ls['ang.mom.vector'][3][:,2]
snapshots = np.flip(snaps['time'], axis=0)[:len(halt_dists[3])]
plt.figure(1)
plt.figure(figsize=(10, 8))
plt.plot(snapshots, ls_1)
plt.xlim(0, 13.9)
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('L$_{z}$ [km s$^{-1}$ kpc]', fontsize=28)
plt.title('m12i, Halo 4', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/ang_halo_4.pdf')
plt.close()
"""

File1 = open(home_dir+'/orbit_plots/test_data.p','wb')
pickle.dump(orbit_info, File1)
pickle.dump(infall_info, File1)
File1.close()
