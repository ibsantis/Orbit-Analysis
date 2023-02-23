#!/usr/bin/env python3
#SBATCH --job-name=mass_profile_all
##SBATCH --partition=high2    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --partition=skx-normal
##SBATCH --mem=100G
#SBATCH --nodes=1
##SBATCH --ntasks=4    # processes total
#SBATCH --tasks-per-node=4
#SBATCH --time=04:00:00
##SBATCH --output=/home/ibsantis/scripts/jobs/mass_profiles/mass_profile_all_%j.txt
#SBATCH --output=/home1/05400/ibsantis/scripts/jobs/mass_profiles/m12z_mass_profile_all_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

"""

  ==========================
  = Mass Profile Evolution =
  ==========================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Summer Quarter, 2021

  Calculate what the mass profile is accounting for ALL particles
  within bins of spherical r = [0.1, 500] kpc

  NOTE:
    August 18th: running on all snapshots now, but at 25 distances, from 5-500 kpc

"""
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import orbit_io
from numba import jit
import sys
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1=sys.argv[1], location='stampede')
print('Set paths')

# Set up snapshot array to loop through
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir)
#snaps = snaps['index']
snaps = np.arange(int(sys.argv[2]), int(sys.argv[3]), -1)

# Create a distance array to loop through
#rs = np.logspace(np.log10(5), np.log10(500), 25) # OLD
rs = np.array([  0.        ,   5.        ,   6.05763829,   7.33899634,
         8.89139705,  10.77217345,  13.05078608,  15.8113883 ,
        19.15593425,  23.20794417,  28.11706626,  34.06460345,
        41.27020926,  50.        ,  60.57638293,  73.38996338,
        88.9139705 , 100.        , 107.7217345 , 130.50786078,
       150., 158.11388301, 191.55934248, 232.07944168, 281.1706626 ,
       340.64603453, 412.70209263, 500.        ])

@jit
def mass_evolution(snap, sim_data, rs):
    """
        Function that reads in particles, then computes the enclosed mass
        for a given distance array.
    """
    if sim_data.num_gal == 1:
        # Set up an empty array to save to
        mass_array = np.zeros(len(rs)-1)
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', snap, properties=['mass', 'position'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        #
        # Compute the 1D distances for all species once
        d_star_tot = part['star'].prop('host.distance.total')
        d_gas_tot = part['gas'].prop('host.distance.total')
        d_dark_tot = part['dark'].prop('host.distance.total')
        #
        # Loop through the distance array
        for j in range(0, len(rs)-1):
            # Find the particles within the distance bin
            star_inds = ut.array.get_indices(d_star_tot, [rs[j], rs[j+1]])
            gas_inds = ut.array.get_indices(d_gas_tot, [rs[j], rs[j+1]])
            dark_inds = ut.array.get_indices(d_dark_tot, [rs[j], rs[j+1]])
            # Sum the mass of all particles in the bin
            mass_array[j] = np.sum(part['star']['mass'][star_inds]) + np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
        #
        # Cumulatively sum the mass array to get the enclosed mass
        mass_array = np.cumsum(mass_array)
        #
        # Save this data to a file
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/all_snapshots/'+sim_data.galaxy+'/'+sim_data.galaxy+'_mass_profile_'+str(snap), dict_or_array_to_write=mass_array, verbose=True)
    #
    # Check if two hosts
    if sim_data.num_gal == 2:
        # Set up empty arrays to save to
        mass_array_1 = np.zeros(len(rs)-1)
        mass_array_2 = np.zeros(len(rs)-1)
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'index', snap, properties=['mass', 'position'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        #
        # Compute the 1D distances for all species once
        d_star_tot = part['star'].prop('host.distance.total')
        d_gas_tot = part['gas'].prop('host.distance.total')
        d_dark_tot = part['dark'].prop('host.distance.total')
        #
        d2_star_tot = part['star'].prop('host2.distance.total')
        d2_gas_tot = part['gas'].prop('host2.distance.total')
        d2_dark_tot = part['dark'].prop('host2.distance.total')
        #
        # Loop over the distance array
        for j in range(0, len(rs)-1):
            # Find the particles within the distance bin
            star_inds_1 = ut.array.get_indices(d_star_tot, [rs[j], rs[j+1]])
            gas_inds_1 = ut.array.get_indices(d_gas_tot, [rs[j], rs[j+1]])
            dark_inds_1 = ut.array.get_indices(d_dark_tot, [rs[j], rs[j+1]])
            # Sum the mass of all particles in the bin
            mass_array_1[j] = np.sum(part['star']['mass'][star_inds_1]) + np.sum(part['gas']['mass'][gas_inds_1]) + np.sum(part['dark']['mass'][dark_inds_1])
            #
            # Find the particles within the distance bin
            star_inds_2 = ut.array.get_indices(d2_star_tot, [rs[j], rs[j+1]])
            gas_inds_2 = ut.array.get_indices(d2_gas_tot, [rs[j], rs[j+1]])
            dark_inds_2 = ut.array.get_indices(d2_dark_tot, [rs[j], rs[j+1]])
            # Sum the mass of all particles in the bin
            mass_array_2[j] = np.sum(part['star']['mass'][star_inds_2]) + np.sum(part['gas']['mass'][gas_inds_2]) + np.sum(part['dark']['mass'][dark_inds_2])
        #
        # Cumulatively sum the mass of the arrays to get the enclosed mass
        mass_array_1 = np.cumsum(mass_array_1)
        mass_array_2 = np.cumsum(mass_array_2)
        #
        # Save this data to a file
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/all_snapshots/'+sim_data.gal_1+'/'+sim_data.gal_1+'_mass_profile_'+str(snap), dict_or_array_to_write=mass_array_1, verbose=True)
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/all_snapshots/'+sim_data.gal_2+'/'+sim_data.gal_2+'_mass_profile_'+str(snap), dict_or_array_to_write=mass_array_2, verbose=True)

# List of arguments to pass to the function above
args_list = [(snapshot, sim_data, rs) for snapshot in snaps]

# Command to run the function above
ut.io.run_in_parallel(mass_evolution, args_list, proc_number=4) # ADD VERBOSE
print('All done.')
