#!/usr/bin/env python3
#SBATCH --job-name=m12i_host_potential
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=480G
#SBATCH --nodes=1
#SBATCH --ntasks=3    # processes total
#SBATCH --time=00:10:00
#SBATCH --output=/home/ibsantis/scripts/jobs/potentials/m12i_host_potential_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

"""

  ==================
  = Host potential =
  ==================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Summer Quarter, 2022

  Calculate the potential of the host at a given distance using all particles
  within some spherical shell, and save to a file

  NOTE:
    R&R and m12z don't have potentials saved in the data

"""

import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
from orbit_analysis import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

# Set up snapshot array to loop through
snaps = np.array([600,587,582])#,578,573,569,564,560,556,551,547,544,539,534,530,525,521,517,513,509,504,484,463,443,424,404,385,365,346,327,308,289,270,250,231,211,190,169,147,124,99,72,42])
times = np.array([13.8,13.7,13.6])#,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8,11.3,10.8,10.3,9.8,9.3,8.8,8.3,7.8,7.3,6.8,6.3,5.8,5.3,4.8,4.3,3.8,3.3,2.8,2.3,1.8,1.3,0.8])

# Pick a distance to calculate the potential
dist = 200
# Choose the width of the shell
delta = np.array([5,10,15,20])

def calc_potential(snap, sim_data, dist, delta):
    """
        Read in the data, find the particles in a shell, calculate and save the
        potential to a file
    """
    # Check if one or two hosts; currently only written to work on one
    if sim_data.num_gal == 1:
        # Create empty arrays to save to
        pot_array_med = np.zeros(len(delta))
        pot_array_mean = np.zeros(len(delta))
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', snap, properties=['mass', 'position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        print('Particles at snapshot {0} read in.'.format(snap))
        #
        # Compute the 1D distances for the particles once
        d_star_tot = part['star'].prop('host.distance.total')
        d_gas_tot = part['gas'].prop('host.distance.total')
        d_dark_tot = part['dark'].prop('host.distance.total')
        #
        # Loop through each spherical shell
        for j in range(0, len(delta)):
            # Select particles in the shell
            star_inds = ut.array.get_indices(d_star_tot, [dist-delta[j], dist+delta[j]])
            gas_inds = ut.array.get_indices(d_gas_tot, [dist-delta[j], dist+delta[j]])
            dark_inds = ut.array.get_indices(d_dark_tot, [dist-delta[j], dist+delta[j]])
            #
            # Calculate the median and mean potential in the shell
            pot_array_med[j] = np.median(np.hstack((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
            pot_array_mean[j] = np.mean(np.hstack((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
            print('Done with step {0} at snapshot {1}'.format(j, snap))
        #
        # Save the median and mean potentials to a dictionary
        data_dict = dict()
        data_dict['potential.median'] = pot_array_med
        data_dict['potential.mean'] = pot_array_mean
        #
        # Save this data to a file
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/host_potentials/'+sim_data.galaxy+'_host_potential_'+str(snap), dict_or_array_to_write=data_dict, verbose=True)
        print('Done with snapshot {0}'.format(snap))

# Create an array of arguments for the function above
args_list = [(snapshot, sim_data, dist, delta) for snapshot in snaps]

# Run the function using the arguments above in parallel
ut.io.run_in_parallel(calc_potential, args_list, proc_number=3) # ADD VERBOSE
print('All done.')
