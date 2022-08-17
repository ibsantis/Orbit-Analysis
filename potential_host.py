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

  ==========================
  = Mass Profile Evolution =
  ==========================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Summer Quarter, 2021

  Calculate what the mass profile is accounting for ALL particles
  within bins of spherical r, out to 500 kpc

  Do this every 100 Myr going back to 1 Gyr (11 snapshots total)

  IDEAS
    - numba decorator
    - numpy vectorize
    - Try with larger r_bins

"""
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

# Set up snapshot array to loop through
#snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,543,538,534,530,525,521,517,513,509,504,486,446,412,382,356,332,312,294,277,262,248,236,225,214,204,195,187,179,172,165,159,153,148,142,137,133,128,124])
# New snapshot array just for the LG pairs
snaps = np.array([600,587,582])#,578,573,569,564,560,556,551,547,544,539,534,530,525,521,517,513,509,504,484,463,443,424,404,385,365,346,327,308,289,270,250,231,211,190,169,147,124,99,72,42])
times = np.array([13.8,13.7,13.6])#,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8,11.3,10.8,10.3,9.8,9.3,8.8,8.3,7.8,7.3,6.8,6.3,5.8,5.3,4.8,4.3,3.8,3.3,2.8,2.3,1.8,1.3,0.8])
dist = 200
delta = np.array([5,10,15,20])

# function from first loop down # TRY RUNNING A COUPLE SNAPSHOTS IN A LOOP FIRST TO MAKE SURE IT WORKS
def calc_potential(snap, sim_data, dist, delta):
    # Check if one or two hosts
    if sim_data.num_gal == 1:
        #
        pot_array_med = np.zeros(len(delta))
        pot_array_mean = np.zeros(len(delta))
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', snap, properties=['mass', 'position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        print('Particles at snapshot {0} read in.'.format(snap))
        #
        # Find the enclosed mass of all particles within 0.1 < R < 500 kpc
        print('starting the loop')
        for j in range(0, len(delta)):
            star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [dist-delta[j], dist+delta[j]])
            gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [dist-delta[j], dist+delta[j]])
            dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [dist-delta[j], dist+delta[j]])
            pot_array_med[j] = np.median(np.hstack((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
            pot_array_mean[j] = np.mean(np.hstack((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
            print('Done with step {0} at snapshot {1}'.format(j, snap))
        #
        data_dict = dict()
        data_dict['potential.median'] = pot_array_med
        data_dict['potential.mean'] = pot_array_mean
        print('Made the dictionary')
        #
        # Save this data to a file ADD VERBOSE
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/host_potentials/'+sim_data.galaxy+'_host_potential_'+str(snap), dict_or_array_to_write=data_dict, verbose=True)
        print('Done with snapshot {0}'.format(snap))


args_list = [
    (snapshot, sim_data, dist, delta) for snapshot in snaps
    ]

ut.io.run_in_parallel(calc_potential, args_list, proc_number=3) # ADD VERBOSE

print('All done?')
