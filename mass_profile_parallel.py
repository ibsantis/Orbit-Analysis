#!/usr/bin/env python3
#SBATCH --job-name=m12w_mass_profile
##SBATCH --partition=high2    # peloton node: 32 cores, 7.8 GB per core, 250 GB total
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=480G
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=4    # MPI tasks per node
#SBATCH --ntasks=8    # processes total
##SBATCH --cpus-per-task=1    # OpenMP threads per MPI task
#SBATCH --time=4:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/mass_profiles/m12w_mass_profile_%j.txt
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

"""
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12w', location='peloton')
print('Set paths')

# Set up snapshot array to loop through
#snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,543,538,534,530,525,521,517,513,509,504,486,446,412,382,356,332,312,294,277,262,248,236,225,214,204,195,187,179,172,165,159,153,148,142,137,133,128,124])
# New snapshot array just for the LG pairs
snaps = np.array([600,587,582,578,573,569,564,560,556,551,547,544,539,534,530,525,521,517,513,509,504,484,463,443,424,404,385,365,346,327,308,289,270,250,231,211,190,169,147,124,99,72,42])
times = np.array([13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8,11.3,10.8,10.3,9.8,9.3,8.8,8.3,7.8,7.3,6.8,6.3,5.8,5.3,4.8,4.3,3.8,3.3,2.8,2.3,1.8,1.3,0.8])
rs = np.logspace(np.log10(0.1), np.log10(500), 100)

# function from first loop down # TRY RUNNING A COUPLE SNAPSHOTS IN A LOOP FIRST TO MAKE SURE IT WORKS
def mass_evolution(snap, sim_data, rs):
    # Check if one or two hosts
    if sim_data.num_gal == 1:
        #
        mass_array = np.zeros(len(rs)-1)
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', snap, simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        #print('Particles at snapshot {0} read in.'.format(snap))
        #
        # Find the enclosed mass of all particles within 0.1 < R < 500 kpc
        for j in range(0, len(rs)-1):
            star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [rs[j], rs[j+1]])
            gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[j], rs[j+1]])
            dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[j], rs[j+1]])
            mass_array[j] = np.sum(part['star']['mass'][star_inds]) + np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
            #print('Done with step', j)
        #
        # Save this data to a file ADD VERBOSE
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/may_18_new/'+sim_data.galaxy+'/'+sim_data.galaxy+'_mass_profile_evolution_'+str(snap), dict_or_array_to_write=mass_array, verbose=True)
        print('Done with snapshot {0}'.format(snap))
    #
    # Check if one or two hosts
    if sim_data.num_gal == 2:
        #
        mass_array_1 = np.zeros(len(rs)-1)
        mass_array_2 = np.zeros(len(rs)-1)
        #
        # Read in the data
        part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', snap, simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
        #print('Particles at snapshot {0} read in.'.format(snap))
        #
        # Find the enclosed mass of all particles within 0.1 < R < 500 kpc
        for j in range(0, len(rs)-1):
            star_inds_1 = ut.array.get_indices(part['star'].prop('host.distance.total'), [rs[j], rs[j+1]])
            gas_inds_1 = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[j], rs[j+1]])
            dark_inds_1 = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[j], rs[j+1]])
            mass_array_1[j] = np.sum(part['star']['mass'][star_inds_1]) + np.sum(part['gas']['mass'][gas_inds_1]) + np.sum(part['dark']['mass'][dark_inds_1])
            #
            star_inds_2 = ut.array.get_indices(part['star'].prop('host2.distance.total'), [rs[j], rs[j+1]])
            gas_inds_2 = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [rs[j], rs[j+1]])
            dark_inds_2 = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [rs[j], rs[j+1]])
            mass_array_2[j] = np.sum(part['star']['mass'][star_inds_2]) + np.sum(part['gas']['mass'][gas_inds_2]) + np.sum(part['dark']['mass'][dark_inds_2])
            #print('Done with step', j)
        #
        # Save this data to a file ADD VERBOSE
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/may_18_new/'+sim_data.gal_1+'/'+sim_data.gal_1+'_mass_profile_evolution_'+str(snap), dict_or_array_to_write=mass_array_1, verbose=True)
        ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/may_18_new/'+sim_data.gal_2+'/'+sim_data.gal_2+'_mass_profile_evolution_'+str(snap), dict_or_array_to_write=mass_array_2, verbose=True)
        print('Done with snapshot {0}'.format(snap))

args_list = [
    (snapshot, sim_data, rs) for snapshot in snaps
    ]

ut.io.run_in_parallel(mass_evolution, args_list, proc_number=8) # ADD VERBOSE
# Try 4 at first, then try 8
# How many snaps can I read in simultaneously? Divide total mem by snapshots to get proc number
#
# proc_num and n_tasks in SBATCH stuff must be equal

print('All done?')
