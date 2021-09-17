#!/usr/bin/env python3
#SBATCH --job-name=m12i_potential_test_sub50_N1e6
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
##SBATCH --mem=480G
#SBATCH --nodes=1
#SBATCH --ntasks=4    # processes total
#SBATCH --time=04:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/potentials/m12i_potential_test_sub50_N1e6_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

"""

    ======================
    = Subhalo potentials =
    ======================

    Calculate:
        - The host halo potential at 2*R_200m +/- 5 kpc using ALL particles
        - Each subhalo potential at +/- 5 kpc from their radius using DM particles

    NOTES:
        - The mean and median host potential are almost identical
        - The potential of the host galaxy using only DM particles is
          almost identical.
        - The subhalo potential using R-5kpc < d < R gives almost the same
          results as when doing R +/- 5 kpc

    COULDNT RUN ON M12Z OR ROMULUS & REMUS!

"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import spatial
import time
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')
#
if sim_data.num_gal == 1:
    #
    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    start = time.time()
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
    halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
    part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, properties=['position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    end = time.time()
    print('Tree and particles at z = 0 read in in {0} seconds'.format(end-start))
    #
    # Set up the halo inds and KDTree
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1)
    start = time.time()
    orbit_tree = orbit_io.OrbitTree(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1, particles=part, subsampling=50)
    end = time.time()
    print('KDTree created in {0} seconds'.format(end-start))
    #
    # Create a binning scheme and mass bin array to loop over
    def mass_binning(mass_array, mass_range):
        mass_array = np.log10(mass_array)
        #
        mask = (mass_array > mass_range[0])*(mass_array < mass_range[1])
        #
        return mask
    #
    halo_mass_bins = np.array([8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5])
    #
    # Create a dictionary to save the data to
    data_dict = dict()
    data_dict['mass.bin'] = halo_mass_bins
    #
    # Loop over each mass bin
    for i in range(0, len(halo_mass_bins)-1):
        start = time.time()
        # Get the halos in a mass bin
        mass_mask = mass_binning(halt.prop('mass.peak', orbits.sub_inds[:,0]),(halo_mass_bins[i],halo_mass_bins[i+1]))
        #
        # If there are halos, continue
        if np.sum(mass_mask) != 0:
            # Get the halo positions and max halo radius in the mass bin
            halo_pos = halt['position'][orbits.sub_inds[:,0][mass_mask]]
            dmax = np.around(np.max(halt['radius'][orbits.sub_inds[:,0][mass_mask]]))
            #
            # Query the particle tree and save the distances and indices
            ndist, nind = orbit_tree.neighbors(centers=halo_pos, neigh_num_max=1e6, neigh_dist_max=dmax, workerss=4)
            #
            # Create empty arrays to save data to
            potential_array = np.zeros(len(halo_pos))
            particle_num = np.zeros(len(halo_pos))
            #
            # Loop over the number of halos
            for j in range(0, len(halo_pos)):
                # Find the particles within +/- 5 kpc of the halo radius, then save the potential and particle number
                part_mask = (ndist[j][np.isfinite(ndist[j])] < (halt['radius'][orbits.sub_inds[:,0][mass_mask]][j]+5))*(ndist[j][np.isfinite(ndist[j])] > (halt['radius'][orbits.sub_inds[:,0][mass_mask]][j]-5))
                potential_array[j] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind[j][np.isfinite(ndist[j])][part_mask]])
                particle_num[j] = np.sum(part_mask)
            data_dict['potential.bin'+str(i)] = potential_array
            data_dict['particle.num.bin'+str(i)] = particle_num
            data_dict['halo.inds.bin'+str(i)] = orbits.sub_inds[:,0][mass_mask]
        #
        # If no halos, save empty arrays
        else:
            data_dict['potential.bin'+str(i)] = np.array([])
            data_dict['particle.num.bin'+str(i)] = np.array([])
            data_dict['halo.inds.bin'+str(i)] = np.array([])
            print('No halos between {0} and {1}'.format(halo_mass_bins[i],halo_mass_bins[i+1]))
        end = time.time()
        print('Done with mass bin {0} in {1} seconds'.format(i, end-start))
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.galaxy+'_potential_test', dict_or_array_to_write=data_dict, verbose=True)
