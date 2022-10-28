#!/usr/bin/env python3
#SBATCH --job-name=m12m_subhalo_potential_all_snaps
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
##SBATCH --partition=skx-normal
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks=3    # processes total
##SBATCH --tasks-per-node=1    # MPI tasks per node
#SBATCH --cpus-per-task=1    # OpenMP threads per MPI task
#SBATCH --time=06:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/potentials/all_snapshots/m12m_subhalo_potential_all_snaps_%j.txt
##SBATCH --output=/home1/05400/ibsantis/scripts/jobs/potentials/all_snapshots/RJ_subhalo_potential_all_snaps_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin
##SBATCH --account=TG-AST140064

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
import time
print('Read in the tools')

### Set path and initial parameters
loc = 'peloton'
sim_data = orbit_io.OrbitRead(gal1='m12m', location=loc)
print('Set paths')

# Read in snapshot dictionary and the halo tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location=loc, host=1)
#orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=1)
print('Read in halo tree and set up subhalo indices')

# Set up the snapshot array to loop through
snaps = np.flip(snaps['index'])[:len(orbits.sub_inds[0])]

# Set up the host indices
host_inds = halt.prop('progenitor.main.indices', halt['host.index'][0])
host_snaps = halt['snapshot'][host_inds]

def calc_sub_potential(snap, simdata, orbit_class):
    #
    if simdata.num_gal == 1:
        # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
        start = time.time()
        #
        # For luminous & ALL subhalos
        part = gizmo.io.Read.read_snapshots('dark', 'snapshot', snap, properties=['position', 'potential'], simulation_directory=simdata.simulation_dir, assign_hosts_rotation=True)
        end = time.time()
        print('Particles at snapshot {0} read in in {1} seconds'.format(snap, end-start))
        #
        # # Set up the halo inds and KDTree
        start = time.time()
        orbit_tree = orbit_io.OrbitTree(tree=halt, gal1=simdata.galaxy, location=loc, host=1, particles=part, subsampling=15)
        end = time.time()
        print('KDTree created in {0} seconds'.format(end-start))
        #
        # Create a binning scheme and mass bin array to loop over
        def mass_binning(mass_array, mass_range):
            mass_array = np.log10(mass_array)
            mask = (mass_array > mass_range[0])*(mass_array < mass_range[1])
            return mask
        #
        halo_mass_bins = np.array([7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.])
        #
        # Create a dictionary to save the data to
        data_dict = dict()
        data_dict['mass.bin'] = halo_mass_bins
        data_dict['halo.inds'] = (-1)*np.ones(len(orbit_class.sub_inds[:,600-snap]), dtype=int)
        data_dict['halo.bin'] = (-1)*np.ones(len(orbit_class.sub_inds[:,600-snap]), dtype=int)
        data_dict['subhalo.potential'] = (-1)*np.ones(len(orbit_class.sub_inds[:,600-snap]))
        data_dict['particle.num'] = (-1)*np.ones(len(orbit_class.sub_inds[:,600-snap]), dtype=int)
        temp = np.arange(len(orbit_class.sub_inds[:,600-snap]))
        print('Set up a null dictionary for the data at snapshot {0}'.format(snap))
        #
        if snap in host_snaps:
            # Find the potential of the host at 100 kpc
            ndist, nind = orbit_tree.neighbors(centers=halt['position'][host_inds[600-snap]], neigh_num_max=1e8, neigh_dist_max=100, workerss=4)
            part_mask = (ndist[np.isfinite(ndist)] < (100+5))*(ndist[np.isfinite(ndist)] > (100-5))
            data_dict['host.potential.100kpc'] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind[np.isfinite(ndist)][part_mask]])
            print('Calculated potential at 100 kpc at snapshot {0}'.format(snap))
        if snap not in host_snaps:
            data_dict['host.potential.100kpc'] = np.nan
            print('No well defined host at snapshot {0}'.format(snap))
        #
        if snap == 600:
            # Find the potential of the host within R200
            ndist, nind = orbit_tree.neighbors(centers=halt['position'][halt['host.index'][0]], neigh_num_max=1e8, neigh_dist_max=halt['radius'][halt['host.index'][0]], workerss=4)
            part_mask = (ndist[np.isfinite(ndist)] < (halt['radius'][halt['host.index'][0]]+5))*(ndist[np.isfinite(ndist)] > (halt['radius'][halt['host.index'][0]]-5))
            data_dict['host.potential.R200m'] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind[np.isfinite(ndist)][part_mask]])
            data_dict['host.particle.num'] = np.sum(part_mask)
            G = 6.67*10**(-11)*(1.988*10**(30))/((1000**2)*(3.086*10**(19)))
            data_dict['KE.at.Rvir'] = 0.5*G*halt['mass'][halt['host.index'][0]]/halt['radius'][halt['host.index'][0]]
            #
            print('Finished calculating host data for snapshot 600')
        #
        # Loop over each mass bin
        print('Starting loop over the mass bins')
        for i in range(0, len(halo_mass_bins)-1):
            start = time.time()
            # Get the halos in a mass bin
            real_halos = (orbit_class.sub_inds[:,600-snap] >= 0)
            mass_mask = mass_binning(halt.prop('mass.peak', orbit_class.sub_inds[:,600-snap]), (halo_mass_bins[i], halo_mass_bins[i+1]))
            data_dict['halo.inds'][real_halos*mass_mask] = orbit_class.sub_inds[:,600-snap][real_halos*mass_mask]
            data_dict['halo.bin'][real_halos*mass_mask] = i
            #
            # If there are halos, continue
            if np.sum(mass_mask) != 0:
                # Get the halo positions and max halo radius in the mass bin
                halo_pos = halt['position'][orbit_class.sub_inds[:,600-snap][real_halos*mass_mask]]
                dmax = np.around(np.max(halt['radius'][orbit_class.sub_inds[:,600-snap][real_halos*mass_mask]]))
                #
                # Query the particle tree and save the distances and indices
                ndist, nind = orbit_tree.neighbors(centers=halo_pos, neigh_num_max=1e6, neigh_dist_max=dmax, workerss=4)
                #
                # Loop over the number of halos
                for j in range(0, len(halo_pos)):
                    # Find the particles within +/- 5 kpc of the halo radius, then save the potential and particle number
                    part_mask = (ndist[j][np.isfinite(ndist[j])] < (halt['radius'][orbit_class.sub_inds[:,600-snap][real_halos*mass_mask]][j]+5))*(ndist[j][np.isfinite(ndist[j])] > (halt['radius'][orbit_class.sub_inds[:,600-snap][real_halos*mass_mask]][j]-5))
                    data_dict['subhalo.potential'][temp[real_halos*mass_mask][j]] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind[j][np.isfinite(ndist[j])][part_mask]])
                    data_dict['particle.num'][temp[real_halos*mass_mask][j]] = np.sum(part_mask)
            #
            # If no halos, say so
            else:
                print('No halos between {0} and {1}'.format(halo_mass_bins[i],halo_mass_bins[i+1]))
            end = time.time()
            #
            print('Done with mass bin {0} in {1} seconds'.format(i, end-start))
        #
        ut.io.file_hdf5(file_name_base=simdata.home_dir+'/orbit_data/hdf5_files/potentials/all_snapshots/'+simdata.galaxy+'/'+simdata.galaxy+'_potentials_'+str(snap), dict_or_array_to_write=data_dict, verbose=True)
        print('Completely finished writing snapshot {0}'.format(snap))

    if simdata.num_gal == 2:
        #
        orbit_class_1 = orbit_class
        orbit_class_2 = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=2)
        # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
        start = time.time()
        #
        # For luminous & ALL subhalos
        part = gizmo.io.Read.read_snapshots('dark', 'snapshot', snap, properties=['position', 'potential'], simulation_directory=simdata.simulation_dir, assign_hosts_rotation=True)
        end = time.time()
        print('Particles at snapshot {0} read in in {1} seconds'.format(snap, end-start))
        #
        # Set up the halo inds and KDTree
        start = time.time()
        orbit_tree = orbit_io.OrbitTree(tree=halt, gal1=simdata.gal_1, location=loc, host=1, particles=part, subsampling=15)
        end = time.time()
        print('KDTree created in {0} seconds'.format(end-start))
        #
        # Create a binning scheme and mass bin array to loop over
        def mass_binning(mass_array, mass_range):
            mass_array = np.log10(mass_array)
            mask = (mass_array > mass_range[0])*(mass_array < mass_range[1])
            return mask
        #
        halo_mass_bins = np.array([7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.])
        #
        # Create a dictionary to save the data to
        data_dict_1 = dict()
        data_dict_1['mass.bin'] = halo_mass_bins
        data_dict_1['halo.inds'] = (-1)*np.ones(len(orbit_class_1.sub_inds[:,600-snap]), dtype=int)
        data_dict_1['halo.bin'] = (-1)*np.ones(len(orbit_class_1.sub_inds[:,600-snap]), dtype=int)
        data_dict_1['subhalo.potential'] = (-1)*np.ones(len(orbit_class_1.sub_inds[:,600-snap]))
        data_dict_1['particle.num'] = (-1)*np.ones(len(orbit_class_1.sub_inds[:,600-snap]), dtype=int)
        temp_1 = np.arange(len(orbit_class_1.sub_inds[:,600-snap]))
        #
        data_dict_2 = dict()
        data_dict_2['mass.bin'] = halo_mass_bins
        data_dict_2['halo.inds'] = (-1)*np.ones(len(orbit_class_2.sub_inds[:,600-snap]), dtype=int)
        data_dict_2['halo.bin'] = (-1)*np.ones(len(orbit_class_2.sub_inds[:,600-snap]), dtype=int)
        data_dict_2['subhalo.potential'] = (-1)*np.ones(len(orbit_class_2.sub_inds[:,600-snap]))
        data_dict_2['particle.num'] = (-1)*np.ones(len(orbit_class_2.sub_inds[:,600-snap]), dtype=int)
        temp_2 = np.arange(len(orbit_class_2.sub_inds[:,600-snap]))
        #
        print('Set up a null dictionary for the data at snapshot {0}'.format(snap))
        #
        if snap == 600:
            # Find the potential of the host within R200
            ndist_1, nind_1 = orbit_tree.neighbors(centers=halt['position'][halt['host.index'][0]], neigh_num_max=1e8, neigh_dist_max=halt['radius'][halt['host.index'][0]], workerss=4)
            part_mask_1 = (ndist_1[np.isfinite(ndist_1)] < (halt['radius'][halt['host.index'][0]]+5))*(ndist_1[np.isfinite(ndist_1)] > (halt['radius'][halt['host.index'][0]]-5))
            data_dict_1['host.potential.R200m'] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind_1[np.isfinite(ndist_1)][part_mask_1]])
            data_dict_1['host.particle.num'] = np.sum(part_mask_1)
            G = 6.67*10**(-11)*(1.988*10**(30))/((1000**2)*(3.086*10**(19)))
            data_dict_1['KE.at.Rvir'] = 0.5*G*halt['mass'][halt['host.index'][0]]/halt['radius'][halt['host.index'][0]]
            #
            # Find the potential of the host at 100 kpc
            ndist_1, nind_1 = orbit_tree.neighbors(centers=halt['position'][halt['host.index'][0]], neigh_num_max=1e8, neigh_dist_max=100, workerss=4)
            part_mask_1 = (ndist_1[np.isfinite(ndist_1)] < (100+5))*(ndist_1[np.isfinite(ndist_1)] > (100-5))
            data_dict_1['host.potential.100kpc'] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind[np.isfinite(ndist_1)][part_mask_1]])
            #
            # Find the potential of the host within R200
            n_dist_2, nind_2 = orbit_tree.neighbors(centers=halt['position'][halt['host2.index'][0]], neigh_num_max=1e8, neigh_dist_max=halt['radius'][halt['host2.index'][0]], workerss=4)
            part_mask_2 = (n_dist_2[np.isfinite(n_dist_2)] < (halt['radius'][halt['host2.index'][0]]+5))*(n_dist_2[np.isfinite(n_dist_2)] > (halt['radius'][halt['host2.index'][0]]-5))
            data_dict_2['host.potential.R200m'] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind_2[np.isfinite(n_dist_2)][part_mask_2]])
            data_dict_2['host.particle.num'] = np.sum(part_mask_2)
            G = 6.67*10**(-11)*(1.988*10**(30))/((1000**2)*(3.086*10**(19)))
            data_dict_2['KE.at.Rvir'] = 0.5*G*halt['mass'][halt['host2.index'][0]]/halt['radius'][halt['host2.index'][0]]
            #
            # Find the potential of the host at 100 kpc
            n_dist_2, nind_2 = orbit_tree.neighbors(centers=halt['position'][halt['host2.index'][0]], neigh_num_max=1e8, neigh_dist_max=100, workerss=4)
            part_mask_2 = (n_dist_2[np.isfinite(n_dist_2)] < (100+5))*(n_dist_2[np.isfinite(n_dist_2)] > (100-5))
            data_dict_2['host.potential.100kpc'] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind[np.isfinite(ndist_2)][part_mask_2]])
            #
            print('Finished calculating host data for snapshot 600')
        #
        # Loop over each mass bin
        print('Starting loop over the mass bins')
        for i in range(0, len(halo_mass_bins)-1):
            start = time.time()
            # Get the halos in a mass bin
            real_halos_1 = (orbit_class_1.sub_inds[:,600-snap] >= 0)
            mass_mask_1 = mass_binning(halt.prop('mass.peak', orbit_class_1.sub_inds[:,600-snap]), (halo_mass_bins[i], halo_mass_bins[i+1]))
            #
            real_halos_2 = (orbit_class_2.sub_inds[:,600-snap] >= 0)
            mass_mask_2 = mass_binning(halt.prop('mass.peak', orbit_class_2.sub_inds[:,600-snap]), (halo_mass_bins[i], halo_mass_bins[i+1]))
            #
            data_dict_1['halo.inds'][real_halos_1*mass_mask_1] = orbit_class_1.sub_inds[:,600-snap][real_halos_1*mass_mask_1]
            data_dict_1['halo.bin'][real_halos_1*mass_mask_1] = i
            #
            data_dict_2['halo.inds'][real_halos_2*mass_mask_2] = orbit_class_2.sub_inds[:,600-snap][real_halos_2*mass_mask_2]
            data_dict_2['halo.bin'][real_halos_2*mass_mask_2] = i
            #
            # If there are halos, continue
            if np.sum(mass_mask_1) != 0:
                # Get the halo positions and max halo radius in the mass bin
                halo_pos_1 = halt['position'][orbit_class_1.sub_inds[:,600-snap][real_halos_1*mass_mask_1]]
                dmax_1 = np.around(np.max(halt['radius'][orbit_class_1.sub_inds[:,600-snap][real_halos_1*mass_mask_1]]))
                #
                # Query the particle tree and save the distances and indices
                ndist_1, nind_1 = orbit_tree.neighbors(centers=halo_pos_1, neigh_num_max=1e6, neigh_dist_max=dmax_1, workerss=4)
                #
                # Loop over the number of halos
                for j in range(0, len(halo_pos_1)):
                    # Find the particles within +/- 5 kpc of the halo radius, then save the potential and particle number
                    part_mask_1 = (ndist_1[j][np.isfinite(ndist_1[j])] < (halt['radius'][orbit_class_1.sub_inds[:,600-snap][real_halos_1*mass_mask_1]][j]+5))*(ndist_1[j][np.isfinite(ndist_1[j])] > (halt['radius'][orbit_class_1.sub_inds[:,600-snap][real_halos_1*mass_mask_1]][j]-5))
                    data_dict_1['subhalo.potential'][temp_1[real_halos_1*mass_mask_1][j]] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind_1[j][np.isfinite(ndist_1[j])][part_mask_1]])
                    data_dict_1['particle.num'][temp_1[real_halos_1*mass_mask_1][j]] = np.sum(part_mask_1)
            #
            if np.sum(mass_mask_2) != 0:
                # Get the halo positions and max halo radius in the mass bin
                halo_pos_2 = halt['position'][orbit_class_2.sub_inds[:,600-snap][real_halos_2*mass_mask_2]]
                dmax_2 = np.around(np.max(halt['radius'][orbit_class_2.sub_inds[:,600-snap][real_halos_2*mass_mask_2]]))
                #
                # Query the particle tree and save the distances and indices
                ndist_2, nind_2 = orbit_tree.neighbors(centers=halo_pos_2, neigh_num_max=1e6, neigh_dist_max=dmax_2, workerss=4)
                #
                # Loop over the number of halos
                for j in range(0, len(halo_pos_2)):
                    # Find the particles within +/- 5 kpc of the halo radius, then save the potential and particle number
                    part_mask_2 = (ndist_2[j][np.isfinite(ndist_2[j])] < (halt['radius'][orbit_class_2.sub_inds[:,600-snap][real_halos_2*mass_mask_2]][j]+5))*(ndist_2[j][np.isfinite(ndist_2[j])] > (halt['radius'][orbit_class_2.sub_inds[:,600-snap][real_halos_2*mass_mask_2]][j]-5))
                    data_dict_2['subhalo.potential'][temp_2[real_halos_2*mass_mask_2][j]] = np.mean(part['dark']['potential'][::orbit_tree.subsampling][nind_2[j][np.isfinite(ndist_2[j])][part_mask_2]])
                    data_dict_2['particle.num'][temp_2[real_halos_2*mass_mask_2][j]] = np.sum(part_mask_2)
            #
            # If no halos, say so
            else:
                print('No halos between {0} and {1}'.format(halo_mass_bins[i],halo_mass_bins[i+1]))
            end = time.time()
            print('Done with mass bin {0} in {1} seconds'.format(i, end-start))
        #
        ut.io.file_hdf5(file_name_base=simdata.home_dir+'/orbit_data/hdf5_files/potentials/all_snapshots/'+simdata.gal_1+'/'+simdata.gal_1+'_potentials_'+str(snap), dict_or_array_to_write=data_dict_1, verbose=True)
        ut.io.file_hdf5(file_name_base=simdata.home_dir+'/orbit_data/hdf5_files/potentials/all_snapshots/'+simdata.gal_2+'/'+simdata.gal_2+'_potentials_'+str(snap), dict_or_array_to_write=data_dict_2, verbose=True)
        print('Completely finished writing snapshot {0}'.format(snap))


print('Setting up arguments')
# Create an array of arguments for the function above
args_list = [(snapshot, sim_data, orbits) for snapshot in snaps]

print('Starting to run on data')
# Run the function using the arguments above in parallel
ut.io.run_in_parallel(calc_sub_potential, args_list, proc_number=3, verbose=True) # ADD VERBOSE

print('All done.')
