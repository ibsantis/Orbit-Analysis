#!/usr/bin/env python3
#SBATCH --job-name=TL_potential
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=450G
#SBATCH --nodes=1
#SBATCH --ntasks=1    # processes total
#SBATCH --time=04:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/potentials/TL_potential_%j.txt
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

"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Thelma', location='peloton')
print('Set paths')

if sim_data.num_gal == 1:

    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
    halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
    part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, properties=['position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    print('Particles at z = 0 read in')

    # Get the mean + median potential at 100 +/- 5 kpc in the host galaxy using all particles
    star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [95, 105])
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [95, 105])
    dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [95, 105])
    potential_host_100 = np.mean(np.concatenate((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
    #potential_median = np.median(np.concatenate((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
    #potential_mean_dark = np.mean(part['dark']['potential'][dark_inds])

    # Get the mean potential at R_200
    host_R200 = halt['radius'][halt['host.index'][0]]
    #
    star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [host_R200-5, host_R200+5])
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [host_R200-5, host_R200+5])
    dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [host_R200-5, host_R200+5])
    potential_host_R200 = np.mean(np.concatenate((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))

    import time
    start = time.time()
    # Find the potential in the outer regions of the subhalos
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1)
    #
    sub_potentials = np.zeros(orbits.shape[0])
    for i in range(0, orbits.shape[0]):
        inds = ut.array.get_indices(np.linalg.norm(part['dark']['position']-halt['position'][orbits.sub_inds[i][0]],axis=1), [halt['radius'][orbits.sub_inds[i][0]]-5, halt['radius'][orbits.sub_inds[i][0]]+5])
        sub_potentials[i] = np.mean(part['dark']['potential'][inds])
    end = time.time()
    print('Done with {0} subhalo potentials in {1} seconds'.format(orbits.shape[0], end-start))

    # Make plots showing the difference of the subhalo potential with the host at the two checks
    plt.figure(figsize=(10,8))
    plt.scatter(sub_potentials/10000, (potential_host_100-sub_potentials)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    plt.xlabel('$\Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, 100\ kpc} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.galaxy, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.galaxy+'_potential_100kpc.pdf')
    plt.close()
    #
    plt.figure(figsize=(10,8))
    plt.scatter(sub_potentials/10000, (potential_host_R200-sub_potentials)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    plt.xlabel('$\Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, R200} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.galaxy, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.galaxy+'_potential_R200.pdf')
    plt.close()

    # Make plots of the difference vs distance
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    plt.scatter(halt.prop('host.distance.total', orbits.sub_inds[:,0]), (potential_host_100-sub_potentials)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    ax.axvspan(95,105, color='k', alpha=0.3)
    plt.xlabel('$d_{\\rm host}$ [kpc]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, 100\ kpc} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.galaxy, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.galaxy+'_potential_vs_d_100kpc.pdf')
    plt.close()
    #
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    plt.scatter(halt.prop('host.distance.total', orbits.sub_inds[:,0]), (potential_host_R200-sub_potentials)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    ax.axvspan(95,105, color='k', alpha=0.3)
    plt.xlabel('$d_{\\rm host}$ [kpc]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, R200} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.galaxy, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.galaxy+'_potential_vs_d_R200.pdf')
    plt.close()

    # Save the data to a file
    data_dict = dict()
    data_dict['host.potential.100kpc'] = potential_host_100
    data_dict['host.potential.R200m'] = potential_host_R200
    data_dict['subhalo.potential'] = sub_potentials
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.galaxy+'_potentials', dict_or_array_to_write=data_dict, verbose=True)

if sim_data.num_gal == 2:

    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
    halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
    part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, properties=['position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    print('Particles at z = 0 read in')
    #
    ### GALAXY 1
    # Get the mean + median potential at 100 +/- 5 kpc in the host galaxy using all particles
    star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [95, 105])
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [95, 105])
    dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [95, 105])
    potential_host_1_100 = np.mean(np.concatenate((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
    #
    # Get the mean potential at R_200
    host_1_R200 = halt['radius'][halt['host.index'][0]]
    #
    star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [host_1_R200-5, host_1_R200+5])
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [host_1_R200-5, host_1_R200+5])
    dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [host_1_R200-5, host_1_R200+5])
    potential_host_1_R200 = np.mean(np.concatenate((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
    #
    import time
    start = time.time()
    # Find the potential in the outer regions of the subhalos
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1)
    #
    sub_potentials_1 = np.zeros(orbits.shape[0])
    for i in range(0, orbits.shape[0]):
        inds = ut.array.get_indices(np.linalg.norm(part['dark']['position']-halt['position'][orbits.sub_inds[i][0]],axis=1), [halt['radius'][orbits.sub_inds[i][0]]-5, halt['radius'][orbits.sub_inds[i][0]]+5])
        sub_potentials_1[i] = np.mean(part['dark']['potential'][inds])
    end = time.time()
    print('Done with {0} subhalo potentials in {1} seconds'.format(orbits.shape[0], end-start))
    #
    # Make plots showing the difference of the subhalo potential with the host at the two checks
    plt.figure(figsize=(10,8))
    plt.scatter(sub_potentials_1/10000, (potential_host_1_100-sub_potentials_1)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    plt.xlabel('$\Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, 100\ kpc} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_1, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_1+'_potential_100kpc.pdf')
    plt.close()
    #
    plt.figure(figsize=(10,8))
    plt.scatter(sub_potentials_1/10000, (potential_host_1_R200-sub_potentials_1)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    plt.xlabel('$\Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, R200} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_1, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_1+'_potential_R200.pdf')
    plt.close()
    #
    # Make plots of the difference vs distance
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    plt.scatter(halt.prop('host.distance.total', orbits.sub_inds[:,0]), (potential_host_1_100-sub_potentials_1)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    ax.axvspan(95,105, color='k', alpha=0.3)
    plt.xlabel('$d_{\\rm host}$ [kpc]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, 100\ kpc} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_1, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_1+'_potential_vs_d_100kpc.pdf')
    plt.close()
    #
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    plt.scatter(halt.prop('host.distance.total', orbits.sub_inds[:,0]), (potential_host_1_R200-sub_potentials_1)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    ax.axvspan(95,105, color='k', alpha=0.3)
    plt.xlabel('$d_{\\rm host}$ [kpc]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, R200} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_1, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_1+'_potential_vs_d_R200.pdf')
    plt.close()
    #
    # Save the data to a file
    data_dict = dict()
    data_dict['host.potential.100kpc'] = potential_host_1_100
    data_dict['host.potential.R200m'] = potential_host_1_R200
    data_dict['subhalo.potential'] = sub_potentials_1
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.gal_1+'_potentials', dict_or_array_to_write=data_dict, verbose=True)
    #
    ### GALAXY 2
    # Get the mean + median potential at 100 +/- 5 kpc in the host galaxy using all particles
    star_inds = ut.array.get_indices(part['star'].prop('host2.distance.total'), [95, 105])
    gas_inds = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [95, 105])
    dark_inds = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [95, 105])
    potential_host_2_100 = np.mean(np.concatenate((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
    #
    # Get the mean potential at 2*R_200
    host_2_R200 = halt['radius'][halt['host2.index'][0]]
    #
    star_inds = ut.array.get_indices(part['star'].prop('host2.distance.total'), [host_2_R200-5, host_2_R200+5])
    gas_inds = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [host_2_R200-5, host_2_R200+5])
    dark_inds = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [host_2_R200-5, host_2_R200+5])
    potential_host_2_R200 = np.mean(np.concatenate((part['star']['potential'][star_inds], part['gas']['potential'][gas_inds], part['dark']['potential'][dark_inds])))
    #
    import time
    start = time.time()
    # Find the potential in the outer regions of the subhalos
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=2)
    #
    sub_potentials_2 = np.zeros(orbits.shape[0])
    for i in range(0, orbits.shape[0]):
        inds = ut.array.get_indices(np.linalg.norm(part['dark']['position']-halt['position'][orbits.sub_inds[i][0]],axis=1), [halt['radius'][orbits.sub_inds[i][0]]-5, halt['radius'][orbits.sub_inds[i][0]]+5])
        sub_potentials_2[i] = np.mean(part['dark']['potential'][inds])
    end = time.time()
    print('Done with {0} subhalo potentials in {1} seconds'.format(orbits.shape[0], end-start))
    #
    # Make plots showing the difference of the subhalo potential with the host at the two checks
    plt.figure(figsize=(10,8))
    plt.scatter(sub_potentials_2/10000, (potential_host_2_100-sub_potentials_2)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    plt.xlabel('$\Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, 100\ kpc} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_2, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_2+'_potential_100kpc.pdf')
    plt.close()
    #
    plt.figure(figsize=(10,8))
    plt.scatter(sub_potentials_2/10000, (potential_host_2_R200-sub_potentials_2)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    plt.xlabel('$\Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, R200} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_2, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_2+'_potential_R200.pdf')
    plt.close()
    #
    # Make plots of the difference vs distance
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    plt.scatter(halt.prop('host2.distance.total', orbits.sub_inds[:,0]), (potential_host_2_100-sub_potentials_2)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    ax.axvspan(95,105, color='k', alpha=0.3)
    plt.xlabel('$d_{\\rm host}$ [kpc]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, 100\ kpc} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_2, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_2+'_potential_vs_d_100kpc.pdf')
    plt.close()
    #
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    plt.scatter(halt.prop('host2.distance.total', orbits.sub_inds[:,0]), (potential_host_2_R200-sub_potentials_2)/10000, s=30, facecolors='None', edgecolors='k', marker='o', alpha=0.6)
    ax.axvspan(95,105, color='k', alpha=0.3)
    plt.xlabel('$d_{\\rm host}$ [kpc]', fontsize=28)
    plt.ylabel('$\Phi_{\\rm host, R200} - \Phi_{\\rm sub}$ [$10^4$ km$^2$ s$^{-2}$]', fontsize=28)
    plt.title(sim_data.gal_2, fontsize=28)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/potential/'+sim_data.gal_2+'_potential_vs_d_R200.pdf')
    plt.close()
    #
    # Save the data to a file
    data_dict = dict()
    data_dict['host.potential.100kpc'] = potential_host_2_100
    data_dict['host.potential.R200m'] = potential_host_2_R200
    data_dict['subhalo.potential'] = sub_potentials_2
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.gal_2+'_potentials', dict_or_array_to_write=data_dict, verbose=True)
