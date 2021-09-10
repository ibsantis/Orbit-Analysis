#!/usr/bin/env python3
#SBATCH --job-name=m12i_potential_test
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
##SBATCH --mem=480G
#SBATCH --nodes=1
#SBATCH --ntasks=1    # processes total
#SBATCH --time=08:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/potentials/m12i_potential_test_%j.txt
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
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')
#
if sim_data.num_gal == 1:
    #
    # Read in the snapshot dictionary, halo tree, and z = 0 snapshot
    snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
    halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
    part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, properties=['position', 'potential'], simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True)
    print('Particles at z = 0 read in')
    #
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1)
    #
    halo_pos = halt['position'][orbits.sub_inds[:,0]]
    part_pos = part['dark']['position'][::2].astype(np.float32)
    max_dist = np.max(halt['radius'][orbits.sub_inds[:,0]])
    #
    ptree = ut.coordinate.get_neighbors(center_positions=halo_pos, neig_positions=part_pos, neig_distance_max=max_dist, exclude_self=False, workers=4)
    #
    data_dict = dict()
    data_dict['distances'] = ptree[0]
    data_dict['indices'] = ptree[1]
    #
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.galaxy+'_potential_test', dict_or_array_to_write=data_dict, verbose=True)

print('all done')
