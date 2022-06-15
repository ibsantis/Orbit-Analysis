#!/usr/bin/env python3
#SBATCH --job-name=m12i_orbits
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1    # OpenMP threads per MPI task
#SBATCH --time=0:05:00
#SBATCH --output=/home/ibsantis/scripts/jobs/animations/checking_data_read_in_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
#from IPython.display import HTML
from celluloid import Camera
#%matplotlib qt
import utilities as ut
import orbit_io

sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_m12i')
print('Read in the data')
data['d.sim'][data['d.sim'] == -1] = np.nan
print('Done converting null values')
for i in range(0, len(data['d.sim'])):
    data['d.sim'][i][(data['first.infall.snap'][i]+1):] = np.nan
print('Done setting null values prior to infall')
traj_X = np.flip(data['d.sim'][:,:,0], axis=1)
traj_Y = np.flip(data['d.sim'][:,:,1], axis=1)
traj_Z = np.flip(data['d.sim'][:,:,2], axis=1)
print('Finished setting up plotting arrays')
