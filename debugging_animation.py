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

# Display a single trajectory
R200m = data['host.radius'][0]+10
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
ax[0].set_xlabel('X [kpc]', fontsize=20)
ax[0].set_ylabel('Y [kpc]', fontsize=20)
#
ax[1].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
ax[1].set_xlabel('X [kpc]', fontsize=20)
ax[1].set_ylabel('Z [kpc]', fontsize=20)
plt.suptitle(sim_data.galaxy+' satellites', fontsize=28)
#
# Set up the colors
colorss = np.array(['#ff0000','#c71585','#40e0d0','#00ff00','#0000ff','#1e90ff'])
#
ax[0].plot(np.nan, np.nan, marker='o', markersize=3, markeredgecolor=colorss[0], markerfacecolor=colorss[0], alpha=0.5, label='$M_{\\rm star} < 10^5 M_{\\odot}$')
ax[0].plot(np.nan, np.nan, marker='o', markersize=4, markeredgecolor=colorss[1], markerfacecolor=colorss[1], alpha=0.5, label='$M_{\\rm star} = [10^5,10^6] M_{\\odot}$')
ax[0].plot(np.nan, np.nan, marker='o', markersize=5, markeredgecolor=colorss[2], markerfacecolor=colorss[2], alpha=0.5, label='$M_{\\rm star} = [10^6,10^7] M_{\\odot}$')
ax[0].plot(np.nan, np.nan, marker='o', markersize=6, markeredgecolor=colorss[3], markerfacecolor=colorss[3], alpha=0.5, label='$M_{\\rm star} = [10^7,10^8] M_{\\odot}$')
ax[0].plot(np.nan, np.nan, marker='o', markersize=7, markeredgecolor=colorss[4], markerfacecolor=colorss[4], alpha=0.5, label='$M_{\\rm star} = [10^8,10^9] M_{\\odot}$')
ax[0].plot(np.nan, np.nan, marker='o', markersize=8, markeredgecolor=colorss[5], markerfacecolor=colorss[5], alpha=0.5, label='$M_{\\rm star} = [10^9,10^{10}] M_{\\odot}$')
ax[0].plot(np.nan, np.nan, marker='o', markersize=9, markeredgecolor='k', markerfacecolor='k', alpha=0.5, label='$M_{\\rm star} > 10^{10} M_{\\odot}$')
ax[0].legend(prop={'size': 16}, loc='best')
print('Finished setting up axes')
