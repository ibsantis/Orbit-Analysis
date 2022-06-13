#!/usr/bin/env python3
#SBATCH --job-name=m12i_orbits
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=480G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1    # OpenMP threads per MPI task
#SBATCH --time=2:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/animations/m12i_orbits_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from celluloid import Camera
#%matplotlib qt
import utilities as ut
import orbit_io

sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_m12i')
data['d.sim'][data['d.sim'] == -1] = np.nan
for i in range(0, len(data['d.sim'])):
    data['d.sim'][i][(data['first.infall.snap'][i]+1):] = np.nan
traj_X = np.flip(data['d.sim'][:,:,0], axis=1)
traj_Y = np.flip(data['d.sim'][:,:,1], axis=1)
traj_Z = np.flip(data['d.sim'][:,:,2], axis=1)


# Display a single trajectory
R200m = data['host.radius'][0]+5
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
ax[0].set_xlabel('X', fontsize=15)
ax[0].set_ylabel('Y', fontsize=15)
#
ax[1].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
ax[1].set_xlabel('X', fontsize=15)
ax[1].set_ylabel('Z', fontsize=15)
plt.title(sim_data.galaxy, fontsize=28)


# Initiate camera
camera = Camera(fig)

"""
# Create individual frames
for j in range(1,traj_X.shape[1]+1):

    # Projectile's trajectory
    x32 = traj_X[32][0:j]
    y32 = traj_Y[32][0:j]
    z32 = traj_Z[32][0:j]
    x34 = traj_X[34][0:j]
    y34 = traj_Y[34][0:j]
    z34 = traj_Z[34][0:j]

    # Show Projectile's location
    #ax[0].plot(x32[-1], y32[-1], marker='o', markersize=3, markeredgecolor='b', markerfacecolor='b', alpha=0.5)
    #ax[0].plot(x34[-1], y34[-1], marker='o', markersize=3, markeredgecolor='g', markerfacecolor='g', alpha=0.5)
    ax[0].plot(xs[])
    ax[0].plot(0, 0, marker='x', markersize=10, markeredgecolor='k', markerfacecolor='k', alpha=0.5)
    #
    ax[1].plot(x32[-1], z32[-1], marker='o', markersize=3, markeredgecolor='b', markerfacecolor='b', alpha=0.5)
    ax[1].plot(x34[-1], z34[-1], marker='o', markersize=3, markeredgecolor='g', markerfacecolor='g', alpha=0.5)
    ax[1].plot(0, 0, marker='x', markersize=10, markeredgecolor='k', markerfacecolor='k', alpha=0.5)

    # Show Projectile's trajectory
    ax[0].plot(x32, y32, color='k', lw=1, linestyle='--', alpha=0.2)
    ax[0].plot(x34, y34, color='k', lw=1, linestyle='--', alpha=0.2)
    #
    ax[1].plot(x32, z32, color='k', lw=1, linestyle='--', alpha=0.2)
    ax[1].plot(x34, z34, color='k', lw=1, linestyle='--', alpha=0.2)

    # Show the time
    ax[0].text(-200, 250, 't = '+str(np.around(data['time.sim'][j], 2))+' Gyr')
    #ax[1].text(-200, 250, 't = '+str(np.around(data['time.sim'][j], 2))+' Gyr')

    # Capture frame
    camera.snap()
"""

for j in range(1,traj_X.shape[1]+1):
    #
    # Projectile's trajectory
    xs = traj_X[data['infall.check']][:,:j]
    ys = traj_Y[data['infall.check']][:,:j]
    zs = traj_Z[data['infall.check']][:,:j]
    #
    ax[0].plot(0, 0, marker='x', markersize=10, markeredgecolor='k', markerfacecolor='k', alpha=0.5)
    ax[1].plot(0, 0, marker='x', markersize=10, markeredgecolor='k', markerfacecolor='k', alpha=0.5)
    #
    ax[0].text(-200, 250, 't = '+str(np.around(data['time.sim'][j], 2))+' Gyr')
    #
    for i in range(0, len(xs)):
        # Show Projectile's location
        ax[0].plot(xs[i][-1], ys[i][-1], marker='o', markersize=3, markeredgecolor='b', markerfacecolor='b', alpha=0.5)
        ax[1].plot(xs[i][-1], zs[i][-1], marker='o', markersize=3, markeredgecolor='b', markerfacecolor='b', alpha=0.5)
        #
        # Show Projectile's trajectory
        ax[0].plot(xs[i], ys[i], color='k', lw=1, linestyle='--', alpha=0.2)
        ax[1].plot(xs[i], zs[i], color='k', lw=1, linestyle='--', alpha=0.2)
        #
        # Capture frame
        camera.snap()
    print('Done with t = {}'.format(str(np.around(data['time.sim'][j], 2))))


# Create animation
anim = camera.animate(interval = 50, repeat = True, repeat_delay = 500)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0)

writergif = animation.PillowWriter(fps=60)
anim.save(sim_data.home_dir+'/orbit_data/animations/test_m12i.gif',writer=writergif)

# Inline display
#HTML(anim.to_html5_video())
