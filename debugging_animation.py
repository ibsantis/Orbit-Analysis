#!/usr/bin/env python3
#SBATCH --job-name=m12i_orbits
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1    # OpenMP threads per MPI task
#SBATCH --time=0:10:00
#SBATCH --output=/home/ibsantis/scripts/jobs/animations/checking_data_read_in_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#from IPython.display import HTML
from celluloid import Camera
#%matplotlib qt
import utilities as ut
import orbit_io
import time

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

# Set up the colors
colorss = np.array(['#ff0000','#c71585','#40e0d0','#00ff00','#0000ff','#1e90ff'])
print('Setting up color and size arrays')
cc = []
ss = []
for i in range(0, traj_X[data['infall.check']].shape[0]):
    if (data['M.star.z0'][data['infall.check']][i] < 1e5):
        cc.append(colorss[0])
        ss.append(3)
    elif (data['M.star.z0'][data['infall.check']][i] > 1e5)&(data['M.star.z0'][data['infall.check']][i] < 1e6):
        cc.append(colorss[1])
        ss.append(4)
    elif (data['M.star.z0'][data['infall.check']][i] > 1e6)&(data['M.star.z0'][data['infall.check']][i] < 1e7):
        cc.append(colorss[2])
        ss.append(5)
    elif (data['M.star.z0'][data['infall.check']][i] > 1e7)&(data['M.star.z0'][data['infall.check']][i] < 1e8):
        cc.append(colorss[3])
        ss.append(6)
    elif (data['M.star.z0'][data['infall.check']][i] > 1e8)&(data['M.star.z0'][data['infall.check']][i] < 1e9):
        cc.append(colorss[4])
        ss.append(7)
    elif (data['M.star.z0'][data['infall.check']][i] > 1e9)&(data['M.star.z0'][data['infall.check']][i] < 1e10):
        cc.append(colorss[5])
        ss.append(8)
    else:
        cc.append('k')
        ss.append(9)

# Set up the graph using Matplotlib
start = time.time()
R200m = data['host.radius'][0]+10
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
ax.set_xlabel('X [kpc]', fontsize=28)
ax.set_ylabel('Y [kpc]', fontsize=28)
#ax[1].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
#ax[1].set_xlabel('X [kpc]', fontsize=28)
#ax[1].set_ylabel('Z [kpc]', fontsize=28)

# Initiate camera
camera = Camera(fig)

# Create individual frames
for j in range(1,traj_X.shape[1]+1):
    for i in range(0, traj_X[data['infall.check']].shape[0]):
        pick_traj = i
        # Projectile's trajectory
        x = traj_X[data['infall.check']][pick_traj][0:j]
        y = traj_Y[data['infall.check']][pick_traj][0:j]
        z = traj_Z[data['infall.check']][pick_traj][0:j]
        #
        #Plot the host position
        ax.plot(0, 0, marker='x', color='k', markersize=9, alpha=0.5)
        #
        # Show Projectile's location
        ax.plot(x[-1], y[-1], marker='o', markersize=ss[i], markeredgecolor=cc[i], markerfacecolor=cc[i], alpha=0.5)

        # Show Projectile's trajectory
        ax.plot(x, y, color='b', lw=1, linestyle='--', alpha=0.15)

    # Capture frame
    camera.snap()

end = time.time()
print('Finished the loop in {0} seconds'.format(end-start))

# Create animation
start = end
ax.tick_params(axis='both', which='both', bottom=True, labelsize=20)
#ax[1].tick_params(axis='both', which='both', bottom=True, labelsize=20)
#plt.tight_layout()
#plt.subplots_adjust(wspace=0.3, hspace=0)
anim = camera.animate(interval = 40, repeat = True, repeat_delay = 500)
end = time.time()
print('Finished animating in {0} seconds'.format(end-start))


# Inline display
#start = end
#writergif = animation.PillowWriter(fps=30)
#anim.save(sim_data.home_dir+'/orbit_data/animations/test_m12i.gif', writer=writergif)
#end = time.time()
#print('Finished saving the file in {0} seconds'.format(end-start))
#HTML(anim.to_html5_video())
