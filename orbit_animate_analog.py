
"""
    ...
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from celluloid import Camera
import utilities as ut
import orbit_io
import time
import pandas as pd
import satellite_io
import matplotlib.patches as mpatches
print('Read in the tools')

print('Starting the script at {0}'.format(time.strftime("%H:%M:%S", time.localtime())))

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
sat_analysis = satellite_io.SatelliteAnalysis(gal1='m12i', location=loc)
host = 1
print('Set paths')

# Read in the snapshot dictionary and the entire tree
lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']


# Loop through all of the satellites and create a figure for each of them
plot_data = dict()
plot_data['Hosts'] = []
#
galaxy = 'Carina'
#
gal_data = sat_analysis.read_subhalo_matches(galaxy)
satellite_name = galaxy.replace(' ', '_')
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
plot_data = dict()
plot_data['Hosts'] = []
#
name = 'm12m'
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)
#
# get the matches for a simulation host
mask = (gal_data['Host'] == name)
# get an array of the tree indices that are matches in m12m
subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
# Get indices within the mini data to select properties
mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
#
if len(mini_data_match_inds != 0):
    plot_data['Hosts'].append(name)
    # Snapshot at match
    time_at_match = snaps['time'][gal_data['Snapshot at match'][mask]]
    #
    plot_data['orbit.distance.X.'+name] = (-1)*np.ones(mini_data['d.sim'][mini_data_match_inds][:,:,0].shape)
    plot_data['orbit.distance.Y.'+name] = (-1)*np.ones(mini_data['d.sim'][mini_data_match_inds][:,:,1].shape)
    plot_data['orbit.distance.Z.'+name] = (-1)*np.ones(mini_data['d.sim'][mini_data_match_inds][:,:,2].shape)
    #
    plot_data['orbit.distance.X.shift.'+name] = (-1)*np.ones(mini_data['d.sim'][mini_data_match_inds][:,:,0].shape)
    plot_data['orbit.distance.Y.shift.'+name] = (-1)*np.ones(mini_data['d.sim'][mini_data_match_inds][:,:,1].shape)
    plot_data['orbit.distance.Z.shift.'+name] = (-1)*np.ones(mini_data['d.sim'][mini_data_match_inds][:,:,2].shape)
    #
    for i in range(0, len(mini_data_match_inds)):
        #
        # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
        time_ind = snaps['index'][-1] - np.where(np.min(np.abs(time_at_match[i] - snaps['time'])) == np.abs(time_at_match[i] - snaps['time']))[0][0]
        plot_data['orbit.distance.X.'+name][i][time_ind:] = mini_data['d.sim'][mini_data_match_inds][i][:,0][time_ind:]
        plot_data['orbit.distance.Y.'+name][i][time_ind:] = mini_data['d.sim'][mini_data_match_inds][i][:,1][time_ind:]
        plot_data['orbit.distance.Z.'+name][i][time_ind:] = mini_data['d.sim'][mini_data_match_inds][i][:,2][time_ind:]
        #
        mask = (plot_data['orbit.distance.X.'+name][i] != -1)
        plot_data['orbit.distance.X.shift.'+name][i][:np.sum(mask)] = plot_data['orbit.distance.X.'+name][i][mask]
        plot_data['orbit.distance.Y.shift.'+name][i][:np.sum(mask)] = plot_data['orbit.distance.Y.'+name][i][mask]
        plot_data['orbit.distance.Z.shift.'+name][i][:np.sum(mask)] = plot_data['orbit.distance.Z.'+name][i][mask]
    #
    plot_data['orbit.time.lb.'+name] = snaps['time'][-1] - np.flip(snaps['time'])[:mini_data['d.tot.sim'][mini_data_match_inds].shape[1]]
#
plot_data['orbit.distance.X.shift.'+name][plot_data['orbit.distance.X.shift.'+name] == -1] = np.nan 
plot_data['orbit.distance.Y.shift.'+name][plot_data['orbit.distance.Y.shift.'+name] == -1] = np.nan 
plot_data['orbit.distance.Z.shift.'+name][plot_data['orbit.distance.Z.shift.'+name] == -1] = np.nan 
print('Done converting null values')

# Don't have to worry too much about pre-infall because they're already analogs!


# Set up the colors
colorss = np.array(['#ff0000','#c71585','#40e0d0','#00ff00','#0000ff','#1e90ff'])
print('Setting up color and size arrays')
cc = []
ss = []
for i in range(0, plot_data['orbit.distance.X.'+name].shape[0]):
    if (mini_data['M.halo.z0'][mini_data['infall.check']][i] < 1e7):
        cc.append(colorss[0])
        ss.append(3)
    elif (mini_data['M.halo.z0'][mini_data['infall.check']][i] > 1e7)&(mini_data['M.halo.z0'][mini_data['infall.check']][i] < 3.16e7):
        cc.append(colorss[1])
        ss.append(4)
    elif (mini_data['M.halo.z0'][mini_data['infall.check']][i] > 3.16e7)&(mini_data['M.halo.z0'][mini_data['infall.check']][i] < 1e8):
        cc.append(colorss[2])
        ss.append(5)
    elif (mini_data['M.halo.z0'][mini_data['infall.check']][i] > 1e8)&(mini_data['M.halo.z0'][mini_data['infall.check']][i] < 3.16e8):
        cc.append(colorss[3])
        ss.append(6)
    elif (mini_data['M.halo.z0'][mini_data['infall.check']][i] > 3.161e8)&(mini_data['M.halo.z0'][mini_data['infall.check']][i] < 1e9):
        cc.append(colorss[4])
        ss.append(7)
    elif (mini_data['M.halo.z0'][mini_data['infall.check']][i] > 1e9)&(mini_data['M.halo.z0'][mini_data['infall.check']][i] < 1e10):
        cc.append(colorss[5])
        ss.append(8)
    else:
        cc.append('k')
        ss.append(9)

# Set up the graph using Matplotlib
start = time.time()
R200m = mini_data['host.radius'][0]+10
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
ax[0].set_xlabel('X [kpc]', fontsize=28)
ax[0].set_ylabel('Y [kpc]', fontsize=28)
ax[0].set(xlim=((-1)*R200m, R200m), ylim=((-1)*R200m, R200m))
ax[0].set_xlabel('X [kpc]', fontsize=28)
ax[0].set_ylabel('Z [kpc]', fontsize=28)
plt.suptitle(sim_data.galaxy+' satellites', fontsize=28)
#

# Initiate camera
camera = Camera(fig)



########## DO A TEST RUN FOR JUST ONE SATELLITE!!!!
# Create individual frames
for j in range(1,plot_data['orbit.distance.X.shift.'+name].shape[1]+1):
    #ax[0].text(-200, 200,'T = {0} Gyr'.format(np.around(data['time.sim'][j], 2)))  ##### Don't want to do this because "t = 0" is different for each satellite and the times in the sanpshot array aren't spaced equally
    for i in range(0, plot_data['orbit.distance.X.shift.'+name].shape[0]):
        pick_traj = i
        # Projectile's trajectory
        x = plot_data['orbit.distance.X.shift.'+name][pick_traj][0:j]
        y = plot_data['orbit.distance.Y.shift.'+name][pick_traj][0:j]
        z = plot_data['orbit.distance.Z.shift.'+name][pick_traj][0:j]
        #
        #Plot the host position
        ax[0].plot(0, 0, marker='x', color='k', markersize=9, alpha=0.5)
        ax[1].plot(0, 0, marker='x', color='k', markersize=9, alpha=0.5)
        #
        # Show Projectile's location
        ax[0].plot(x[-1], y[-1], marker='o', markersize=ss[i], markeredgecolor=cc[i], markerfacecolor=cc[i], alpha=0.5)
        ax[1].plot(x[-1], z[-1], marker='o', markersize=ss[i], markeredgecolor=cc[i], markerfacecolor=cc[i], alpha=0.5)

        # Show Projectile's trajectory
        ax[0].plot(x, y, color='k', lw=1, linestyle='--', alpha=0.15)
        ax[1].plot(x, z, color='k', lw=1, linestyle='--', alpha=0.15)

    # Capture frame
    camera.snap()

end = time.time()
print('Finished the loop in {0} seconds'.format(end-start))

# Create animation
start = end
ax[0].tick_params(axis='both', which='both', bottom=True, labelsize=20)
ax[1].tick_params(axis='both', which='both', bottom=True, labelsize=20)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0)
anim = camera.animate(interval = 40, repeat = True, repeat_delay = 500)
end = time.time()
print('Finished animating in {0} seconds'.format(end-start))


# Inline display
start = end
writergif = animation.PillowWriter(fps=60)
anim.save(sim_data.home_dir+'/orbit_data/animations/'+sim_data.galaxy+'_'+galaxy+'_analogs.gif', writer=writergif)
end = time.time()
print('Finished saving the file in {0} seconds'.format(end-start))
print('Done with script at {0}'.format(time.strftime("%H:%M:%S", time.localtime())))
