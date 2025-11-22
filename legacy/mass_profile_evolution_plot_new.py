from orbit_analysis import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
#sim_data.galaxy = 'Remus' # this is only necessary for LG pairs
print('Set paths')

# Read in the data
mass_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'_mass_profile_all_new')
rs = np.array([  0.        ,   5.        ,   6.05763829,   7.33899634,
         8.89139705,  10.77217345,  13.05078608,  15.8113883 ,
        19.15593425,  23.20794417,  28.11706626,  34.06460345,
        41.27020926,  50.        ,  60.57638293,  73.38996338,
        88.9139705 , 100.        , 107.7217345 , 130.50786078,
       150., 158.11388301, 191.55934248, 232.07944168, 281.1706626 ,
       340.64603453, 412.70209263, 500.        ])


"""
    Average over the last 2 Gyrs
"""
avg_mask = (mass_data['time'][-1]-mass_data['time'] < 2)
mass_prof_avg_2G = np.average(mass_data['mass.profile'][avg_mask], axis=0)

inds = np.arange(len(mass_data['time'][avg_mask]))
times = np.around(13.8-mass_data['time'][avg_mask], decimals=2)

def color_cycle(cycle_length=len(mass_data['time'][avg_mask]), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(mass_data['time'][avg_mask]), cmap_name='plasma', low=0, high=1)

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in inds:
    plt.plot(rs[1:], np.flip(mass_data['mass.profile'], axis=0)[i]/mass_prof_avg_2G, color=colorss[i])
plt.xlim(5, 350)
#plt.ylim(0.95, 1.06)
#plt.ylim(0.96, 1.07)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm avg, 2\ Gyr}$(<r)', fontsize=32)
plt.title(sim_data.galaxy, fontsize=32)
cmap = plt.get_cmap('plasma', len(mass_data['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.set_label('Lookback time [Gyr]', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/individual/'+sim_data.galaxy+'_mass_profile_evolution_2Gyr_avg.pdf')
plt.close()


"""
    Snapshots relative to z = 0
"""
ideal_times = np.linspace(2.8, 13.8, 23)
inds = np.zeros(len(ideal_times), int)
times = np.zeros(len(ideal_times))
for i in range(0, len(ideal_times)):
    inds[i] = np.where(np.min(np.abs(ideal_times[i]-mass_data['time'])) == np.abs(ideal_times[i]-mass_data['time']))[0]
    times[i] = np.around(13.8-mass_data['time'][inds[i]], decimals=2)

def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=1)

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in range(0, len(inds)):
    plt.plot(rs[1:], mass_data['mass.profile'][inds[i]]/mass_data['mass.profile'][-1], color=np.flip(colorss, axis=0)[i])
plt.xlim(5, 350)
plt.ylim(0, 1.2)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm z = 0}$(<r)', fontsize=32)
plt.title(sim_data.galaxy+', 0.5 Gyr spacing', fontsize=32)
cmap = plt.get_cmap('plasma', len(mass_data['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.set_label('Lookback time [Gyr]', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/individual/'+sim_data.galaxy+'_mass_profile_evolution_z0.pdf')
plt.close()



plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
plt.plot(rs[1:], mass_data['mass.profile'][599-148]/mass_data['mass.profile'][-1], color='k', label='3.3 Gyr ago')
plt.plot(rs[1:], mass_data['mass.profile'][599-178]/mass_data['mass.profile'][-1], color='b', label='4.1 Gyr ago')
plt.xlim(5, 350)
plt.ylim(0, 1.2)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm z = 0}$(<r)', fontsize=32)
#plt.title(sim_data.galaxy+', 0.5 Gyr spacing', fontsize=32)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/individual/'+sim_data.galaxy+'_sub_10_check_at_100kpc.pdf')
plt.close()
