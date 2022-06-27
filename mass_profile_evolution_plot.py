#!/usr/bin/python3

"""

  ===============================
  = Mass Profile Evolution Plot =
  ===============================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  Plot the mass profile at given snapshots to the average mass profile to see
  how it has changed over the last Gyr.

  Uses data calculated from "mass_profile_evolution.py"

"""
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
sim_data = orbit_io.OrbitRead(gal1='m12b', location='mac')
#sim_data.galaxy = 'Remus' # this is only necessary for LG pairs
print('Set paths')

# Read in the data
mass_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'_full_mass_profile')
rs = np.logspace(np.log10(0.1), np.log10(500), 100)


"""
    Average over the last 2 Gyrs
"""
mass_prof_avg_2G = np.average(np.cumsum(mass_data['mass.profile'][:21], axis=1), axis=0)

inds = np.arange(len(mass_data['time'][:21]))
times = np.around(13.8-mass_data['time'][:21], decimals=2)

def color_cycle(cycle_length=len(mass_data['time'][:21]), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(mass_data['time'][:21]), cmap_name='plasma', low=0, high=1)

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in inds:
    plt.plot(rs[1:], np.cumsum(mass_data['mass.profile'][i])/mass_prof_avg_2G, color=colorss[i])
plt.xlim(5, 350)
plt.ylim(0.95, 1.06)
#plt.ylim(0.96,1.07)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm avg, 2\ Gyr}$(<r)', fontsize=32)
plt.title(sim_data.galaxy+', 0.1 Gyr spacing', fontsize=32)
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
inds = np.concatenate((np.arange(len(mass_data['time']))[:21][::5], np.arange(len(mass_data['time']))[21:39]))
times = np.around(13.8-np.concatenate((mass_data['time'][:21][::5], mass_data['time'][21:39])), decimals=2)

def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=1)

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in range(0, len(inds)):
    plt.plot(rs[1:], np.cumsum(mass_data['mass.profile'][inds[i]])/np.cumsum(mass_data['mass.profile'][0]), color=colorss[i])
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


#####################################################################################


import orbit_io
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
print('Set paths')

# Read in the data
mass_m12b = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12b_full_mass_profile')
mass_m12c = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12c_full_mass_profile')
mass_m12f = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12f_full_mass_profile')
mass_m12i = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12i_full_mass_profile')
mass_m12m = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12m_full_mass_profile')
mass_m12r = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12r_full_mass_profile')
mass_m12w = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12w_full_mass_profile')
mass_m12z = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12z_full_mass_profile')
mass_Romeo = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romeo_full_mass_profile')
mass_Juliet = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Juliet_full_mass_profile')
mass_Thelma = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Thelma_full_mass_profile')
mass_Louise = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Louise_full_mass_profile')
mass_Romulus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romulus_full_mass_profile')
mass_Remus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Remus_full_mass_profile')
rs = np.logspace(np.log10(0.1), np.log10(500), 100)


"""
    Plot the median at each snapshot relative to the z = 0 mass profile
        - Want to take the ratio for each host before taking the medians
"""

# Cumulatively sum each and put them all in one array
mass_prof_all = np.array([np.cumsum(mass_m12b['mass.profile'],axis=1)/np.cumsum(mass_m12b['mass.profile'][0]),\
                          np.cumsum(mass_m12c['mass.profile'],axis=1)/np.cumsum(mass_m12c['mass.profile'][0]),\
                          np.cumsum(mass_m12f['mass.profile'],axis=1)/np.cumsum(mass_m12f['mass.profile'][0]),\
                          np.cumsum(mass_m12i['mass.profile'],axis=1)/np.cumsum(mass_m12i['mass.profile'][0]),\
                          np.cumsum(mass_m12m['mass.profile'],axis=1)/np.cumsum(mass_m12m['mass.profile'][0]),\
                          np.cumsum(mass_m12r['mass.profile'],axis=1)/np.cumsum(mass_m12r['mass.profile'][0]),\
                          np.cumsum(mass_m12w['mass.profile'],axis=1)/np.cumsum(mass_m12w['mass.profile'][0]),\
                          np.cumsum(mass_m12z['mass.profile'],axis=1)/np.cumsum(mass_m12z['mass.profile'][0]),\
                          np.cumsum(mass_Romeo['mass.profile'],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0]),\
                          np.cumsum(mass_Juliet['mass.profile'],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0]),\
                          np.cumsum(mass_Thelma['mass.profile'],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0]),\
                          np.cumsum(mass_Louise['mass.profile'],axis=1)/np.cumsum(mass_Louise['mass.profile'][0]),\
                          np.cumsum(mass_Romulus['mass.profile'],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0]),\
                          np.cumsum(mass_Remus['mass.profile'],axis=1)/np.cumsum(mass_Remus['mass.profile'][0])])

mass_prof_all_med = np.median(mass_prof_all, axis=0)

inds = np.concatenate((np.arange(len(mass_m12b['time']))[:21][::5], np.arange(len(mass_m12b['time']))[21:39]))
times = np.around(13.8-np.concatenate((mass_m12b['time'][:21][::5], mass_m12b['time'][21:39])), decimals=2)

def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=1)

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in range(0, len(inds)):
    plt.plot(rs[1:], mass_prof_all_med[i], color=colorss[i])
plt.xlim(5, 350)
plt.ylim(0.85, 1.03)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M_{\\rm median}(<r)$ / $M_{\\rm z = 0, median}$(<r)', fontsize=32)
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.set_label('Lookback time [Gyr]', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/mass_profile_evolution_z0_median.pdf')
plt.close()



"""
    Plot the median at each snapshot relative to the average profile over the last 2 Gyr
        - Want to take the ratio for each host before taking the medians
"""

# Cumulatively sum each and put them all in one array
mass_prof_avg_all = np.array([np.cumsum(mass_m12b['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12b['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12c['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12c['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12f['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12f['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12i['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12i['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12m['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12m['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12r['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12r['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12w['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12w['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12z['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12z['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Romeo['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Romeo['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Juliet['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Juliet['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Thelma['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Thelma['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Louise['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Louise['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Romulus['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Romulus['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Remus['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Remus['mass.profile'][:21], axis=1), axis=0)])

mass_prof_avg_all_med = np.median(mass_prof_avg_all, axis=0)

inds = np.arange(len(mass_m12b['time'][:21]))
times = np.around(13.8-mass_m12b['time'][:21], decimals=2)

def color_cycle(cycle_length=len(mass_m12b['time'][:21]), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(mass_m12b['time'][:21]), cmap_name='plasma', low=0, high=1)

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in inds:
    plt.plot(rs[1:], mass_prof_avg_all_med[i], color=colorss[i])
plt.xlim(5, 350)
plt.ylim(0.95, 1.05)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M_{\\rm median}(<r)$ / $M_{\\rm avg, 2\ Gyr}$(<r)', fontsize=32)
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.set_label('Lookback time [Gyr]', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/mass_profile_evolution_2Gyr_avg_median.pdf')
plt.close()






##### OLD AND DONT WANT TO KEEP ANYMORE BUT TALK TO ANDREW ABOUT THIS FIRST CAUSE HE MIGHT WANT THESE
# Sigma percents
onesigp = 84.13
onesigm = 15.87
#
twosigp = 97.72
twosigm = 2.28
#
thrsigp = 99.87
thrsigm = 0.13
#
upper_one = np.percentile((mass_prof_med_all/mass_prof_med_all[0]), onesigp, axis=0)
lower_one = np.percentile((mass_prof_med_all/mass_prof_med_all[0]), onesigm, axis=0)
upper_two = np.percentile((mass_prof_med_all/mass_prof_med_all[0]), twosigp, axis=0)
lower_two = np.percentile((mass_prof_med_all/mass_prof_med_all[0]), twosigm, axis=0)
upper_thr = np.percentile((mass_prof_med_all/mass_prof_med_all[0]), thrsigp, axis=0)
lower_thr = np.percentile((mass_prof_med_all/mass_prof_med_all[0]), thrsigm, axis=0)

colorss = ['#93E9BE', '#008080', '#0080FF']

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
plt.fill_between(rs[1:], upper_thr, lower_thr, color='#9966cc', alpha=0.3)
plt.fill_between(rs[1:], upper_two, lower_two, color='#9966cc', alpha=0.5)
plt.fill_between(rs[1:], upper_one, lower_one, color='#9966cc', alpha=1)
plt.xlim(5, 350)
plt.ylim(0, 1.2)
plt.hlines(1, 0.1, 500, color='k', alpha=1, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
#plt.ylabel('$M$(<r) / $M_{\\rm avg}$(<r)', fontsize=32)
plt.ylabel('$M$(<r) / $M_{z=0}$(<r)', fontsize=32)
#plt.title('Average over last 11.5 Gyr', fontsize=32)
#plt.title('Average over last 500 Myr', fontsize=32)
#plt.title('Average over last 1 Gyr', fontsize=32)
plt.title(' ', fontsize=32)
#
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/mass_profile_evolution_avg_all_shaded.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/mass_profile_evolution_avg_500Myr_shaded.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/mass_profile_evolution_avg_1Gyr_shaded.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/mass_profile_evolution_avg_z0_shaded.pdf')
plt.close()
