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
import summary_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
print('Set paths')

# Read in the data
mass_m12b = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12b_full_mass_profile')
mass_m12c = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12c_full_mass_profile')
mass_m12f = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12f_full_mass_profile')
mass_m12i = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12i_full_mass_profile')
mass_m12m = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12m_full_mass_profile')
#mass_m12r = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12r_full_mass_profile')
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
    plt.plot(rs[1:], mass_prof_all_med[inds[i]], color=colorss[i])
plt.xlim(5, 350)
plt.ylim(0, 1.05)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M(<r, z)$ / $M(<r, z=0)$', fontsize=32)
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.set_label('Lookback time [Gyr]', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_z0_median.pdf')
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
    plt.plot(rs[1:], mass_prof_avg_all_med[inds[i]], color=colorss[i])
plt.xlim(5, 350)
plt.ylim(0.95, 1.05)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M(<r,z)$ / $M_{\\rm 2\ Gyr\ avg}(<r)$', fontsize=32)
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.set_label('Lookback time [Gyr]', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_2Gyr_avg_median.pdf')
plt.close()

# Plotting the 68%, 95%, and 100% scatters across all hosts and times for each distance
onesigp = 84.13
onesigm = 15.87
#
twosigp = 97.72
twosigm = 2.28
#
thrsigp = 100
thrsigm = 0
#
upper_one = np.percentile(mass_prof_avg_all, onesigp, axis=(0,1))
lower_one = np.percentile(mass_prof_avg_all, onesigm, axis=(0,1))
upper_two = np.percentile(mass_prof_avg_all, twosigp, axis=(0,1))
lower_two = np.percentile(mass_prof_avg_all, twosigm, axis=(0,1))
upper_thr = np.percentile(mass_prof_avg_all, thrsigp, axis=(0,1))
lower_thr = np.percentile(mass_prof_avg_all, thrsigm, axis=(0,1))

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
plt.fill_between(rs[1:], upper_thr, lower_thr, color='#9966cc', alpha=0.15)
plt.fill_between(rs[1:], upper_two, lower_two, color='#9966cc', alpha=0.5)
plt.fill_between(rs[1:], upper_one, lower_one, color='#9966cc', alpha=1)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xlim(5, 350)
plt.ylim(0.8, 1.2)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M(<r, z)$ / $M_{\\rm 2\ Gyr\ avg}$(<r)', fontsize=32)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_2Gyr_shaded.pdf')
plt.close()


"""
    Plotting the enclosed mass within 100 kpc for each host at each snapshot
"""

mass_prof_all_100 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:81]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:81]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:81]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:81]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:81]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:81]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:81])])
#
times = np.around(13.8-mass_m12b['time'], decimals=2)
#
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in range(0, len(mass_prof_all_100)):
    plt.plot(times, mass_prof_all_100[i][:,-1], summary_plot.colors[i], label=summary_plot.host_names['all'][i])
plt.xlim(13, 0)
#plt.ylim(1e11, 2e12)
#plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Lookback time [Gyr]', fontsize=32)
plt.ylabel('$M(z)/M(z=0)$', fontsize=32)
plt.title('Mass enclosed within 100 kpc')
plt.legend(ncol=3)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_100kpc_ind.pdf')
plt.close()


## Same as above, but showing just the scatters
# Sigma percents
onesigp = 84.13
onesigm = 15.87
#
twosigp = 100
twosigm = 0
#
upper_one = np.percentile(mass_prof_all_100, onesigp, axis=0)[:,-1]
lower_one = np.percentile(mass_prof_all_100, onesigm, axis=0)[:,-1]
upper_two = np.percentile(mass_prof_all_100, twosigp, axis=0)[:,-1]
lower_two = np.percentile(mass_prof_all_100, twosigm, axis=0)[:,-1]
#
med = np.median(mass_prof_all_100, axis=0)

colorss = ['#93E9BE', '#008080']

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
plt.plot(times, med[:,-1], color='#9966cc', alpha=1)
plt.fill_between(times, upper_two, lower_two, color='#9966cc', alpha=0.3)
plt.fill_between(times, upper_one, lower_one, color='#9966cc', alpha=0.5)
plt.hlines(1, 13, 0, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xlim(13, 0)
#plt.ylim(0.95, 1.05)
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Lookback time [Gyr]', fontsize=32)
plt.ylabel('$M(z)/M(z=0)$', fontsize=32)
plt.title('Mass enclosed within 100 kpc')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_100kpc_scatter.pdf')
plt.close()



"""
    Plotting the enclosed mass within 50 kpc for each host at each snapshot
"""

mass_prof_all_50 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:73]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:73]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:73]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:73]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:73]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:73]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:73])])
#
times = np.around(13.8-mass_m12b['time'], decimals=2)
#
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
for i in range(0, len(mass_prof_all_50)):
    plt.plot(times, mass_prof_all_50[i][:,-1], summary_plot.colors[i], label=summary_plot.host_names['all'][i])
plt.xlim(13, 0)
#plt.ylim(1e11, 2e12)
#plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Lookback time [Gyr]', fontsize=32)
plt.ylabel('$M(z)/M(z=0)$', fontsize=32)
plt.title('Mass enclosed within 50 kpc')
plt.legend(ncol=3)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_50kpc_ind.pdf')
plt.close()


onesigp = 84.13
onesigm = 15.87
#
twosigp = 100
twosigm = 0
#
upper_one = np.percentile(mass_prof_all_50, onesigp, axis=0)[:,-1]
lower_one = np.percentile(mass_prof_all_50, onesigm, axis=0)[:,-1]
upper_two = np.percentile(mass_prof_all_50, twosigp, axis=0)[:,-1]
lower_two = np.percentile(mass_prof_all_50, twosigm, axis=0)[:,-1]
#
med = np.median(mass_prof_all_50, axis=0)

colorss = ['#93E9BE', '#008080']

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
plt.plot(times, med[:,-1], color='#9966cc', alpha=1)
plt.fill_between(times, upper_two, lower_two, color='#9966cc', alpha=0.3)
plt.fill_between(times, upper_one, lower_one, color='#9966cc', alpha=0.5)
plt.hlines(1, 13, 0, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xlim(13, 0)
#plt.ylim(0.95, 1.05)
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Lookback time [Gyr]', fontsize=32)
plt.ylabel('$M(z)/M(z=0)$', fontsize=32)
plt.title('Mass enclosed within 50 kpc')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_50kpc_scatter.pdf')
plt.close()




mass_prof_all_50 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:73]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:73]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:73]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:73]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:73]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:73]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:73])])


mass_prof_all_100 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:81]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:81]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:81]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:81]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:81]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:81]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:81])])

mass_prof_all_150 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:86]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:86]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:86]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:86]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:86]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:86]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:86])])



upper_one_50 = np.percentile(mass_prof_all_50, onesigp, axis=0)[:,-1]
lower_one_50 = np.percentile(mass_prof_all_50, onesigm, axis=0)[:,-1]
upper_two_50 = np.percentile(mass_prof_all_50, twosigp, axis=0)[:,-1]
lower_two_50 = np.percentile(mass_prof_all_50, twosigm, axis=0)[:,-1]
#
med_50 = np.median(mass_prof_all_50, axis=0)
med_100 = np.median(mass_prof_all_100, axis=0)
med_150 = np.median(mass_prof_all_150, axis=0)

colorss = ['#93E9BE', '#008080']
times = np.around(13.8-mass_m12b['time'], decimals=2)
#
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
plt.plot(times, med_50[:,-1], color=summary_plot.colors[1], alpha=1, label='50 kpc')
plt.fill_between(times, upper_two_50, lower_two_50, color=summary_plot.colors[1], alpha=0.15)
plt.fill_between(times, upper_one_50, lower_one_50, color=summary_plot.colors[1], alpha=0.3)
plt.plot(times, med_100[:,-1], color=summary_plot.colors[3], alpha=1, label='100 kpc')
plt.plot(times, med_150[:,-1], color=summary_plot.colors[6], alpha=1, label='150 kpc')
plt.hlines(1, 13, 0, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xlim(13, 0)
#plt.ylim(0.95, 1.05)
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Lookback time [Gyr]', fontsize=32)
plt.ylabel('$M(<d, z)$ / $M(<d, z=0)$', fontsize=32)
plt.legend(prop={'size': 24}, loc='best')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_evolution/mass_profile_evolution_enclosed.pdf')
plt.close()
