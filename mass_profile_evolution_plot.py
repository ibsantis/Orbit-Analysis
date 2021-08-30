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
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
#sim_data.galaxy = 'Remus' # this is only necessary for LG pairs
print('Set paths')

# Read in the data
mass_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'/'+sim_data.galaxy+'_full_mass_profile')

#inds = np.concatenate((np.arange(len(mass_data['time']))[:21], np.arange(len(mass_data['time']))[21::2]))
inds = np.arange(len(mass_data['time']))[::2]
times = np.around(13.8-mass_data['time'], decimals=2)

# Cumulatively sum each array, then take the average over all of them
mass_prof_avg_500M = np.average(np.cumsum(mass_data['mass.profile'][:6], axis=1), axis=0)
mass_prof_avg_1G = np.average(np.cumsum(mass_data['mass.profile'][:11], axis=1), axis=0)

# Plot each snapshot mass profile to the average mass profile
rs = np.logspace(np.log10(0.1), np.log10(500), 100)
#colors = ['#800000', '#9A6324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', '#3cb44b', '#42d4f4', '#911eb4', '#f032e6']

def color_cycle(cycle_length=len(mass_data['time']), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(mass_data['time']), cmap_name='plasma', low=0, high=1)

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
for i in inds:
    plt.plot(rs[1:], np.cumsum(mass_data['mass.profile'][i])/mass_prof_avg_500M, color=colorss[i], label=str(times[i])+' Gyr ago')
plt.xlim(5, 350)
#plt.ylim(0.97, 1.04)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm avg, 500\ Myr}$(<r)', fontsize=32)
plt.title(sim_data.galaxy, fontsize=32)
plt.legend(prop={'size': 12}, ncol=4)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/'+sim_data.galaxy+'_mass_profile_evolution_500Myr_avg.pdf')


plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
#
for i in inds:
    plt.plot(rs[1:], np.cumsum(mass_data['mass.profile'][i])/np.cumsum(mass_data['mass.profile'][0]), color=colorss[i], label=str(times[i])+' Gyr ago')
plt.xlim(5, 350)
#plt.ylim(0.95, 1.02)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm z = 0}$(<r)', fontsize=32)
plt.title(sim_data.galaxy, fontsize=32)
plt.legend(prop={'size': 12}, ncol=4)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/'+sim_data.galaxy+'_mass_profile_evolution_z0.pdf')


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
mass_m12b = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12b_mass_profile_evolution')
mass_m12c = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12c_mass_profile_evolution')
mass_m12f = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12f_mass_profile_evolution')
mass_m12i = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12i_mass_profile_evolution')
mass_m12m = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12m_mass_profile_evolution')
mass_m12r = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12r_mass_profile_evolution')
mass_m12w = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12w_mass_profile_evolution')
mass_m12z = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12z_mass_profile_evolution')
mass_Romeo = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romeo_mass_profile_evolution')
mass_Juliet = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Juliet_mass_profile_evolution')
mass_Thelma = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Thelma_mass_profile_evolution')
mass_Louise = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Louise_mass_profile_evolution')
mass_Romulus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romulus_mass_profile_evolution')
mass_Remus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Remus_mass_profile_evolution')

# Cumulatively sum each and put them all in one array
mass_prof_all = np.array([np.cumsum(mass_m12b['mass.profile'],axis=1),\
                          np.cumsum(mass_m12c['mass.profile'],axis=1),\
                          np.cumsum(mass_m12f['mass.profile'],axis=1),\
                          np.cumsum(mass_m12i['mass.profile'],axis=1),\
                          np.cumsum(mass_m12m['mass.profile'],axis=1),\
                          np.cumsum(mass_m12r['mass.profile'],axis=1),\
                          np.cumsum(mass_m12w['mass.profile'],axis=1),\
                          np.cumsum(mass_m12z['mass.profile'],axis=1),\
                          np.cumsum(mass_Romeo['mass.profile'],axis=1),\
                          np.cumsum(mass_Juliet['mass.profile'],axis=1),\
                          np.cumsum(mass_Thelma['mass.profile'],axis=1),\
                          np.cumsum(mass_Louise['mass.profile'],axis=1),\
                          np.cumsum(mass_Romulus['mass.profile'],axis=1),\
                          np.cumsum(mass_Remus['mass.profile'],axis=1)])

# Take the average
mass_prof_med_all = np.median(mass_prof_all, axis=0)
mass_prof_avg_all = np.average(mass_prof_med_all, axis=0)

# Plot each snapshot mass profile to the average mass profile
rs = np.logspace(np.log10(0.1), np.log10(500), 100)
#
def color_cycle(cycle_length=len(mass_m12b['time']), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(mass_m12b['time']), cmap_name='plasma', low=0, high=1)
#
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
#
for i in range(0, len(mass_prof_med_all)):
    plt.plot(rs, mass_prof_med_all[i]/mass_prof_avg_all, color=colorss[i], label=str(mass_m12b['time'][i])+' Gyr')
plt.xlim(5, 500)
#plt.ylim(0.97, 1.04)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M_{\\rm med}$(<r) / $M_{\\rm med,avg}$(<r)', fontsize=32)
#plt.title(sim_data.galaxy, fontsize=32)
plt.legend(prop={'size': 16}, ncol=2)
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/'+sim_data.galaxy+'_mass_profile_evolution.pdf')



# Sigma percents
onesigp = 84.13
onesigm = 15.87
#
twosigp = 97.72
twosigm = 2.28
#
thrsigp = 99.87
thrsigm = 0.13
