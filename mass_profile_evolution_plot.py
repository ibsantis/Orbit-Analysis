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
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='mac')
print('Set paths')

sim_data.galaxy = 'Remus' # this is only necessary for LG pairs

# Read in the data
mass_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'_mass_profile_evolution')

# Cumulatively sum each array, then take the average over all of them
mass_prof_avg = np.average(np.cumsum(mass_data['mass.profile'], axis=1), axis=0)

# Plot each snapshot mass profile to the average mass profile
rs = np.logspace(np.log10(0.1), np.log10(500), 100)
#colors = ['#800000', '#9A6324', '#808000', '#469990', '#000075', '#e6194B', '#f58231', '#3cb44b', '#42d4f4', '#911eb4', '#f032e6']

def color_cycle(cycle_length=len(mass_data['time']), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(mass_data['time']), cmap_name='plasma', low=0, high=1)
#
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
#
for i in range(0, len(mass_data['mass.profile'])):
    plt.plot(rs[1:], np.cumsum(mass_data['mass.profile'][i])[:-1]/mass_prof_avg[:-1], color=colorss[i], label=str(mass_data['time'][i])+' Gyr')
plt.xlim(5, 500)
plt.ylim(0.97, 1.04)
plt.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm avg}$(<r)', fontsize=32)
plt.title(sim_data.galaxy, fontsize=32)
plt.legend(prop={'size': 16}, ncol=2)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/'+sim_data.galaxy+'_mass_profile_evolution.pdf')


plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
#
for i in range(0, len(mass_data['mass.profile'])):
    plt.plot(rs[1:], np.cumsum(mass_data['mass.profile'][i])[:-1]/np.cumsum(mass_data['mass.profile'][0])[:-1], color=colorss[i], label=str(mass_data['time'][i])+' Gyr')
plt.xlim(5, 500)
plt.ylim(0.95, 1.02)
plt.xscale('log')
plt.xlabel('r [kpc]', fontsize=32)
plt.ylabel('$M$(<r) / $M_{\\rm z = 0}$(<r)', fontsize=32)
plt.title(sim_data.galaxy, fontsize=32)
plt.legend(prop={'size': 16}, ncol=2)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/mass_profiles/'+sim_data.galaxy+'_mass_profile_evolution_z0.pdf')
