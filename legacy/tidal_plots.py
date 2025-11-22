#!/usr/bin/python3

"""

  ========================
  = Tidal field plotting =
  ========================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Summer Quarter, 2022

  Reads in the enclosed mass profiles at all snapshots, then calculates and
  plots both the tidal acceleration and |da/dr|

  NOTE:
    Units of a and |da/dr| are in SI

"""

from galpy.orbit import Orbit
from orbit_analysis import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import splrep, splev
from astropy import units as u
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
#sim_data.galaxy = sim_data.gal_2
print('Set paths')

# Read in the data
masses = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+sim_data.galaxy+'_mass_profile_all')
rs = np.logspace(np.log10(5), np.log10(500), 25)
rs_new = np.logspace(np.log10(rs[0]), np.log10(rs[-1]), 1000)
print('Read the data in')

# Set up empty arrays to save to
tidal_field = np.zeros(masses['mass.profile'].shape)

# Loop through each snapshot
for i in range(0, masses['mass.profile'].shape[0]):
    # Loop through each value of enclosed mass (i.e. loop through distance)
    for j in range(0, masses['mass.profile'].shape[1]):
        # Calculate the tidal acceleration
        tidal_field[i,j] = (masses['mass.profile'][i][j]/rs[j+1]**2)*(6.67*10**(-11))*(2*10**(30)*100)/((1000**2)*(3.086*10**(16))**2) # in cm/s^2

# Calculate the tidal acceleration
da_dr = np.zeros((masses['mass.profile'].shape[0], len(rs_new)))
for i in range(0, masses['mass.profile'].shape[0]):
    #f = interp1d(rs[1:], tidal_field[i], kind='cubic')
    #da_dr[i] = (np.diff(f(rs_new))/np.diff(rs_new))*(1/1000)*(1/(3.086*10**(18))) # Needed to convert the denominator to cm
    spline_fit = splrep(x=rs[1:], y=tidal_field[i])
    da_dr[i] = splev(rs_new, spline_fit, der=1)*(1/1000)*(1/(3.086*10**(18))) # Needed to convert the denominator to cm

# Create an array of times that I'm interested in
ts = np.linspace(0.8, 13.8, 66) # every 0.2 Gyr

# Set empty arrays to save to
times = np.zeros(len(ts))
inds = np.zeros(len(ts),dtype=int)

# Loop through the array of times
for i in range(0, len(ts)):
    # Find the closest snapshot corresponding to the times I'm interested in
    temp = np.abs(ts[i]-masses['time'])
    times[i] = masses['time'][np.where(np.min(temp) == temp)[0][0]]
    inds[i] = masses['snapshot'][np.where(np.min(temp) == temp)[0][0]]
times = np.around(times[-1]-times, decimals=2)

# Create the color array for all of the curves
def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=1)

# Plot both a and |da/dr| on the same plot
# Flip the arrays so that it starts at early times and ends at z = 0
times = np.flip(times)
inds = np.flip(inds)
#
plt.rcParams["font.family"] = "serif"
f, ax = plt.subplots(2, 1, figsize=(10,8))
for i in range(0, len(inds)):
    ax[0].plot(rs[1:], tidal_field[inds[i]-2], color=colorss[i])
    ax[1].plot(rs_new, np.abs(da_dr[inds[i]-2]), color=colorss[i])
ax[0].set_xlim(5, 500)
ax[1].set_xlim(5, 500)
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
ax[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
ax[1].set_xlabel('r [kpc]', fontsize=32)
ax[0].set_title(sim_data.galaxy, fontsize=26)
ax[0].set_ylabel('$G \ M(<r) / r^2$ [cm s$^{-2}$]', fontsize=22)
ax[1].set_ylabel('$|da/dr|$ [s$^{-2}$]', fontsize=22)
cmap = plt.get_cmap('plasma', len(masses['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = f.colorbar(sm, ax=ax[:])
cbar.set_label('Lookback time [Gyr]', fontsize=28)
#plt.tight_layout() # Can't use tight_layout, it messes up the colorbar
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2_fix/tidal_field/'+sim_data.galaxy+'_tidal_0.2Gyr.pdf', bbox_inches='tight')
#plt.show()
plt.close()


"""
    Make a median plot of all hosts
"""
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')
#
# Read in the data
mass_m12b = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12b_mass_profile_all')
mass_m12c = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12c_mass_profile_all')
mass_m12f = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12f_mass_profile_all')
mass_m12i = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12i_mass_profile_all')
mass_m12m = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12m_mass_profile_all')
mass_m12w = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12w_mass_profile_all')
mass_m12z = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12z_mass_profile_all')
mass_Romeo = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romeo_mass_profile_all')
mass_Juliet = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Juliet_mass_profile_all')
mass_Thelma = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Thelma_mass_profile_all')
mass_Louise = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Louise_mass_profile_all')
mass_Romulus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romulus_mass_profile_all')
mass_Remus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Remus_mass_profile_all')
#
masses_all = np.array([mass_m12b['mass.profile'],\
                       mass_m12c['mass.profile'],\
                       mass_m12f['mass.profile'],\
                       mass_m12i['mass.profile'],\
                       mass_m12m['mass.profile'],\
                       mass_m12w['mass.profile'],\
                       mass_m12z['mass.profile'],\
                       mass_Romeo['mass.profile'],\
                       mass_Juliet['mass.profile'],\
                       mass_Thelma['mass.profile'],\
                       mass_Louise['mass.profile'],\
                       mass_Romulus['mass.profile'],\
                       mass_Remus['mass.profile']])
#
rs = np.logspace(np.log10(5), np.log10(500), 25)
rs_new = np.logspace(np.log10(rs[1]), np.log10(rs[-1]), 1000)
print('Read the data in')

# Set up empty arrays to save to
tidal_field = np.zeros((masses_all.shape[0], masses_all.shape[1], masses_all.shape[2]))
da_dr = np.zeros((masses_all.shape[0], masses_all.shape[1], len(rs_new)))

# Loop through each host
for n in range(0, masses_all.shape[0]):
    # Loop through each snapshot
    for i in range(0, masses_all.shape[1]):
        # Loop through each value of enclosed mass (i.e. loop through distance)
        for j in range(0, masses_all.shape[2]):
            # Calculate the tidal acceleration
            tidal_field[n,i,j] = (masses_all[n][i][j]/rs[j+1]**2)*(6.67*10**(-11))*(2*10**(30)*100)/((1000**2)*(3.086*10**(16))**2) # in cm/s^2
#
tidal_field_med = np.median(tidal_field, axis=0)

# Calculate the tidal acceleration
# Loop over each host
for n in range(0, masses_all.shape[0]):
    # Loop over each snapshot
    for i in range(0, masses_all.shape[1]):
        spline_fit = splrep(x=rs[1:], y=tidal_field[n,i])
        da_dr[n,i] = splev(rs_new, spline_fit, der=1)*(1/1000)*(1/(3.086*10**(18))) # Needed to convert the denominator to cm
da_dr_med = np.median(da_dr, axis=0)

# Create an array of times that I'm interested in
ts = np.linspace(0.8, 13.8, 66) # every 0.2 Gyr

# Set empty arrays to save to
times = np.zeros(len(ts))
inds = np.zeros(len(ts),dtype=int)

# Loop through the array of times
for i in range(0, len(ts)):
    # Find the closest snapshot corresponding to the times I'm interested in
    temp = np.abs(ts[i]-mass_m12b['time'])
    times[i] = mass_m12b['time'][np.where(np.min(temp) == temp)[0][0]]
    inds[i] = mass_m12b['snapshot'][np.where(np.min(temp) == temp)[0][0]]
times = np.around(times[-1]-times, decimals=2)

# Create the color array for all of the curves
def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=1)

# Plot both a and |da/dr| on the same plot
# Flip the arrays so that it starts at early times and ends at z = 0
times = np.flip(times)[:-1] # need to ignore snapshot 42 because the median is zero and it looks funny
inds = np.flip(inds)[:-1]
#
plt.rcParams["font.family"] = "serif"
f, ax = plt.subplots(2, 1, figsize=(10,8))
for i in range(0, len(inds)):
    ax[0].plot(rs[1:], tidal_field_med[inds[i]-2], color=colorss[i])
    ax[1].plot(rs_new, np.abs(da_dr_med[inds[i]-2]), color=colorss[i])
ax[0].set_xlim(5, 500)
ax[1].set_xlim(5, 500)
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
ax[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
ax[1].set_xlabel('r [kpc]', fontsize=32)
ax[0].set_title('Median, 0.2 Gyr spacing', fontsize=26)
ax[0].set_ylabel('$G \ M(<r) / r^2$ [cm s$^{-2}$]', fontsize=22)
ax[1].set_ylabel('$|da/dr|$ [s$^{-2}$]', fontsize=22)
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = f.colorbar(sm, ax=ax[:])
cbar.set_label('Lookback time [Gyr]', fontsize=28)
#plt.tight_layout() # Can't use tight_layout, it messes up the colorbar
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2_fix/tidal_field/median_tidal_0.2Gyr.pdf', bbox_inches='tight')
#plt.show()
plt.close()


# Create an array of times that I'm interested in
ts = np.linspace(0.8, 13.8, 27) # every 0.5 Gyr

# Set empty arrays to save to
times = np.zeros(len(ts))
inds = np.zeros(len(ts),dtype=int)

# Loop through the array of times
for i in range(0, len(ts)):
    # Find the closest snapshot corresponding to the times I'm interested in
    temp = np.abs(ts[i]-mass_m12b['time'])
    times[i] = mass_m12b['time'][np.where(np.min(temp) == temp)[0][0]]
    inds[i] = mass_m12b['snapshot'][np.where(np.min(temp) == temp)[0][0]]
times = np.around(times[-1]-times, decimals=2)

# Create the color array for all of the curves
def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=1)

# Plot both a and |da/dr| on the same plot
# Flip the arrays so that it starts at early times and ends at z = 0
times = np.flip(times)[:-1] # need to ignore snapshot 42 because the median is zero and it looks funny
inds = np.flip(inds)[:-1]
#
plt.rcParams["font.family"] = "serif"
f, ax = plt.subplots(2, 1, figsize=(10,8))
for i in range(0, len(inds)):
    ax[0].plot(rs[1:], tidal_field_med[inds[i]-2], color=colorss[i])
    ax[1].plot(rs_new, np.abs(da_dr_med[inds[i]-2]), color=colorss[i])
ax[0].set_xlim(5, 500)
ax[1].set_xlim(5, 500)
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
ax[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
ax[1].set_xlabel('r [kpc]', fontsize=32)
ax[0].set_title('Median, 0.5 Gyr spacing', fontsize=26)
ax[0].set_ylabel('$G \ M(<r) / r^2$ [cm s$^{-2}$]', fontsize=22)
ax[1].set_ylabel('$|da/dr|$ [s$^{-2}$]', fontsize=22)
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = f.colorbar(sm, ax=ax[:])
cbar.set_label('Lookback time [Gyr]', fontsize=28)
#plt.tight_layout() # Can't use tight_layout, it messes up the colorbar
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2_fix/tidal_field/median_tidal_0.5Gyr.pdf', bbox_inches='tight')
#plt.show()
plt.close()
