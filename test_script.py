#!/usr/bin/python3

"""
Write a simple script to:
    1. Select halos that follow the following criteria:
    2. Calculate what their smallest pericenter was and what time it happened at
    3. Plot the distance of the halo from the host as a function of time
    4. Save this data to a file
"""

# Import all of the tools for analysis
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import orbit_times as ot
import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12i'
loc = 'stampede'

if gal1 == 'Romeo':
    gal2 = 'Juliet'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res3500'
    num_gal = 2
elif gal1 == 'Thelma':
    gal2 = 'Louise'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res4000'
    num_gal = 2
elif gal1 == 'Romulus':
    gal2 = 'Remus'
    galaxy = 'm12_elvis_'+gal1+gal2
    resolution = '_res4000'
    num_gal = 2
else:
    galaxy = gal1
    resolution = '_res7100'
    num_gal = 1

if loc == 'mac':
    home_dir = '/Users/isaiahsantistevan/simulation'
else:
    home_dir = '/home1/05400/ibsantis/scripts'
simulation_dir = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'+galaxy+resolution
print('Set paths')

# Read in the entire tree
snaps = ut.simulation.read_snapshot_times(directory=simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory='/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12i_res7100')

orbits = ot.OrbitAnalysis()
#################################################################################################
# Select the luminous subhalos first
subhalo_inds = get_luminous_halos(halt) # Originally "halt_inds_w_star_all_new"

####
# New test
subhalo_inds_new = orbits.get_luminous_halos(halt)
#################################################################################################

# Get the distances from the host for each halo for all snapshots
# This goes from z = 0 to z'
halt_dists = halo_distances(halt, subhalo_inds) # Originally had the function written out here

# Make a plot of this halo's distance vs time
# Ignoring first four snapshots because there aren't any halos there...
plt.figure(1)
plt.figure(figsize=(10, 8))
for i in range(0, len(halt_dists)):
    snapshots = np.flip(np.arange(4,601))[:len(halt_dists[i])]
    plt.plot(snapshots, halt_dists[i])
plt.xlabel('snapshot (time)', fontsize=28)
plt.ylabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.title('Luminous subhalos in '+gal1, fontsize=32)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/d_host_'+gal1+'.pdf')
plt.close()


# Figure out what the virial radius is for the host and make another plot
# These go from z = 0 to z'
host_radii = halt['radius'][subhalo_inds[0]] # Want to divide the other distances by this distance

# Need to use the mask for each halo on these radii so that the lengths are equal
# Some of the halos existed longer than the "host", that's okay?
# Goes from z = 0 to z'
halt_dists_norm = halo_distances_norm(halt_dists, host_radii) # Originally had the function written out here

# Make another plot
plt.figure(2)
plt.figure(figsize=(10, 8))
for i in range(0, len(halt_dists_norm)):
    snapshots = np.flip(np.arange(4,601))[:len(halt_dists[i])]
    plt.plot(snapshots[:len(halt_dists_norm[i])], halt_dists_norm[i])
plt.xlim(-0.1, 600.1)
plt.ylim(0.1, 3)
plt.hlines(y=1, xmin=0, xmax=600, linestyles='dotted', color='k')
plt.xlabel('snapshot (time)', fontsize=28)
plt.ylabel('d$_{\\rm host}$/R$_{\\rm host,200m}$', fontsize=28)
plt.title('Luminous subhalos in '+gal1, fontsize=32)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/d_host_'+gal1+'_norm.pdf')
plt.close()


#################################################################################################
"""
Infall times

    Read in the normalized distances of subhalos from the host
    Returns:
        1. Number of subhalos that did, and didn't, fall into the host
        2. Time at first infall

NOTE: Probably want to just add this to the previous function
"""

# Calculate what the infall times were (i.e. when are the normalized distances < 1?)
# Goes from z = 0 to z'
infall_info = first_infall_times(halt_dists_norm, snaps)
#infall_info['snapshot']
#infall_info['time']

# Save these values to a file or something...?
#################################################################################################

# Figure out how to calculate minima in their orbits...
# Do an example case first
peri_tot = []
time_tot = []
check_tot = []
peri_spl = []
time_spl = []
for k in range(0, len(halt_dists_norm)):
    temp_halo = halt_dists_norm[k] # Goes from z_form to z = 0
    peri_list = []
    time_list = []
    temp_peri = temp_halo[4]
    temp_check = np.zeros(len(temp_halo))
    temp_peri_spl = []
    temp_time_spl = []
    for i in range(4, len(temp_halo)-4):
        if (temp_peri < temp_halo[i+1]) and (temp_peri < temp_halo[i+2]) and (temp_peri < temp_halo[i+3])and (temp_peri < temp_halo[i+4]) and (temp_peri < temp_halo[i-1]) and (temp_peri < temp_halo[i-2]) and (temp_peri < temp_halo[i-3])and (temp_peri < temp_halo[i-4]) and (temp_peri < 1):
            temp_check[i] = 1
            peri_list.append(temp_halo[i])
            time_list.append(snaps['time'][600-i])
            temp_peri_spl.append(temp_halo[i-4:i+4])
            temp_time_spl.append(snaps['time'][600-i-4:600-i+4])
            temp_peri = temp_halo[i+1]
        else:
            temp_peri = temp_halo[i+1]
    peri_tot.append(peri_list)
    time_tot.append(time_list)
    check_tot.append(temp_check)
    peri_spl.append(temp_peri_spl)
    time_spl.append(temp_time_spl)

# Try to make a boolean mask to tell if there is pericenter
peri_bool = []
for i in range(0, len(peri_tot)):
    if (len(peri_tot[i]) != 0) and (np.sum(np.asarray(peri_tot[i]) < 1) > 0):
        peri_bool.append(True)
    else:
        peri_bool.append(False)

peri_new_spl = []
time_new_spl = []
for i in range(0, len(peri_spl)):
    if (len(peri_spl[i]) !=0):
        temp_peri_new_spl = []
        temp_time_new_spl = []
        for j in range(0, len(peri_spl[i])):
            temp_dist = peri_spl[i][j]
            temp_time = time_spl[i][j]
            f = interp1d(temp_time, temp_dist, kind='cubic')
            x_new = np.linspace(temp_time[0], temp_time[-1], 100)
            temp_peri_new_spl.append(np.min(f(x_new)))
            temp_time_new_spl.append(x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
        peri_new_spl.append(temp_peri_new_spl)
        time_new_spl.append(temp_time_new_spl)
    else:
        temp_peri_new_spl = []
        temp_time_new_spl = []
        peri_new_spl.append(temp_peri_new_spl)
        time_new_spl.append(temp_time_new_spl)


# Play around with spline fitting

temp_dist = peri_spl[32][5]
temp_time = time_spl[32][5]
f = interp1d(temp_time, temp_dist, kind='cubic')
x_new = np.linspace(temp_time[0], temp_time[-1], 100)
# Find the minimum of the spline
peri_new_spl = np.min(f(x_new))
# Time at new peri
time_new_spl = x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]]
plt.figure(1)
plt.figure(figsize=(10, 8))
plt.plot(temp_time, temp_dist, 'o') # data
plt.plot(x_new, f(x_new), '-')
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('d$_{\\rm host}$/R$_{\\rm host,200m}$', fontsize=28)
plt.title('m12i, Halo 33', fontsize=24)
plt.legend(['data', 'cubic'], loc='best', prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/spl_halo_33_norm.pdf')
plt.close()

###############################################
peri_tot_2 = []
time_tot_2 = []
check_tot_2 = []
peri_spl_2 = []
time_spl_2 = []
for k in range(0, len(halt_dists)): # Want this one because the un-normalized vector is sometimes longer, with nulls
    temp_halo = halt_dists[k] # Goes from z = 0 to z_form
    temp_halo_2 = halt_dists_norm[k]
    peri_list = []
    time_list = []
    temp_peri = temp_halo[4]
    temp_check = np.zeros(len(temp_halo))
    temp_peri_spl = []
    temp_time_spl = []
    for i in range(4, len(temp_halo_2)-4):
        if (temp_peri < temp_halo[i+1]) and (temp_peri < temp_halo[i+2]) and (temp_peri < temp_halo[i+3])and (temp_peri < temp_halo[i+4]) and (temp_peri < temp_halo[i-1]) and (temp_peri < temp_halo[i-2]) and (temp_peri < temp_halo[i-3])and (temp_peri < temp_halo[i-4]) and (temp_peri/host_radii[i] < 1):
            print(k, temp_peri, host_radii[i], snaps['time'][600-i])
            temp_check[i] = 1
            peri_list.append(temp_halo[i])
            time_list.append(snaps['time'][600-i])
            temp_peri_spl.append(temp_halo[i-4:i+4])
            temp_time_spl.append(snaps['time'][600-i-4:600-i+4])
            temp_peri = temp_halo[i+1]
        else:
            temp_peri = temp_halo[i+1]
    peri_tot_2.append(peri_list)
    time_tot_2.append(time_list)
    check_tot_2.append(temp_check)
    peri_spl_2.append(temp_peri_spl)
    time_spl_2.append(temp_time_spl)

peri_new_spl_2 = []
time_new_spl_2 = []
for i in range(0, len(peri_spl_2)):
    if (len(peri_spl_2[i]) !=0):
        temp_peri_new_spl = []
        temp_time_new_spl = []
        for j in range(0, len(peri_spl_2[i])):
            temp_dist = peri_spl_2[i][j]
            temp_time = time_spl_2[i][j]
            f = interp1d(temp_time, temp_dist, kind='cubic')
            x_new = np.linspace(temp_time[0], temp_time[-1], 100)
            temp_peri_new_spl.append(np.min(f(x_new)))
            temp_time_new_spl.append(x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
        peri_new_spl_2.append(temp_peri_new_spl)
        time_new_spl_2.append(temp_time_new_spl)
    else:
        temp_peri_new_spl = []
        temp_time_new_spl = []
        peri_new_spl_2.append(temp_peri_new_spl)
        time_new_spl_2.append(temp_time_new_spl)

temp_dist = peri_spl_2[32][4]
temp_time = time_spl_2[32][4]
f = interp1d(temp_time, temp_dist, kind='cubic')
x_new = np.linspace(temp_time[0], temp_time[-1], 100)
plt.figure(1)
plt.figure(figsize=(10, 8))
plt.plot(temp_time, temp_dist, 'o') # data
plt.plot(x_new, f(x_new), '-')
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.title('m12i, Halo 33', fontsize=24)
plt.legend(['data', 'cubic'], loc='best', prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/spl_halo_33.pdf')
plt.close()
