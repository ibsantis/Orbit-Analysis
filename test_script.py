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
import orbit_io
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12i'
loc = 'peloton'

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
elif loc == 'peloton' and num_gal == 1:
    home_dir = '/home/ibsantis/scripts'
    simulation_dir = '/home/awetzel/scratch/'+galaxy+'/'+galaxy+resolution
elif loc == 'peloton' and num_gal == 2:
    home_dir = '/home/ibsantis/scripts'
    simulation_dir = '/home/awetzel/scratch/m12_elvis/'+galaxy+resolution
else:
    home_dir = '/home1/05400/ibsantis/scripts'
    simulation_dir = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'+galaxy+resolution
print('Set paths')

# Read in the entire tree
#
snaps = ut.simulation.read_snapshot_times(directory=simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
#
halt = halo.io.IO.read_tree(simulation_directory=simulation_dir, file_kind='hdf5', species='star')
# Read in the halo potential file
halo_potential = ut.io.file_hdf5(home_dir+'/orbit_data/hdf5_files/'+galaxy+'_halo_potentials.hdf5')
print('Done reading in the data.')

orbits = orbit_io.OrbitAnalysis()
orbit_plot = orbit_io.OrbitPlot()
#
subhalo_inds = orbits.get_luminous_halos(halt)
halt_dists = orbits.halo_distances(halt, subhalo_inds) # Originally had the function written out here
halt_vels = orbits.halo_velocities(halt, subhalo_inds)
host_radii = halt['radius'][subhalo_inds[0][subhalo_inds[0] >= 0]] # Want to divide the other distances by this distance
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
infall_info = orbits.first_infall_times(halt_dists_norm, snaps)
peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
angs = orbits.angular_momentum(tree=halt, sub_inds=subhalo_inds)
pot_norm = orbits.potential_norm(tree=halt, potential=halo_potential, sub_inds=subhalo_inds)
energies = orbits.orbit_energy(tree=halt, potential_norm=pot_norm, sub_inds=subhalo_inds)

sub = 3
orbit_plot.distance_plot(tree=halt, sub_inds=subhalo_inds, subhalo_num=sub, comp='all', infall_array=infall_info, pericenter_array=peris, apocenter_array=apos, time_array=snaps, file_name='distance_total_subhalo_'+str(sub))
orbit_plot.velocity_plot(tree=halt, sub_inds=subhalo_inds, subhalo_num=sub, comp='three', infall_array=infall_info, pericenter_array=peris, apocenter_array=apos, time_array=snaps, file_name='velocities_subhalo_'+str(sub))
orbit_plot.angular_momentum_plot(ell=angs, subhalo_num=sub, comp='all', infall_array=infall_info, pericenter_array=peris, apocenter_array=apos, time_array=snaps, file_name='ang_total_subhalo_'+str(sub))
orbit_plot.orbit_energy_plot(tree=halt, potential_norm=np.abs(pot_norm), energy_tot=np.abs(energies), sub_inds=subhalo_inds, subhalo_num=sub, infall_array=infall_info, pericenter_array=peris, apocenter_array=apos, time_array=snaps, file_name='energies_subhalo_'+str(sub))


#################################################################################################
# Select the luminous subhalos first
subhalo_inds = orbits.get_luminous_halos(halt) # Originally "halt_inds_w_star_all_new"
#################################################################################################

# Get the distances from the host for each halo for all snapshots
# This goes from z = 0 to z'
halt_dists = orbits.halo_distances(halt, subhalo_inds) # Originally had the function written out here
halt_vels = orbits.halo_velocities(halt, subhalo_inds)

# Make a plot of this halo's distance vs time
# Ignoring first four snapshots because there aren't any halos there...
plt.figure(1)
plt.figure(figsize=(8, 6))
for i in range(0, len(halt_dists)):
    snapshots = np.flip(np.arange(4,601))[:len(halt_dists[i])]
    plt.plot(snapshots, halt_dists[i])
plt.xlabel('snapshot (time)', fontsize=22)
plt.ylabel('d$_{\\rm host}$ [kpc]', fontsize=22)
plt.title('Luminous subhalos in '+gal1, fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/d_host_'+gal1+'.pdf')
plt.close()

# Figure out what the virial radius is for the host and make another plot
# These go from z = 0 to z'
host_radii = halt['radius'][subhalo_inds[0]] # Want to divide the other distances by this distance

# Need to use the mask for each halo on these radii so that the lengths are equal
# Some of the halos existed longer than the "host", that's okay?
# Goes from z = 0 to z'
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii) # Originally had the function written out here

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
plt.savefig(home_dir+'/orbit_data/plots/d_host_'+gal1+'_norm.pdf')
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
infall_info = orbits.first_infall_times(halt_dists_norm, snaps)
#infall_info['snapshot']
#infall_info['time']

# Save these values to a file or something...?
#################################################################################################

###############################################
check_tot_2 = []
peri_spl_2 = []
peri_vel_spl_2 = []
time_spl_2 = []
for k in range(0, len(halt_dists)): # Want this one because the un-normalized vector is sometimes longer, with nulls
    temp_halo_d = halt_dists[k] # Goes from z = 0 to z_form
    temp_halo_v = halt_vels[k]
    temp_peri = temp_halo_d[4]
    temp_check = np.zeros(len(temp_halo_d))
    temp_peri_spl = []
    temp_peri_vel_spl = []
    temp_time_spl = []
    for i in range(4, len(temp_halo_d)-4):
        if (temp_peri < temp_halo_d[i+1]) and (temp_peri < temp_halo_d[i+2]) and (temp_peri < temp_halo_d[i+3])and (temp_peri < temp_halo_d[i+4]) and (temp_peri < temp_halo_d[i-1]) and (temp_peri < temp_halo_d[i-2]) and (temp_peri < temp_halo_d[i-3])and (temp_peri < temp_halo_d[i-4]) and (temp_peri/host_radii[i] < 1):
            temp_check[i] = 1
            temp_peri_spl.append(temp_halo_d[i-4:i+4])
            temp_peri_vel_spl.append(temp_halo_v[i-4:i+4])
            temp_time_spl.append(snaps['time'][600-i-4:600-i+4])
            temp_peri = temp_halo_d[i+1]
        else:
            temp_peri = temp_halo_d[i+1]
    check_tot_2.append(temp_check)
    peri_spl_2.append(temp_peri_spl)
    peri_vel_spl_2.append(temp_peri_vel_spl)
    time_spl_2.append(temp_time_spl)

peri_new_spl_2 = []
peri_vel_new_spl_2 = []
time_new_spl_2 = []
for i in range(0, len(peri_spl_2)):
    if (len(peri_spl_2[i]) !=0):
        temp_peri_new_spl = []
        temp_peri_vel_new_spl = []
        temp_time_new_spl = []
        for j in range(0, len(peri_spl_2[i])):
            temp_dist = peri_spl_2[i][j]
            temp_vel = peri_vel_spl_2[i][j]
            temp_time = time_spl_2[i][j]
            f = interp1d(temp_time, temp_dist, kind='cubic')
            f2 = interp1d(temp_time, temp_vel, kind='cubic')
            x_new = np.linspace(temp_time[0], temp_time[-1], 100)
            temp_peri_new_spl.append(np.min(f(x_new)))
            temp_time_new_spl.append(x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
            temp_peri_vel_new_spl.append(f2(x_new)[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
        peri_new_spl_2.append(temp_peri_new_spl)
        peri_vel_new_spl_2.append(temp_peri_vel_new_spl)
        time_new_spl_2.append(temp_time_new_spl)
    else:
        temp_peri_new_spl = []
        temp_peri_vel_new_spl = []
        temp_time_new_spl = []
        peri_new_spl_2.append(temp_peri_new_spl)
        peri_vel_new_spl_2.append(temp_peri_vel_new_spl)
        time_new_spl_2.append(temp_time_new_spl)

temp_dist = peri_spl_2[23][0]
temp_vel = peri_vel_spl_2[23][0]
temp_time = time_spl_2[23][0]
f = interp1d(temp_time, temp_dist, kind='cubic')
f2 = interp1d(temp_time, temp_vel, kind='cubic')
x_new = np.linspace(temp_time[0], temp_time[-1], 100)
plt.figure(1)
plt.figure(figsize=(10, 8))
plt.plot(temp_time, temp_dist, 'o') # data
plt.plot(x_new, f(x_new), '-')
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.title('Subhalo 23', fontsize=24)
plt.legend(['data', 'cubic'], loc='best', prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/spl_dist_subhalo_23.pdf')
plt.close()
#
plt.figure(2)
plt.figure(figsize=(10, 8))
plt.plot(temp_time, temp_vel, 'o') # data
plt.plot(x_new, f2(x_new), '-')
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('v$_{\\rm host}$ [kpc]', fontsize=28)
plt.title('Subhalo 23', fontsize=24)
plt.legend(['data', 'cubic'], loc='best', prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/spl_vel_subhalo_23.pdf')
plt.close()

###################################
check = []
apo_spl = []
apo_vel_spl = []
time_spl = []
# Loop through the number of subhalos
for k in range(0, len(halt_dists)):
    temp_halo_d = halt_dists[k] # Now goes from z = 0 to z_form (un-normalized)
    temp_halo_v = halt_vels[k] # Same as above
    # Want initial element to be this because we check +- 4 neighbors on each side
    temp_apo = temp_halo_d[4]
    temp_check = np.zeros(len(temp_halo_d))
    temp_apo_spl = []
    temp_apo_vel_spl = []
    temp_time_spl = []
    # Loop through each subhalo
    for i in range(4, len(temp_halo_d)-4):
        if (temp_apo > temp_halo_d[i+1]) and (temp_apo > temp_halo_d[i+2]) and (temp_apo > temp_halo_d[i+3])and (temp_apo > temp_halo_d[i+4]) and (temp_apo > temp_halo_d[i-1]) and (temp_apo > temp_halo_d[i-2]) and (temp_apo > temp_halo_d[i-3])and (temp_apo > temp_halo_d[i-4]):
            temp_check[i] = 1
            temp_apo_spl.append(temp_halo_d[i-4:i+4])
            temp_apo_vel_spl.append(temp_halo_v[i-4:i+4])
            temp_time_spl.append(snaps['time'][600-i-4:600-i+4])
            temp_apo = temp_halo_d[i+1]
        else:
            temp_apo = temp_halo_d[i+1]
    check.append(temp_check)
    apo_spl.append(temp_apo_spl)
    apo_vel_spl.append(temp_apo_vel_spl)
    time_spl.append(temp_time_spl)

apocenter_spline = []
apo_vel_spline = []
time_spline = []
# Loop over all of the subhalos
for i in range(0, len(apo_spl)):
    # Check if subhalo experienced apocenter. If so, continue.
    if (len(apo_spl[i]) != 0):
        temp_apo_new_spl = []
        temp_apo_vel_new_spl = []
        temp_time_new_spl = []
        # Loop over the number of apocenter events
        for j in range(0, len(apo_spl[i])):
            temp_dist = apo_spl[i][j]
            temp_vel = apo_vel_spl[i][j]
            temp_time = time_spl[i][j]
            # Work on distance
            f = interp1d(temp_time, temp_dist, kind='cubic')
            f2 = interp1d(temp_time, temp_vel, kind='cubic')
            x_new = np.linspace(temp_time[0], temp_time[-1], 100)
            temp_apo_new_spl.append(np.max(f(x_new)))
            temp_time_new_spl.append(x_new[np.where(f(x_new) == np.max(f(x_new)))[0][0]])
            temp_apo_vel_new_spl.append(f2(x_new)[np.where(f(x_new) == np.max(f(x_new)))[0][0]])
        apocenter_spline.append(temp_apo_new_spl)
        apo_vel_spline.append(temp_apo_vel_new_spl)
        time_spline.append(temp_time_new_spl)
    else:
        temp_apo_new_spl = []
        temp_apo_vel_new_spl = []
        temp_time_new_spl = []
        apocenter_spline.append(temp_apo_new_spl)
        apo_vel_spline.append(temp_apo_vel_new_spl)
        time_spline.append(temp_time_new_spl)

temp_dist = apo_spl[23][0]
temp_vel = apo_vel_spl[23][0]
temp_time = time_spl[23][0]
f = interp1d(temp_time, temp_dist, kind='cubic')
f2 = interp1d(temp_time, temp_vel, kind='cubic')
x_new = np.linspace(temp_time[0], temp_time[-1], 100)
plt.figure(1)
plt.figure(figsize=(10, 8))
plt.plot(temp_time, temp_dist, 'o') # data
plt.plot(x_new, f(x_new), '-')
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.title('Subhalo 23', fontsize=24)
plt.legend(['data', 'cubic'], loc='best', prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/spl_apo_dist_subhalo_23.pdf')
plt.close()
#
plt.figure(2)
plt.figure(figsize=(10, 8))
plt.plot(temp_time, temp_vel, 'o') # data
plt.plot(x_new, f2(x_new), '-')
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('v$_{\\rm host}$ [kpc]', fontsize=28)
plt.title('Subhalo 23', fontsize=24)
plt.legend(['data', 'cubic'], loc='best', prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/spl_apo_vel_subhalo_23.pdf')
plt.close()

########

orbit_plot = orbit_io.OrbitPlot()
sub = 39
orbit_plot.distance_plot(tree=halt, sub_inds=subhalo_inds, subhalo_num=sub, comp='all', infall_array=infall_info, pericenter_array=peris, apocenter_array=apos, time_array=snaps, file_name='/'+gal1+'/distance_total_subhalo'+str(sub)+'_'+gal1)
orbit_plot.velocity_plot(tree=halt, sub_inds=subhalo_inds, subhalo_num=sub, comp='all', infall_array=infall_info, pericenter_array=peris, apocenter_array=apos, time_array=snaps, file_name='/'+gal1+'/velocity_total_subhalo'+str(sub)+'_'+gal1)
orbit_plot.angular_momentum_plot(ell=angs, subhalo_num=sub, comp='all', infall_array=infall_info, pericenter_array=peris, apocenter_array=apos, time_array=snaps, file_name='/'+gal1+'/ang_total_subhalo'+str(sub)+'_'+gal1)

#########

kin = 0.5*halt.prop('host.velocity.total', subhalo_inds[3])**2
times = np.flip(snaps['time'], axis=0)[:len(energies[3])]
plt.figure(1)
plt.figure(figsize=(10, 8))
plt.plot(times, halo_potential['halo.potentials'][subhalo_inds[3]], label='U')
plt.plot(times, kin, label='K')
plt.plot(times, energies[3], label='E$_{tot}$', alpha=0.5, linestyle='dotted')
plt.xlim(0, 13.8)
plt.xlabel('time [Gyr]', fontsize=28)
plt.ylabel('Energy', fontsize=28)
plt.title('Subhalo 3, m12i', fontsize=24)
plt.legend(loc='best', prop={'size': 24})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/sub3m12i_energies.pdf')
plt.close()

####################

# Looking into how U_sub - U_host looks


plt.figure(1)
plt.figure(figsize=(10, 8))
#
plt.plot(lookbacks[:len(energies[3][mask3])], energies[3][mask3]/1000, alpha=0.8, label='subhalo 3', color=colors[0])
peri = peris['pericenter.check'][3]
if peri == True:
    mask = (peris['pericenter.time.lb'][3] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][3])
        [plt.vlines(peri_times[i], -45, -35, color=colors[0], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.plot(lookbacks[:len(energies[9][mask9])], energies[9][mask9]/1000, alpha=0.8, label='subhalo 9', color=colors[1])
peri = peris['pericenter.check'][9]
if peri == True:
    mask = (peris['pericenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][9])
        [plt.vlines(peri_times[i], -45, -35, color=colors[1], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.plot(lookbacks[:len(energies[10][mask10])], energies[10][mask10]/1000, alpha=0.8, label='subhalo 10', color=colors[6])
peri = peris['pericenter.check'][10]
if peri == True:
    mask = (peris['pericenter.time.lb'][10] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][10])
        [plt.vlines(peri_times[i], -45, -35, color=colors[6], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.plot(lookbacks[:len(energies[16][mask16])], energies[16][mask16]/1000, alpha=0.8, label='subhalo 16', color=colors[3])
peri = peris['pericenter.check'][16]
if peri == True:
    mask = (peris['pericenter.time.lb'][16] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][16])
        [plt.vlines(peri_times[i], -45, -35, color=colors[3], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.plot(lookbacks[:len(energies[32][mask32])], energies[32][mask32]/1000, alpha=0.8, label='subhalo 32', color=colors[4])
peri = peris['pericenter.check'][32]
if peri == True:
    mask = (peris['pericenter.time.lb'][32] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][32])
        [plt.vlines(peri_times[i], -45, -35, color=colors[4], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-45, 50)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('E$_{\\rm sub}$(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('Luminous subhalos in m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/energy_tot_diff.pdf')
plt.close()

###################

mask3 =


lookbacks = np.flip(snaps['time'][-1] - snaps['time'])
delta_t = np.asarray([(lookbacks[i] + lookbacks[i+1])/2 for i in range(0, len(lookbacks)-1)])

delta_e_3 = np.asarray([energies[3][mask3][i] - energies[3][mask3][i+1] for i in range(0, len(energies[3][mask3])-1)])/1000
delta_e_9 = np.asarray([energies[9][mask3][i] - energies[9][mask3][i+1] for i in range(0, len(energies[9][mask3])-1)])/1000
delta_e_10 = np.asarray([energies[10][mask3][i] - energies[10][mask3][i+1] for i in range(0, len(energies[10][mask3])-1)])/1000
delta_e_16 = np.asarray([energies[16][mask3][i] - energies[16][mask3][i+1] for i in range(0, len(energies[16][mask3])-1)])/1000
delta_e_32 = np.asarray([energies[32][mask3][i] - energies[32][mask3][i+1] for i in range(0, len(energies[32][mask3])-1)])/1000

kinetic_3 = (0.5*halt.prop('host.velocity.total', subhalo_inds[3][mask3])**2)
kinetic_9 = (0.5*halt.prop('host.velocity.total', subhalo_inds[9][mask9])**2)
kinetic_10 = (0.5*halt.prop('host.velocity.total', subhalo_inds[10][mask10])**2)
kinetic_16 = (0.5*halt.prop('host.velocity.total', subhalo_inds[16][mask16])**2)
kinetic_32 = (0.5*halt.prop('host.velocity.total', subhalo_inds[32][mask32])**2)

delta_k_3 = np.asarray([kinetic_3[i] - kinetic_3[i+1] for i in range(0, len(kinetic_3)-1)])/1000
delta_k_9 = np.asarray([kinetic_9[i] - kinetic_9[i+1] for i in range(0, len(kinetic_9)-1)])/1000
delta_k_10 = np.asarray([kinetic_10[i] - kinetic_10[i+1] for i in range(0, len(kinetic_10)-1)])/1000
delta_k_16 = np.asarray([kinetic_16[i] - kinetic_16[i+1] for i in range(0, len(kinetic_16)-1)])/1000
delta_k_32 = np.asarray([kinetic_32[i] - kinetic_32[i+1] for i in range(0, len(kinetic_32)-1)])/1000

delta_u_3 = np.asarray([pot_norm[3][mask3][i] - pot_norm[3][mask3][i+1] for i in range(0, len(pot_norm[3][mask3])-1)])/1000
delta_u_9 = np.asarray([pot_norm[9][mask9][i] - pot_norm[9][mask9][i+1] for i in range(0, len(pot_norm[9][mask9])-1)])/1000
delta_u_10 = np.asarray([pot_norm[10][mask10][i] - pot_norm[10][mask10][i+1] for i in range(0, len(pot_norm[10][mask10])-1)])/1000
delta_u_16 = np.asarray([pot_norm[16][mask16][i] - pot_norm[16][mask16][i+1] for i in range(0, len(pot_norm[16][mask16])-1)])/1000
delta_u_32 = np.asarray([pot_norm[32][mask32][i] - pot_norm[32][mask32][i+1] for i in range(0, len(pot_norm[32][mask32])-1)])/1000

#sub3
plt.figure(1)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(delta_e_3)], delta_e_3, alpha=0.8, label='$\Delta$E')
plt.plot(delta_t[:len(delta_k_3)], delta_k_3, alpha=0.8, label='$\Delta$K')
plt.plot(delta_t[:len(delta_u_3)], delta_u_3, alpha=0.8, label='$\Delta$U')
peri = peris['pericenter.check'][3]
if peri == True:
    mask = (peris['pericenter.time.lb'][3] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][3])
        [plt.vlines(peri_times[i], -1, -0.75, color=colors[0], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-1, 1)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 3, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_e_3.pdf')
plt.close()

plt.figure(2)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(delta_e_9)], delta_e_9, alpha=0.8, label='$\Delta$E')
plt.plot(delta_t[:len(delta_k_9)], delta_k_9, alpha=0.8, label='$\Delta$K')
plt.plot(delta_t[:len(delta_u_9)], delta_u_9, alpha=0.8, label='$\Delta$U')
peri = peris['pericenter.check'][9]
if peri == True:
    mask = (peris['pericenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][9])
        [plt.vlines(peri_times[i], -2, -1.5, color=colors[1], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-2, 2)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 9, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_e_9.pdf')
plt.close()

plt.figure(3)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(delta_e_10)], delta_e_10, alpha=0.8, label='$\Delta$E')
plt.plot(delta_t[:len(delta_k_10)], delta_k_10, alpha=0.8, label='$\Delta$K')
plt.plot(delta_t[:len(delta_u_10)], delta_u_10, alpha=0.8, label='$\Delta$U')
peri = peris['pericenter.check'][10]
if peri == True:
    mask = (peris['pericenter.time.lb'][10] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][10])
        [plt.vlines(peri_times[i], -2, -2.5, color=colors[6], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-2, 2)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 10, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_e_10.pdf')
plt.close()

plt.figure(4)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(delta_e_16)], delta_e_16, alpha=0.8, label='$\Delta$E')
plt.plot(delta_t[:len(delta_k_16)], delta_k_16, alpha=0.8, label='$\Delta$K')
plt.plot(delta_t[:len(delta_u_16)], delta_u_16, alpha=0.8, label='$\Delta$U')
peri = peris['pericenter.check'][16]
if peri == True:
    mask = (peris['pericenter.time.lb'][16] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][16])
        [plt.vlines(peri_times[i], -2, -1, color=colors[3], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-2, 2)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 16, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_e_16.pdf')
plt.close()

plt.figure(5)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(delta_e_32)], delta_e_32, alpha=0.8, label='$\Delta$E')
plt.plot(delta_t[:len(delta_k_32)], delta_k_32, alpha=0.8, label='$\Delta$K')
plt.plot(delta_t[:len(delta_u_32)], delta_u_32, alpha=0.8, label='$\Delta$U')
peri = peris['pericenter.check'][32]
if peri == True:
    mask = (peris['pericenter.time.lb'][32] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][32])
        [plt.vlines(peri_times[i], -3, -2, color=colors[4], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-3, 3)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 32, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_e_32.pdf')
plt.close()

##################
# Plot the different energies vs distance


plt.figure(1)
plt.figure(figsize=(10, 8))
#
plt.plot(halt_dists[3][mask3], energies[3][mask3]/1000, alpha=0.8, label='K + U')
plt.plot(halt_dists[3][mask3], kinetic_3/1000, alpha=0.8, label='K')
plt.plot(halt_dists[3][mask3], pot_norm[3][mask3]/1000, alpha=0.8, label='U$_{\\rm norm}$')
plt.plot(halt_dists[3][mask3], halo_potential['halo.potentials'][subhalo_inds[3][mask3]]/10000, alpha=0.8, label='U/10')
#plt.xlim(lookbacks[-1], lookbacks[0])
#plt.ylim(-1, 1)
plt.xlabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.ylabel('E [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 3, m12i', fontsize=24)
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/evd_3.pdf')
plt.close()

plt.figure(2)
plt.figure(figsize=(10, 8))
#
plt.plot(halt_dists[9][mask9], energies[9][mask9]/1000, alpha=0.8, label='K + U')
plt.plot(halt_dists[9][mask9], kinetic_9/1000, alpha=0.8, label='K')
plt.plot(halt_dists[9][mask9], pot_norm[9][mask9]/1000, alpha=0.8, label='U$_{\\rm norm}$')
plt.plot(halt_dists[9][mask9], halo_potential['halo.potentials'][subhalo_inds[9][mask9]]/10000, alpha=0.8, label='U/10')
#plt.xlim(lookbacks[-1], lookbacks[0])
#plt.ylim(-1, 1)
plt.xlabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.ylabel('E [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 9, m12i', fontsize=24)
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/evd_9.pdf')
plt.close()

plt.figure(3)
plt.figure(figsize=(10, 8))
#
plt.plot(halt_dists[10][mask10], energies[10][mask10]/1000, alpha=0.8, label='K + U')
plt.plot(halt_dists[10][mask10], kinetic_10/1000, alpha=0.8, label='K')
plt.plot(halt_dists[10][mask10], pot_norm[10][mask10]/1000, alpha=0.8, label='U$_{\\rm norm}$')
plt.plot(halt_dists[10][mask10], halo_potential['halo.potentials'][subhalo_inds[10][mask10]]/10000, alpha=0.8, label='U/10')
#plt.xlim(lookbacks[-1], lookbacks[0])
#plt.ylim(-1, 1)
plt.xlabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.ylabel('E [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 10, m12i', fontsize=24)
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/evd_10.pdf')
plt.close()

plt.figure(4)
plt.figure(figsize=(10, 8))
#
plt.plot(halt_dists[16][mask16], energies[16][mask16]/1000, alpha=0.8, label='K + U')
plt.plot(halt_dists[16][mask16], kinetic_16/1000, alpha=0.8, label='K')
plt.plot(halt_dists[16][mask16], pot_norm[16][mask16]/1000, alpha=0.8, label='U$_{\\rm norm}$')
plt.plot(halt_dists[16][mask16], halo_potential['halo.potentials'][subhalo_inds[16][mask16]]/10000, alpha=0.8, label='U/10')
#plt.xlim(lookbacks[-1], lookbacks[0])
#plt.ylim(-1, 1)
plt.xlabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.ylabel('E [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 16, m12i', fontsize=24)
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/evd_16.pdf')
plt.close()

plt.figure(5)
plt.figure(figsize=(10, 8))
#
plt.plot(halt_dists[32][mask32], energies[32][mask32]/1000, alpha=0.8, label='K + U')
plt.plot(halt_dists[32][mask32], kinetic_32/1000, alpha=0.8, label='K')
plt.plot(halt_dists[32][mask32], pot_norm[32][mask32]/1000, alpha=0.8, label='U$_{\\rm norm}$')
plt.plot(halt_dists[32][mask32], halo_potential['halo.potentials'][subhalo_inds[32][mask32]]/10000, alpha=0.8, label='U/10')
#plt.xlim(lookbacks[-1], lookbacks[0])
#plt.ylim(-1, 1)
plt.xlabel('d$_{\\rm host}$ [kpc]', fontsize=28)
plt.ylabel('E [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 32, m12i', fontsize=24)
plt.legend(loc='best', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/evd_32.pdf')
plt.close()

##########
# Check the potential energy vs the normalized potential energy differences

d_u_3 = np.asarray([halo_potential['halo.potentials'][subhalo_inds[3][mask3]][i] - halo_potential['halo.potentials'][subhalo_inds[3][mask3]][i+1] for i in range(0, len(halo_potential['halo.potentials'][subhalo_inds[3][mask3]])-1)])/1000
d_u_9 = np.asarray([halo_potential['halo.potentials'][subhalo_inds[9][mask9]][i] - halo_potential['halo.potentials'][subhalo_inds[9][mask9]][i+1] for i in range(0, len(halo_potential['halo.potentials'][subhalo_inds[9][mask9]])-1)])/1000
d_u_10 = np.asarray([halo_potential['halo.potentials'][subhalo_inds[10][mask10]][i] - halo_potential['halo.potentials'][subhalo_inds[10][mask10]][i+1] for i in range(0, len(halo_potential['halo.potentials'][subhalo_inds[10][mask10]])-1)])/1000
d_u_16 = np.asarray([halo_potential['halo.potentials'][subhalo_inds[16][mask16]][i] - halo_potential['halo.potentials'][subhalo_inds[16][mask16]][i+1] for i in range(0, len(halo_potential['halo.potentials'][subhalo_inds[16][mask16]])-1)])/1000
d_u_32 = np.asarray([halo_potential['halo.potentials'][subhalo_inds[32][mask32]][i] - halo_potential['halo.potentials'][subhalo_inds[32][mask32]][i+1] for i in range(0, len(halo_potential['halo.potentials'][subhalo_inds[32][mask32]])-1)])/1000

plt.figure(1)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(d_u_3)], d_u_3, alpha=0.8, label='$\Delta$U')
plt.plot(delta_t[:len(delta_u_3)], delta_u_3, alpha=0.8, label='$\Delta$U$_{\\rm norm}$')
peri = peris['pericenter.check'][3]
if peri == True:
    mask = (peris['pericenter.time.lb'][3] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][3])
        [plt.vlines(peri_times[i], -1, -0.75, color=colors[0], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-1, 1)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 3, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_u_3.pdf')
plt.close()

plt.figure(2)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(d_u_9)], d_u_9, alpha=0.8, label='$\Delta$U')
plt.plot(delta_t[:len(delta_u_9)], delta_u_9, alpha=0.8, label='$\Delta$U$_{\\rm norm}$')
peri = peris['pericenter.check'][9]
if peri == True:
    mask = (peris['pericenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][9])
        [plt.vlines(peri_times[i], -2, -1.5, color=colors[1], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-2, 2)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 9, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_u_9.pdf')
plt.close()

plt.figure(3)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(d_u_10)], d_u_10, alpha=0.8, label='$\Delta$U')
plt.plot(delta_t[:len(delta_u_10)], delta_u_10, alpha=0.8, label='$\Delta$U$_{\\rm norm}$')
peri = peris['pericenter.check'][10]
if peri == True:
    mask = (peris['pericenter.time.lb'][10] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][10])
        [plt.vlines(peri_times[i], -2, -2.5, color=colors[6], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-2, 2)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 10, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_u_10.pdf')
plt.close()

plt.figure(4)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(d_u_16)], d_u_16, alpha=0.8, label='$\Delta$U')
plt.plot(delta_t[:len(delta_u_16)], delta_u_16, alpha=0.8, label='$\Delta$U$_{\\rm norm}$')
peri = peris['pericenter.check'][16]
if peri == True:
    mask = (peris['pericenter.time.lb'][16] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][16])
        [plt.vlines(peri_times[i], -2, -1, color=colors[3], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-2, 2)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 16, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_u_16.pdf')
plt.close()

plt.figure(5)
plt.figure(figsize=(10, 8))
#
plt.plot(delta_t[:len(d_u_32)], d_u_32, alpha=0.8, label='$\Delta$U')
plt.plot(delta_t[:len(delta_u_32)], delta_u_32, alpha=0.8, label='$\Delta$U$_{\\rm norm}$')
peri = peris['pericenter.check'][32]
if peri == True:
    mask = (peris['pericenter.time.lb'][32] > 0)
    if np.sum(mask) > 0:
        peri_times = np.asarray(peris['pericenter.time.lb'][32])
        [plt.vlines(peri_times[i], -3, -2, color=colors[4], alpha=0.5, linestyles='solid') for i in range(0, len(peri_times))]
#
plt.xlim(lookbacks[-1], lookbacks[0])
plt.ylim(-3, 3)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('$\Delta$E(t) [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
plt.title('subhalo 32, m12i', fontsize=24)
plt.legend(loc='upper left', prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/delta_u_32.pdf')
plt.close()
