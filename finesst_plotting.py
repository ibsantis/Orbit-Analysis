#!/usr/bin/python3

"""

    ========================
    = Integrating subhalos =
    ========================

    Integrate subhalos in custom potential
        - Disk (radial and vertical) model
        - DM halo model

"""

# Import packages
from galpy.orbit import Orbit
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
from astropy import units as u
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12i'
loc = 'peloton'
#
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
#
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

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)

orbits = orbit_io.OrbitAnalysis(tree=halt, gal1='m12i', location='peloton')
orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1='m12i', location='peloton')
#
halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
halt_vels = orbits.halo_velocities(halt)
host_radii = halt['radius'][orbits.sub_inds[0][orbits.sub_inds[0] >= 0]] # Want to divide the other distances by this distance
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
infall_info = orbits.first_infall_times(halt_dists_norm, snaps)
peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps)
apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
angs = orbits.angular_momentum(tree=halt)

# Import the potentials and create custom ones
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import NFWPotential # For DM halos
#
disk_outer = DoubleExponentialDiskPotential(amp=1.17e9*2.24*u.solMass/u.kpc**3, hr=2.76*u.kpc, hz=0.41*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=1.59e10*2.24*u.solMass/u.kpc**3, hr=0.61*u.kpc, hz=0.41*u.kpc)
nfw = NFWPotential(amp=5.41e11*u.solMass, a=18.73*u.kpc)
potential_total = disk_inner+disk_outer+nfw

# Get halos that fell into the host; these are IDs at z = 0
halo_1 = 3257469
halo_2 = 3502033
halo_3 = 6719222

# Get the halo properties necessary for orbit initialization
print(halt.prop('host.distance.principal', halo_1))
print(halt.prop('host.distance.principal.cylidnrical', halo_1))
print(halt.prop('host.velocity.principal.cylindrical', halo_1))
print(halt.prop('host.velocity.tan', halo_1))
#
print(halt.prop('host.distance.principal', halo_2))
print(halt.prop('host.distance.principal.cylidnrical', halo_2))
print(halt.prop('host.velocity.principal.cylindrical', halo_2))
print(halt.prop('host.velocity.tan', halo_2))
#
print(halt.prop('host.distance.principal', halo_3))
print(halt.prop('host.distance.principal.cylidnrical', halo_3))
print(halt.prop('host.velocity.principal.cylindrical', halo_3))
print(halt.prop('host.velocity.tan', halo_3))

# Initialize the orbits (R, vR, vT, z, vz, phi)
orb_3 = Orbit([231.64*u.kpc, 12.99*u.km/u.s, 86.77*u.km/u.s, 19.84*u.kpc, -39.72*u.km/u.s, 37.89*u.deg])
orb_9 = Orbit([271.18*u.kpc, 36.82*u.km/u.s, 35.85*u.km/u.s, 220.78*u.kpc, 24.67*u.km/u.s, 45.41*u.deg])
orb_32 = Orbit([80.01*u.kpc, -4.90*u.km/u.s, 102.53*u.km/u.s, -60.53*u.kpc, -61.92*u.km/u.s, 44.53*u.deg])

# Set up time array (negative because integrating backward)
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr

orb_3.integrate(ts, potential_total, method='odeint')
d_model = orb_3._parse_plot_quantity(quant='r')
print('Pericenter distance for halo 3 in potential_total: {0}'.format(orb_3.rperi()))
print('Apocenter distance for halo 3 in potential_total: {0}'.format(orb_3.rap()))

d_mask = (halt_dists[3] >= 0)
ds = halt_dists[3][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
#plt.xlim(lookback_time[-1], lookback_time[0])
#plt.ylim(0, np.nanmax(ds))
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][3]
peri = peris['pericenter.check'][3]
apo = apos['apocenter.check'][3]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][3]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][3] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][3][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][3] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][3][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
plt.xlim(times[-1], times[0])
plt.ylim(0, np.nanmax(ds))
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('Subhalo 3', fontsize=24)
plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_3_sim_n_data.pdf')
plt.close()



## HALO 9
#
# Initialize the orbit and integrate it
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_9.integrate(ts, potential_total, method='odeint')
d_model = orb_9._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 2 in potential_total: {0}'.format(orb_9.rperi()))
print('Apocenter distance for halo 2 in potential_total: {0}'.format(orb_9.rap()))
#
d_mask = (halt_dists[9] >= 0)
ds = halt_dists[9][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 5))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
"""
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][9]
peri = peris['pericenter.check'][9]
apo = apos['apocenter.check'][9]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][9]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][9][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][9][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
"""
plt.xlim(times[-1], times[0])
plt.ylim(0, np.nanmax(ds))
plt.xlabel('lookback time [Gyr]', fontsize=32)
plt.ylabel('r [kpc]', fontsize=32)
plt.title('Satellite 1', fontsize=30)
plt.legend(prop={'size': 24})
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_9_sim_n_data.pdf')
plt.close()
#
#
#
v_model = orb_9._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[9] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[9][orbits.sub_inds[9]>=0])[:,0]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 5))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(vs)]
# Plot the data and set the limits
plt.plot(times, vs, label='simulation')
plt.plot(-1*ts, v_model, label='galpy')
"""
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][9]
peri = peris['pericenter.check'][9]
apo = apos['apocenter.check'][9]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][9]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][9][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][9][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
"""
plt.xlim(times[-1], times[0])
plt.ylim(np.nanmin(v_model), np.nanmax(vs))
plt.xlabel('lookback time [Gyr]', fontsize=32)
plt.ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
plt.title('Satellite 1', fontsize=30)
#plt.legend(prop={'size': 24})
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_9_sim_n_data_vel.pdf')
plt.close()
#
#
#
#L_model = np.linalg.norm(orb_9._parse_plot_quantity(quant='L'), axis=1)/1000
#Ls = angs['ang.mom.total'][9]/1000
L_model = orb_9._parse_plot_quantity(quant='Lz')/1000
Ls = angs['ang.mom.vector'][9][:,-1]/1000
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 5))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(Ls)]
# Plot the data and set the limits
plt.plot(times, Ls, label='simulation')
plt.plot(-1*ts, L_model, label='galpy')
"""
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][9]
peri = peris['pericenter.check'][9]
apo = apos['apocenter.check'][9]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][9]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][9][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][9] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][9][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
"""
plt.xlim(times[-1], times[0])
plt.ylim(np.nanmin(Ls), np.nanmax(Ls))
plt.xlabel('lookback time [Gyr]', fontsize=32)
plt.ylabel('$L_{\\rm z}$ [10$^3$ kpc km s$^{-1}$]', fontsize=27)
plt.title('Satellite 1', fontsize=30)
#plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_9_sim_n_data_ellz.pdf')
plt.close()









## HALO 33
#
# Initialize the orbit and integrate it
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_32.integrate(ts, potential_total, method='odeint')
d_model = orb_32._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 32 in potential_total: {0}'.format(orb_32.rperi()))
print('Apocenter distance for halo 32 in potential_total: {0}'.format(orb_32.rap()))

d_mask = (halt_dists[32] >= 0)
ds = halt_dists[32][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
#plt.xlim(lookback_time[-1], lookback_time[0])
#plt.ylim(0, np.nanmax(ds))
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][32]
peri = peris['pericenter.check'][32]
apo = apos['apocenter.check'][32]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][32]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][32] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][32][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][32] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][32][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
plt.xlim(times[-1], times[0])
plt.ylim(0, np.nanmax(ds))
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('Subhalo 32', fontsize=24)
plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_32_sim_n_data.pdf')
plt.close()



# Subhalo 5
print(halt.prop('host.distance.principal', orbits.sub_inds[5][0]))
print(halt.prop('host.distance.principal.cylidnrical', orbits.sub_inds[5][0]))
print(halt.prop('host.velocity.principal.cylindrical', orbits.sub_inds[5][0]))
print(halt.prop('host.velocity.tan', orbits.sub_inds[5][0]))

orb_5 = Orbit([267.57*u.kpc, -70.60*u.km/u.s, 121.62*u.km/u.s, -102.80*u.kpc, 81.65*u.km/u.s, 86.11*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_5.integrate(ts, potential_total, method='odeint')
d_model = orb_5._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 5 in potential_total: {0}'.format(orb_5.rperi()))
print('Apocenter distance for halo 5 in potential_total: {0}'.format(orb_5.rap()))

d_mask = (halt_dists[5] >= 0)
ds = halt_dists[5][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
#plt.xlim(lookback_time[-1], lookback_time[0])
#plt.ylim(0, np.nanmax(ds))
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][5]
peri = peris['pericenter.check'][5]
apo = apos['apocenter.check'][5]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][5]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][5] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][5][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][5] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][5][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
plt.xlim(times[-1], times[0])
plt.ylim(0, np.nanmax(ds))
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('Subhalo 5', fontsize=24)
plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_5_sim_n_data.pdf')
plt.close()




# Subhalo 8
print(halt.prop('host.distance.principal', orbits.sub_inds[8][0]))
print(halt.prop('host.distance.principal.cylidnrical', orbits.sub_inds[8][0]))
print(halt.prop('host.velocity.principal.cylindrical', orbits.sub_inds[8][0]))
print(halt.prop('host.velocity.tan', orbits.sub_inds[8][0]))

orb_8 = Orbit([324.23*u.kpc, 113.92*u.km/u.s, 98.79*u.km/u.s, -78.20*u.kpc, 66.47*u.km/u.s, -83.71*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_8.integrate(ts, potential_total, method='odeint')
d_model = orb_8._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 8 in potential_total: {0}'.format(orb_8.rperi()))
print('Apocenter distance for halo 8 in potential_total: {0}'.format(orb_8.rap()))

d_mask = (halt_dists[8] >= 0)
ds = halt_dists[8][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
#plt.xlim(lookback_time[-1], lookback_time[0])
#plt.ylim(0, np.nanmax(ds))
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][8]
peri = peris['pericenter.check'][8]
apo = apos['apocenter.check'][8]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][8]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][8] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][8][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][8] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][8][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
plt.xlim(times[-1], times[0])
plt.ylim(0, 1000)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('Subhalo 8', fontsize=24)
plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_8_sim_n_data.pdf')
plt.close()




# Subhalo 10
print(halt.prop('host.distance.principal', orbits.sub_inds[10][0]))
print(halt.prop('host.distance.principal.cylidnrical', orbits.sub_inds[10][0]))
print(halt.prop('host.velocity.principal.cylindrical', orbits.sub_inds[10][0]))
print(halt.prop('host.velocity.tan', orbits.sub_inds[10][0]))

orb_10 = Orbit([135.74*u.kpc, -106.50*u.km/u.s, 98.79*u.km/u.s, -85.83*u.kpc, 32.45*u.km/u.s, -69.70*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_10.integrate(ts, potential_total, method='odeint')
d_model = orb_10._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 10 in potential_total: {0}'.format(orb_10.rperi()))
print('Apocenter distance for halo 10 in potential_total: {0}'.format(orb_10.rap()))

d_mask = (halt_dists[10] >= 0)
ds = halt_dists[10][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
#plt.xlim(lookback_time[-1], lookback_time[0])
#plt.ylim(0, np.nanmax(ds))
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][10]
peri = peris['pericenter.check'][10]
apo = apos['apocenter.check'][10]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][10]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][10] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][10][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][10] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][10][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
plt.xlim(times[-1], times[0])
plt.ylim(0, np.nanmax(ds))
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('Subhalo 10', fontsize=24)
plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_10_sim_n_data.pdf')
plt.close()





# Subhalo 11
print(halt.prop('host.distance.principal', orbits.sub_inds[11][0]))
print(halt.prop('host.distance.principal.cylidnrical', orbits.sub_inds[11][0]))
print(halt.prop('host.velocity.principal.cylindrical', orbits.sub_inds[11][0]))
print(halt.prop('host.velocity.tan', orbits.sub_inds[11][0]))

orb_11 = Orbit([18.36*u.kpc, 102.84*u.km/u.s, 134.58*u.km/u.s, -163.88*u.kpc, 115.71*u.km/u.s, -52.31*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_11.integrate(ts, potential_total, method='odeint')
d_model = orb_11._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 11 in potential_total: {0}'.format(orb_11.rperi()))
print('Apocenter distance for halo 11 in potential_total: {0}'.format(orb_11.rap()))

d_mask = (halt_dists[11] >= 0)
ds = halt_dists[11][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
#plt.xlim(lookback_time[-1], lookback_time[0])
#plt.ylim(0, np.nanmax(ds))
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][11]
peri = peris['pericenter.check'][11]
apo = apos['apocenter.check'][11]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][11]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][11] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][11][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][11] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][11][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
plt.xlim(times[-1], times[0])
plt.ylim(0, np.nanmax(ds))
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('Subhalo 11', fontsize=24)
plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_11_sim_n_data.pdf')
plt.close()




# Subhalo 30
print(halt.prop('host.distance.principal', orbits.sub_inds[30][0]))
print(halt.prop('host.distance.principal.cylidnrical', orbits.sub_inds[30][0]))
print(halt.prop('host.velocity.principal.cylindrical', orbits.sub_inds[30][0]))
print(halt.prop('host.velocity.tan', orbits.sub_inds[30][0]))

orb_30 = Orbit([32.50*u.kpc, -81.20*u.km/u.s, 79.74*u.km/u.s, -160.75*u.kpc, 30.26*u.km/u.s, -85.70*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_30.integrate(ts, potential_total, method='odeint')
d_model = orb_30._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 30 in potential_total: {0}'.format(orb_30.rperi()))
print('Apocenter distance for halo 30 in potential_total: {0}'.format(orb_30.rap()))

d_mask = (halt_dists[30] >= 0)
ds = halt_dists[30][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 8))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
#plt.xlim(lookback_time[-1], lookback_time[0])
#plt.ylim(0, np.nanmax(ds))
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][30]
peri = peris['pericenter.check'][30]
apo = apos['apocenter.check'][30]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][30]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][30] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][30][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][30] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][30][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
plt.xlim(times[-1], times[0])
plt.ylim(0, np.nanmax(ds))
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('Subhalo 30', fontsize=24)
plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_30_sim_n_data.pdf')
plt.close()






# Subhalo 22
print(halt.prop('host.distance.principal', orbits.sub_inds[22][0]))
print(halt.prop('host.distance.principal.cylidnrical', orbits.sub_inds[22][0]))
print(halt.prop('host.velocity.principal.cylindrical', orbits.sub_inds[22][0]))
print(halt.prop('host.velocity.tan', orbits.sub_inds[22][0]))

orb_22 = Orbit([92.24*u.kpc, 62.08*u.km/u.s, 127.85*u.km/u.s, 106.65*u.kpc, -105.09*u.km/u.s, -82.18*u.deg])
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr
orb_22.integrate(ts, potential_total, method='odeint')
d_model = orb_22._parse_plot_quantity(quant='r')
#
# Print out the pericenter distance
print('Pericenter distance for halo 22 in potential_total: {0}'.format(orb_22.rperi()))
print('Apocenter distance for halo 22 in potential_total: {0}'.format(orb_22.rap()))

d_mask = (halt_dists[22] >= 0)
ds = halt_dists[22][d_mask]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 5))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(ds)]
# Plot the data and set the limits
plt.plot(times, ds, label='simulation')
plt.plot(-1*ts, d_model, label='galpy')
"""
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][22]
peri = peris['pericenter.check'][22]
apo = apos['apocenter.check'][22]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][22]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][22] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][22][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][22] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][22][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
"""
plt.xlim(10, times[0])
plt.ylim(0, 200)
plt.xlabel('lookback time [Gyr]', fontsize=32)
plt.ylabel('r [kpc]', fontsize=32)
plt.title('Satellite 2', fontsize=30)
plt.legend(prop={'size': 24})
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_22_sim_n_data.pdf')
plt.close()
#
#
#
v_model = orb_22._parse_plot_quantity(quant='vR')
v_mask = (halt_vels[22] >= 0)
vs = halt.prop('host.velocity.principal.spherical', orbits.sub_inds[22][orbits.sub_inds[22]>=0])[:,0]
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 5))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(vs)]
# Plot the data and set the limits
plt.plot(times, vs, label='simulation')
plt.plot(-1*ts, v_model, label='galpy')
"""
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][22]
peri = peris['pericenter.check'][22]
apo = apos['apocenter.check'][22]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][22]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][22] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][22][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][22] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][22][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
"""
plt.xlim(10, times[0])
plt.ylim(np.nanmin(v_model), np.nanmax(vs))
plt.xlabel('lookback time [Gyr]', fontsize=32)
plt.ylabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=32)
plt.title('Satellite 2', fontsize=30)
#plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_22_sim_n_data_vel.pdf')
plt.close()
#
#
#
#L_model = np.linalg.norm(orb_22._parse_plot_quantity(quant='L'), axis=1)/1000
#Ls = angs['ang.mom.total'][22]/1000
L_model = orb_22._parse_plot_quantity(quant='Lz')/1000
Ls = angs['ang.mom.vector'][22][:,-1]/1000
# Plot the data from the simulation and model
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 5))
lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
times = lookback_time[:len(Ls)]
# Plot the data and set the limits
plt.plot(times, Ls, label='simulation')
plt.plot(-1*ts, L_model, label='galpy')
"""
# Check to see if there were infall, pericenter, or apocenter events
infall = infall_info['check'][22]
peri = peris['pericenter.check'][22]
apo = apos['apocenter.check'][22]
# If there are, plot when they occurred
if infall == True:
    infall_time = infall_info['time.lb'][22]
    plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
if peri == True:
    mask = (peris['pericenter.time.lb'][22] > 0)
    if np.sum(mask) > 0:
        peri_times = peris['pericenter.time.lb'][22][mask]
        [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
if apo == True:
    mask = (apos['apocenter.time.lb'][22] > 0)
    if np.sum(mask) > 0:
        apo_times = apos['apocenter.time.lb'][22][mask]
        [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
#
"""
plt.xlim(10, times[0])
plt.ylim(np.nanmin(Ls), np.nanmax(Ls)+1)
plt.xlabel('lookback time [Gyr]', fontsize=32)
plt.ylabel('$L_{\\rm z}$ [10$^3$ kpc km s$^{-1}$]', fontsize=27)
plt.title('Satellite 2', fontsize=30)
#plt.legend(prop={'size': 18})
plt.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.savefig('/home/ibsantis/scripts/orbit_data/plots/sub_22_sim_n_data_ellz.pdf')
plt.close()
