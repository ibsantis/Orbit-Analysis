#!/usr/bin/python3

"""

  =============================
  = Correcting halo amplitude =
  =============================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  - Set up halos in circular orbits
      - Integrate the halos in the 2P halo potential
        - Vary the amplitude until you get the most circular orbit

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
from scipy.interpolate import interp1d
from astropy import units as u
import pandas as pd
from scipy import special
import time

print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12f', location='mac')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory='simulation') # Saves snapshots, redshifts, lookback times, etc. to an array

# Read in the fitting parameters
fitting_data_2p = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params.csv', index_col=0)
fitting_data_nfw = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params_nfw.csv', index_col=0)
fitting_data_nfw_v2 = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params_nfw_v2.csv', index_col=0)
#
# Read in the subhalo initial conditions
sub_ics = pd.read_csv(sim_data.home_dir+'/orbit_data/'+sim_data.galaxy+'_subhalo_ics.csv', index_col=0)
ts = np.linspace(0.0, -13.78, 1379)*u.Gyr

# Define the 2P halo enclosed mass profile
def halo_mass(r, gal):
    A_halo = fitting_data_2p['A_halo'][gal]
    a_halo = fitting_data_2p['a_halo'][gal]
    alpha = fitting_data_2p['alpha'][gal]
    beta = fitting_data_2p['beta'][gal]
    #
    return ((A_halo/a_halo**3)*(r**(3-alpha))*(a_halo+r)**(alpha-beta)*(r/a_halo+1)**(beta-1)*a_halo**beta/(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a_halo)
#
# Define the circular velocity function
def vcirc(r,m):
    G = 6.67*10**(-11)*1000**(-3) # km^3 kg^(-1) s^(-2)
    mass = m*2*10**(30) # kg
    distance = r*10**3*3.086*10**(13) # km
    return np.sqrt(np.array(G*mass/distance, dtype=np.float64))


# Import the potentials
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential


# Find what the circular velocity should be for a given radius in 2P NFW

vc = vcirc(15.0, halo_mass(15.0, sim_data.galaxy))
#
# Find out what is the best fix to the amplitude to ensure most circular orbit
fixes = np.linspace(1.0, 2.0, 201)
vars = np.zeros(len(fixes))
for i in range(0, len(fixes)):
    start = time.time()
    potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*fixes[i]*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
    orb = Orbit([15.0*(u.kpc), 0.0*(u.km/u.s), (vc)*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
    orb.integrate(ts, potential_two_power, method='odeint')
    vars[i] = np.var(orb.r(ts))
    end = time.time()
    print('Done with step {0} in {1} seconds'.format(i, end-start))
print(np.where(np.min(vars) == vars)[0], np.min(vars), fixes[np.where(np.min(vars) == vars)[0]])


# Plot the orbit with different corrections
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
#
potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*0.9*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
orb = Orbit([15.0*(u.kpc), 0.0*(u.km/u.s), vc*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb.integrate(ts, potential_two_power, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='0.9*amp')
#
potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*1.*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
orb = Orbit([15.0*(u.kpc), 0.0*(u.km/u.s), vc*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb.integrate(ts, potential_two_power, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='1.5*amp')
#
potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*1.05*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
orb = Orbit([15.0*(u.kpc), 0.0*(u.km/u.s), vc*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb.integrate(ts, potential_two_power, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='1.615*amp')
#
potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*1.1*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
orb = Orbit([15.0*(u.kpc), 0.0*(u.km/u.s), vc*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb.integrate(ts, potential_two_power, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='1.7*amp')
#
plt.xlabel('Lookback time [Gyr]', fontsize=32)
plt.ylabel('r [kpc]', fontsize=32)
plt.xlim(13.8, 0)
plt.legend(prop={'size': 16})
plt.tight_layout()


###########################################################################################
"""
    Find the correction to the circular orbit for an actual subhalo in the sims

    Plot the corrected potential and other potentials to see the comparisons
"""
# Find what the circular velocity should be for a given radius in 2P NFW
vc = vcirc(sub_ics['R'][20], halo_mass(sub_ics['R'][20], sim_data.galaxy))
#
# Find out what is the best fix to the amplitude to ensure most circular orbit
fixes = np.linspace(0.8, 3.0, 441)
vars = np.zeros(len(fixes))
for i in range(0, len(fixes)):
    potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*fixes[i]*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
    orb = Orbit([sub_ics['R'][20]*(u.kpc), 0.0*(u.km/u.s), (vc)*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
    orb.integrate(ts, potential_two_power, method='odeint')
    vars[i] = np.var(orb.r(ts))
print(np.where(np.min(vars) == vars)[0], np.min(vars), fixes[np.where(np.min(vars) == vars)[0]])

# Create the different potentials
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
halo_two_power_1 = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
halo_two_power_2 = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*1.47*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
halo_nfw_1 = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)
halo_nfw_2 = NFWPotential(amp=fitting_data_nfw_v2['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw_v2['a_halo'][sim_data.galaxy]*u.kpc)
#
potential_two_power_1 = disk_inner+disk_outer+halo_two_power_1
potential_two_power_2 = disk_inner+disk_outer+halo_two_power_2
potential_nfw_1 = disk_inner+disk_outer+halo_nfw_1
potential_nfw_2 = disk_inner+disk_outer+halo_nfw_2
#
#potential_two_power_1 = halo_two_power_1
#potential_two_power_2 = halo_two_power_2
#potential_nfw_1 = halo_nfw_1
#potential_nfw_2 = halo_nfw_2

# Plot the orbits in the different potentials
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
#
orb = Orbit([sub_ics['R'][20]*(u.kpc), sub_ics['vR'][20]*(u.km/u.s), sub_ics['vT'][20]*(u.km/u.s), sub_ics['z'][20]*(u.kpc), sub_ics['vz'][20]*(u.km/u.s), sub_ics['phi'][20]*(u.deg)])
orb.integrate(ts, potential_two_power_1, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='2P Original')
#
orb = Orbit([sub_ics['R'][20]*(u.kpc), sub_ics['vR'][20]*(u.km/u.s), sub_ics['vT'][20]*(u.km/u.s), sub_ics['z'][20]*(u.kpc), sub_ics['vz'][20]*(u.km/u.s), sub_ics['phi'][20]*(u.deg)])
orb.integrate(ts, potential_two_power_2, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='2P Corrected')
#
orb = Orbit([sub_ics['R'][20]*(u.kpc), sub_ics['vR'][20]*(u.km/u.s), sub_ics['vT'][20]*(u.km/u.s), sub_ics['z'][20]*(u.kpc), sub_ics['vz'][20]*(u.km/u.s), sub_ics['phi'][20]*(u.deg)])
orb.integrate(ts, potential_nfw_1, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='NFW')
#
orb = Orbit([sub_ics['R'][20]*(u.kpc), sub_ics['vR'][20]*(u.km/u.s), sub_ics['vT'][20]*(u.km/u.s), sub_ics['z'][20]*(u.kpc), sub_ics['vz'][20]*(u.km/u.s), sub_ics['phi'][20]*(u.deg)])
orb.integrate(ts, potential_nfw_2, method='odeint')
plt.plot(-1*ts, orb.r(ts), label='NFW v2')
#
plt.xlabel('Lookback time [Gyr]', fontsize=32)
plt.ylabel('r [kpc]', fontsize=32)
plt.xlim(13.8, 0)
plt.legend(prop={'size': 16})
plt.title('Subhalo 20, '+sim_data.galaxy+', w/disk', fontsize=32)
plt.tight_layout()
