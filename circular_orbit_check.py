#!/usr/bin/python3

"""

  =====================================
  = Testing galpy integrating methods =
  =====================================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  - Set up halos in circular orbits
      - Integrate the halos in NFW potential using
        - odeint
        - leapfrog
      - Integrate the halos in NFW + disk potential using
        - odeint
        - leapfrog

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

print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory='simulation') # Saves snapshots, redshifts, lookback times, etc. to an array

# Read in the fitting parameters
fitting_data_2p = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params.csv', index_col=0)
fitting_data_nfw = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params_nfw.csv', index_col=0)
#
# Read in the subhalo initial conditions
sub_ics = pd.read_csv(sim_data.home_dir+'/orbit_data/m12i_subhalo_ics.csv', index_col=0)

"""
    First test the effects of just the halo potential
"""

# Define the enclosed mass profiles
ts = np.linspace(0.0, -13.78, 1379)*u.Gyr
#
def halo_mass(r, gal):
    A_halo = fitting_data_2p['A_halo'][gal]
    a_halo = fitting_data_2p['a_halo'][gal]
    alpha = fitting_data_2p['alpha'][gal]
    beta = fitting_data_2p['beta'][gal]
    #
    return ((A_halo/a_halo**3)*(r**(3-alpha))*(a_halo+r)**(alpha-beta)*(r/a_halo+1)**(beta-1)*a_halo**beta/(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a_halo)
#
def nfw_halo_mass(r, gal):
    A_halo = fitting_data_nfw['A_halo'][gal]
    a_halo = fitting_data_nfw['a_halo'][gal]
    return (A_halo)*(np.log((a_halo+r)/a_halo) + (a_halo/(a_halo+r)) - 1)
#

# Define circular velocity function
def vcirc(r,m):
    G = 6.67*10**(-11)*1000**(-3) # km^3 kg^(-1) s^(-2)
    mass = m*2*10**(30) # kg
    distance = r*10**3*3.086*10**(13) # km
    return np.sqrt(np.array(G*mass/distance, dtype=np.float64))


# Import the potentials
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential
#
potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
potential_nfw = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)


# Set up orbit for a subhalo and integrate using odeint
orb_2p_odeint = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_2p_odeint.integrate(ts, potential_two_power, method='odeint')
orb_nfw_odeint = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_odeint.integrate(ts, potential_nfw, method='odeint')
#
# Integrate using leapfrog
orb_2p_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_2p_leapfrog.integrate(ts, potential_two_power, method='leapfrog')
orb_nfw_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_leapfrog.integrate(ts, potential_nfw, method='leapfrog')
#
# Find the circular velocity at this radius
v_c_2p = vcirc(sub_ics['R'][9], halo_mass(sub_ics['R'][9], gal=sim_data.galaxy))
v_c_nfw = vcirc(sub_ics['R'][9], nfw_halo_mass(sub_ics['R'][9], gal=sim_data.galaxy))
#
# Integrate the circular orbits using odeint
orb_2p_c_odeint = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_2p*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_2p_c_odeint.integrate(ts, potential_two_power, method='odeint')
orb_nfw_c_odeint = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_c_odeint.integrate(ts, potential_nfw, method='odeint')
#
# Integrate the circular orbits using leapfrog
orb_2p_c_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_2p*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_2p_c_leapfrog.integrate(ts, potential_two_power, method='leapfrog')
orb_nfw_c_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_c_leapfrog.integrate(ts, potential_nfw, method='leapfrog')


# Plot in a 2x2 figure
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222, sharey=ax1)
ax3 = plt.subplot(223, sharex=ax1)
ax4 = plt.subplot(224, sharex=ax2, sharey=ax3)
#
ax1.plot(-1*ts, orb_2p_odeint.r(ts), 'k', alpha=0.5)
ax1.plot(-1*ts, orb_nfw_odeint.r(ts), 'b', alpha=0.5)
ax1.set_xlim(13.8, 0)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=22)
ax1.text(9, 600, 'odeint')
#
ax3.plot(-1*ts, orb_2p_leapfrog.r(ts), 'k', alpha=0.5)
ax3.plot(-1*ts, orb_nfw_leapfrog.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
ax3.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax3.text(9, 600, 'leapfrog')
#
ax2.plot(-1*ts, orb_2p_c_odeint.r(ts), 'k', label='2P', alpha=0.5)
ax2.plot(-1*ts, orb_nfw_c_odeint.r(ts), 'b', label='NFW', alpha=0.5)
ax2.set_xlim(13.8, 0)
ax2.label_outer()
ax2.legend(prop={'size': 16})
ax2.text(12, 100, 'circular orbit, odeint')
#
ax4.plot(-1*ts, orb_2p_c_leapfrog.r(ts), 'k', alpha=0.5)
ax4.plot(-1*ts, orb_nfw_c_leapfrog.r(ts), 'b', alpha=0.5)
ax4.set_xlim(13.8, 0)
ax4.label_outer()
ax4.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax4.text(12, 100, 'circular orbit, leapfrog')
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)



"""
    Now test the effects with halo + disk potential
"""
# Import the potentials
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
halo_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
halo_nfw = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)
#
potential_two_power = disk_inner+disk_outer+halo_two_power
potential_nfw = disk_inner+disk_outer+halo_nfw

# Set up orbit for a subhalo and integrate using odeint
orb_2p_odeint = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_2p_odeint.integrate(ts, potential_two_power, method='odeint')
orb_nfw_odeint = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_odeint.integrate(ts, potential_nfw, method='odeint')
#
# Integrate using leapfrog
orb_2p_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_2p_leapfrog.integrate(ts, potential_two_power, method='leapfrog')
orb_nfw_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_leapfrog.integrate(ts, potential_nfw, method='leapfrog')
#
# Find the circular velocity at this radius
v_c_2p = vcirc(sub_ics['R'][9], halo_mass(sub_ics['R'][9], gal=sim_data.galaxy))
v_c_nfw = vcirc(sub_ics['R'][9], nfw_halo_mass(sub_ics['R'][9], gal=sim_data.galaxy))
#
# Integrate the circular orbits using odeint
orb_2p_c_odeint = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_2p*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_2p_c_odeint.integrate(ts, potential_two_power, method='odeint')
orb_nfw_c_odeint = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_c_odeint.integrate(ts, potential_nfw, method='odeint')
#
# Integrate the circular orbits using leapfrog
orb_2p_c_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_2p*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_2p_c_leapfrog.integrate(ts, potential_two_power, method='leapfrog')
orb_nfw_c_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_c_leapfrog.integrate(ts, potential_nfw, method='leapfrog')


# Plot in a 2x2 figure
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222, sharey=ax1)
ax3 = plt.subplot(223, sharex=ax1)
ax4 = plt.subplot(224, sharex=ax2, sharey=ax3)
#
ax1.plot(-1*ts, orb_2p_odeint.r(ts), 'k', alpha=0.5)
ax1.plot(-1*ts, orb_nfw_odeint.r(ts), 'b', alpha=0.5)
ax1.set_xlim(13.8, 0)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=22)
ax1.text(9, 200, 'odeint')
#
ax3.plot(-1*ts, orb_2p_leapfrog.r(ts), 'k', alpha=0.5)
ax3.plot(-1*ts, orb_nfw_leapfrog.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
ax3.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax3.text(9, 200, 'leapfrog')
#
ax2.plot(-1*ts, orb_2p_c_odeint.r(ts), 'k', label='2P', alpha=0.5)
ax2.plot(-1*ts, orb_nfw_c_odeint.r(ts), 'b', label='NFW', alpha=0.5)
ax2.set_xlim(13.8, 0)
ax2.label_outer()
ax2.legend(prop={'size': 16})
ax2.text(12, 150, 'circular orbit, odeint')
#
ax4.plot(-1*ts, orb_2p_c_leapfrog.r(ts), 'k', alpha=0.5)
ax4.plot(-1*ts, orb_nfw_c_leapfrog.r(ts), 'b', alpha=0.5)
ax4.set_xlim(13.8, 0)
ax4.label_outer()
ax4.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax4.text(12, 100, 'circular orbit, leapfrog')
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)



"""
    Now test several C methods in the NFW + disk potential for non-circular orbit
"""
# Import the potentials
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
halo_nfw = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)
#
potential_nfw = disk_inner+disk_outer+halo_nfw

# Set up orbit for a subhalo and integrate using odeint
orb_nfw_odeint = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_odeint.integrate(ts, potential_nfw, method='odeint')
orb_nfw_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_leapfrog.integrate(ts, potential_nfw, method='leapfrog')
orb_nfw_leapfrogc = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_leapfrogc.integrate(ts, potential_nfw, method='leapfrog_c')
orb_nfw_symplec4c = Orbit([sub_ics['R'][9]*(u.kpc), sub_ics['vR'][9]*(u.km/u.s), sub_ics['vT'][9]*(u.km/u.s), sub_ics['z'][9]*(u.kpc), sub_ics['vz'][9]*(u.km/u.s), sub_ics['phi'][9]*(u.deg)])
orb_nfw_symplec4c.integrate(ts, potential_nfw, method='symplec4_c')


# Plot in a 2x2 figure
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
ax1 = plt.subplot(411)
ax2 = plt.subplot(412, sharey=ax1)
ax3 = plt.subplot(413, sharex=ax2)
ax4 = plt.subplot(414, sharex=ax3)
#
ax1.plot(-1*ts, orb_nfw_odeint.r(ts), 'b', alpha=0.5)
ax1.set_xlim(13.8, 0)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=22)
ax1.text(8, 100, 'odeint')
#
ax2.plot(-1*ts, orb_nfw_leapfrog.r(ts), 'b', alpha=0.5)
ax2.set_xlim(13.8, 0)
ax2.label_outer()
ax2.set_ylabel('r [kpc]', fontsize=22)
ax2.text(8, 100, 'leapfrog')
#
ax3.plot(-1*ts, orb_nfw_leapfrogc.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
ax3.text(8, 100, 'leapfrog_c')
#
ax4.plot(-1*ts, orb_nfw_symplec4c.r(ts), 'b', alpha=0.5)
ax4.set_xlim(13.8, 0)
ax4.label_outer()
ax4.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax4.set_ylabel('r [kpc]', fontsize=22)
ax4.text(8, 100, 'symplec4_c')
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)



"""
    Now test several C methods in the NFW + disk potential for circular orbit
"""
# Import the potentials
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
halo_nfw = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)
#
potential_nfw = disk_inner+disk_outer+halo_nfw

v_c_nfw = vcirc(sub_ics['R'][9], nfw_halo_mass(sub_ics['R'][9], gal=sim_data.galaxy))

# Set up orbit for a subhalo and integrate using odeint
orb_nfw_odeint = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_odeint.integrate(ts, potential_nfw, method='odeint')
orb_nfw_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_leapfrog.integrate(ts, potential_nfw, method='leapfrog')
orb_nfw_leapfrogc = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_leapfrogc.integrate(ts, potential_nfw, method='leapfrog_c')
orb_nfw_symplec4c = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_symplec4c.integrate(ts, potential_nfw, method='symplec4_c')


# Plot in a 2x2 figure
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
ax1 = plt.subplot(411)
ax2 = plt.subplot(412, sharey=ax1)
ax3 = plt.subplot(413, sharex=ax2)
ax4 = plt.subplot(414, sharex=ax3)
#
ax1.plot(-1*ts, orb_nfw_odeint.r(ts), 'b', alpha=0.5)
ax1.set_xlim(13.8, 0)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=22)
#ax1.text(8, 100, 'odeint')
#
ax2.plot(-1*ts, orb_nfw_leapfrog.r(ts), 'b', alpha=0.5)
ax2.set_xlim(13.8, 0)
ax2.label_outer()
ax2.set_ylabel('r [kpc]', fontsize=22)
#ax2.text(8, 100, 'leapfrog')
#
ax3.plot(-1*ts, orb_nfw_leapfrogc.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
#ax3.text(8, 100, 'leapfrog_c')
#
ax4.plot(-1*ts, orb_nfw_symplec4c.r(ts), 'b', alpha=0.5)
ax4.set_xlim(13.8, 0)
ax4.label_outer()
ax4.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax4.set_ylabel('r [kpc]', fontsize=22)
#ax4.text(8, 100, 'symplec4_c')
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)


"""
    Now test several C methods in the NFW potential for circular orbit
"""
# Import the potentials
from galpy.potential import NFWPotential
#
halo_nfw = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)
#
potential_nfw = halo_nfw

v_c_nfw = vcirc(sub_ics['R'][9], nfw_halo_mass(sub_ics['R'][9], gal=sim_data.galaxy))

# Set up orbit for a subhalo and integrate using odeint
orb_nfw_odeint = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_odeint.integrate(ts, potential_nfw, method='odeint')
orb_nfw_leapfrog = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_leapfrog.integrate(ts, potential_nfw, method='leapfrog')
orb_nfw_leapfrogc = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_leapfrogc.integrate(ts, potential_nfw, method='leapfrog_c')
orb_nfw_symplec4c = Orbit([sub_ics['R'][9]*(u.kpc), 0.0*(u.km/u.s), v_c_nfw*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_symplec4c.integrate(ts, potential_nfw, method='symplec4_c')


# Plot in a 2x2 figure
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
ax1 = plt.subplot(411)
ax2 = plt.subplot(412, sharey=ax1)
ax3 = plt.subplot(413, sharex=ax2)
ax4 = plt.subplot(414, sharex=ax3)
#
ax1.plot(-1*ts, orb_nfw_odeint.r(ts), 'b', alpha=0.5)
ax1.set_xlim(13.8, 0)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=22)
#ax1.text(8, 100, 'odeint')
#
ax2.plot(-1*ts, orb_nfw_leapfrog.r(ts), 'b', alpha=0.5)
ax2.set_xlim(13.8, 0)
ax2.label_outer()
ax2.set_ylabel('r [kpc]', fontsize=22)
#ax2.text(8, 100, 'leapfrog')
#
ax3.plot(-1*ts, orb_nfw_leapfrogc.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
#ax3.text(8, 100, 'leapfrog_c')
#
ax4.plot(-1*ts, orb_nfw_symplec4c.r(ts), 'b', alpha=0.5)
ax4.set_xlim(13.8, 0)
ax4.label_outer()
ax4.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax4.set_ylabel('r [kpc]', fontsize=22)
#ax4.text(8, 100, 'symplec4_c')
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
