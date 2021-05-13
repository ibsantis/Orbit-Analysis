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
fitting_data_2p = pd.read_csv(sim_data.home_dir+'/orbit_data/param_2p_all.csv', index_col=0)
fitting_data_2p_2 = pd.read_csv(sim_data.home_dir+'/orbit_data/param_2p_gasdm.csv', index_col=0)
fitting_data_nfw = pd.read_csv(sim_data.home_dir+'/orbit_data/param_nfw_all.csv', index_col=0)
fitting_data_nfw_2 = pd.read_csv(sim_data.home_dir+'/orbit_data/param_nfw_gasdm.csv', index_col=0)
#
# Read in the subhalo initial conditions
sub_ics = pd.read_csv(sim_data.home_dir+'/orbit_data/'+sim_data.galaxy+'_subhalo_ics.csv', index_col=0)

#######################################################
#######################################################
"""
    First test the effects of just the halo potential
"""
#######################################################
#######################################################

# Define the enclosed mass profiles
ts = np.linspace(0.0, -13.78, 1379)*u.Gyr
#
def halo_mass(r, gal):
    A_halo = fitting_data_2p['A_halo'][gal]
    a_halo = fitting_data_2p['a_halo'][gal]
    alpha = fitting_data_2p['alpha'][gal]
    beta = fitting_data_2p['beta'][gal]
    #
    return (A_halo/(3-alpha))*((r/a_halo)**(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a_halo)
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
potential_two_power_2 = TwoPowerSphericalPotential(amp=fitting_data_2p_2['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p_2['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p_2['alpha'][sim_data.galaxy], beta=fitting_data_2p_2['beta'][sim_data.galaxy])
potential_nfw = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)
potential_nfw_2 = NFWPotential(amp=fitting_data_nfw_2['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw_2['a_halo'][sim_data.galaxy]*u.kpc)


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
#ax1.text(9, 600, 'odeint')
#
ax3.plot(-1*ts, orb_2p_leapfrog.r(ts), 'k', alpha=0.5)
ax3.plot(-1*ts, orb_nfw_leapfrog.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
ax3.set_xlabel('Lookback time [Gyr]', fontsize=22)
#ax3.text(9, 600, 'leapfrog')
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


# Plot many different orbits for 2P potential using all particles r > 10 kpc
# Did the same thing for the other halo models
r5 = 5
r10 = 10
r20 = 20
r50 = 50
r100 = 100
r300 = 300
#
vc_5 = vcirc(r5, halo_mass(r5, gal=sim_data.galaxy))
vc_10 = vcirc(r10, halo_mass(r10, gal=sim_data.galaxy))
vc_20 = vcirc(r20, halo_mass(r20, gal=sim_data.galaxy))
vc_50 = vcirc(r50, halo_mass(r50, gal=sim_data.galaxy))
vc_100 = vcirc(r100, halo_mass(r100, gal=sim_data.galaxy))
vc_300 = vcirc(r300, halo_mass(r300, gal=sim_data.galaxy))
#
orb_5 = Orbit([r5*(u.kpc), 0.0*(u.km/u.s), vc_5*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_5.integrate(ts, potential_two_power, method='odeint')
orb_10 = Orbit([r10*(u.kpc), 0.0*(u.km/u.s), vc_10*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_10.integrate(ts, potential_two_power, method='odeint')
orb_20 = Orbit([r20*(u.kpc), 0.0*(u.km/u.s), vc_20*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_20.integrate(ts, potential_two_power, method='odeint')
orb_50 = Orbit([r50*(u.kpc), 0.0*(u.km/u.s), vc_50*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_50.integrate(ts, potential_two_power, method='odeint')
orb_100 = Orbit([r100*(u.kpc), 0.0*(u.km/u.s), vc_100*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_100.integrate(ts, potential_two_power, method='odeint')
orb_300 = Orbit([r300*(u.kpc), 0.0*(u.km/u.s), vc_300*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_300.integrate(ts, potential_two_power, method='odeint')
#
#
# Plot
plt.figure(figsize=(10,8))
plt.plot(-1*ts, orb_5.r(ts))
plt.plot(-1*ts, orb_10.r(ts))
plt.plot(-1*ts, orb_20.r(ts))
plt.plot(-1*ts, orb_50.r(ts))
plt.plot(-1*ts, orb_100.r(ts))
plt.plot(-1*ts, orb_300.r(ts))
plt.xlim(13.8, 0)
plt.xlabel('lookback time [Gyr]', fontsize=28)
plt.ylabel('r [kpc]', fontsize=28)
plt.title('2P NFW (all particles r > 10 kpc)')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/galpy_tests/circular_orbit_2p_all.pdf')
plt.close()


# Plot all models on the same image for circular orbit
rr = 300 # kpc
vc_2p_1 = vcirc(rr, halo_mass(rr, gal=sim_data.galaxy))
vc_2p_2 = vcirc(rr, halo_mass_2(rr, gal=sim_data.galaxy))
vc_nfw_1 = vcirc(rr, nfw_halo_mass(rr, gal=sim_data.galaxy))
vc_nfw_2 = vcirc(rr, nfw_halo_mass_2(rr, gal=sim_data.galaxy))
#
orb_2p_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_2p_1*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_2p_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_2p_2*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_nfw_1*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_nfw_2*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
#
orb_2p_1.integrate(ts, potential_two_power, method='odeint')
orb_2p_2.integrate(ts, potential_two_power_2, method='odeint')
orb_nfw_1.integrate(ts, potential_nfw, method='odeint')
orb_nfw_2.integrate(ts, potential_nfw_2, method='odeint')
#
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
ax1 = plt.subplot(411)
ax2 = plt.subplot(412, sharey=ax1)
ax3 = plt.subplot(413, sharex=ax2)
ax4 = plt.subplot(414, sharex=ax3)
#
ax1.plot(-1*ts, orb_2p_1.r(ts), 'b', alpha=0.5)
ax1.set_xlim(13.8, 0)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=22)
ax1.text(10, (np.max(orb_2p_1.r(ts))+np.min(orb_2p_1.r(ts)))/2, '2P (all)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
ax2.plot(-1*ts, orb_2p_2.r(ts), 'b', alpha=0.5)
ax2.set_xlim(13.8, 0)
ax2.label_outer()
ax2.set_ylabel('r [kpc]', fontsize=22)
ax2.text(10, (np.max(orb_2p_2.r(ts))+np.min(orb_2p_2.r(ts)))/2, '2P (gas and dm only)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
ax3.plot(-1*ts, orb_nfw_1.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
ax3.text(10, (np.max(orb_nfw_1.r(ts))+np.min(orb_nfw_1.r(ts)))/2, 'NFW (all)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
ax4.plot(-1*ts, orb_nfw_2.r(ts), 'b', alpha=0.5)
ax4.set_xlim(13.8, 0)
ax4.label_outer()
ax4.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax4.set_ylabel('r [kpc]', fontsize=22)
ax4.text(10, (np.max(orb_nfw_2.r(ts))+np.min(orb_nfw_2.r(ts)))/2, 'NFW (gas and dm only)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)


#######################################################
#######################################################
"""
    Now test several C methods in the NFW potential for circular orbit
"""
#######################################################
#######################################################

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


#######################################################
#######################################################
"""
    Add in the disk component and see how the circular orbits change
"""
#######################################################
#######################################################

from galpy.potential import DoubleExponentialDiskPotential # For disks
#
#disk_outer = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
#disk_inner = DoubleExponentialDiskPotential(amp=fitting_data_2p['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data_2p['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data_2p['h_z'][sim_data.galaxy]*u.kpc)
disk_outer = DoubleExponentialDiskPotential(amp=(7.98410662301394E+08)*u.solMass/u.kpc**3, hr=4.4078991378626*u.kpc, hz=0.640059160684297*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=(6.46926397526533E+09)*u.solMass/u.kpc**3, hr=0.79064659178449*u.kpc, hz=0.640059160684297*u.kpc)
halo_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
halo_two_power_2 = TwoPowerSphericalPotential(amp=fitting_data_2p_2['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_2p_2['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p_2['alpha'][sim_data.galaxy], beta=fitting_data_2p_2['beta'][sim_data.galaxy])
halo_nfw = NFWPotential(amp=fitting_data_nfw['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw['a_halo'][sim_data.galaxy]*u.kpc)
halo_nfw_2 = NFWPotential(amp=fitting_data_nfw_2['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data_nfw_2['a_halo'][sim_data.galaxy]*u.kpc)
#
potential_two_power = disk_inner+disk_outer+halo_two_power
potential_two_power_2 = disk_inner+disk_outer+halo_two_power_2
potential_nfw = disk_inner+disk_outer+halo_nfw
potential_nfw_2 = disk_inner+disk_outer+halo_nfw_2
#
#potential_two_power = halo_two_power
#potential_two_power_2 = halo_two_power_2
#potential_nfw = halo_nfw
#potential_nfw_2 = halo_nfw_2

def disk_mass(r, gal):
    #A_disk_in = fitting_data_2p['A_disk_in'][gal]
    #r_in = fitting_data_2p['r_in'][gal]
    #A_disk_out = fitting_data_2p['A_disk_out'][gal]
    #r_out = fitting_data_2p['r_out'][gal]
    #h_z = fitting_data_2p['h_z'][gal]
    A_disk_in = 6.46926397526533E+09
    r_in = 0.79064659178449
    A_disk_out = 7.98410662301394E+08
    r_out = 4.4078991378626
    h_z = 0.640059160684297
    #
    mass_in = 4*np.pi*A_disk_in*h_z*r_in*(r_in-np.exp(-r/r_in)*(r_in+r))
    mass_out = 4*np.pi*A_disk_out*h_z*r_out*(r_out-np.exp(-r/r_out)*(r_out+r))
    return mass_in+mass_out


rr = 300
ts = np.linspace(0.0, -13.78, 1379)*u.Gyr

total_mass_1 = halo_mass(rr, gal=sim_data.galaxy)+disk_mass(rr, gal=sim_data.galaxy)
total_mass_2 = halo_mass_2(rr, gal=sim_data.galaxy)+disk_mass(rr, gal=sim_data.galaxy)
total_mass_nfw_1 = nfw_halo_mass(rr, gal=sim_data.galaxy)+disk_mass(rr, gal=sim_data.galaxy)
total_mass_nfw_2 = nfw_halo_mass_2(rr, gal=sim_data.galaxy)+disk_mass(rr, gal=sim_data.galaxy)

vc_2p_1 = vcirc(rr, total_mass_1)
vc_2p_2 = vcirc(rr, total_mass_2)
vc_nfw_1 = vcirc(rr, total_mass_nfw_1)
vc_nfw_2 = vcirc(rr, total_mass_nfw_2)

orb_2p_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_2p_1*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_2p_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_2p_2*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_nfw_1*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
orb_nfw_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), vc_nfw_2*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
#
#orb_2p_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), np.sqrt(0.5)*vc_2p_1*(u.km/u.s), 0.0*(u.kpc), np.sqrt(0.5)*vc_2p_1*(u.km/u.s), 0.0*(u.deg)]) # for polar orbits (kinda)
#orb_2p_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), np.sqrt(0.5)*vc_2p_2*(u.km/u.s), 0.0*(u.kpc), np.sqrt(0.5)*vc_2p_2*(u.km/u.s), 0.0*(u.deg)])
#orb_nfw_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), np.sqrt(0.5)*vc_nfw_1*(u.km/u.s), 0.0*(u.kpc), np.sqrt(0.5)*vc_nfw_1*(u.km/u.s), 0.0*(u.deg)])
#orb_nfw_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), np.sqrt(0.5)*vc_nfw_2*(u.km/u.s), 0.0*(u.kpc), np.sqrt(0.5)*vc_nfw_2*(u.km/u.s), 0.0*(u.deg)])
#
#orb_2p_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), 1*(u.km/u.s), 0.0*(u.kpc), (vc_2p_1-1)*(u.km/u.s), 0.0*(u.deg)]) # for near-polar orbits
#orb_2p_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), 1*(u.km/u.s), 0.0*(u.kpc), (vc_2p_2-1)*(u.km/u.s), 0.0*(u.deg)])
#orb_nfw_1 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), 1*(u.km/u.s), 0.0*(u.kpc), (vc_nfw_1-1)*(u.km/u.s), 0.0*(u.deg)])
#orb_nfw_2 = Orbit([rr*(u.kpc), 0.0*(u.km/u.s), 1*(u.km/u.s), 0.0*(u.kpc), (vc_nfw_2-1)*(u.km/u.s), 0.0*(u.deg)])
#
orb_2p_1.integrate(ts, potential_two_power, method='odeint')
orb_2p_2.integrate(ts, potential_two_power_2, method='odeint')
orb_nfw_1.integrate(ts, potential_nfw, method='odeint')
orb_nfw_2.integrate(ts, potential_nfw_2, method='odeint')
#
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(10, 12))
ax1 = plt.subplot(411)
ax2 = plt.subplot(412, sharey=ax1)
ax3 = plt.subplot(413, sharex=ax2)
ax4 = plt.subplot(414, sharex=ax3)
#
ax1.plot(-1*ts, orb_2p_1.r(ts), 'b', alpha=0.5)
ax1.set_xlim(13.8, 0)
ax1.label_outer()
ax1.set_ylabel('r [kpc]', fontsize=22)
ax1.text(10, (np.max(orb_2p_1.r(ts))+np.min(orb_2p_1.r(ts)))/2, '2P (all)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
ax2.plot(-1*ts, orb_2p_2.r(ts), 'b', alpha=0.5)
ax2.set_xlim(13.8, 0)
ax2.label_outer()
ax2.set_ylabel('r [kpc]', fontsize=22)
ax2.text(10, (np.max(orb_2p_2.r(ts))+np.min(orb_2p_2.r(ts)))/2, '2P (gas and dm only)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
ax3.plot(-1*ts, orb_nfw_1.r(ts), 'b', alpha=0.5)
ax3.set_xlim(13.8, 0)
ax3.label_outer()
ax3.set_ylabel('r [kpc]', fontsize=22)
ax3.text(10, (np.max(orb_nfw_1.r(ts))+np.min(orb_nfw_1.r(ts)))/2, 'NFW (all)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
ax4.plot(-1*ts, orb_nfw_2.r(ts), 'b', alpha=0.5)
ax4.set_xlim(13.8, 0)
ax4.label_outer()
ax4.set_xlabel('Lookback time [Gyr]', fontsize=22)
ax4.set_ylabel('r [kpc]', fontsize=22)
ax4.text(10, (np.max(orb_nfw_2.r(ts))+np.min(orb_nfw_2.r(ts)))/2, 'NFW (gas and dm only)', fontsize=18, bbox=dict(facecolor='black', alpha=0.3))
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)


############################################################################################################
