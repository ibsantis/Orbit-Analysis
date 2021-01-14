#!/usr/bin/python3

"""
============================
=  Varying MWPotential2014 =
============================

    - Plot the rotation curve for MWPotential2014
    - Plug in all of the same physical parameters for MWPotential2014
        - Check that each component returns the same parameters
    - Vary the disk scale height to see if there are any drastic
      changes to the orbital parameters
    - Vary the disk scale radius to see if there are any drastic
      changes to the orbital parameters
"""

from astropy.constants import G, M_sun
from galpy.orbit import Orbit
from galpy.potential import MiyamotoNagaiPotential, NFWPotential, PowerSphericalPotentialwCutoff, MWPotential2014
from galpy.potential import plotRotcurve
import numpy
import astropy.units as u
from matplotlib import pyplot as plt

########
# Plot v_rot using the potential with the default parameters
bp = PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
mp = MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
np = NFWPotential(a=16/8.,normalize=.35)
plotRotcurve(hp+mp+np,Rrange=[0.01,10.],grid=1001,yrange=[0.,1.2])


# Reconstruct bulge parameters and plot again
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
mp = MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
np = NFWPotential(a=16/8.,normalize=.35)
plotRotcurve(bulge+mp+np, Rrange=[0.8,80.], grid=1001, yrange=[0.,264.0])


# Reconstruct disk parameters and plot again
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
np = NFWPotential(a=16/8.,normalize=.35)
plotRotcurve(bulge+disk+np, Rrange=[0.8, 80.0], grid=1001, yrange=[0.0, 264.0])


# Reconstruct halo parameters and plot again
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
halo = NFWPotential(amp=G*0.8e12*M_sun, a=16.0*u.kpc, normalize=.35, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
plotRotcurve(bulge+disk+halo, Rrange=[0.8, 80.0], grid=1001, yrange=[0.0, 264.0])


################################################################################
"""
    Vary the scale height and see how some orbital parameters change
"""

# Generate orbits for 3 subhalos
orb_1 = Orbit([231.64*u.kpc, 12.99*u.km/u.s, 86.77*u.km/u.s, 3.80*u.kpc, -76.66*u.km/u.s, 37.89*u.deg])
orb_2 = Orbit([271.18*u.kpc, 36.82*u.km/u.s, 35.85*u.km/u.s, 3.93*u.kpc, -35.61*u.km/u.s, 45.41*u.deg])
orb_3 = Orbit([80.01*u.kpc, -4.90*u.km/u.s, 102.53*u.km/u.s, 3.92*u.kpc, -88.17*u.km/u.s, 44.53*u.deg])
#
# Checking same subhalos on purely R orbits
orb_1 = Orbit([231.64*u.kpc, 12.99*u.km/u.s, 86.77*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 37.89*u.deg])
orb_2 = Orbit([271.18*u.kpc, 36.82*u.km/u.s, 35.85*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 45.41*u.deg])
orb_3 = Orbit([80.01*u.kpc, -4.90*u.km/u.s, 102.53*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 44.53*u.deg])
#
# Checking same subhalos on purely z orbits
orb_1 = Orbit([231.64*u.kpc, 0.0*u.km/u.s, 86.77*u.km/u.s, 3.80*u.kpc, -76.66*u.km/u.s, 0.0*u.deg])
orb_2 = Orbit([271.18*u.kpc, 0.0*u.km/u.s, 35.85*u.km/u.s, 3.93*u.kpc, -35.61*u.km/u.s, 0.0*u.deg])
orb_3 = Orbit([80.01*u.kpc, 0.0*u.km/u.s, 102.53*u.km/u.s, 3.92*u.kpc, -88.17*u.km/u.s, 0.0*u.deg])
#
# Checking same subhalos on purely circular orbits
orb_1 = Orbit([231.64*u.kpc, 0.0*u.km/u.s, 86.77*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 37.89*u.deg])
orb_2 = Orbit([271.18*u.kpc, 0.0*u.km/u.s, 35.85*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 45.41*u.deg])
orb_3 = Orbit([80.01*u.kpc, 0.0*u.km/u.s, 102.53*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 44.53*u.deg])

#
ts = numpy.linspace(0.0, -13.78, 1378)*u.Gyr


# Use the standard potential to integrate the orbits and print out properties
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
halo = NFWPotential(amp=G*0.8e12*M_sun, a=16.0*u.kpc, normalize=.35, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
#MWP = bulge+disk+halo
MWP = disk
orb_1.integrate(ts, MWP, method='odeint')
orb_2.integrate(ts, MWP, method='odeint')
orb_3.integrate(ts, MWP, method='odeint')
props_1_orig = numpy.array([orb_1.rap(), orb_1.rperi(), orb_1.E(), orb_1.Lz()])
props_2_orig = numpy.array([orb_2.rap(), orb_2.rperi(), orb_2.E(), orb_2.Lz()])
props_3_orig = numpy.array([orb_3.rap(), orb_3.rperi(), orb_3.E(), orb_3.Lz()])


# Set up emtpy array to save orbital parameters to as you vary disk scale height
props_1 = numpy.zeros((41, 4))
props_2 = numpy.zeros((41, 4))
props_3 = numpy.zeros((41, 4))
#
# Set bulge and halo potential terms
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
halo = NFWPotential(amp=G*0.8e12*M_sun, a=16.0*u.kpc, normalize=.35, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
#
# Set up array of disk scale heights
zs = numpy.arange(0, 4.1, 0.1)
#
for i in range(0, len(zs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=zs[i]*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    #mwp = bulge+disk+halo
    mwp = disk
    #
    orb_1.integrate(ts, mwp, method='odeint')
    props_1[i,0] = orb_1.rap()
    props_1[i,1] = orb_1.rperi()
    props_1[i,2] = orb_1.E()
    props_1[i,3] = orb_1.Lz()
    #
    orb_2.integrate(ts, mwp, method='odeint')
    props_2[i,0] = orb_2.rap()
    props_2[i,1] = orb_2.rperi()
    props_2[i,2] = orb_2.E()
    props_2[i,3] = orb_2.Lz()
    #
    orb_3.integrate(ts, mwp, method='odeint')
    props_3[i,0] = orb_3.rap()
    props_3[i,1] = orb_3.rperi()
    props_3[i,2] = orb_3.E()
    props_3[i,3] = orb_3.Lz()


# Plot the original paramters and the other values
# Subhalo 3
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(0.28, props_1_orig[0], '.k')
axs[0,0].plot(zs, props_1[:,0])
axs[0,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{apo}$ [kpc]', fontsize=16)
#
axs[0,1].plot(0.28, props_1_orig[1], '.k')
axs[0,1].plot(zs, props_1[:,1])
axs[0,1].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
#
axs[1,0].plot(0.28, props_1_orig[2], '.k')
axs[1,0].plot(zs, props_1[:,2])
axs[1,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[1,0].set_ylabel('Energy', fontsize=16)
#
axs[1,1].plot(0.28, props_1_orig[3], '.k')
axs[1,1].plot(zs, props_1[:,3])
axs[1,1].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[1,1].set_ylabel('L$_{z}$', fontsize=16)
#
#plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('Subhalo 3, R = vR = vT = 0', fontsize=16, y=0.99)
plt.tight_layout()
plt.show()

# Subhalo 9
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(0.28, props_2_orig[0], '.k')
axs[0,0].plot(zs, props_2[:,0])
axs[0,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{apo}$ [kpc]', fontsize=16)
#
axs[0,1].plot(0.28, props_2_orig[1], '.k')
axs[0,1].plot(zs, props_2[:,1])
axs[0,1].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
#
axs[1,0].plot(0.28, props_2_orig[2], '.k')
axs[1,0].plot(zs, props_2[:,2])
axs[1,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[1,0].set_ylabel('Energy', fontsize=16)
#
axs[1,1].plot(0.28, props_2_orig[3], '.k')
axs[1,1].plot(zs, props_2[:,3])
axs[1,1].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[1,1].set_ylabel('L$_{z}$', fontsize=16)
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('Subhalo 9, R = vR = vT = 0', fontsize=16, y=0.99)
plt.tight_layout()

# Subhalo 32
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(0.28, props_3_orig[0], '.k')
axs[0,0].plot(zs, props_3[:,0])
axs[0,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{apo}$ [kpc]', fontsize=16)
#
axs[0,1].plot(0.28, props_3_orig[1], '.k')
axs[0,1].plot(zs, props_3[:,1])
axs[0,1].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
#
axs[1,0].plot(0.28, props_3_orig[2], '.k')
axs[1,0].plot(zs, props_3[:,2])
axs[1,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[1,0].set_ylabel('Energy', fontsize=16)
#
axs[1,1].plot(0.28, props_3_orig[3], '.k')
axs[1,1].plot(zs, props_3[:,3])
axs[1,1].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[1,1].set_ylabel('L$_{z}$', fontsize=16)
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('Subhalo 32, R = vR = vT = 0', fontsize=16, y=0.99)
plt.tight_layout()

# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(2,2)
#
#axs[0,0].plot(0.28, props_1_orig[1], '.k')
axs[0,0].plot(zs, props_1[:,1])
axs[0,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
axs[0,0].set_title('Subhalo 3', fontsize=16)
#
#axs[0,1].plot(0.28, props_2_orig[1], '.k')
axs[0,1].plot(zs, props_2[:,1])
axs[0,1].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
axs[0,1].set_title('Subhalo 9', fontsize=16)
#
#axs[1,0].plot(0.28, props_3_orig[1], '.k')
axs[1,0].plot(zs, props_3[:,1])
axs[1,0].set_xlabel('Scale Height [kpc]', fontsize=16)
axs[1,0].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
axs[1,0].set_title('Subhalo 32', fontsize=16)
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
#fig.suptitle('Subhalo 32, z = 0, vz = 0', fontsize=16, y=0.99)


################################################################################
"""
    Vary the scale radius and see how some orbital parameters change
"""

# Set up emtpy array to save orbital parameters to as you vary disk scale height
props_1 = numpy.zeros((101, 4))
props_2 = numpy.zeros((101, 4))
props_3 = numpy.zeros((101, 4))
#
# Set bulge and halo potential terms
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
halo = NFWPotential(amp=G*0.8e12*M_sun, a=16.0*u.kpc, normalize=.35, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
#
# Set up array of disk scale radii
rs = numpy.arange(0, 10.1, 0.1)
#
for i in range(0, len(rs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=rs[i]*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    #mwp = bulge+disk+halo
    mwp = disk
    #
    orb_1.integrate(ts, mwp, method='odeint')
    props_1[i,0] = orb_1.rap()
    props_1[i,1] = orb_1.rperi()
    props_1[i,2] = orb_1.E()
    props_1[i,3] = orb_1.Lz()
    #
    orb_2.integrate(ts, mwp, method='odeint')
    props_2[i,0] = orb_2.rap()
    props_2[i,1] = orb_2.rperi()
    props_2[i,2] = orb_2.E()
    props_2[i,3] = orb_2.Lz()
    #
    orb_3.integrate(ts, mwp, method='odeint')
    props_3[i,0] = orb_3.rap()
    props_3[i,1] = orb_3.rperi()
    props_3[i,2] = orb_3.E()
    props_3[i,3] = orb_3.Lz()


# Plot the original paramters and the other values
# Subhalo 3
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(3.0, props_1_orig[0], '.k')
axs[0,0].plot(rs, props_1[:,0])
axs[0,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{apo}$ [kpc]', fontsize=16)
#
axs[0,1].plot(3.0, props_1_orig[1], '.k')
axs[0,1].plot(rs, props_1[:,1])
axs[0,1].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
#
axs[1,0].plot(3.0, props_1_orig[2], '.k')
axs[1,0].plot(rs, props_1[:,2])
axs[1,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[1,0].set_ylabel('Energy', fontsize=16)
#
axs[1,1].plot(3.0, props_1_orig[3], '.k')
axs[1,1].plot(rs, props_1[:,3])
axs[1,1].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[1,1].set_ylabel('L$_{z}$', fontsize=16)
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('Subhalo 3, R = vR = vT = 0', fontsize=16, y=0.99)
plt.tight_layout()

# Subhalo 9
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(3.0, props_2_orig[0], '.k')
axs[0,0].plot(rs, props_2[:,0])
axs[0,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{apo}$ [kpc]', fontsize=16)
#
axs[0,1].plot(3.0, props_2_orig[1], '.k')
axs[0,1].plot(rs, props_2[:,1])
axs[0,1].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
#
axs[1,0].plot(3.0, props_2_orig[2], '.k')
axs[1,0].plot(rs, props_2[:,2])
axs[1,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[1,0].set_ylabel('Energy', fontsize=16)
#
axs[1,1].plot(3.0, props_2_orig[3], '.k')
axs[1,1].plot(rs, props_2[:,3])
axs[1,1].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[1,1].set_ylabel('L$_{z}$', fontsize=16)
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('Subhalo 9, R = vR = vT = 0', fontsize=16, y=0.99)
plt.tight_layout()

# Subhalo 32
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(3.0, props_3_orig[0], '.k')
axs[0,0].plot(rs, props_3[:,0])
axs[0,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{apo}$ [kpc]', fontsize=16)
#
axs[0,1].plot(3.0, props_3_orig[1], '.k')
axs[0,1].plot(rs, props_3[:,1])
axs[0,1].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
#
axs[1,0].plot(3.0, props_3_orig[2], '.k')
axs[1,0].plot(rs, props_3[:,2])
axs[1,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[1,0].set_ylabel('Energy', fontsize=16)
#
axs[1,1].plot(3.0, props_3_orig[3], '.k')
axs[1,1].plot(rs, props_3[:,3])
axs[1,1].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[1,1].set_ylabel('L$_{z}$', fontsize=16)
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('Subhalo 32, R = vR = vT = 0', fontsize=16, y=0.99)
plt.tight_layout()

# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(2,2)
#
#axs[0,0].plot(3.0, props_1_orig[1], '.k')
axs[0,0].plot(rs, props_1[:,1])
axs[0,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,0].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
axs[0,0].set_title('Subhalo 3', fontsize=16)
#
#axs[0,1].plot(3.0, props_2_orig[1], '.k')
axs[0,1].plot(rs, props_2[:,1])
axs[0,1].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[0,1].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
axs[0,1].set_title('Subhalo 9', fontsize=16)
#
#axs[1,0].plot(3.0, props_3_orig[1], '.k')
axs[1,0].plot(rs, props_3[:,1])
axs[1,0].set_xlabel('Scale Radius [kpc]', fontsize=16)
axs[1,0].set_ylabel('r$_{peri}$ [kpc]', fontsize=16)
axs[1,0].set_title('Subhalo 32', fontsize=16)
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)


#######################################################
"""
Play with subhalos that are on purely circular orbits and see how
changing the scale length and height changes their pericenters (and other props)
"""
# First change the R position to see how this changes the pericenters
orb_1 = Orbit([231.64*u.kpc, 0.0*u.km/u.s, 86.77*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 37.89*u.deg])
orb_2 = Orbit([271.18*u.kpc, 0.0*u.km/u.s, 35.85*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 45.41*u.deg])
orb_3 = Orbit([80.01*u.kpc, 0.0*u.km/u.s, 102.53*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 44.53*u.deg])

# First change the R position to see how this changes the pericenters
Rs = numpy.linspace(20, 300, 29)
orbits = []
for i in range(0, len(Rs)):
    orbits.append(Orbit([Rs[i]*u.kpc, 0.0*u.km/u.s, 86.77*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 37.89*u.deg]))
orbits_tot = Orbit(orbits)

# Set up array of disk scale heights
zs = numpy.arange(0, 4.1, 0.1)

props = numpy.zeros((len(zs), len(orbits_tot), 4))
for i in range(0, len(zs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=zs[i]*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = disk
    orbits_tot.integrate(ts, mwp, method='odeint')
    for j in range(0, len(orbits_tot)):
        props[i,j,0] = orbits_tot[j].rap()
        props[i,j,1] = orbits_tot[j].rperi()
        props[i,j,2] = orbits_tot[j].E()
        props[i,j,3] = orbits_tot[j].Lz()

# props has the shape (number of disks)x(number of subhalos)x(number of properties)
"""
Want to plot a single subhalo for all potentials -> props[:,i,N]
"""
# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(3,2)
#
axs[0,0].plot(zs, props[:,0,1], label='R = 20 kpc')
axs[0,0].plot(zs, props[:,1,1], label='R = 30 kpc')
axs[0,0].plot(zs, props[:,2,1], label='R = 40 kpc')
axs[0,0].plot(zs, props[:,3,1], label='R = 50 kpc')
axs[0,0].plot(zs, props[:,4,1], label='R = 60 kpc')
axs[0,0].legend()
#
axs[0,1].plot(zs, props[:,5,1], label='R = 70 kpc')
axs[0,1].plot(zs, props[:,6,1], label='R = 80 kpc')
axs[0,1].plot(zs, props[:,7,1], label='R = 90 kpc')
axs[0,1].plot(zs, props[:,8,1], label='R = 100 kpc')
axs[0,1].plot(zs, props[:,9,1], label='R = 110 kpc')
axs[0,1].legend()
#
axs[1,0].plot(zs, props[:,10,1], label='R = 120 kpc')
axs[1,0].plot(zs, props[:,11,1], label='R = 130 kpc')
axs[1,0].plot(zs, props[:,12,1], label='R = 140 kpc')
axs[1,0].plot(zs, props[:,13,1], label='R = 150 kpc')
axs[1,0].plot(zs, props[:,14,1], label='R = 160 kpc')
axs[1,0].legend()
#
axs[1,1].plot(zs, props[:,15,1], label='R = 170 kpc')
axs[1,1].plot(zs, props[:,16,1], label='R = 180 kpc')
axs[1,1].plot(zs, props[:,17,1], label='R = 190 kpc')
axs[1,1].plot(zs, props[:,18,1], label='R = 200 kpc')
axs[1,1].plot(zs, props[:,19,1], label='R = 210 kpc')
axs[1,1].legend()
#
axs[2,0].plot(zs, props[:,20,1], label='R = 220 kpc')
axs[2,0].plot(zs, props[:,21,1], label='R = 230 kpc')
axs[2,0].plot(zs, props[:,22,1], label='R = 240 kpc')
axs[2,0].plot(zs, props[:,23,1], label='R = 250 kpc')
axs[2,0].plot(zs, props[:,24,1], label='R = 260 kpc')
axs[2,0].legend()
#
axs[2,1].plot(zs, props[:,25,1], label='R = 270 kpc')
axs[2,1].plot(zs, props[:,26,1], label='R = 280 kpc')
axs[2,1].plot(zs, props[:,27,1], label='R = 290 kpc')
axs[2,1].plot(zs, props[:,28,1], label='R = 300 kpc')
axs[2,1].legend()
#
for ax in axs.flat:
    ax.set(xlabel='Scale Height [kpc]', ylabel='$r_{\\rm apo}$')
for ax in axs.flat:
    ax.label_outer()
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('R varying, vT = 86.77 km/s, $\phi=37.89$ deg', fontsize=16, y=0.999)


# Set up array of disk scale lengths
rs = numpy.arange(0, 10.1, 0.1)
#
props = numpy.zeros((len(rs), len(orbits_tot), 4))
for i in range(0, len(rs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=rs[i]*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = disk
    orbits_tot.integrate(ts, mwp, method='odeint')
    for j in range(0, len(orbits_tot)):
        props[i,j,0] = orbits_tot[j].rap()
        props[i,j,1] = orbits_tot[j].rperi()
        props[i,j,2] = orbits_tot[j].E()
        props[i,j,3] = orbits_tot[j].Lz()

# props has the shape (number of disks)x(number of subhalos)x(number of properties)
"""
Want to plot a single subhalo for all potentials -> props[:,i,N]
"""
# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(3,2)
#
axs[0,0].plot(rs, props[:,0,1], label='R = 20 kpc')
axs[0,0].plot(rs, props[:,1,1], label='R = 30 kpc')
axs[0,0].plot(rs, props[:,2,1], label='R = 40 kpc')
axs[0,0].plot(rs, props[:,3,1], label='R = 50 kpc')
axs[0,0].plot(rs, props[:,4,1], label='R = 60 kpc')
axs[0,0].legend()
#
axs[0,1].plot(rs, props[:,5,1], label='R = 70 kpc')
axs[0,1].plot(rs, props[:,6,1], label='R = 80 kpc')
axs[0,1].plot(rs, props[:,7,1], label='R = 90 kpc')
axs[0,1].plot(rs, props[:,8,1], label='R = 100 kpc')
axs[0,1].plot(rs, props[:,9,1], label='R = 110 kpc')
axs[0,1].legend()
#
axs[1,0].plot(rs, props[:,10,1], label='R = 120 kpc')
axs[1,0].plot(rs, props[:,11,1], label='R = 130 kpc')
axs[1,0].plot(rs, props[:,12,1], label='R = 140 kpc')
axs[1,0].plot(rs, props[:,13,1], label='R = 150 kpc')
axs[1,0].plot(rs, props[:,14,1], label='R = 160 kpc')
axs[1,0].legend()
#
axs[1,1].plot(rs, props[:,15,1], label='R = 170 kpc')
axs[1,1].plot(rs, props[:,16,1], label='R = 180 kpc')
axs[1,1].plot(rs, props[:,17,1], label='R = 190 kpc')
axs[1,1].plot(rs, props[:,18,1], label='R = 200 kpc')
axs[1,1].plot(rs, props[:,19,1], label='R = 210 kpc')
axs[1,1].legend()
#
axs[2,0].plot(rs, props[:,20,1], label='R = 220 kpc')
axs[2,0].plot(rs, props[:,21,1], label='R = 230 kpc')
axs[2,0].plot(rs, props[:,22,1], label='R = 240 kpc')
axs[2,0].plot(rs, props[:,23,1], label='R = 250 kpc')
axs[2,0].plot(rs, props[:,24,1], label='R = 260 kpc')
axs[2,0].legend()
#
axs[2,1].plot(rs, props[:,25,1], label='R = 270 kpc')
axs[2,1].plot(rs, props[:,26,1], label='R = 280 kpc')
axs[2,1].plot(rs, props[:,27,1], label='R = 290 kpc')
axs[2,1].plot(rs, props[:,28,1], label='R = 300 kpc')
axs[2,1].legend()
#
for ax in axs.flat:
    ax.set(xlabel='Scale Radius [kpc]', ylabel='$r_{\\rm apo}$')
for ax in axs.flat:
    ax.label_outer()
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('R varying, vT = 86.77 km/s, $\phi=37.89$ deg', fontsize=16, y=0.999)


#######################################################
# First change the R position to see how this changes the pericenters
orb_1 = Orbit([231.64*u.kpc, 0.0*u.km/u.s, 86.77*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 37.89*u.deg])
orb_2 = Orbit([271.18*u.kpc, 0.0*u.km/u.s, 35.85*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 45.41*u.deg])
orb_3 = Orbit([80.01*u.kpc, 0.0*u.km/u.s, 102.53*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 44.53*u.deg])

# Change vT to see how this changes the pericenters
vTs = numpy.linspace(10, 200, 20)
orbits = []
for i in range(0, len(vTs)):
    orbits.append(Orbit([100*u.kpc, 0.0*u.km/u.s, vTs[i]*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 37.89*u.deg]))
orbits_tot = Orbit(orbits)

# Set up array of disk scale heights
zs = numpy.arange(0, 4.1, 0.1)

props = numpy.zeros((len(zs), len(orbits_tot), 4))
for i in range(0, len(zs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=zs[i]*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = disk
    orbits_tot.integrate(ts, mwp, method='odeint')
    for j in range(0, len(orbits_tot)):
        props[i,j,0] = orbits_tot[j].rap()
        props[i,j,1] = orbits_tot[j].rperi()
        props[i,j,2] = orbits_tot[j].E()
        props[i,j,3] = orbits_tot[j].Lz()

# props has the shape (number of disks)x(number of subhalos)x(number of properties)
"""
Want to plot a single subhalo for all potentials -> props[:,i,N]
"""
# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(zs, props[:,0,1], label='$v_{\\rm tan}$ = 10 km/s')
axs[0,0].plot(zs, props[:,1,1], label='$v_{\\rm tan}$ = 20 km/s')
axs[0,0].plot(zs, props[:,2,1], label='$v_{\\rm tan}$ = 30 km/s')
axs[0,0].plot(zs, props[:,3,1], label='$v_{\\rm tan}$ = 40 km/s')
axs[0,0].plot(zs, props[:,4,1], label='$v_{\\rm tan}$ = 50 km/s')
axs[0,0].legend()
#
axs[0,1].plot(zs, props[:,5,1], label='$v_{\\rm tan}$ = 60 km/s')
axs[0,1].plot(zs, props[:,6,1], label='$v_{\\rm tan}$ = 70 km/s')
axs[0,1].plot(zs, props[:,7,1], label='$v_{\\rm tan}$ = 80 km/s')
axs[0,1].plot(zs, props[:,8,1], label='$v_{\\rm tan}$ = 90 km/s')
axs[0,1].plot(zs, props[:,9,1], label='$v_{\\rm tan}$ = 100 km/s')
axs[0,1].legend()
#
axs[1,0].plot(zs, props[:,10,1], label='$v_{\\rm tan}$ = 110 km/s')
axs[1,0].plot(zs, props[:,11,1], label='$v_{\\rm tan}$ = 120 km/s')
axs[1,0].plot(zs, props[:,12,1], label='$v_{\\rm tan}$ = 130 km/s')
axs[1,0].plot(zs, props[:,13,1], label='$v_{\\rm tan}$ = 140 km/s')
axs[1,0].plot(zs, props[:,14,1], label='$v_{\\rm tan}$ = 150 km/s')
axs[1,0].legend()
#
axs[1,1].plot(zs, props[:,15,1], label='$v_{\\rm tan}$ = 160 km/s')
axs[1,1].plot(zs, props[:,16,1], label='$v_{\\rm tan}$ = 170 km/s')
axs[1,1].plot(zs, props[:,17,1], label='$v_{\\rm tan}$ = 180 km/s')
axs[1,1].plot(zs, props[:,18,1], label='$v_{\\rm tan}$ = 190 km/s')
axs[1,1].plot(zs, props[:,19,1], label='$v_{\\rm tan}$ = 200 km/s')
axs[1,1].legend()
#
for ax in axs.flat:
    ax.set(xlabel='Scale Height [kpc]', ylabel='$r_{\\rm apo}$')
for ax in axs.flat:
    ax.label_outer()
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('R = 100 kpc, vT varies, $\phi=37.89$ deg', fontsize=16, y=0.999)


# Set up array of disk scale lengths
rs = numpy.arange(0, 10.1, 0.1)
#
props = numpy.zeros((len(rs), len(orbits_tot), 4))
for i in range(0, len(rs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=rs[i]*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = disk
    orbits_tot.integrate(ts, mwp, method='odeint')
    for j in range(0, len(orbits_tot)):
        props[i,j,0] = orbits_tot[j].rap()
        props[i,j,1] = orbits_tot[j].rperi()
        props[i,j,2] = orbits_tot[j].E()
        props[i,j,3] = orbits_tot[j].Lz()

# props has the shape (number of disks)x(number of subhalos)x(number of properties)
"""
Want to plot a single subhalo for all potentials -> props[:,i,N]
"""
# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(2,2)
#
axs[0,0].plot(rs, props[:,0,1], label='$v_{\\rm tan}$ = 10 km/s')
axs[0,0].plot(rs, props[:,1,1], label='$v_{\\rm tan}$ = 20 km/s')
axs[0,0].plot(rs, props[:,2,1], label='$v_{\\rm tan}$ = 30 km/s')
axs[0,0].plot(rs, props[:,3,1], label='$v_{\\rm tan}$ = 40 km/s')
axs[0,0].plot(rs, props[:,4,1], label='$v_{\\rm tan}$ = 50 km/s')
axs[0,0].legend()
#
axs[0,1].plot(rs, props[:,5,1], label='$v_{\\rm tan}$ = 60 km/s')
axs[0,1].plot(rs, props[:,6,1], label='$v_{\\rm tan}$ = 70 km/s')
axs[0,1].plot(rs, props[:,7,1], label='$v_{\\rm tan}$ = 80 km/s')
axs[0,1].plot(rs, props[:,8,1], label='$v_{\\rm tan}$ = 90 km/s')
axs[0,1].plot(rs, props[:,9,1], label='$v_{\\rm tan}$ = 100 km/s')
axs[0,1].legend()
#
axs[1,0].plot(rs, props[:,10,1], label='$v_{\\rm tan}$ = 110 km/s')
axs[1,0].plot(rs, props[:,11,1], label='$v_{\\rm tan}$ = 120 km/s')
axs[1,0].plot(rs, props[:,12,1], label='$v_{\\rm tan}$ = 130 km/s')
axs[1,0].plot(rs, props[:,13,1], label='$v_{\\rm tan}$ = 140 km/s')
axs[1,0].plot(rs, props[:,14,1], label='$v_{\\rm tan}$ = 150 km/s')
axs[1,0].legend()
#
axs[1,1].plot(rs, props[:,15,1], label='$v_{\\rm tan}$ = 160 km/s')
axs[1,1].plot(rs, props[:,16,1], label='$v_{\\rm tan}$ = 170 km/s')
axs[1,1].plot(rs, props[:,17,1], label='$v_{\\rm tan}$ = 180 km/s')
axs[1,1].plot(rs, props[:,18,1], label='$v_{\\rm tan}$ = 190 km/s')
axs[1,1].plot(rs, props[:,19,1], label='$v_{\\rm tan}$ = 200 km/s')
axs[1,1].legend()
#
for ax in axs.flat:
    ax.set(xlabel='Scale Radius [kpc]', ylabel='$r_{\\rm apo}$')
for ax in axs.flat:
    ax.label_outer()
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('R = 100 kpc, vT varies, $\phi=37.89$ deg', fontsize=16, y=0.999)


#######################################################
orb_1 = Orbit([231.64*u.kpc, 0.0*u.km/u.s, 86.77*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 37.89*u.deg])
orb_2 = Orbit([271.18*u.kpc, 0.0*u.km/u.s, 35.85*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 45.41*u.deg])
orb_3 = Orbit([80.01*u.kpc, 0.0*u.km/u.s, 102.53*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, 44.53*u.deg])

# Change phi to see how this changes the pericenters
phis = numpy.linspace(0, 360, 37)
orbits = []
for i in range(0, len(phis)):
    orbits.append(Orbit([100*u.kpc, 0.0*u.km/u.s, 50.0*u.km/u.s, 0.0*u.kpc, 0.0*u.km/u.s, phis[i]*u.deg]))
orbits_tot = Orbit(orbits)

# Set up array of disk scale heights
zs = numpy.arange(0, 4.1, 0.1)

props = numpy.zeros((len(zs), len(orbits_tot), 4))
for i in range(0, len(zs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=zs[i]*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = disk
    orbits_tot.integrate(ts, mwp, method='odeint')
    for j in range(0, len(orbits_tot)):
        props[i,j,0] = orbits_tot[j].rap()
        props[i,j,1] = orbits_tot[j].rperi()
        props[i,j,2] = orbits_tot[j].E()
        props[i,j,3] = orbits_tot[j].Lz()

# props has the shape (number of disks)x(number of subhalos)x(number of properties)
"""
Want to plot a single subhalo for all potentials -> props[:,i,N]
"""
# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(3,2)
#
axs[0,0].plot(zs, props[:,0,1], label='$\phi$ = 0 deg')
axs[0,0].plot(zs, props[:,1,1], label='$\phi$ = 10 deg')
axs[0,0].plot(zs, props[:,2,1], label='$\phi$ = 20 deg')
axs[0,0].plot(zs, props[:,3,1], label='$\phi$ = 30 deg')
axs[0,0].plot(zs, props[:,4,1], label='$\phi$ = 40 deg')
axs[0,0].plot(zs, props[:,5,1], label='$\phi$ = 50 deg')
axs[0,0].plot(zs, props[:,6,1], label='$\phi$ = 60 deg')
axs[0,0].legend()
#
axs[0,1].plot(zs, props[:,7,1], label='$\phi$ = 70 deg')
axs[0,1].plot(zs, props[:,8,1], label='$\phi$ = 80 deg')
axs[0,1].plot(zs, props[:,9,1], label='$\phi$ = 90 deg')
axs[0,1].plot(zs, props[:,10,1], label='$\phi$ = 100 deg')
axs[0,1].plot(zs, props[:,11,1], label='$\phi$ = 110 deg')
axs[0,1].plot(zs, props[:,12,1], label='$\phi$ = 120 deg')
axs[0,1].legend()
#
axs[1,0].plot(zs, props[:,13,1], label='$\phi$ = 130 deg')
axs[1,0].plot(zs, props[:,14,1], label='$\phi$ = 140 deg')
axs[1,0].plot(zs, props[:,15,1], label='$\phi$ = 150 deg')
axs[1,0].plot(zs, props[:,16,1], label='$\phi$ = 160 deg')
axs[1,0].plot(zs, props[:,17,1], label='$\phi$ = 170 deg')
axs[1,0].plot(zs, props[:,18,1], label='$\phi$ = 180 deg')
axs[1,0].legend()
#
axs[1,1].plot(zs, props[:,19,1], label='$\phi$ = 190 deg')
axs[1,1].plot(zs, props[:,20,1], label='$\phi$ = 200 deg')
axs[1,1].plot(zs, props[:,21,1], label='$\phi$ = 210 deg')
axs[1,1].plot(zs, props[:,22,1], label='$\phi$ = 220 deg')
axs[1,1].plot(zs, props[:,23,1], label='$\phi$ = 230 deg')
axs[1,1].plot(zs, props[:,24,1], label='$\phi$ = 240 deg')
axs[1,1].legend()
#
axs[2,0].plot(zs, props[:,25,1], label='$\phi$ = 250 deg')
axs[2,0].plot(zs, props[:,26,1], label='$\phi$ = 260 deg')
axs[2,0].plot(zs, props[:,27,1], label='$\phi$ = 270 deg')
axs[2,0].plot(zs, props[:,28,1], label='$\phi$ = 280 deg')
axs[2,0].plot(zs, props[:,29,1], label='$\phi$ = 290 deg')
axs[2,0].plot(zs, props[:,30,1], label='$\phi$ = 300 deg')
axs[2,0].legend()
#
axs[2,1].plot(zs, props[:,31,1], label='$\phi$ = 310 deg')
axs[2,1].plot(zs, props[:,32,1], label='$\phi$ = 320 deg')
axs[2,1].plot(zs, props[:,33,1], label='$\phi$ = 330 deg')
axs[2,1].plot(zs, props[:,34,1], label='$\phi$ = 340 deg')
axs[2,1].plot(zs, props[:,35,1], label='$\phi$ = 350 deg')
axs[2,1].plot(zs, props[:,36,1], label='$\phi$ = 360 deg')
axs[2,1].legend()
#
for ax in axs.flat:
    ax.set(xlabel='Scale Height [kpc]', ylabel='$r_{\\rm apo}$')
for ax in axs.flat:
    ax.label_outer()
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('R = 100 kpc, vT = 50 km/s, $\phi$ varies', fontsize=16, y=0.999)


# Set up array of disk scale lengths
rs = numpy.arange(0, 10.1, 0.1)
#
props = numpy.zeros((len(rs), len(orbits_tot), 4))
for i in range(0, len(rs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=rs[i]*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = disk
    orbits_tot.integrate(ts, mwp, method='odeint')
    for j in range(0, len(orbits_tot)):
        props[i,j,0] = orbits_tot[j].rap()
        props[i,j,1] = orbits_tot[j].rperi()
        props[i,j,2] = orbits_tot[j].E()
        props[i,j,3] = orbits_tot[j].Lz()

# props has the shape (number of disks)x(number of subhalos)x(number of properties)
"""
Want to plot a single subhalo for all potentials -> props[:,i,N]
"""
# Plot all changes in pericenters on same plot
fig, axs = plt.subplots(3,2)
#
axs[0,0].plot(rs, props[:,0,1], label='$\phi$ = 0 deg')
axs[0,0].plot(rs, props[:,1,1], label='$\phi$ = 10 deg')
axs[0,0].plot(rs, props[:,2,1], label='$\phi$ = 20 deg')
axs[0,0].plot(rs, props[:,3,1], label='$\phi$ = 30 deg')
axs[0,0].plot(rs, props[:,4,1], label='$\phi$ = 40 deg')
axs[0,0].plot(rs, props[:,5,1], label='$\phi$ = 50 deg')
axs[0,0].plot(rs, props[:,6,1], label='$\phi$ = 60 deg')
axs[0,0].legend()
#
axs[0,1].plot(rs, props[:,7,1], label='$\phi$ = 70 deg')
axs[0,1].plot(rs, props[:,8,1], label='$\phi$ = 80 deg')
axs[0,1].plot(rs, props[:,9,1], label='$\phi$ = 90 deg')
axs[0,1].plot(rs, props[:,10,1], label='$\phi$ = 100 deg')
axs[0,1].plot(rs, props[:,11,1], label='$\phi$ = 110 deg')
axs[0,1].plot(rs, props[:,12,1], label='$\phi$ = 120 deg')
axs[0,1].legend()
#
axs[1,0].plot(rs, props[:,13,1], label='$\phi$ = 130 deg')
axs[1,0].plot(rs, props[:,14,1], label='$\phi$ = 140 deg')
axs[1,0].plot(rs, props[:,15,1], label='$\phi$ = 150 deg')
axs[1,0].plot(rs, props[:,16,1], label='$\phi$ = 160 deg')
axs[1,0].plot(rs, props[:,17,1], label='$\phi$ = 170 deg')
axs[1,0].plot(rs, props[:,18,1], label='$\phi$ = 180 deg')
axs[1,0].legend()
#
axs[1,1].plot(rs, props[:,19,1], label='$\phi$ = 190 deg')
axs[1,1].plot(rs, props[:,20,1], label='$\phi$ = 200 deg')
axs[1,1].plot(rs, props[:,21,1], label='$\phi$ = 210 deg')
axs[1,1].plot(rs, props[:,22,1], label='$\phi$ = 220 deg')
axs[1,1].plot(rs, props[:,23,1], label='$\phi$ = 230 deg')
axs[1,1].plot(rs, props[:,24,1], label='$\phi$ = 240 deg')
axs[1,1].legend()
#
axs[2,0].plot(rs, props[:,25,1], label='$\phi$ = 250 deg')
axs[2,0].plot(rs, props[:,26,1], label='$\phi$ = 260 deg')
axs[2,0].plot(rs, props[:,27,1], label='$\phi$ = 270 deg')
axs[2,0].plot(rs, props[:,28,1], label='$\phi$ = 280 deg')
axs[2,0].plot(rs, props[:,29,1], label='$\phi$ = 290 deg')
axs[2,0].plot(rs, props[:,30,1], label='$\phi$ = 300 deg')
axs[2,0].legend()
#
axs[2,1].plot(rs, props[:,31,1], label='$\phi$ = 310 deg')
axs[2,1].plot(rs, props[:,32,1], label='$\phi$ = 320 deg')
axs[2,1].plot(rs, props[:,33,1], label='$\phi$ = 330 deg')
axs[2,1].plot(rs, props[:,34,1], label='$\phi$ = 340 deg')
axs[2,1].plot(rs, props[:,35,1], label='$\phi$ = 350 deg')
axs[2,1].plot(rs, props[:,36,1], label='$\phi$ = 360 deg')
axs[2,1].legend()
#
for ax in axs.flat:
    ax.set(xlabel='Scale Radius [kpc]', ylabel='$r_{\\rm apo}$')
for ax in axs.flat:
    ax.label_outer()
#
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle('R = 100 kpc, vT = 50 km/s, $\phi$ varies', fontsize=16, y=0.999)










"""
##########################
# Generate a bunch of random orbits to test how varying the potential changes things
test = []
for i in range(0, 10):
    test.append([np.random.uniform(30,1000)*u.kpc, np.random.uniform(-125,125)*u.km/u.s, np.random.uniform(30, 200)*u.km/u.s, np.random.uniform(-700, 900)*u.kpc, np.random.uniform(-100, 160)*u.km/u.s, np.random.uniform(-90, 90)*u.deg])
    #test.append(Orbit(temp))

os = Orbit(test)
"""
