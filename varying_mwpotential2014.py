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
ts = numpy.linspace(0.0, -13.78, 1378)*u.Gyr


# Use the standard potential to integrate the orbits and print out properties
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
halo = NFWPotential(amp=G*0.8e12*M_sun, a=16.0*u.kpc, normalize=.35, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
MWP = bulge+disk+halo
orb_1.integrate(ts, MWP, method='odeint')
orb_2.integrate(ts, MWP, method='odeint')
orb_3.integrate(ts, MWP, method='odeint')
props_1_orig = numpy.array([orb_1.rap(), orb_1.rperi(), orb_1.E(), orb_1.Lz()])
props_2_orig = numpy.array([orb_2.rap(), orb_2.rperi(), orb_2.E(), orb_2.Lz()])
props_3_orig = numpy.array([orb_3.rap(), orb_3.rperi(), orb_3.E(), orb_3.Lz()])


# Set up emtpy array to save orbital parameters to as you vary disk scale height
props_1 = numpy.zeros((101, 4))
props_2 = numpy.zeros((101, 4))
props_3 = numpy.zeros((101, 4))
#
# Set bulge and halo potential terms
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
halo = NFWPotential(amp=G*0.8e12*M_sun, a=16.0*u.kpc, normalize=.35, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
#
# Set up array of disk scale heights
zs = numpy.arange(0, 10.1, 0.1)
#
for i in range(0, len(zs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=3.*u.kpc, b=zs[i]*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = bulge+disk+halo
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
plt.subplots_adjust(wspace=0.3, hspace=0.3)

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


################################################################################
"""
    Vary the scale radius and see how some orbital parameters change
"""

# Set up emtpy array to save orbital parameters to as you vary disk scale height
props_1 = numpy.zeros((191, 4))
props_2 = numpy.zeros((191, 4))
props_3 = numpy.zeros((191, 4))
#
# Set bulge and halo potential terms
bulge = PowerSphericalPotentialwCutoff(amp=G*0.5e10*u.M_sun/(4/3*numpy.pi*(1.9*u.kpc)**3), alpha=-1.8, rc=1.9*u.kpc, normalize=0.05, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
halo = NFWPotential(amp=G*0.8e12*M_sun, a=16.0*u.kpc, normalize=.35, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
#
# Set up array of disk scale radii
rs = numpy.arange(1, 20.1, 0.1)
#
for i in range(0, len(rs)):
    disk = MiyamotoNagaiPotential(amp=G*6.8e10*M_sun, a=rs[i]*u.kpc, b=0.28*u.kpc, normalize=.6, ro=8.0*u.kpc, vo=220.0*u.km/u.s)
    mwp = bulge+disk+halo
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









"""
##########################
# Generate a bunch of random orbits to test how varying the potential changes things
test = []
for i in range(0, 10):
    test.append([np.random.uniform(30,1000)*u.kpc, np.random.uniform(-125,125)*u.km/u.s, np.random.uniform(30, 200)*u.km/u.s, np.random.uniform(-700, 900)*u.kpc, np.random.uniform(-100, 160)*u.km/u.s, np.random.uniform(-90, 90)*u.deg])
    #test.append(Orbit(temp))

os = Orbit(test)
"""
