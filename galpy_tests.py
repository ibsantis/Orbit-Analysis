

"""

    ===============
    = Galpy Tests =
    ===============

    [Write something later...]

"""

# Import packages
from galpy.orbit import Orbit
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

# Import the potentials and create custom ones
from galpy.potential import MWPotential2014
from galpy.potential import MiyamotoNagaiPotential # For disks
from galpy.potential import NFWPotential # For DM halos
from galpy.potential import KeplerPotential
from galpy.util import bovy_conversion
#
miya = MiyamotoNagaiPotential(a=3.0/8.0, b=0.28/8.0, normalize=0.6)
nfw = NFWPotential(a=16.0/8.0, normalize=0.35)
MWPotential_2 = miya+nfw
#
miya = MiyamotoNagaiPotential(a=3.0/8.0, b=0.28/8.0, normalize=0.6)
nfw = NFWPotential(a=16.0/8.0, normalize=0.35)
kep = KeplerPotential(amp=4*10**6./bovy_conversion.mass_in_msol(220.,8.))
MWPotential_3 = miya+nfw+kep

# Read in the halo tree
halt = halo.io.IO.read_tree(simulation_directory=simulation_dir, file_kind='hdf5', species='star')

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
orb_1 = Orbit([231.64*u.kpc, 12.99*u.km/u.s, 86.77*u.km/u.s, 3.80*u.kpc, -76.66*u.km/u.s, 37.89*u.deg])
orb_2 = Orbit([271.18*u.kpc, 36.82*u.km/u.s, 35.85*u.km/u.s, 3.93*u.kpc, -35.61*u.km/u.s, 45.41*u.deg])
orb_3 = Orbit([80.01*u.kpc, -4.90*u.km/u.s, 102.53*u.km/u.s, 3.92*u.kpc, -88.17*u.km/u.s, 44.53*u.deg])

# Set up time array (negative because integrating backward)
ts = np.linspace(0.0, -13.78, 1378)*u.Gyr

### MWPotential2014
## HALO 1
#
# Set up time array to integrate the orbits
orb_1.integrate(ts, MWPotential2014, method='odeint')
#
# Plot a 2D and 3D version of the plot and save to a figure
plot_1_2D_MWP, = orb_1.plot(d1='x', d2='y')
plot_1_3D_MWP, = orb_1.plot3d()
#
plot_1_2D_MWP.figure.savefig('halo_1_2D_MWP')
plot_1_3D_MWP.figure.savefig('halo_1_3D_MWP')
#
# Print out the pericenter distance
print('Pericenter distance for halo 1 in MWPotential2014: {0}'.format(orb_1.rperi()))
print('Apocenter distance for halo 1 in MWPotential2014: {0}'.format(orb_1.rap()))

## HALO 2
#
# Initialize the orbit and integrate it
orb_2.integrate(ts, MWPotential2014, method='odeint')
#
# Plot a 2D and 3D version of the plot and save to a figure
plot_2_2D_MWP, = orb_2.plot(d1='x', d2='y')
plot_2_3D_MWP, = orb_2.plot3d()
#
plot_2_2D_MWP.figure.savefig('halo_2_2D_MWP')
plot_2_3D_MWP.figure.savefig('halo_2_3D_MWP')
#
# Print out the pericenter distance
print('Pericenter distance for halo 2 in MWPotential2014: {0}'.format(orb_2.rperi()))
print('Apocenter distance for halo 2 in MWPotential2014: {0}'.format(orb_2.rap()))

## HALO 3
orb_3.integrate(ts, MWPotential2014, method='odeint')
#
# Plot a 2D and 3D version of the plot and save to a figure
plot_3_2D_MWP, = orb_3.plot(d1='x', d2='y')
plot_3_3D_MWP, = orb_3.plot3d()
#
plot_3_2D_MWP.figure.savefig('halo_3_2D_MWP')
plot_3_3D_MWP.figure.savefig('halo_3_3D_MWP')
#
# Print out the pericenter distance
print('Pericenter distance for halo 3 in MWPotential2014: {0}'.format(orb_3.rperi()))
print('Apocenter distance for halo 3 in MWPotential2014: {0}'.format(orb_3.rap()))

## ALL HALOS
# Combine the orbits and integrate them at the same time
#orbs = [orb_1, orb_2, orb_3]
#orb_tot = Orbit(orbs)
#orb_tot.integrate(ts, MWPotential2014, method='odeint')
#
# Plot them on the same figure
plot_tot_2D_MWP, = orb_1.plot(d1='x', d2='y')
plot_tot_2D_MWP, = orb_2.plot(d1='x', d2='y', overplot=True)
plot_tot_2D_MWP, = orb_3.plot(d1='x', d2='y', overplot=True)
plot_tot_3D_MWP, = orb_1.plot3d()
plot_tot_3D_MWP, = orb_2.plot3d(overplot=True)
plot_tot_3D_MWP, = orb_3.plot3d(overplot=True)
#
plot_tot_2D_MWP.figure.savefig('halos_2D_MWP')
plot_tot_3D_MWP.figure.savefig('halos_3D_MWP')


### Combine both the disk and halo potentials and re-do everything; using same params as the tutorial
#
orb_1.integrate(ts, MWPotential_2, method='odeint')
#
plot_1_2D_nfw_miya, = orb_1.plot(d1='x', d2='y')
plot_1_3D_nfw_miya, = orb_1.plot3d()
#
plot_1_2D_nfw_miya.figure.savefig('halo_1_2D_nfw_miya')
plot_1_3D_nfw_miya.figure.savefig('halo_1_3D_nfw_miya')
#
print('Pericenter distance for halo 1 in NFW+Disk: {0}'.format(orb_1.rperi()))
print('Apocenter distance for halo 1 in NFW+Disk: {0}'.format(orb_1.rap()))
#
orb_2.integrate(ts, MWPotential_2, method='odeint')
#
plot_2_2D_nfw_miya, = orb_2.plot(d1='x', d2='y')
plot_2_3D_nfw_miya, = orb_2.plot3d()
#
plot_2_2D_nfw_miya.figure.savefig('halo_2_2D_nfw_miya')
plot_2_3D_nfw_miya.figure.savefig('halo_2_3D_nfw_miya')
#
# Print out the pericenter distance
print('Pericenter distance for halo 2 in NFW+Disk: {0}'.format(orb_2.rperi()))
print('Apocenter distance for halo 2 in NFW+Disk: {0}'.format(orb_2.rap()))
#
orb_3.integrate(ts, MWPotential_2, method='odeint')
#
plot_3_2D_nfw_miya, = orb_3.plot(d1='x', d2='y')
plot_3_3D_nfw_miya, = orb_3.plot3d()
#
plot_3_2D_nfw_miya.figure.savefig('halo_3_2D_nfw_miya')
plot_3_3D_nfw_miya.figure.savefig('halo_3_3D_nfw_miya')
#
# Print out the pericenter distance
print('Pericenter distance for halo 3 in NFW+Disk: {0}'.format(orb_3.rperi()))
print('Apocenter distance for halo 3 in NFW+Disk: {0}'.format(orb_3.rap()))

## BOTH HALOS
# Combine the orbits and integrate them at the same time
#orbs = [orb_1, orb_2]
#orb_tot = Orbit(orbs)
#orb_tot.integrate(ts, MWPotential_2, method='odeint')
#
# Plot them on the same figure
plot_tot_2D_nfw_miya, = orb_1.plot(d1='x', d2='y')
plot_tot_2D_nfw_miya, = orb_2.plot(d1='x', d2='y', overplot=True)
plot_tot_2D_nfw_miya, = orb_3.plot(d1='x', d2='y', overplot=True)
plot_tot_3D_nfw_miya, = orb_1.plot3d()
plot_tot_3D_nfw_miya, = orb_2.plot3d(overplot=True)
plot_tot_3D_nfw_miya, = orb_3.plot3d(overplot=True)
#
plot_tot_2D_nfw_miya.figure.savefig('halos_2D_nfw_miya')
plot_tot_3D_nfw_miya.figure.savefig('halos_3D_nfw_miya')


### Combine both the disk and halo potentials with a blackhole
#
orb_1.integrate(ts, MWPotential_3, method='odeint')
#
plot_1_2D_nfw_miya_bh, = orb_1.plot(d1='x', d2='y')
plot_1_3D_nfw_miya_bh, = orb_1.plot3d()
#
plot_1_2D_nfw_miya_bh.figure.savefig('halo_1_2D_nfw_miya_bh')
plot_1_3D_nfw_miya_bh.figure.savefig('halo_1_3D_nfw_miya_bh')
#
print('Pericenter distance for halo 1 in NFW+Disk+BH: {0}'.format(orb_1.rperi()))
print('Apocenter distance for halo 1 in NFW+Disk+BH: {0}'.format(orb_1.rap()))
#
orb_2.integrate(ts, MWPotential_3, method='odeint')
#
plot_2_2D_nfw_miya_bh, = orb_2.plot(d1='x', d2='y')
plot_2_3D_nfw_miya_bh, = orb_2.plot3d()
#
plot_2_2D_nfw_miya_bh.figure.savefig('halo_2_2D_nfw_miya_bh')
plot_2_3D_nfw_miya_bh.figure.savefig('halo_2_3D_nfw_miya_bh')
#
# Print out the pericenter distance
print('Pericenter distance for halo 2 in NFW+Disk+BH: {0}'.format(orb_2.rperi()))
print('Apocenter distance for halo 2 in NFW+Disk+BH: {0}'.format(orb_2.rap()))
#
orb_3.integrate(ts, MWPotential_3, method='odeint')
#
plot_3_2D_nfw_miya_bh, = orb_3.plot(d1='x', d2='y')
plot_3_3D_nfw_miya_bh, = orb_3.plot3d()
#
plot_3_2D_nfw_miya_bh.figure.savefig('halo_3_2D_nfw_miya_bh')
plot_3_3D_nfw_miya_bh.figure.savefig('halo_3_3D_nfw_miya_bh')
#
# Print out the pericenter distance
print('Pericenter distance for halo 3 in NFW+Disk+BH: {0}'.format(orb_3.rperi()))
print('Apocenter distance for halo 3 in NFW+Disk+BH: {0}'.format(orb_3.rap()))

## BOTH HALOS
# Combine the orbits and integrate them at the same time
#orbs = [orb_1, orb_2]
#orb_tot = Orbit(orbs)
#orb_tot.integrate(ts, MWPotential_3, method='odeint')
#
# Plot them on the same figure
plot_tot_2D_nfw_miya_bh, = orb_1.plot(d1='x', d2='y')
plot_tot_2D_nfw_miya_bh, = orb_2.plot(d1='x', d2='y', overplot=True)
plot_tot_2D_nfw_miya_bh, = orb_3.plot(d1='x', d2='y', overplot=True)
plot_tot_3D_nfw_miya_bh, = orb_1.plot3d()
plot_tot_3D_nfw_miya_bh, = orb_2.plot3d(overplot=True)
plot_tot_3D_nfw_miya_bh, = orb_3.plot3d(overplot=True)
#
plot_tot_2D_nfw_miya_bh.figure.savefig('halos_2D_nfw_miya_bh')
plot_tot_3D_nfw_miya_bh.figure.savefig('halos_3D_nfw_miya_bh')
