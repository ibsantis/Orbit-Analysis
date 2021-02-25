import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from scipy import special
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12m'
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
elif gal1 == 'm12z':
    galaxy = gal1
    resolution = '_res4200'
    num_gal = 1
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

# Read in the data
part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', 0, simulation_directory=simulation_dir, assign_hosts_rotation=True, assign_formation_coordinates=True)
print('Particles at z = 0 read in')


"""
 Find the hot gas in the disk
"""
# Model the radial profile
rs = np.linspace(0, 20, 41)
mass = np.zeros(len(rs)-1)
density = np.zeros(len(rs)-1)
#
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [1e5, np.inf])
gas_z_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [0, 3], gas_temp_inds)
#
for i in range(0, len(rs)-1):
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [rs[i], rs[i+1]], gas_z_inds)
    mass[i] = np.sum(part['gas']['mass'][gas_inds])
    density[i] = mass[i]/(np.pi*(rs[i+1]**2 - rs[i]**2))
    print('Done with step', i)

d_hot = dict()
d_hot['density'] = density
d_hot['mass'] = mass
d_hot['rs'] = rs

print(np.cumsum(d_hot['mass']))



"""
 Find the cold gas in the halo
"""
# Find the particles that are in the disk first
disk_inds = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [0,20])
disk_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [0,3], disk_inds)
#
# Need to calculate density on my own, the particles won't help
rs = np.logspace(-1, 2.699, 81)
mass = np.zeros(len(rs)-1)
density = np.zeros(len(rs)-1)
#gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5])
for i in range(0, len(rs)-1):
    temp = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[i], rs[i+1]])
    gas_inds = np.setdiff1d(temp, disk_inds)
    gas_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5], gas_inds)
    mass[i] = np.sum(part['gas']['mass'][gas_inds])
    density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
    print('done with step', i)

d_cold = dict()
d_cold['density'] = density
d_cold['mass'] = mass
d_cold['rs'] = rs

print(np.cumsum(d_cold['mass']))



"""
 Find the stellar mass in the stellar halo
"""

disk_inds = ut.array.get_indices(part['star'].prop('host.distance.principal.cylindrical')[:,0], [0,20])
disk_inds = ut.array.get_indices(np.abs(part['star'].prop('host.distance.principal.cylindrical')[:,2]), [0,3], disk_inds)
#
rs = np.logspace(-1, 2.699, 81)
mass = np.zeros(len(rs)-1)
density = np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    temp = ut.array.get_indices(part['star'].prop('host.distance.total'), [rs[i], rs[i+1]])
    star_inds = np.setdiff1d(temp, disk_inds)
    mass[i] = np.sum(part['star']['mass'][star_inds])
    density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
    print('done with step', i)

d_halo = dict()
d_halo['density'] = density
d_halo['mass'] = mass
d_halo['rs'] = rs

print(np.cumsum(d_halo['mass']))
