#!/usr/bin/python3

"""
  ==============================
  = Miyamoto-Nagai Profile Fit =
  ==============================

  Fit data to the Miyamoto-Nagai density/mass profile

"""


"""
    Code for saving the data to a file.
        - This is run on peloton.
"""
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
gal1 = 'Romulus'
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
 Generate data for the model
    - Want cold gas (T < 1e5 K) and stars
    - Getting the surface density here:
        Need to divide by the area of an annulus
            A = pi * (R**2 - r**2)
            R: Outer radius
            r: Inner radius
"""
# Model the radial profile
rs = np.linspace(0, 20, 41)
mass = np.zeros(len(rs)-1)
density = np.zeros(len(rs)-1)
#
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5])
#gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, np.inf])
gas_z_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [0, 3], gas_temp_inds)
star_z_inds = ut.array.get_indices(np.abs(part['star'].prop('host.distance.principal.cylindrical')[:,2]), [0, 3])
#
for i in range(0, len(rs)-1):
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [rs[i], rs[i+1]], gas_z_inds)
    star_inds = ut.array.get_indices(part['star'].prop('host.distance.principal.cylindrical')[:,0], [rs[i], rs[i+1]], star_z_inds)
    mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['star']['mass'][star_inds])
    density[i] = mass[i]/(np.pi*(rs[i+1]**2 - rs[i]**2))
    print('Done with step', i)
#
# Save the data to a dictionary
d_r = dict()
d_r['density'] = density
d_r['mass'] = mass
d_r['rs'] = rs
#
ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/disk/'+gal1+'_disk_radial_profile_fitting_all_gas', dict_or_array_to_write=d_r, verbose=True)


# Model the vertical profile
# This profile is going to be cumulative already, no need to cumulatively sum the mass or density at all...
zs = np.linspace(0, 3, 31)
#
mass_tot = np.zeros(len(zs)-1)
density_tot = np.zeros(len(zs)-1)
#
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5])
#gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, np.inf])
#
gas_r_tot_inds = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [0, 10], gas_temp_inds)
star_r_tot_inds = ut.array.get_indices(part['star'].prop('host.distance.principal.cylindrical')[:,0], [0, 10])
#
for i in range(0, len(zs)-1):
    gas_tot_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], gas_r_tot_inds)
    star_tot_inds = ut.array.get_indices(np.abs(part['star'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], star_r_tot_inds)
    #
    mass_tot[i] = np.sum(part['gas']['mass'][gas_tot_inds]) + np.sum(part['star']['mass'][star_tot_inds])
    density_tot[i] = mass_tot[i]/(np.pi*2*(zs[i+1]-zs[i])*(10**2 - 0**2))
    #
    print('Done with step', i)
#
# Save the data to a dictionary
d_z = dict()
d_z['density.total'] = density_tot
d_z['mass.total'] = mass_tot
d_z['zs'] = zs
#
ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/disk/'+gal1+'_disk_vertical_profile_fitting_all_gas', dict_or_array_to_write=d_z, verbose=True)


if num_gal == 2:
    """
     Generate data for the model
        - Want cold gas (T < 1e5 K) and stars
        - Getting the surface density here:
            Need to divide by the area of an annulus
                A = pi * (R**2 - r**2)
                R: Outer radius
                r: Inner radius
    """
    # Model the radial profile
    rs = np.linspace(0, 20, 41)
    mass = np.zeros(len(rs)-1)
    density = np.zeros(len(rs)-1)
    #
    gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5])
    #gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, np.inf])
    gas_z_inds = ut.array.get_indices(np.abs(part['gas'].prop('host2.distance.principal.cylindrical')[:,2]), [0, 3], gas_temp_inds)
    star_z_inds = ut.array.get_indices(np.abs(part['star'].prop('host2.distance.principal.cylindrical')[:,2]), [0, 3])
    #
    for i in range(0, len(rs)-1):
        gas_inds = ut.array.get_indices(part['gas'].prop('host2.distance.principal.cylindrical')[:,0], [rs[i], rs[i+1]], gas_z_inds)
        star_inds = ut.array.get_indices(part['star'].prop('host2.distance.principal.cylindrical')[:,0], [rs[i], rs[i+1]], star_z_inds)
        mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['star']['mass'][star_inds])
        density[i] = mass[i]/(np.pi*(rs[i+1]**2 - rs[i]**2))
        print('Done with step', i)
    #
    # Save the data to a dictionary
    d_r = dict()
    d_r['density'] = density
    d_r['mass'] = mass
    d_r['rs'] = rs
    #
    ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/disk/'+gal2+'_disk_radial_profile_fitting_all_gas', dict_or_array_to_write=d_r, verbose=True)


    # Model the vertical profile
    # This profile is going to be cumulative already, no need to cumulatively sum the mass or density at all...
    zs = np.linspace(0, 3, 31)
    #
    mass_tot = np.zeros(len(zs)-1)
    density_tot = np.zeros(len(zs)-1)
    #
    gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5])
    #gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, np.inf])
    #
    gas_r_tot_inds = ut.array.get_indices(part['gas'].prop('host2.distance.principal.cylindrical')[:,0], [0, 10], gas_temp_inds)
    star_r_tot_inds = ut.array.get_indices(part['star'].prop('host2.distance.principal.cylindrical')[:,0], [0, 10])
    #
    for i in range(0, len(zs)-1):
        gas_tot_inds = ut.array.get_indices(np.abs(part['gas'].prop('host2.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], gas_r_tot_inds)
        star_tot_inds = ut.array.get_indices(np.abs(part['star'].prop('host2.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], star_r_tot_inds)
        #
        mass_tot[i] = np.sum(part['gas']['mass'][gas_tot_inds]) + np.sum(part['star']['mass'][star_tot_inds])
        density_tot[i] = mass_tot[i]/(np.pi*2*(zs[i+1]-zs[i])*(10**2 - 0**2))
        #
        print('Done with step', i)
    #
    # Save the data to a dictionary
    d_z = dict()
    d_z['density.total'] = density_tot
    d_z['mass.total'] = mass_tot
    d_z['zs'] = zs
    #
    ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/disk/'+gal2+'_disk_vertical_profile_fitting_all_gas', dict_or_array_to_write=d_z, verbose=True)



"""
# Model the whole profile
rs, zs = np.mgrid[0:20.5:0.5, 0:3.5:0.5]
mass = np.zeros((rs.shape[0]-1, rs.shape[1]-1))
density = np.zeros((rs.shape[0]-1, rs.shape[1]-1))
#
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5])
#
for i in range(0, rs.shape[0]-1):
    gas_inds_r = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [rs[i, 0], rs[i+1, 0]], gas_temp_inds)
    star_inds_r = ut.array.get_indices(part['star'].prop('host.distance.principal.cylindrical')[:,0], [rs[i, 0], rs[i+1, 0]])
    for j in range(0, zs.shape[1]-1):
        gas_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i,j], zs[i,j+1]], gas_inds_r)
        star_inds = ut.array.get_indices(np.abs(part['star'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i,j], zs[i,j+1]], star_inds_r)
        print(len(gas_inds), len(star_inds))
        mass[i,j] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['star']['mass'][star_inds])
        density[i,j] = mass[i,j]/(np.pi*2*(zs[i,j+1]-zs[i,j])*(rs[i+1,0]**2 - rs[i,0]**2))
    print('Done with step', i)
#
# Save the data to a dictionary
d1 = dict()
d1['density'] = density
d1['mass'] = mass
d1['rs'] = rs
d1['zs'] = zs
#
ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/disk/'+gal1+'_disk_whole_profile_fitting', dict_or_array_to_write=d1, verbose=True)
"""
