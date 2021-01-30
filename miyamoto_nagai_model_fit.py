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
from astropy.constants import G
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
d1 = dict()
d1['density'] = density
d1['mass'] = mass
d1['rs'] = rs
#
ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/disk/'+gal1+'_disk_radial_profile_fitting', dict_or_array_to_write=d1, verbose=True)


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


# Model the vertical profile
# This profile is going to be cumulative already, no need to cumulatively sum the mass or density at all...
r_in = 0.77
r_out = 4.15
zs = np.linspace(0, 3, 31)
#
mass_in = np.zeros(len(zs)-1)
density_in = np.zeros(len(zs)-1)
#
mass_out = np.zeros(len(zs)-1)
density_out = np.zeros(len(zs)-1)
#
mass_tot = np.zeros(len(zs)-1)
density_tot = np.zeros(len(zs)-1)
#
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [0, 1e5])
#
gas_r_in_inds = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [0, r_in], gas_temp_inds)
star_r_in_inds = ut.array.get_indices(part['star'].prop('host.distance.principal.cylindrical')[:,0], [0, r_in])
#
gas_r_out_inds = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [r_in, r_out], gas_temp_inds)
star_r_out_inds = ut.array.get_indices(part['star'].prop('host.distance.principal.cylindrical')[:,0], [r_in, r_out])
#
gas_r_tot_inds = ut.array.get_indices(part['gas'].prop('host.distance.principal.cylindrical')[:,0], [0, 10], gas_temp_inds)
star_r_tot_inds = ut.array.get_indices(part['star'].prop('host.distance.principal.cylindrical')[:,0], [0, 10])
#
for i in range(0, len(zs)-1):
    gas_in_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], gas_r_in_inds)
    star_in_inds = ut.array.get_indices(np.abs(part['star'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], star_r_in_inds)
    #
    mass_in[i] = np.sum(part['gas']['mass'][gas_in_inds]) + np.sum(part['star']['mass'][star_in_inds])
    density_in[i] = mass_in[i]/(np.pi*2*(zs[i+1]-zs[i])*(r_in**2-0**2))
    #
    gas_out_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], gas_r_out_inds)
    star_out_inds = ut.array.get_indices(np.abs(part['star'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], star_r_out_inds)
    #
    mass_out[i] = np.sum(part['gas']['mass'][gas_out_inds]) + np.sum(part['star']['mass'][star_out_inds])
    density_out[i] = mass_out[i]/(np.pi*2*(zs[i+1]-zs[i])*(r_out**2 - r_in**2))
    #
    gas_tot_inds = ut.array.get_indices(np.abs(part['gas'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], gas_r_tot_inds)
    star_tot_inds = ut.array.get_indices(np.abs(part['star'].prop('host.distance.principal.cylindrical')[:,2]), [zs[i], zs[i+1]], star_r_tot_inds)
    #
    mass_tot[i] = np.sum(part['gas']['mass'][gas_tot_inds]) + np.sum(part['star']['mass'][star_tot_inds])
    density_tot[i] = mass_out[i]/(np.pi*2*(zs[i+1]-zs[i])*(10**2 - 0**2))
    #
    print('Done with step', i)
#
# Save the data to a dictionary
d2 = dict()
d2['density.inner'] = density_in
d2['mass.inner'] = mass_in
d2['density.outer'] = density_out
d2['mass.outer'] = mass_out
d2['density.total'] = density_tot
d2['mass.total'] = mass_tot
d2['zs'] = zs
d2['r1'] = np.array([r_in])
d2['r2'] = np.array([r_out])
#
ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/disk/'+gal1+'_disk_vertical_profile_fitting', dict_or_array_to_write=d2, verbose=True)


################################################################################


"""
    Code for fitting the models
        - This is run on my local machine
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
gal1 = 'm12i'
loc = 'mac'

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

# Read in the data
data_rad = ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+gal1+'_disk_radial_profile_fitting_z3')
density_rad = data_rad['surf.density']
mass_rad = data_rad['mass']
rs = data_rad['rs']


##############################################################################
"""
    - Define the M-N density model

        - Fit the model to the radial profile ONLY

        - Want to exclude the inner 1 kpc
"""
# alpha/beta vary
@custom_model
def mn_surface_density(r, M=8e10, a=1):
    return ((a*M)/(2*np.pi))*(1/((r**2 + a**2)**1.5))


# Fit the model to the data for various cutoff radii
model_init = mn_surface_density(bounds={'M':(1e9, 5e11), 'a':(0.1, 10)})
fit = LevMarLSQFitter()
model_05 = fit(model_init, rs[1:], density_rad, maxiter=100000)
model_10 = fit(model_init, rs[2:], density_rad[1:], maxiter=100000)
model_15 = fit(model_init, rs[3:], density_rad[2:], maxiter=100000)
model_20 = fit(model_init, rs[4:], density_rad[3:], maxiter=100000)
model_25 = fit(model_init, rs[5:], density_rad[4:], maxiter=100000)
model_30 = fit(model_init, rs[6:], density_rad[5:], maxiter=100000)
model_35 = fit(model_init, rs[7:], density_rad[6:], maxiter=100000)
model_40 = fit(model_init, rs[8:], density_rad[7:], maxiter=100000)
print(model_05)
print(model_10)
print(model_15)
print(model_20)
print(model_25)
print(model_30)
print(model_35)
print(model_40)

# Plot all of the profiles
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density_rad, 'k.', label='data')
plt.plot(rs[1:], model_05(rs[1:]), '-', label='0.5 kpc')
plt.plot(rs[2:], model_10(rs[2:]), '-', label='1.0 kpc')
plt.plot(rs[3:], model_15(rs[3:]), '-', label='1.5 kpc')
plt.plot(rs[4:], model_20(rs[4:]), '-', label='2.0 kpc')
plt.plot(rs[5:], model_25(rs[5:]), '-', label='2.5 kpc')
plt.plot(rs[6:], model_30(rs[6:]), '-', label='3.0 kpc')
plt.plot(rs[7:], model_35(rs[7:]), '-', label='3.5 kpc')
plt.plot(rs[8:], model_40(rs[8:]), '-', label='4.0 kpc')
plt.yscale('log')
#plt.hlines(y=1,xmin=1,xmax=10.5,linestyles='dotted')
plt.xlim(xmin=0,xmax=20.1)
#plt.ylim(ymin=1e8)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$\\Sigma(R)$ [$M_{\\odot} \ kpc^{-2}$]', fontsize=28)
plt.title(gal1+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/miyamoto_nagai/'+gal1+'/'+gal1+'_disk_radial_profile_z3.pdf')
plt.close()

"""
    Make a plot of the enclosed mass
"""
m_encl = np.cumsum(mass_rad)[-1]
print('The enclosed mass is: {0:.3g}'.format(m_encl))
#
mass_05 = np.zeros(len(rs)-1)
mass_10 = np.zeros(len(rs)-1)
mass_15 = np.zeros(len(rs)-1)
mass_20 = np.zeros(len(rs)-1)
mass_25 = np.zeros(len(rs)-1)
mass_30 = np.zeros(len(rs)-1)
mass_35 = np.zeros(len(rs)-1)
mass_40 = np.zeros(len(rs)-1)
#
for i in range(0, len(rs)-1):
    area = np.pi*(rs[i+1]**2 - rs[i]**2)
    mass_05[i] = model_05(rs[i+1])*area
    mass_10[i] = model_10(rs[i+1])*area
    mass_15[i] = model_15(rs[i+1])*area
    mass_20[i] = model_20(rs[i+1])*area
    mass_25[i] = model_25(rs[i+1])*area
    mass_30[i] = model_30(rs[i+1])*area
    mass_35[i] = model_35(rs[i+1])*area
    mass_40[i] = model_40(rs[i+1])*area
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass_rad), 'k.', label='data')
plt.plot(rs[1:], np.cumsum(mass_05), '-', label='0.5 kpc')
plt.plot(rs[1:], np.cumsum(mass_10), '-', label='1.0 kpc')
plt.plot(rs[1:], np.cumsum(mass_15), '-', label='1.5 kpc')
plt.plot(rs[1:], np.cumsum(mass_20), '-', label='2.0 kpc')
plt.plot(rs[1:], np.cumsum(mass_25), '-', label='2.5 kpc')
plt.plot(rs[1:], np.cumsum(mass_30), '-', label='3.0 kpc')
plt.plot(rs[1:], np.cumsum(mass_35), '-', label='3.5 kpc')
plt.plot(rs[1:], np.cumsum(mass_40), '-', label='4.0 kpc')
plt.yscale('log')
plt.hlines(y=1,xmin=1,xmax=10.5,linestyles='dotted')
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=1e9, ymax=1.5e11)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('M(<R) [$M_{\\odot}$]', fontsize=28)
plt.title(gal1+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/miyamoto_nagai/'+gal1+'/'+gal1+'_enclosed_disk_mass_z3.pdf')
plt.close()
