#!/usr/bin/python3

"""

  ==================================
  = Double Exponential Profile Fit =
  ==================================

  Fit density/mass profiles for radial and vertical

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
from astropy.constants import G
from scipy import special
print('Read in the tools')

### Set path and initial parameters
gal1 = 'Romulus'
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
data_rad = ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+gal1+'_disk_radial_profile_fitting')
density_rad = data_rad['density']
mass_rad = data_rad['mass']
rs = data_rad['rs']
#
data_vert = ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+gal1+'_disk_vertical_profile_fitting')
density_total = data_vert['density.total']
mass_total = data_vert['mass.total']
zs = data_vert['zs']
#
##############################################################################
"""
    - Define the double exponential density model

        - Fit the model to the radial profile ONLY
"""
# Double exponential model
@custom_model
def double_exponential_density(r, amp1=1e9, r1=1, amp2=1e9, r2=1):
    return amp1*np.exp(-r/r1) + amp2*np.exp(-r/r2)
#
# Fit the model to the data for various cutoff radii
model_init = double_exponential_density(bounds={'amp1':(1e7, 5e11), 'r1':(1e-1, 10), 'amp2':(1e8, 5e12), 'r2':(1e-2, 10)})
fit = LevMarLSQFitter()
model_de = fit(model_init, rs[1:], density_rad, maxiter=100000)
print(model_de)

# Inner region
@custom_model
def double_exponential_density1(r, amp1=1e9, r1=1):
    return amp1*np.exp(-r/r1)
#
# Fit the model to the data for various cutoff radii
model_init = double_exponential_density1(bounds={'amp1':(1e7, 5e11), 'r1':(1e-1, 10)})
fit = LevMarLSQFitter()
model_de1 = fit(model_init, rs[1:17], density_rad[:16], maxiter=100000)
print(model_de1)
#
# Outer region
@custom_model
def double_exponential_density2(r, amp2=1e9, r2=1):
    return amp2*np.exp(-r/r2)
#
# Fit the model to the data for various cutoff radii
model_init = double_exponential_density2(bounds={'amp2':(-1e8, 5e12), 'r2':(-1e-2, 10)})
fit = LevMarLSQFitter()
model_de2 = fit(model_init, rs[17:], density_rad[16:], maxiter=100000)
print(model_de2)
#
model_de = model_de1+model_de2
print(model_de)

"""
    Make a plot of the profiles
"""
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density_rad, 'k.', label='Data')
plt.plot(rs, model_de(rs), '-', label='Two Exponentials')
plt.yscale('log')
plt.xlim(xmin=0,xmax=20.1)
plt.ylim(ymin=1e6, ymax=1e10)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$\\rho(R)$ [$M_{\\odot} \ kpc^{-3}$]', fontsize=28)
plt.title(gal1+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/double_exponential/'+gal1+'/'+gal1+'_disk_radial_profile.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density_rad/model_de(rs[1:]), '-')
plt.xlim(xmin=0,xmax=20.1)
plt.hlines(y=1,xmin=1,xmax=20.1,linestyles='dotted')
plt.ylim(ymin=0.5, ymax=2)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}(R)/\\rho_{\\rm model}(R)$', fontsize=28)
plt.title(gal1+', |Z| < 3 kpc', fontsize=28)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/double_exponential/'+gal1+'/'+gal1+'_disk_radial_profile_ratio.pdf')
plt.close()


"""
    Make a plot of the enclosed mass
"""
m_encl = np.cumsum(mass_rad)[-1]
print('The enclosed mass is: {0:.3g}'.format(m_encl))
#
mass_de = np.zeros(len(rs)-1)
#
for i in range(0, len(rs)-1):
    area = np.pi*(rs[i+1]**2 - rs[i]**2)
    mass_de[i] = model_de(rs[i+1])*area
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass_rad), 'k.', label='Data')
plt.plot(rs[1:], np.cumsum(mass_de), '-', label='Two Exponentials')
plt.yscale('log')
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=3e9, ymax=2e11)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('M(<R) [$M_{\\odot}$]', fontsize=28)
plt.title(gal1+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/double_exponential/'+gal1+'/'+gal1+'_enclosed_disk_mass.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass_rad)/np.cumsum(mass_de), '-')
plt.hlines(y=1,xmin=0,xmax=20.1,linestyles='dotted')
plt.xlim(xmin=0, xmax=20.1)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm data}(<R)/M_{\\rm model}(R)$', fontsize=28)
plt.title(gal1+', |Z| < 3 kpc', fontsize=28)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/double_exponential/'+gal1+'/'+gal1+'_enclosed_disk_mass_ratio.pdf')
plt.close()



##############################################################################
"""
    - Define the exponential density model

        - Fit the model to the vertical profile ONLY
"""
# Single exponential model
@custom_model
def exponential_vert_density(z, amp1=1e7, z1=0.1):
    return amp1*np.exp(-np.abs(z)/z1)
#
# Fit the model to the data for various cutoff radii
model_init = exponential_vert_density(bounds={'amp1':(1e5, 5e11), 'z1':(1e-2, 10)})
fit = LevMarLSQFitter()
model_dev = fit(model_init, zs[1:], density_total, maxiter=10000000)
print(model_dev)

"""
    Make a plot of the profiles
"""
plt.figure(figsize=(10,8))
plt.plot(zs[1:], density_total, 'k.', label='Data')
plt.plot(zs, model_dev(zs), '-', label='Exponential')
plt.yscale('log')
plt.xlim(xmin=0,xmax=3.1)
#plt.ylim(ymin=1e6, ymax=1e8)
plt.xlabel('|Z| [kpc]', fontsize=28)
plt.ylabel('$\\rho(Z)$ [$M_{\\odot} \ kpc^{-3}$]', fontsize=28)
plt.title(gal1, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/double_exponential/'+gal1+'/'+gal1+'_disk_vertical_profile.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(zs[1:-1], density_total[:-1]/model_dev(zs[1:-1]), '-')
plt.xlim(xmin=0,xmax=3.1)
plt.hlines(y=1,xmin=0,xmax=3.1,linestyles='dotted')
plt.xlabel('|Z| [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}(Z)/\\rho_{\\rm model}(Z)$', fontsize=28)
plt.title(gal1, fontsize=28)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/double_exponential/'+gal1+'/'+gal1+'_disk_vertical_profile_ratio.pdf')
plt.close()
