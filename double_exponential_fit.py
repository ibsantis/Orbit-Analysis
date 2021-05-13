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
import orbit_io
import model_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')

# Read in the data
data_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.galaxy+'_disk_radial_profile_fitting')
density_rad = data_rad['density']
mass_rad = data_rad['mass']
rs = data_rad['rs']
#
data_vert = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.galaxy+'_disk_vertical_profile_fitting')
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
def double_exponential_density(r, amp1=1e10, r1=1, amp2=1e10, r2=1):
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


############################################################################################
############################################################################################
############################################################################################

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
from orbit_analysis import orbit_io
import model_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='mac')
print('Set paths')

# Read in the data
#data_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.galaxy+'_disk_radial_profile_fitting')
data_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.gal_2+'_disk_radial_profile_fitting')
density_rad = data_rad['density']
mass_rad = data_rad['mass']
rs = data_rad['rs']
#
#data_vert = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.galaxy+'_disk_vertical_profile_fitting')
data_vert = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.gal_2+'_disk_vertical_profile_fitting')
density_total = data_vert['density.total']
mass_total = data_vert['mass.total']
zs = data_vert['zs']
#

# Initialize the class (not sure if this is how to explain this?)
disk_models = model_io.MassModelFit()

# Get a fit for hz to the vertical mass profile
disk_v = disk_models.disk_vert_mass_model(distances=zs, masses=mass_total, Amp=1e7, hz=1, Amp_bounds=(1e5, 5e11), hz_bounds=(1e-2, 1000))

# Get a fit for the other disk parameters from the radial mass profile
disk_r = disk_models.disk_rad_mass_model(distances=rs, masses=mass_rad, A_in=1e10, r_in=1, A_out=1e8, r_out=10, hz=disk_v.z1.value, A_in_bounds=(1e7,5e11), r_in_bounds=(1e-1,10), A_out_bounds=(1e6,5e10), r_out_bounds=(1e-2,20))

# Plot the radial mass profile
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass_rad), 'k.', label='Data')
plt.plot(rs, disk_r(rs), '-', label='Two Exponentials')
plt.yscale('log')
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=3e9, ymax=2e11)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('M(<R) [$M_{\\odot}$]', fontsize=28)
#plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
plt.title(sim_data.gal_2+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_enclosed_disk_mass.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_enclosed_disk_mass.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], disk_r(rs[1:])/np.cumsum(mass_rad), '-')
plt.hlines(y=1,xmin=0,xmax=20.1,linestyles='dotted')
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=0.95, ymax=1.05)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<R)/M_{\\rm sim}(<R)$', fontsize=28)
#plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
plt.title(sim_data.gal_2+', |Z| < 3 kpc', fontsize=28)
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_enclosed_disk_mass_ratio.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_enclosed_disk_mass_ratio.pdf')
plt.close()
