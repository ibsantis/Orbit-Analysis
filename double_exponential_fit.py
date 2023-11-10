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
sim_data = orbit_io.OrbitRead(gal1='m12g', location='mac')
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

# Initialize the class (not sure if this is how to explain this?)
disk_models = model_io.DensityModelFit()

# Get a fit for hz to the vertical density profile
disk_v = disk_models.disk_vert_dens_model(distances=zs, densities=density_total, Amp=1e7, hz=1, Amp_bounds=(1e5, 5e11), hz_bounds=(1e-2, 1000))

# Get a fit for the other parameters in the radial density profile
#disk_r = disk_models.disk_rad_dens_model(distances=rs, densities=density_rad, A_in=1e10, r_in=1, A_out=1e9, r_out=1, A_in_bounds=(1e7,5e11), r_in_bounds=(1e-1,10), A_out_bounds=(1e5, 5e11), r_out_bounds=(1e-2,20))
disk_r = disk_models.disk_rad_dens_model(distances=rs, densities=density_rad, A_in=1e10, r_in=1, A_out=1e9, r_out=1, A_in_bounds=(1e7,5e11), r_in_bounds=(1e-1,10), A_out_bounds=(1e5, 5e11), r_out_bounds=(1e-2,20), r_cut=8)

# Make a plot of the model and data
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density_rad, 'k.', label='Data')
plt.plot(rs, disk_r(rs), '-', label='Two Exponentials')
plt.yscale('log')
plt.xlim(xmin=0,xmax=20.1)
plt.ylim(ymin=1e6, ymax=1e10)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$\\rho(R)$ [$M_{\\odot} \ kpc^{-3}$]', fontsize=28)
plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential/'+sim_data.galaxy+'/'+sim_data.galaxy+'_disk_radial_profile.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], disk_r(rs[1:])/density_rad, '-')
plt.xlim(xmin=0,xmax=20.1)
plt.hlines(y=1,xmin=1,xmax=20.1,linestyles='dotted')
plt.ylim(ymin=0.5, ymax=2)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm model}(R)/\\rho_{\\rm sim}(R)$', fontsize=28)
plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential/'+sim_data.galaxy+'/'+sim_data.galaxy+'_disk_radial_profile_ratio.pdf')
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
    mass_de[i] = disk_r(rs[i+1])*area
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass_rad), 'k.', label='Data')
plt.plot(rs[1:], np.cumsum(mass_de), '-', label='Two Exponentials')
plt.yscale('log')
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=3e9, ymax=2e11)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('M(<R) [$M_{\\odot}$]', fontsize=28)
plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential/'+sim_data.galaxy+'/'+sim_data.galaxy+'_enclosed_disk_mass.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass_de)/np.cumsum(mass_rad), '-')
plt.hlines(y=1,xmin=0,xmax=20.1,linestyles='dotted')
plt.xlim(xmin=0, xmax=20.1)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<R)/M_{\\rm sim}(<R)$', fontsize=28)
plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential/'+sim_data.galaxy+'/'+sim_data.galaxy+'_enclosed_disk_mass_ratio.pdf')
plt.close()


"""
    - Plot the vertical density profile
"""
plt.figure(figsize=(10,8))
plt.plot(zs[1:], density_total, 'k.', label='Data')
plt.plot(zs, disk_v(zs), '-', label='Exponential')
plt.yscale('log')
plt.xlim(xmin=0,xmax=3.1)
#plt.ylim(ymin=1e6, ymax=1e8)
plt.xlabel('|Z| [kpc]', fontsize=28)
plt.ylabel('$\\rho(Z)$ [$M_{\\odot} \ kpc^{-3}$]', fontsize=28)
plt.title(sim_data.galaxy, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential/'+sim_data.galaxy+'/'+sim_data.galaxy+'_disk_vertical_profile.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(zs[1:-1], disk_v(zs[1:-1])/density_total[:-1], '-')
plt.xlim(xmin=0,xmax=3.1)
plt.hlines(y=1,xmin=0,xmax=3.1,linestyles='dotted')
plt.xlabel('|Z| [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm model}(Z)/\\rho_{\\rm sim}(Z)$', fontsize=28)
plt.title(sim_data.galaxy, fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential/'+sim_data.galaxy+'/'+sim_data.galaxy+'_disk_vertical_profile_ratio.pdf')
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
sim_data = orbit_io.OrbitRead(gal1='m12g', location='mac')
print('Set paths')

# Read in the data
data_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.galaxy+'_disk_radial_profile_fitting')
#data_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.gal_2+'_disk_radial_profile_fitting')
density_rad = data_rad['density']
mass_rad = data_rad['mass']
rs = data_rad['rs']
#
data_vert = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.galaxy+'_disk_vertical_profile_fitting')
#data_vert = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.gal_2+'_disk_vertical_profile_fitting')
density_total = data_vert['density.total']
mass_total = data_vert['mass.total']
zs = data_vert['zs']
#

# Initialize the class (not sure if this is how to explain this?)
disk_models = model_io.MassModelFit()

# Get a fit for hz to the vertical mass profile
disk_v = disk_models.disk_vert_mass_model(distances=zs, masses=mass_total, Amp=1e7, hz=1, Amp_bounds=(1e5, 5e11), hz_bounds=(1e-2, 1000))

# Get a fit for the other disk parameters from the radial mass profile
disk_r = disk_models.disk_rad_mass_model(distances=rs, masses=mass_rad, A_in=1e10, r_in=1, A_out=1e8, r_out=10, hz=disk_v.z1.value, A_in_bounds=(1e6,5e11), r_in_bounds=(1e-1,10), A_out_bounds=(1e5,5e10), r_out_bounds=(1e-2,20))

# Plot the radial mass profile
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass_rad), 'k.', label='Data')
plt.plot(rs, disk_r(rs), '-', label='Two Exponentials')
plt.yscale('log')
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=3e9, ymax=2e11)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('M(<R) [$M_{\\odot}$]', fontsize=28)
plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
#plt.title(sim_data.gal_2+', |Z| < 3 kpc', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_enclosed_disk_mass.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_enclosed_disk_mass.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], disk_r(rs[1:])/np.cumsum(mass_rad), '-')
plt.hlines(y=1,xmin=0,xmax=20.1,linestyles='dotted')
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=0.95, ymax=1.05)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<R)/M_{\\rm sim}(<R)$', fontsize=28)
plt.title(sim_data.galaxy+', |Z| < 3 kpc', fontsize=28)
#plt.title(sim_data.gal_2+', |Z| < 3 kpc', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_enclosed_disk_mass_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/double_exponential_mass_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_enclosed_disk_mass_ratio.pdf')
plt.close()
