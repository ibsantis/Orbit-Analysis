#!/usr/bin/python3

"""

  ==================================
  = Spherical mass ratio check NFW =
  ==================================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  Check the full disk + NFW halo model against the data

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
import pandas as pd
from orbit_analysis import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='mac')
sim_data.gal_1 = 'Romulus'
sim_data.gal_2 = 'Remus'
print('Set paths')


"""
    Plot the full data with halo DENSITY model
"""

# Read in the data
#masses = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/full_profile/'+sim_data.galaxy+'_spherical_mass')
masses = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/full_profile/'+sim_data.gal_2+'_spherical_mass')
#
# Read in the fitting parameters
fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params.csv', index_col=0)
fitting_data_nfw = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params_nfw.csv', index_col=0)
fitting_data_nfw_v2 = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params_nfw_v2.csv', index_col=0)

##########################################################################################

"""
    Plot the full data with halo MASS model
"""

# Create the mass model for the disk and halo
def disk_density(r, gal):
    A_disk_in = fitting_data['A_disk_in'][gal]
    r_in = fitting_data['r_in'][gal]
    A_disk_out = fitting_data['A_disk_out'][gal]
    r_out = fitting_data['r_out'][gal]
    h_z = fitting_data['h_z'][gal]
    #
    # Integrate the z comp out which results in just a factor of hz
    #
    disk_inner = A_disk_in*h_z*np.exp(-r/r_in)
    disk_outer = A_disk_out*h_z*np.exp(-r/r_out)
    return disk_inner+disk_outer


rs = np.logspace(np.log10(0.1), np.log10(500), 81)
disk_mass= np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    area = np.pi*(rs[i+1]**2-rs[i]**2)
    #disk_mass[i] = disk_density(rs[i+1], gal=sim_data.galaxy)*area
    disk_mass[i] = disk_density(rs[i+1], gal=sim_data.gal_2)*area

def halo_mass_1(r, gal):
    A_halo = fitting_data_nfw['A_halo'][gal]
    a_halo = fitting_data_nfw['a_halo'][gal]
    #
    return A_halo*(np.log((a_halo+r)/a_halo)+a_halo/(a_halo+r)-1)

def halo_mass_2(r, gal):
    A_halo = fitting_data_nfw_v2['A_halo'][gal]
    a_halo = fitting_data_nfw_v2['a_halo'][gal]
    #
    return A_halo*(np.log((a_halo+r)/a_halo)+a_halo/(a_halo+r)-1)

#total_mass_1 = halo_mass_1(rs[1:], gal=sim_data.galaxy)+np.cumsum(disk_mass)
#total_mass_2 = halo_mass_2(rs[1:], gal=sim_data.galaxy)+np.cumsum(disk_mass)
total_mass_1 = halo_mass_1(rs[1:], gal=sim_data.gal_2)+np.cumsum(disk_mass)
total_mass_2 = halo_mass_2(rs[1:], gal=sim_data.gal_2)+np.cumsum(disk_mass)


# Plot the ratio of the data to the model
plt.figure(figsize=(10,8))
plt.plot(rs[1:], total_mass_1/masses['mass.enclosed'], '-', label='Old NFW Params')
plt.plot(rs[1:], total_mass_2/masses['mass.enclosed'], '-', label='New NFW Params')
plt.xscale('log')
plt.xlim(xmin=5,xmax=500)
plt.hlines(y=1,xmin=5,xmax=500,linestyles='dotted')
plt.ylim(ymin=0.8, ymax=1.1)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(R)/M_{\\rm sim}(R)$', fontsize=28)
#plt.title(sim_data.galaxy+', Full model', fontsize=28)
plt.title(sim_data.gal_2+', Full model', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/full_model_check_nfw/'+sim_data.galaxy+'_full_model_nfw.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/full_model_check_nfw/'+sim_data.gal_2+'_full_model_nfw.pdf')
plt.close()
