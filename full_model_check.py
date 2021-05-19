#!/usr/bin/python3

"""

  ==============================
  = Spherical mass ratio check =
  ==============================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  Calculate what the enclosed mass is for ALL particles
  within bins of spherical r, out to 500 kpc

  Then, combine the disk and halo models together and plot the ratio of
  the model to the simulation data.

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
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12b'
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

"""
    This is for generating and saving the data
        - Need to be on peloton to do this...
"""
# Read in the data
part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, simulation_directory=simulation_dir, assign_hosts_rotation=True)
print('Particles at z = 0 read in')


# Find the enclosed mass of all particles within 5 < R < 500 kpc
rs = np.logspace(np.log10(0.1), np.log10(500), 81)
mass_prof = np.zeros(len(rs)-1)
#
for i in range(0, len(rs)-1):
    star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [rs[i], rs[i+1]])
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[i], rs[i+1]])
    dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[i], rs[i+1]])
    mass_prof[i] = np.sum(part['star']['mass'][star_inds]) + np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
    print('Done with step', i)
#
mass_encl = np.cumsum(mass_prof)
#
# Save this data to a file
data_dict = dict()
data_dict['mass.profile'] = mass_prof
data_dict['mass.enclosed'] = mass_encl
#
ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/'+gal1+'_spherical_mass', dict_or_array_to_write=data_dict, verbose=True)

if num_gal == 2:
    # Find the enclosed mass of all particles within 5 < R < 500 kpc
    rs = np.logspace(np.log10(0.1), np.log10(500), 81)
    mass_prof = np.zeros(len(rs)-1)
    #
    for i in range(0, len(rs)-1):
        star_inds = ut.array.get_indices(part['star'].prop('host2.distance.total'), [rs[i], rs[i+1]])
        gas_inds = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [rs[i], rs[i+1]])
        dark_inds = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [rs[i], rs[i+1]])
        mass_prof[i] = np.sum(part['star']['mass'][star_inds]) + np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
        print('Done with step', i)
    #
    mass_encl = np.cumsum(mass_prof)
    #
    # Save this data to a file
    data_dict = dict()
    data_dict['mass.profile'] = mass_prof
    data_dict['mass.enclosed'] = mass_encl
    #
    ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/'+gal2+'_spherical_mass', dict_or_array_to_write=data_dict, verbose=True)


################################################################################
################################################################################
################################################################################

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
import orbit_io
import model_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='mac')
print('Set paths')


"""
    Plot the full data with halo DENSITY model
"""

# Read in the data
#masses = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/full_profile/'+sim_data.galaxy+'_spherical_mass')
masses = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/full_profile/'+sim_data.gal_2+'_spherical_mass')

models = model_io.Profiles(directory=sim_data.home_dir)
rs = np.logspace(np.log10(0.1), np.log10(500), 81)


#disk_density = models.disk_density(distances=rs, fitting_csv=models.fitting_data_1, gal=sim_data.galaxy)
#disk_density_mass= np.zeros(len(rs)-1)
#for i in range(0, len(rs)-1):
#    area = np.pi*(rs[i+1]**2-rs[i]**2)
#    disk_density_mass[i] = disk_density[i+1]*area

#disk_mass = models.disk_radial_mass(distances=rs[1:], gal=sim_data.galaxy)
disk_mass = models.disk_radial_mass(distances=rs[1:], gal=sim_data.gal_2)

#halo_mass_2p = models.halo_2p_nfw_mass(distances=rs[1:], gal=sim_data.galaxy)
halo_mass_2p = models.halo_2p_nfw_mass(distances=rs[1:], gal=sim_data.gal_2)

# THESE WERE WHEN I WAS TESTING ALL MODELS
#halo_mass_nfw_1 = models.halo_nfw_mass(distances=rs[1:], fitting_csv=models.fitting_data_nfw_1, gal=sim_data.gal_2)
#halo_mass_nfw_2 = models.halo_nfw_mass(distances=rs[1:], fitting_csv=models.fitting_data_nfw_2, gal=sim_data.gal_2)
#halo_mass_2p_nfw_1 = models.halo_2p_nfw_mass(distances=rs[1:], fitting_csv=models.fitting_data_1, gal=sim_data.gal_2)
#halo_mass_2p_nfw_2 = models.halo_2p_nfw_mass(distances=rs[1:], fitting_csv=models.fitting_data_2, gal=sim_data.gal_2)

#total_mass_nfw_1 = halo_mass_nfw_1+disk_mass
#total_mass_nfw_2 = halo_mass_nfw_2+disk_mass
#total_mass_2p_nfw_1 = halo_mass_2p_nfw_1+disk_mass
#total_mass_2p_nfw_2 = halo_mass_2p_nfw_2+disk_mass

total_mass = halo_mass_2p + disk_mass


# Plot the ratio of the data to the model
plt.figure(figsize=(10,8))
plt.plot(rs[1:], total_mass/masses['mass.enclosed'], '-')
#plt.plot(rs[1:], total_mass_2p_nfw_2/masses['mass.enclosed'], '-', label='2P: Gas + DM only')
#plt.plot(rs[1:], total_mass_nfw_1/masses['mass.enclosed'], '-', label='NFW: All r > 10 kpc')
#plt.plot(rs[1:], total_mass_nfw_2/masses['mass.enclosed'], '-', label='NFW: Gas + DM only')
plt.xscale('log')
plt.xlim(xmin=5,xmax=500)
plt.hlines(y=1,xmin=5,xmax=500,linestyles='dotted')
plt.ylim(ymin=0.8, ymax=1.1)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<R)/M_{\\rm sim}(<R)$', fontsize=28)
#plt.title(sim_data.galaxy+', Full model', fontsize=28)
plt.title(sim_data.gal_2+', Full model', fontsize=28)
#plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/full_model_ratios/'+sim_data.galaxy+'_full_model.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/full_model_ratios/'+sim_data.gal_2+'_full_model.pdf')
plt.close()


##############################################################################################################################
"""
    Compare the mass from halo density model to the mass from halo mass model


plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(halo_density_mass)/halo_mass(rs[1:], gal1), '-')
plt.xscale('log')
plt.xlim(xmin=5,xmax=500)
plt.hlines(y=1,xmin=5,xmax=500,linestyles='dotted')
#plt.ylim(ymin=0.5, ymax=4)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm mass}(R)/M_{\\rm density}(R)$', fontsize=28)
plt.title(gal1+', halo model ratios', fontsize=28)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/full_model_ratios/'+gal1+'_halo_density_to_halo_mass_models.pdf')
plt.close()
"""

##############################################################################################################################
"""
    Compare the regular and 2-power NFW mass profiles
"""

# Plot the ratio of the data to the model
plt.figure(figsize=(10,8))
plt.plot(rs[1:], halo_mass_nfw_1/halo_mass_2p_nfw_1, '-', label='All particles')
plt.plot(rs[1:], halo_mass_nfw_2/halo_mass_2p_nfw_2, '-', label='Gas+DM only')
plt.xscale('log')
plt.xlim(xmin=5,xmax=500)
plt.hlines(y=1,xmin=5,xmax=500,linestyles='dotted')
#plt.ylim(ymin=0.8, ymax=2)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm NFW}(<R)/M_{\\rm 2P}(<R)$', fontsize=28)
plt.title(sim_data.galaxy, fontsize=28)
#plt.title(sim_data.gal_2, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_vs_nfw/'+sim_data.galaxy+'_model_compare.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_vs_nfw/'+sim_data.gal_2+'_model_compare.pdf')
plt.close()
