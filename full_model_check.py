#!/usr/bin/python3

"""

  ==============================
  = Spherical mass ratio check =
  ==============================

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
gal1 = 'Romeo'
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


"""
    Plot the full data with halo DENSITY model
"""

# Read in the data
masses = ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting_data/'+gal1+'_spherical_mass')
#
# Read in the fitting parameters
fitting_data = pd.read_csv(home_dir+'/orbit_data/fitting_params.csv', index_col=0)

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
disk_mass = np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    area = np.pi*(rs[i+1]**2-rs[i]**2)
    disk_mass[i] = disk_density(rs[i+1], gal=gal1)*area

def halo_mass(r, gal):
    A_halo = fitting_data['A_halo'][gal]
    a_halo = fitting_data['a_halo'][gal]
    alpha = fitting_data['alpha'][gal]
    beta = fitting_data['beta'][gal]
    #
    return ((A_halo/a_halo**3)*(r**(3-alpha))*(a_halo+r)**(alpha-beta)*(r/a_halo+1)**(beta-1)*a_halo**beta/(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a_halo)


total_mass = halo_mass(rs[1:], gal=gal1)+np.cumsum(disk_mass)

# Plot the ratio of the data to the model
plt.figure(figsize=(10,8))
plt.plot(rs[1:], total_mass/masses['mass.enclosed'], '-')
plt.xscale('log')
plt.xlim(xmin=5,xmax=500)
plt.hlines(y=1,xmin=5,xmax=500,linestyles='dotted')
plt.ylim(ymin=0.8, ymax=1.1)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(R)/M_{\\rm sim}(R)$', fontsize=28)
plt.title(gal1+', halo mass model', fontsize=28)
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/full_model_check/'+gal1+'_full_model_check.pdf')
plt.close()


##############################################################################################################################
"""
    Compare the mass from halo density model to the mass from halo mass model
"""

rs = np.logspace(np.log10(0.1), np.log10(500), 81)

def halo_density(r, gal):
    A_halo = fitting_data['A_halo'][gal]
    a_halo = fitting_data['a_halo'][gal]
    alpha = fitting_data['alpha'][gal]
    beta = fitting_data['beta'][gal]
    #
    return (A_halo/(4*np.pi*a_halo**3))*(1/(((r/a_halo)**(alpha))*((1+r/a_halo)**(beta-alpha))))

halo_density_mass = np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    den1 = halo_density(rs[i], gal1)
    den2 = halo_density(rs[i+1], gal1)
    den_avg = np.average((den1,den2))
    halo_density_mass[i] = den_avg*volume

def halo_mass(r, gal):
    A_halo = fitting_data['A_halo'][gal]
    a_halo = fitting_data['a_halo'][gal]
    alpha = fitting_data['alpha'][gal]
    beta = fitting_data['beta'][gal]
    #
    return ((A_halo/a_halo**3)*(r**(3-alpha))*(a_halo+r)**(alpha-beta)*(r/a_halo+1)**(beta-1)*a_halo**beta/(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a_halo)


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
plt.savefig(home_dir+'/orbit_data/plots/fitting/full_model_check/'+gal1+'_halo_density_to_halo_mass_models.pdf')
plt.close()
