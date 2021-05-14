
"""
  ===========================
  = NFW Density Profile Fit =
  ===========================

  Collect data to then model with an NFW mass/density profile

"""


"""
    Code for saving the data to a file.
        - This is run on stampede.
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
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12i'
loc = 'stampede'

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
part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, simulation_directory=simulation_dir, assign_hosts_rotation=True, assign_formation_coordinates=True)
print('Stars at z = 0 read in')

# Generate data for the NFW model
gas_inds = ut.array.get_indices(part['gas']['temperature'], [1e5, np.inf])
#
# Need to calculate density on my own, the particles won't help
rs = np.logspace(-1, 2.477, 61)
mass = np.zeros(len(rs)-1)
density = np.zeros(len(rs)-1)
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [1e5, np.inf])
for i in range(0, len(rs)-1):
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[i], rs[i+1]], gas_temp_inds)
    dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[i], rs[i+1]])
    mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
    density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
    print('done with step', i)


###############################################################################
###############################################################################
###############################################################################

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
import orbit_io
import model_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')

## Read in the data
#data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+sim_data.galaxy+'_halo_fitting')
#data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+sim_data.gal_1+'_halo_fitting')
data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/gas_dm_only/'+sim_data.galaxy+'_halo_profile_fitting')
#data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/gas_dm_only/'+sim_data.gal_2+'_halo_profile_fitting')
density = data['density']
mass = data['mass']
rs = data['rs']

halo_models = model_io.MassModelFit()

halo_300 = halo_models.halo_nfw_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, A_halo_bounds=(3e10,2.25e12), a_halo_bounds=(5,30), r_max=300)
halo_350 = halo_models.halo_nfw_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, A_halo_bounds=(3e10,2.25e12), a_halo_bounds=(5,30), r_max=350)
halo_400 = halo_models.halo_nfw_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, A_halo_bounds=(3e10,2.25e12), a_halo_bounds=(5,30), r_max=400)
halo_500 = halo_models.halo_nfw_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, A_halo_bounds=(3e10,2.25e12), a_halo_bounds=(5,30))

# Average the mass ratio from 10 - X kpc
print('Standard deviation offset for 300 kpc cutoff is {0:.4g}'.format(np.std(np.abs(halo_300(rs[54:94])/np.cumsum(mass)[53:93] - 1))))
print('Standard deviation offset for 350 kpc cutoff is {0:.4g}'.format(np.std(np.abs(halo_350(rs[54:96])/np.cumsum(mass)[53:95] - 1))))
print('Standard deviation offset for 400 kpc cutoff is {0:.4g}'.format(np.std(np.abs(halo_400(rs[54:97])/np.cumsum(mass)[53:96] - 1))))
print('Standard deviation offset for 500 kpc cutoff is {0:.4g}'.format(np.std(np.abs(halo_500(rs[54:])/np.cumsum(mass)[53:] - 1))))


# Plot all of the profiles
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs, halo_300(rs), '-', label='300 kpc fit')
plt.plot(rs, halo_350(rs), '-', label='350 kpc fit')
plt.plot(rs, halo_400(rs), '-', label='400 kpc fit')
plt.plot(rs, halo_500(rs), '-', label='500 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1)
plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.title(sim_data.galaxy, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.gal_1+'/'+sim_data.gal_1+'_nfw_mass_profile.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.gal_2+'/'+sim_data.gal_2+'_nfw_mass_profile.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], halo_300(rs[1:])/np.cumsum(mass), '-', label='300 kpc fit')
plt.plot(rs[1:], halo_350(rs[1:])/np.cumsum(mass), '-', label='350 kpc fit')
plt.plot(rs[1:], halo_400(rs[1:])/np.cumsum(mass), '-', label='400 kpc fit')
plt.plot(rs[1:], halo_500(rs[1:])/np.cumsum(mass), '-', label='500 kpc fit')
plt.xscale('log')
plt.xlim(xmin=5)
plt.ylim(ymin=0.5, ymax=1.2)
plt.hlines(y=1, xmin=1, xmax=500, colors='k', linestyles='dotted')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm model}/M(<r)_{\\rm sim}$', fontsize=28)
plt.title(sim_data.galaxy, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.gal_1+'/'+sim_data.gal_1+'_nfw_mass_profile_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile_ratio.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.gal_2+'/'+sim_data.gal_2+'_nfw_mass_profile_ratio.pdf')
plt.close()

# Use the mass model parameters to plot the density profiles
dens_10_300 = np.zeros(len(rs)-1)
dens_10_350 = np.zeros(len(rs)-1)
dens_10_400 = np.zeros(len(rs)-1)
dens_10_500 = np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    dens_10_300[i] = (halo_300(rs[i+1]) - halo_300(rs[i]))/volume
    dens_10_350[i] = (halo_350(rs[i+1]) - halo_350(rs[i]))/volume
    dens_10_400[i] = (halo_400(rs[i+1]) - halo_400(rs[i]))/volume
    dens_10_500[i] = (halo_500(rs[i+1]) - halo_500(rs[i]))/volume

# Plot the derived density profiles on top of the actual density
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
plt.plot(rs[1:], dens_10_300, '-', label='300 kpc fit')
plt.plot(rs[1:], dens_10_350, '-', label='350 kpc fit')
plt.plot(rs[1:], dens_10_400, '-', label='400 kpc fit')
plt.plot(rs[1:], dens_10_500, '-', label='500 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.title(sim_data.galaxy, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile_density.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.gal_1+'/'+sim_data.gal_1+'_nfw_mass_profile_density.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile_density.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.gal_2+'/'+sim_data.gal_2+'_nfw_mass_profile_density.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], dens_10_300/density, '-', label='300 kpc fit')
plt.plot(rs[1:], dens_10_350/density, '-', label='350 kpc fit')
plt.plot(rs[1:], dens_10_400/density, '-', label='400 kpc fit')
plt.plot(rs[1:], dens_10_500/density, '-', label='500 kpc fit')
plt.xscale('log')
plt.xlim(xmin=1)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm model}/\\rho_{\\rm sim}$', fontsize=28)
plt.title(sim_data.galaxy, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile_density_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3/'+sim_data.gal_1+'/'+sim_data.gal_1+'_nfw_mass_profile_density_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_nfw_mass_profile_density_ratio.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/nfw_model_v3_gas_dm/'+sim_data.gal_2+'/'+sim_data.gal_2+'_nfw_mass_profile_density_ratio.pdf')
plt.close()
