
"""
  ===========================
  = NFW Density Profile Fit =
  ===========================

  Find fits to an NFW Potential for m12i, m12f, m12m

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
gal1 = 'Remus'
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
data = ut.io.file_hdf5(file_name_base=home_dir+'/orbit_plots/fitting/fitting_data/'+gal1+'_profile_fitting')
density = data['density']
rs = data['rs']
mass = data['mass']


"""
    - Define the NFW mass model
    - Want to exclude the inner [1, 2, 3, 5, 10] kpc
"""
# alpha/beta vary
@custom_model
def nfw_mass_model(r, amp=1e12, a=10):
    return amp*(np.log((a+r)/a)+a/(a+r)-1)

# Fit the model to the data for various cutoff radii
model_init = nfw_mass_model(bounds={'amp':(3e10, 2.25e12), 'a':(5, 30)})
fit = LevMarLSQFitter()
model_0 = fit(model_init, rs[1:], np.cumsum(mass), maxiter=1000000000)
model_1 = fit(model_init, rs[22:], np.cumsum(mass)[21:], maxiter=1000000000)
model_2 = fit(model_init, rs[28:], np.cumsum(mass)[27:], maxiter=1000000000)
model_3 = fit(model_init, rs[32:], np.cumsum(mass)[31:], maxiter=1000000000)
model_5 = fit(model_init, rs[37:], np.cumsum(mass)[36:], maxiter=1000000000)
model_10 = fit(model_init, rs[44:], np.cumsum(mass)[43:], maxiter=1000000000)
print(model_0)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

# Plot all of the profiles
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1)
plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.title(gal1, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/nfw_model/'+gal1+'/'+gal1+'_nfw_mass_profiles_float.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass)/model_1(rs[1:]), '-', label='1 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_2(rs[1:]), '-', label='2 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_3(rs[1:]), '-', label='3 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_5(rs[1:]), '-', label='5 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_10(rs[1:]), '-', label='10 kpc fit')
plt.xscale('log')
plt.xlim(xmin=1)
plt.hlines(y=1, xmin=1, xmax=500, colors='k', linestyles='dotted')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm data}/M(<r)_{\\rm model}$', fontsize=28)
plt.title(gal1, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/nfw_model/'+gal1+'/'+gal1+'_nfw_mass_profiles_float_ratio.pdf')
plt.close()

# Average the mass ratio from 10 - 500 kpc
print('Standard deviation offset for 1 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_1(rs[44:]) - 1))))
print('Standard deviation offset for 2 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_2(rs[44:]) - 1))))
print('Standard deviation offset for 3 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_3(rs[44:]) - 1))))
print('Standard deviation offset for 5 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_5(rs[44:]) - 1))))
print('Standard deviation offset for 10 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_10(rs[44:]) - 1))))

# Use the mass model parameters to plot the density profiles
dens_1 = np.zeros(len(rs)-1)
dens_2 = np.zeros(len(rs)-1)
dens_3 = np.zeros(len(rs)-1)
dens_5 = np.zeros(len(rs)-1)
dens_10 = np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    dens_1[i] = (model_1(rs[i+1]) - model_1(rs[i]))/volume
    dens_2[i] = (model_2(rs[i+1]) - model_2(rs[i]))/volume
    dens_3[i] = (model_3(rs[i+1]) - model_3(rs[i]))/volume
    dens_5[i] = (model_5(rs[i+1]) - model_5(rs[i]))/volume
    dens_10[i] = (model_10(rs[i+1]) - model_10(rs[i]))/volume

# Plot the derived density profiles on top of the actual density
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
plt.plot(rs[1:], dens_1, '-', label='1 kpc fit')
plt.plot(rs[1:], dens_2, '-', label='2 kpc fit')
plt.plot(rs[1:], dens_3, '-', label='3 kpc fit')
plt.plot(rs[1:], dens_5, '-', label='5 kpc fit')
plt.plot(rs[1:], dens_10, '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.title(gal1, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/nfw_model/'+gal1+'/'+gal1+'_nfw_mass_profiles_density_float.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density/dens_1, '-', label='1 kpc fit')
plt.plot(rs[1:], density/dens_2, '-', label='2 kpc fit')
plt.plot(rs[1:], density/dens_3, '-', label='3 kpc fit')
plt.plot(rs[1:], density/dens_5, '-', label='5 kpc fit')
plt.plot(rs[1:], density/dens_10, '-', label='10 kpc fit')
plt.xscale('log')
plt.xlim(xmin=1)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}/\\rho_{\\rm model}$', fontsize=28)
plt.title(gal1, fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/nfw_model/'+gal1+'/'+gal1+'_nfw_mass_profiles_density_float_ratio.pdf')
plt.close()
