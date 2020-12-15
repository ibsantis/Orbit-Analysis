
"""
  ===============================
  = Fitting to density profiles =
  ===============================

  Find fits to a Miyamoto-Nagai Potential and an NFW Potential for m12i

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


"""
     Fitting the NFW profile I

     - First generate the data to plot densities at different radii
     - Define the NFW model and generate fits with different cutoff radii
     - Plot all of the fits on top of the data
     - Then, derive what the mass profile should be from the data
     - Plot the mass data and mass models on top of each other
"""
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

# Define model for the density profile
@custom_model
def nfw_profile(r, amp=1e7, a=10):
    return amp/( (r/a) * (1+r/a)**2 )

# Fit the model to the data for various cutoff radii
model_init = nfw_profile(bounds={'amp':(1e5, 1e10), 'a':(5,30)})
fit = LevMarLSQFitter()
model_1 = fit(model_init, rs[18:], density[17:], maxiter=10000)
model_2 = fit(model_init, rs[22:], density[21:], maxiter=10000)
model_3 = fit(model_init, rs[25:], density[24:], maxiter=10000)
model_5 = fit(model_init, rs[29:], density[28:], maxiter=10000)
model_10 = fit(model_init, rs[35:], density[34:], maxiter=10000)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

"""
# Plot a single density profile and data
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
plt.plot(rs, model(rs), 'r-', label='fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_fit_10kpc_astropy.pdf')
plt.close()
"""

# Plot all of the profiles
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_density_profiles_astropy.pdf')
plt.close()

###############################################################################

# Calculate what the mass enclosed is, via the model from the density profile
rs_new = rs[1:]
m_encl_1 = np.zeros(len(rs_new)-1)
m_encl_2 = np.zeros(len(rs_new)-1)
m_encl_3 = np.zeros(len(rs_new)-1)
m_encl_5 = np.zeros(len(rs_new)-1)
m_encl_10 = np.zeros(len(rs_new)-1)
m_tot_1 = 0
m_tot_2 = 0
m_tot_3 = 0
m_tot_5 = 0
m_tot_10 = 0
for i in range(0, len(rs_new)-1):
    volume = 4/3*np.pi*(rs_new[i+1]**3-rs_new[i]**3)
    m_encl_1[i] = model_1(rs_new[i+1])*volume
    m_encl_2[i] = model_2(rs_new[i+1])*volume
    m_encl_3[i] = model_3(rs_new[i+1])*volume
    m_encl_5[i] = model_5(rs_new[i+1])*volume
    m_encl_10[i] = model_10(rs_new[i+1])*volume
    m_tot_1 += m_encl_1[i]
    m_tot_2 += m_encl_2[i]
    m_tot_3 += m_encl_3[i]
    m_tot_5 += m_encl_5[i]
    m_tot_10 += m_encl_10[i]
print(m_tot_1)
print(m_tot_2)
print(m_tot_3)
print(m_tot_5)
print(m_tot_10)

"""
# Plot a single mass profile and data
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs_new[1:], np.cumsum(m_encl), 'r-', label='fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_mass_10kpc_astropy.pdf')
plt.close()
"""

# Plot all of the models
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs_new[1:], np.cumsum(m_encl_1), '-', label='1 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_2), '-', label='2 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_3), '-', label='3 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_5), '-', label='5 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_mass_profiles_astropy.pdf')
plt.close()

###############################################################################


# Define model for the mass profile
@custom_model
def nfw_profile(r, amp=1e7, a=10):
    return (4*np.pi*amp*a**3)*(np.log((a+r)/a)+a/(a+r)-1)

# Fit the model to the data for various cutoff radii
model_init = nfw_profile(bounds={'amp':(1e5, 1e9), 'a':(1,100)})
fit = LevMarLSQFitter()
model_1 = fit(model_init, rs[18:], np.cumsum(mass[17:]), maxiter=100000)
model_2 = fit(model_init, rs[22:], np.cumsum(mass[21:]), maxiter=100000)
model_3 = fit(model_init, rs[25:], np.cumsum(mass[24:]), maxiter=100000)
model_5 = fit(model_init, rs[29:], np.cumsum(mass[28:]), maxiter=100000)
model_10 = fit(model_init, rs[35:], np.cumsum(mass[34:]), maxiter=100000)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

"""
# Plot a sinlge mass profile and data
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs, model_10(rs), 'r-', label='fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_mass_model_10kpc_astropy.pdf')
plt.close()
"""

# Plot all mass models and data
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_mass_models_astropy.pdf')
plt.close()

print(model_1(rs)[-1])
print(model_2(rs)[-1])
print(model_3(rs)[-1])
print(model_5(rs)[-1])
print(model_10(rs)[-1])


# Use the mass model parameters to plot the density profiles
rs_new = rs
dens_1 = np.zeros(len(rs_new)-1)
dens_2 = np.zeros(len(rs_new)-1)
dens_3 = np.zeros(len(rs_new)-1)
dens_5 = np.zeros(len(rs_new)-1)
dens_10 = np.zeros(len(rs_new)-1)
for i in range(0, len(rs_new)-1):
    volume = 4/3*np.pi*(rs_new[i+1]**3-rs_new[i]**3)
    dens_1[i] = (model_1(rs_new[i+1]) - model_1(rs_new[i]))/volume
    dens_2[i] = (model_2(rs_new[i+1]) - model_2(rs_new[i]))/volume
    dens_3[i] = (model_3(rs_new[i+1]) - model_3(rs_new[i]))/volume
    dens_5[i] = (model_5(rs_new[i+1]) - model_5(rs_new[i]))/volume
    dens_10[i] = (model_10(rs_new[i+1]) - model_10(rs_new[i]))/volume

# Plot the derived density profiles on top of the actual density
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
plt.plot(rs_new[1:], dens_1, '-', label='1 kpc fit')
plt.plot(rs_new[1:], dens_2, '-', label='2 kpc fit')
plt.plot(rs_new[1:], dens_3, '-', label='3 kpc fit')
plt.plot(rs_new[1:], dens_5, '-', label='5 kpc fit')
plt.plot(rs_new[1:], dens_10, '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_derived_density_astropy.pdf')
plt.close()


###############################################################################

"""
    Fitting the two-power spherical density model
        - Fitting the full profile doesn't really work, so...
        - Try keeping beta = 3 fixed and vary everything else
        - Try keeping alpha = 0.5 fixed and vary everything else
"""
# Define model for the two-power density profile
# Try fixing beta = 3, or alpha = 0.5
@custom_model
def nfw_profile(r, amp=1e7, a=10, alpha=1.5, beta=3.5):
    return amp/( (r/a)**alpha * (1+r/a)**(beta - alpha) )
    #return amp/( (4*np.pi*a**3) * (r/a)**alpha * (1+r/a)**(beta - alpha) )

# Fit the model to the data for various cutoff radii
model_init = nfw_profile(bounds={'amp':(1e3, 1e12), 'a':(1e-1,30), 'alpha':(-5, 5), 'beta':(1e-3, 5)})
#model_init = nfw_profile()
fit = LevMarLSQFitter()
model_0 = fit(model_init, rs[1:], density, maxiter=1000000)
model_1 = fit(model_init, rs[18:], density[17:], maxiter=1000000)
model_2 = fit(model_init, rs[22:], density[21:], maxiter=1000000)
model_3 = fit(model_init, rs[25:], density[24:], maxiter=1000000)
model_5 = fit(model_init, rs[29:], density[28:], maxiter=1000000)
model_10 = fit(model_init, rs[35:], density[34:], maxiter=1000000)
print(model_0)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

# Plot all of the profiles
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/two_power_density_profiles_astropy.pdf')
plt.close()

######## HAVEN"T DONE ANY OF THIS YET 
###############################################################################

# Calculate what the mass enclosed is, via the model from the density profile
rs_new = rs[1:]
m_encl_1 = np.zeros(len(rs_new)-1)
m_encl_2 = np.zeros(len(rs_new)-1)
m_encl_3 = np.zeros(len(rs_new)-1)
m_encl_5 = np.zeros(len(rs_new)-1)
m_encl_10 = np.zeros(len(rs_new)-1)
m_tot_1 = 0
m_tot_2 = 0
m_tot_3 = 0
m_tot_5 = 0
m_tot_10 = 0
for i in range(0, len(rs_new)-1):
    volume = 4/3*np.pi*(rs_new[i+1]**3-rs_new[i]**3)
    m_encl_1[i] = model_1(rs_new[i+1])*volume
    m_encl_2[i] = model_2(rs_new[i+1])*volume
    m_encl_3[i] = model_3(rs_new[i+1])*volume
    m_encl_5[i] = model_5(rs_new[i+1])*volume
    m_encl_10[i] = model_10(rs_new[i+1])*volume
    m_tot_1 += m_encl_1[i]
    m_tot_2 += m_encl_2[i]
    m_tot_3 += m_encl_3[i]
    m_tot_5 += m_encl_5[i]
    m_tot_10 += m_encl_10[i]
print(m_tot_1)
print(m_tot_2)
print(m_tot_3)
print(m_tot_5)
print(m_tot_10)


# Plot a sinlge mass profile and data
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs_new[1:], np.cumsum(m_encl), 'r-', label='fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_mass_10kpc_astropy.pdf')
plt.close()


# Plot all of the models
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs_new[1:], np.cumsum(m_encl_1), '-', label='1 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_2), '-', label='2 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_3), '-', label='3 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_5), '-', label='5 kpc fit')
plt.plot(rs_new[1:], np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_data/plots/fitting/nfw_mass_profiles_astropy.pdf')
plt.close()
