#!/usr/bin/python3

"""
  ===========================================
  = Two-Power Spherical Density Profile Fit =
  ===========================================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  Fit data from a host to the two-power spherical density profile

"""


"""
    Code for saving the data to a file.
        - This is run on Peloton.
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

# Read in the data
part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, simulation_directory=simulation_dir, assign_hosts_rotation=True, assign_formation_coordinates=True)
print('Particles at z = 0 read in')


"""
 Generate data for the model
    - Want ALL particles
"""
# Need to calculate density on my own, the particles won't help
rs = np.logspace(np.log10(0.1), np.log10(500), 100)
mass = np.zeros(len(rs)-1)
density = np.zeros(len(rs)-1)
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [1e5, np.inf])
for i in range(0, len(rs)-1):
    if rs[i] < 10:
        gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[i], rs[i+1]], gas_temp_inds)
        dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[i], rs[i+1]])
        mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
        density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
        print('done with step', i)
    if rs[i] > 10:
        gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[i], rs[i+1]])
        star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [rs[i], rs[i+1]])
        dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[i], rs[i+1]])
        mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['star']['mass'][star_inds]) + np.sum(part['dark']['mass'][dark_inds])
        density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
        print('done with step', i)

d1 = dict()
d1['density'] = density
d1['mass'] = mass
d1['rs'] = rs

ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/halo/'+gal1+'_halo_fitting', dict_or_array_to_write=d1, verbose=True)


if num_gal == 2:
    """
     Generate data for the model
        - Want ALL particles
    """
    # Need to calculate density on my own, the particles won't help
    rs = np.logspace(np.log10(0.1), np.log10(500), 100)
    mass = np.zeros(len(rs)-1)
    density = np.zeros(len(rs)-1)
    gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [1e5, np.inf])
    for i in range(0, len(rs)-1):
        if rs[i] < 10:
            gas_inds = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [rs[i], rs[i+1]], gas_temp_inds)
            dark_inds = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [rs[i], rs[i+1]])
            mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
            density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
            print('done with step', i)
        if rs[i] > 10:
            gas_inds = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [rs[i], rs[i+1]])
            star_inds = ut.array.get_indices(part['star'].prop('host2.distance.total'), [rs[i], rs[i+1]])
            dark_inds = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [rs[i], rs[i+1]])
            mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['star']['mass'][star_inds]) + np.sum(part['dark']['mass'][dark_inds])
            density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
            print('done with step', i)
    #
    d2 = dict()
    d2['density'] = density
    d2['mass'] = mass
    d2['rs'] = rs
    #
    ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/fitting/halo/'+gal2+'_halo_fitting', dict_or_array_to_write=d2, verbose=True)



"""
 Generate data for the model
    - Want hot gas (T > 1e5 K) and dark matter within the virial radius
        - For now, I'm just testing out to 300 kpc

# Need to calculate density on my own, the particles won't help
rs = np.logspace(-1, 2.699, 81)
mass = np.zeros(len(rs)-1)
density = np.zeros(len(rs)-1)
gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [1e5, np.inf])
for i in range(0, len(rs)-1):
    gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[i], rs[i+1]], gas_temp_inds)
    dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[i], rs[i+1]])
    mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
    density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
    print('done with step', i)

d1 = dict()
d1['density'] = density
d1['mass'] = mass
d1['rs'] = rs

ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/plots/fitting/'+gal1+'_profile_fitting', dict_or_array_to_write=d1, verbose=True)
"""

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
from orbit_analysis import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='mac')
sim_data.gal_1 = 'Romulus'
sim_data.gal_2 = 'Remus'
print('Set paths')

# Read in the data
###data = ut.io.file_hdf5(file_name_base=home_dir+'/orbit_plots/fitting/fitting_data/'+gal1+'_profile_fitting')
#data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+sim_data.galaxy+'_halo_fitting')
data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+sim_data.gal_2+'_halo_fitting')
density = data['density']
mass = data['mass']
rs = data['rs']

##############################################################################
"""
    - Define the two-power spherical mass model

    - Fit the model to the data
        - Let both alpha and beta vary
        - Try keeping beta = 3 fixed and vary everything else
        - Try keeping alpha = 0.5 fixed and vary everything else

    - Want to exclude the inner [1, 2, 3, 5, 10] kpc
"""
# alpha/beta vary
@custom_model
def two_power_beta_fixed(r, amp=1e12, a=5, alpha=0.5, beta=2.5):
    #return (4*np.pi*amp*a**3)*(((r/a)**(3.-alpha))/(3.-alpha)*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a))
    return ((amp/a**3)*(r**(3-alpha))*(a+r)**(alpha-beta)*(r/a+1)**(beta-1)*a**beta/(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a)

# Fit the model to the data for various cutoff radii
model_init = two_power_beta_fixed(bounds={'amp':(3e10, 2.25e12), 'a':(0.5, 30), 'alpha':(0, 1.5), 'beta':(0, 5)})
fit = LevMarLSQFitter()
#model_10_100 = fit(model_init, rs[54:81], np.cumsum(mass)[53:80], maxiter=10000000)
#model_10_150 = fit(model_init, rs[54:86], np.cumsum(mass)[53:85], maxiter=10000000)
#model_10_200 = fit(model_init, rs[54:89], np.cumsum(mass)[53:88], maxiter=10000000)
model_10_300 = fit(model_init, rs[54:93], np.cumsum(mass)[53:92], maxiter=10000000)
model_10_350 = fit(model_init, rs[54:95], np.cumsum(mass)[53:94], maxiter=10000000)
model_10_400 = fit(model_init, rs[54:96], np.cumsum(mass)[53:95], maxiter=10000000)
model_10_500 = fit(model_init, rs[54:], np.cumsum(mass)[53:], maxiter=10000000)
#print(model_10_100)
#print(model_10_150)
#print(model_10_200)
print(model_10_300)
print(model_10_350)
print(model_10_400)
print(model_10_500)

# Plot all of the profiles
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
#plt.plot(rs, model_10_100(rs), '-', label='Rmax = 100 kpc')
#plt.plot(rs, model_10_200(rs), '-', label='Rmax = 200 kpc')
plt.plot(rs, model_10_300(rs), '-', label='Rmax = 300 kpc')
plt.plot(rs, model_10_350(rs), '-', label='Rmax = 350 kpc')
plt.plot(rs, model_10_400(rs), '-', label='Rmax = 400 kpc')
plt.plot(rs, model_10_500(rs), '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,600)
plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profile.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_2P_mass_profile.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], model_10_100(rs[1:])/np.cumsum(mass), '-', label='Rmax = 100 kpc')
#plt.plot(rs[1:], model_10_200(rs[1:])/np.cumsum(mass), '-', label='Rmax = 200 kpc')
plt.plot(rs[1:], model_10_300(rs[1:])/np.cumsum(mass), '-', label='Rmax = 300 kpc')
plt.plot(rs[1:], model_10_350(rs[1:])/np.cumsum(mass), '-', label='Rmax = 350 kpc')
plt.plot(rs[1:], model_10_400(rs[1:])/np.cumsum(mass), '-', label='Rmax = 400 kpc')
plt.plot(rs[1:], model_10_500(rs[1:])/np.cumsum(mass), '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.xlim(5,600)
plt.ylim(ymin=0.9,ymax=1.2)
plt.hlines(y=1, xmin=1, xmax=500, colors='k', linestyles='dotted')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm model}/M(<r)_{\\rm data}$', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profile_ratio.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_2P_mass_profile_ratio.pdf')
plt.close()

## Average the mass ratio from 10 - 500 kpc
#print('Standard deviation offset for 1 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_1(rs[44:]) - 1))))
#print('Standard deviation offset for 2 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_2(rs[44:]) - 1))))
#print('Standard deviation offset for 3 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_3(rs[44:]) - 1))))
#print('Standard deviation offset for 5 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_5(rs[44:]) - 1))))
#print('Standard deviation offset for 10 kpc cutoff is {0:.4g}'.format(np.std(np.abs(np.cumsum(mass)[43:]/model_10(rs[44:]) - 1))))

# Use the mass model parameters to plot the density profiles
#dens_10_100 = np.zeros(len(rs)-1)
#dens_10_200 = np.zeros(len(rs)-1)
dens_10_300 = np.zeros(len(rs)-1)
dens_10_350 = np.zeros(len(rs)-1)
dens_10_400 = np.zeros(len(rs)-1)
dens_10_500 = np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    #dens_10_100[i] = (model_10_100(rs[i+1]) - model_10_100(rs[i]))/volume
    #dens_10_200[i] = (model_10_200(rs[i+1]) - model_10_200(rs[i]))/volume
    dens_10_300[i] = (model_10_300(rs[i+1]) - model_10_300(rs[i]))/volume
    dens_10_350[i] = (model_10_350(rs[i+1]) - model_10_350(rs[i]))/volume
    dens_10_400[i] = (model_10_400(rs[i+1]) - model_10_400(rs[i]))/volume
    dens_10_500[i] = (model_10_500(rs[i+1]) - model_10_500(rs[i]))/volume

# Plot the derived density profiles on top of the actual density
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
#plt.plot(rs[1:], dens_10_100, '-', label='Rmax = 100 kpc')
#plt.plot(rs[1:], dens_10_200, '-', label='Rmax = 200 kpc')
plt.plot(rs[1:], dens_10_300, '-', label='Rmax = 300 kpc')
plt.plot(rs[1:], dens_10_350, '-', label='Rmax = 350 kpc')
plt.plot(rs[1:], dens_10_400, '-', label='Rmax = 400 kpc')
plt.plot(rs[1:], dens_10_500, '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,600)
plt.ylim(ymax=1e9)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profiles_density.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_2P_mass_profiles_density.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], dens_10_100/density, '-', label='Rmax = 100 kpc')
#plt.plot(rs[1:], dens_10_200/density, '-', label='Rmax = 200 kpc')
plt.plot(rs[1:], dens_10_300/density, '-', label='Rmax = 300 kpc')
plt.plot(rs[1:], dens_10_350/density, '-', label='Rmax = 350 kpc')
plt.plot(rs[1:], dens_10_400/density, '-', label='Rmax = 400 kpc')
plt.plot(rs[1:], dens_10_500/density, '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.hlines(y=1, xmin=1, xmax=500, colors='k', linestyles='dotted')
plt.xlim(1,600)
plt.ylim(0.5, 2)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm model}/\\rho_{\\rm data}$', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profiles_density_ratio.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_2+'/'+sim_data.gal_2+'_2P_mass_profiles_density_ratio.pdf')
plt.close()








"""
# beta = 3 fixed
@custom_model
def two_power_beta_fixed(r, amp=1e12, a=10, alpha=0.5):
    #return (4*np.pi*amp*a**3)*(((r/a)**(3.-alpha))/(3.-alpha)*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a))
    return ((amp/a**3)*(r**(3-alpha))*(a+r)**(alpha-3)*(r/a+1)**(3-1)*a**3/(3-alpha))*special.hyp2f1(3.-alpha,-alpha+3,4.-alpha,-r/a)

# Fit the model to the data for various cutoff radii
model_init = two_power_beta_fixed(bounds={'amp':(3e11, 2.25e12), 'a':(5, 50), 'alpha':(0, 1.5)})
fit = LevMarLSQFitter()
model_0 = fit(model_init, rs[1:], np.cumsum(mass), maxiter=1000000)
model_1 = fit(model_init, rs[22:], np.cumsum(mass)[21:], maxiter=1000000)
model_2 = fit(model_init, rs[28:], np.cumsum(mass)[27:], maxiter=1000000)
model_3 = fit(model_init, rs[32:], np.cumsum(mass)[31:], maxiter=1000000)
model_5 = fit(model_init, rs[37:], np.cumsum(mass)[36:], maxiter=1000000)
model_10 = fit(model_init, rs[44:], np.cumsum(mass)[43:], maxiter=1000000)
print(model_0)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

# Plot all of the profiles
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
#plt.plot(rs[1:], model_1(rs[1:]), '-', label='1 kpc fit')
#plt.plot(rs[1:], model_2(rs[1:]), '-', label='2 kpc fit')
#plt.plot(rs[1:], model_3(rs[1:]), '-', label='3 kpc fit')
#plt.plot(rs[1:], model_5(rs[1:]), '-', label='5 kpc fit')
#plt.plot(rs[1:], model_10(rs[1:]), '-', label='10 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_1(rs[1:]), '-', label='1 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_2(rs[1:]), '-', label='2 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_3(rs[1:]), '-', label='3 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_5(rs[1:]), '-', label='5 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_10(rs[1:]), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
#plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm data}/M(<r)_{\\rm model}$', fontsize=28)
plt.title('$\\beta = 3$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_mass_profiles_beta_fixed_ratio.pdf')
plt.close()

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
#plt.plot(rs[1:], density, 'k.', label='data')
#plt.plot(rs[1:], dens_1, '-', label='1 kpc fit')
#plt.plot(rs[1:], dens_2, '-', label='2 kpc fit')
#plt.plot(rs[1:], dens_3, '-', label='3 kpc fit')
#plt.plot(rs[1:], dens_5, '-', label='5 kpc fit')
#plt.plot(rs[1:], dens_10, '-', label='10 kpc fit')
plt.plot(rs[1:], density/dens_1, '-', label='1 kpc fit')
plt.plot(rs[1:], density/dens_2, '-', label='2 kpc fit')
plt.plot(rs[1:], density/dens_3, '-', label='3 kpc fit')
plt.plot(rs[1:], density/dens_5, '-', label='5 kpc fit')
plt.plot(rs[1:], density/dens_10, '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}/\\rho_{\\rm model}$', fontsize=28)
plt.title('$\\beta = 3$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_mass_profiles_density_beta_fixed_ratio.pdf')
plt.close()


# alpha = 0.5 fixed
@custom_model
def two_power_beta_fixed(r, amp=1e12, a=10, beta=3):
    #return (4*np.pi*amp*a**3)*(((r/a)**(3.-alpha))/(3.-alpha)*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a))
    return ((amp/a**3)*(r**(3-0.5))*(a+r)**(0.5-beta)*(r/a+1)**(beta-1)*a**beta/(3-0.5))*special.hyp2f1(3.-0.5,-0.5+beta,4.-0.5,-r/a)

# Fit the model to the data for various cutoff radii
model_init = two_power_beta_fixed(bounds={'amp':(3e10, 2.25e12), 'a':(5, 30), 'beta':(2,5)})
fit = LevMarLSQFitter()
model_0 = fit(model_init, rs[1:], np.cumsum(mass), maxiter=1000000)
model_1 = fit(model_init, rs[22:], np.cumsum(mass)[21:], maxiter=1000000)
model_2 = fit(model_init, rs[28:], np.cumsum(mass)[27:], maxiter=1000000)
model_3 = fit(model_init, rs[32:], np.cumsum(mass)[31:], maxiter=1000000)
model_5 = fit(model_init, rs[37:], np.cumsum(mass)[36:], maxiter=1000000)
model_10 = fit(model_init, rs[44:], np.cumsum(mass)[43:], maxiter=1000000)
print(model_0)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

# Plot all of the profiles
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
#plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
#plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
#plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
#plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
#plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_1(rs[1:]), '-', label='1 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_2(rs[1:]), '-', label='2 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_3(rs[1:]), '-', label='3 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_5(rs[1:]), '-', label='5 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/model_10(rs[1:]), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
#plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm data}/M(<r)_{\\rm model}$', fontsize=28)
plt.title('$\\alpha = 0.5$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_mass_profiles_alpha_fixed_ratio.pdf')
plt.close()

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
#plt.plot(rs[1:], density, 'k.', label='data')
#plt.plot(rs[1:], dens_1, '-', label='1 kpc fit')
#plt.plot(rs[1:], dens_2, '-', label='2 kpc fit')
#plt.plot(rs[1:], dens_3, '-', label='3 kpc fit')
#plt.plot(rs[1:], dens_5, '-', label='5 kpc fit')
#plt.plot(rs[1:], dens_10, '-', label='10 kpc fit')
plt.plot(rs[1:], density/dens_1, '-', label='1 kpc fit')
plt.plot(rs[1:], density/dens_2, '-', label='2 kpc fit')
plt.plot(rs[1:], density/dens_3, '-', label='3 kpc fit')
plt.plot(rs[1:], density/dens_5, '-', label='5 kpc fit')
plt.plot(rs[1:], density/dens_10, '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}/\\rho_{\\rm model}$', fontsize=28)
plt.title('$\\alpha = 0.5$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_mass_profiles_density_alpha_fixed_ratio.pdf')
plt.close()
"""

"""
    - Define the two-power spherical density model

    - Fit the model to the data
        - Let both alpha and beta vary
        - Try keeping beta = 3 fixed and vary everything else
        - Try keeping alpha = 0.5 fixed and vary everything else

    - Want to exclude the inner [1, 2, 3, 5, 10] kpc

# alpha/beta vary
@custom_model
def two_power_beta_fixed(r, amp=1e12, a=10, alpha=0.5, beta=3):
    return amp/( (4*np.pi*a**3) * (r/a)**alpha * (1+r/a)**(beta - alpha) )

# Fit the model to the data for various cutoff radii
model_init = two_power_beta_fixed(bounds={'amp':(3e10, 2.25e12), 'a':(5, 30), 'alpha':(0, 1.5), 'beta':(2, 5)})
fit = LevMarLSQFitter()
model_0 = fit(model_init, rs[1:], density, maxiter=1000000)
model_1 = fit(model_init, rs[22:], density[21:], maxiter=1000000)
model_2 = fit(model_init, rs[28:], density[27:], maxiter=1000000)
model_3 = fit(model_init, rs[32:], density[31:], maxiter=1000000)
model_5 = fit(model_init, rs[37:], density[36:], maxiter=1000000)
model_10 = fit(model_init, rs[44:], density[43:], maxiter=1000000)
print(model_0)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

# Plot all of the profiles
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], density, 'k.', label='data')
#plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
#plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
#plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
#plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
#plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.plot(rs[1:], density/model_1(rs[1:]), '-', label='1 kpc fit')
plt.plot(rs[1:], density/model_2(rs[1:]), '-', label='2 kpc fit')
plt.plot(rs[1:], density/model_3(rs[1:]), '-', label='3 kpc fit')
plt.plot(rs[1:], density/model_5(rs[1:]), '-', label='5 kpc fit')
plt.plot(rs[1:], density/model_10(rs[1:]), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
#plt.ylim(0,3)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}/\\rho_{\\rm model}$', fontsize=28)
plt.title('$\\beta$, $\\alpha$ float', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_density_profiles_float_ratio.pdf')
plt.close()

# Calculate what the mass enclosed is, via the model from the density profile
m_encl_1 = np.zeros(len(rs)-1)
m_encl_2 = np.zeros(len(rs)-1)
m_encl_3 = np.zeros(len(rs)-1)
m_encl_5 = np.zeros(len(rs)-1)
m_encl_10 = np.zeros(len(rs)-1)
m_tot_1 = 0
m_tot_2 = 0
m_tot_3 = 0
m_tot_5 = 0
m_tot_10 = 0
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    m_encl_1[i] = model_1(rs[i+1])*volume
    m_encl_2[i] = model_2(rs[i+1])*volume
    m_encl_3[i] = model_3(rs[i+1])*volume
    m_encl_5[i] = model_5(rs[i+1])*volume
    m_encl_10[i] = model_10(rs[i+1])*volume
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

# Plot all of the models
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
#plt.plot(rs[1:], np.cumsum(m_encl_1), '-', label='1 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_2), '-', label='2 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_3), '-', label='3 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_5), '-', label='5 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_1), '-', label='1 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_2), '-', label='2 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_3), '-', label='3 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_5), '-', label='5 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
#plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm data}/M(<r)_{\\rm model}$', fontsize=28)
plt.title('$\\beta$, $\\alpha$ float', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_density_profiles_mass_float_ratio.pdf')
plt.close()


##############################################################################


# Fix beta = 3
@custom_model
def two_power_beta_fixed(r, amp=1e12, a=10, alpha=0.5):
    return amp/( (4*np.pi*a**3) * (r/a)**alpha * (1+r/a)**(3 - alpha) )

# Fit the model to the data for various cutoff radii
model_init = two_power_beta_fixed(bounds={'amp':(3e10, 2.25e12), 'a':(5,30), 'alpha':(0, 1.5)})
fit = LevMarLSQFitter()
model_0 = fit(model_init, rs[1:], density, maxiter=1000000)
model_1 = fit(model_init, rs[22:], density[21:], maxiter=1000000)
model_2 = fit(model_init, rs[28:], density[27:], maxiter=1000000)
model_3 = fit(model_init, rs[32:], density[31:], maxiter=1000000)
model_5 = fit(model_init, rs[37:], density[36:], maxiter=1000000)
model_10 = fit(model_init, rs[44:], density[43:], maxiter=1000000)
print(model_0)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

# Plot all of the profiles
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], density, 'k.', label='data')
#plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
#plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
#plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
#plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
#plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.plot(rs[1:], density/model_1(rs[1:]), '-', label='1 kpc fit')
plt.plot(rs[1:], density/model_2(rs[1:]), '-', label='2 kpc fit')
plt.plot(rs[1:], density/model_3(rs[1:]), '-', label='3 kpc fit')
plt.plot(rs[1:], density/model_5(rs[1:]), '-', label='5 kpc fit')
plt.plot(rs[1:], density/model_10(rs[1:]), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}/\\rho_{\\rm model}$', fontsize=28)
plt.title('$\\beta = 3$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_density_profiles_beta_fixed_ratio.pdf')
plt.close()

# Calculate what the mass enclosed is, via the model from the density profile
m_encl_1 = np.zeros(len(rs)-1)
m_encl_2 = np.zeros(len(rs)-1)
m_encl_3 = np.zeros(len(rs)-1)
m_encl_5 = np.zeros(len(rs)-1)
m_encl_10 = np.zeros(len(rs)-1)
m_tot_1 = 0
m_tot_2 = 0
m_tot_3 = 0
m_tot_5 = 0
m_tot_10 = 0
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    m_encl_1[i] = model_1(rs[i+1])*volume
    m_encl_2[i] = model_2(rs[i+1])*volume
    m_encl_3[i] = model_3(rs[i+1])*volume
    m_encl_5[i] = model_5(rs[i+1])*volume
    m_encl_10[i] = model_10(rs[i+1])*volume
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

# Plot all of the models
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
#plt.plot(rs[1:], np.cumsum(m_encl_1), '-', label='1 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_2), '-', label='2 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_3), '-', label='3 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_5), '-', label='5 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_1), '-', label='1 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_2), '-', label='2 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_3), '-', label='3 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_5), '-', label='5 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
#plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm data}/M(<r)_{\\rm model}$', fontsize=28)
plt.title('$\\beta$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_density_profiles_mass_beta_fixed_ratio.pdf')
plt.close()


##############################################################################


# Fix alpha = 0.5
@custom_model
def two_power_alpha_fixed(r, amp=1e12, a=10, beta=3):
    return amp/( (4*np.pi*a**3) * (r/a)**0.5 * (1+r/a)**(beta - 0.5) )

# Fit the model to the data for various cutoff radii
model_init = two_power_alpha_fixed(bounds={'amp':(3e10, 2.25e12), 'a':(1,30), 'beta':(2, 5)})
fit = LevMarLSQFitter()
model_0 = fit(model_init, rs[1:], density, maxiter=1000000)
model_1 = fit(model_init, rs[22:], density[21:], maxiter=1000000)
model_2 = fit(model_init, rs[28:], density[27:], maxiter=1000000)
model_3 = fit(model_init, rs[32:], density[31:], maxiter=1000000)
model_5 = fit(model_init, rs[37:], density[36:], maxiter=1000000)
model_10 = fit(model_init, rs[44:], density[43:], maxiter=1000000)
print(model_0)
print(model_1)
print(model_2)
print(model_3)
print(model_5)
print(model_10)

# Plot all of the profiles
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], density, 'k.', label='data')
#plt.plot(rs, model_1(rs), '-', label='1 kpc fit')
#plt.plot(rs, model_2(rs), '-', label='2 kpc fit')
#plt.plot(rs, model_3(rs), '-', label='3 kpc fit')
#plt.plot(rs, model_5(rs), '-', label='5 kpc fit')
#plt.plot(rs, model_10(rs), '-', label='10 kpc fit')
plt.plot(rs[1:], density/model_1(rs[1:]), '-', label='1 kpc fit')
plt.plot(rs[1:], density/model_2(rs[1:]), '-', label='2 kpc fit')
plt.plot(rs[1:], density/model_3(rs[1:]), '-', label='3 kpc fit')
plt.plot(rs[1:], density/model_5(rs[1:]), '-', label='5 kpc fit')
plt.plot(rs[1:], density/model_10(rs[1:]), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
#plt.ylim(0,10)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.ylabel('$\\rho_{\\rm data}/\\rho_{\\rm model}$', fontsize=28)
plt.title('$\\alpha = 0.5$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_density_profiles_alpha_fixed_ratio.pdf')
plt.close()

# Calculate what the mass enclosed is, via the model from the density profile
m_encl_1 = np.zeros(len(rs)-1)
m_encl_2 = np.zeros(len(rs)-1)
m_encl_3 = np.zeros(len(rs)-1)
m_encl_5 = np.zeros(len(rs)-1)
m_encl_10 = np.zeros(len(rs)-1)
m_tot_1 = 0
m_tot_2 = 0
m_tot_3 = 0
m_tot_5 = 0
m_tot_10 = 0
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    m_encl_1[i] = model_1(rs[i+1])*volume
    m_encl_2[i] = model_2(rs[i+1])*volume
    m_encl_3[i] = model_3(rs[i+1])*volume
    m_encl_5[i] = model_5(rs[i+1])*volume
    m_encl_10[i] = model_10(rs[i+1])*volume
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

# Plot all of the models
plt.figure(figsize=(10,8))
#plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
#plt.plot(rs[1:], np.cumsum(m_encl_1), '-', label='1 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_2), '-', label='2 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_3), '-', label='3 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_5), '-', label='5 kpc fit')
#plt.plot(rs[1:], np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_1), '-', label='1 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_2), '-', label='2 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_3), '-', label='3 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_5), '-', label='5 kpc fit')
plt.plot(rs[1:], np.cumsum(mass)/np.cumsum(m_encl_10), '-', label='10 kpc fit')
plt.xscale('log')
#plt.yscale('log')
plt.xlim(xmin=1)
#plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
#plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm data}/M(<r)_{\\rm model}$', fontsize=28)
plt.title('$\\alpha$ fixed', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
plt.savefig(home_dir+'/orbit_plots/fitting/two_power_model/'+gal1+'/'+gal1+'_2P_density_profiles_mass_alpha_fixed_ratio.pdf')
plt.close()
"""
