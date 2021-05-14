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
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='peloton')
print('Set paths')


# Read in the data
part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'redshift', 0, simulation_directory=sim_data.simulation_dir, assign_hosts_rotation=True, assign_formation_coordinates=True)
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
"""
# Need to calculate density on my own, the particles won't help
rs = np.logspace(np.log10(0.1), np.log10(500), 100)
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

ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting/halo/'+sim_data.gal_1+'_halo_profile_fitting', dict_or_array_to_write=d1, verbose=True)

if sim_data.num_gal == 2:
    rs = np.logspace(np.log10(0.1), np.log10(500), 100)
    mass = np.zeros(len(rs)-1)
    density = np.zeros(len(rs)-1)
    gas_temp_inds = ut.array.get_indices(part['gas']['temperature'], [1e5, np.inf])
    for i in range(0, len(rs)-1):
        gas_inds = ut.array.get_indices(part['gas'].prop('host2.distance.total'), [rs[i], rs[i+1]], gas_temp_inds)
        dark_inds = ut.array.get_indices(part['dark'].prop('host2.distance.total'), [rs[i], rs[i+1]])
        mass[i] = np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
        density[i] = mass[i]/(4/3*np.pi*(rs[i+1]**3-rs[i]**3))
        print('done with step', i)

    d2 = dict()
    d2['density'] = density
    d2['mass'] = mass
    d2['rs'] = rs

    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting/halo/'+sim_data.gal_2+'_halo_profile_fitting', dict_or_array_to_write=d2, verbose=True)


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

# Read in the data
data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/gas_dm_only/'+sim_data.galaxy+'_halo_profile_fitting')
#data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/gas_dm_only/'+sim_data.gal_1+'_halo_profile_fitting')
#data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+sim_data.galaxy+'_halo_fitting')
#data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+sim_data.gal_1+'_halo_fitting')
density = data['density']
mass = data['mass']
rs = data['rs']

halo_models = model_io.MassModelFit()

halo_300 = halo_models.halo_2p_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, slope_in=0.5, slope_out=3, A_halo_bounds=(3e9,2.25e12), a_halo_bounds=(0.5,50), slope_in_bounds=(0,2.5), slope_out_bounds=(-2,5), r_max=300)
halo_350 = halo_models.halo_2p_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, slope_in=0.5, slope_out=3, A_halo_bounds=(3e9,2.25e12), a_halo_bounds=(0.5,50), slope_in_bounds=(0,2.5), slope_out_bounds=(-2,5), r_max=350)
halo_400 = halo_models.halo_2p_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, slope_in=0.5, slope_out=3, A_halo_bounds=(3e9,2.25e12), a_halo_bounds=(0.5,50), slope_in_bounds=(0,2.5), slope_out_bounds=(-2,5), r_max=400)
halo_500 = halo_models.halo_2p_mass_model(distances=rs, masses=mass, A_halo=1e12, a_halo=10, slope_in=0.5, slope_out=3, A_halo_bounds=(3e9,2.25e12), a_halo_bounds=(0.5,50), slope_in_bounds=(0,2.5), slope_out_bounds=(-2,5))

# Average the mass ratio from 10 - X kpc
print('Standard deviation offset for 300 kpc cutoff is {0:.4g}'.format(np.std(np.abs(model_10_300(rs[54:94])/np.cumsum(mass)[53:93] - 1))))
print('Standard deviation offset for 350 kpc cutoff is {0:.4g}'.format(np.std(np.abs(model_10_350(rs[54:96])/np.cumsum(mass)[53:95] - 1))))
print('Standard deviation offset for 400 kpc cutoff is {0:.4g}'.format(np.std(np.abs(model_10_400(rs[54:97])/np.cumsum(mass)[53:96] - 1))))
print('Standard deviation offset for 500 kpc cutoff is {0:.4g}'.format(np.std(np.abs(model_10_500(rs[54:])/np.cumsum(mass)[53:] - 1))))

# Plot all of the profiles
plt.figure(figsize=(10,8))
plt.plot(rs[1:], np.cumsum(mass), 'k.', label='data')
plt.plot(rs, halo_300(rs), '-', label='Rmax = 300 kpc')
plt.plot(rs, halo_350(rs), '-', label='Rmax = 350 kpc')
plt.plot(rs, halo_400(rs), '-', label='Rmax = 400 kpc')
plt.plot(rs, halo_500(rs), '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.yscale('log')
plt.xlim(5,600)
plt.ylim(ymin=1e8)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('M(<r) [M$_{\\odot}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profile.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profile.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profile.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profile.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], halo_300(rs[1:])/np.cumsum(mass), '-', label='Rmax = 300 kpc')
plt.plot(rs[1:], halo_350(rs[1:])/np.cumsum(mass), '-', label='Rmax = 350 kpc')
plt.plot(rs[1:], halo_400(rs[1:])/np.cumsum(mass), '-', label='Rmax = 400 kpc')
plt.plot(rs[1:], halo_500(rs[1:])/np.cumsum(mass), '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.xlim(5,600)
plt.ylim(ymin=0.9,ymax=1.2)
plt.hlines(y=1, xmin=1, xmax=500, colors='k', linestyles='dotted')
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M(<r)_{\\rm model}/M(<r)_{\\rm sim}$', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profile_ratio.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profile_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profile_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profile_ratio.pdf')
plt.close()

# Use the mass model parameters to plot the density profiles
dens_300 = np.zeros(len(rs)-1)
dens_350 = np.zeros(len(rs)-1)
dens_400 = np.zeros(len(rs)-1)
dens_500 = np.zeros(len(rs)-1)
for i in range(0, len(rs)-1):
    volume = 4/3*np.pi*(rs[i+1]**3-rs[i]**3)
    dens_300[i] = (halo_300(rs[i+1]) - halo_300(rs[i]))/volume
    dens_350[i] = (halo_350(rs[i+1]) - halo_350(rs[i]))/volume
    dens_400[i] = (halo_400(rs[i+1]) - halo_400(rs[i]))/volume
    dens_500[i] = (halo_500(rs[i+1]) - halo_500(rs[i]))/volume

# Plot the derived density profiles on top of the actual density
plt.figure(figsize=(10,8))
plt.plot(rs[1:], density, 'k.', label='data')
plt.plot(rs[1:], dens_300, '-', label='Rmax = 300 kpc')
plt.plot(rs[1:], dens_350, '-', label='Rmax = 350 kpc')
plt.plot(rs[1:], dens_400, '-', label='Rmax = 400 kpc')
plt.plot(rs[1:], dens_500, '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.yscale('log')
plt.xlim(5,600)
plt.ylim(ymax=1e9)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho$ [M$_{\\odot}$ kpc$^{-3}$]', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profiles_density.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profiles_density.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profiles_density.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profiles_density.pdf')
plt.close()
#
plt.figure(figsize=(10,8))
plt.plot(rs[1:], dens_300/density, '-', label='Rmax = 300 kpc')
plt.plot(rs[1:], dens_350/density, '-', label='Rmax = 350 kpc')
plt.plot(rs[1:], dens_400/density, '-', label='Rmax = 400 kpc')
plt.plot(rs[1:], dens_500/density, '-', label='Rmax = 500 kpc')
plt.xscale('log')
plt.hlines(y=1, xmin=1, xmax=500, colors='k', linestyles='dotted')
plt.xlim(5,600)
plt.ylim(0.5, 2)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$\\rho_{\\rm model}/\\rho_{\\rm sim}$', fontsize=28)
plt.legend(prop={'size': 18})
plt.tight_layout()
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profiles_density_ratio.pdf')
plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model_gas_dm/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profiles_density_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_2P_mass_profiles_density_ratio.pdf')
#plt.savefig(sim_data.home_dir+'/orbit_data/plots/fitting/two_power_model/'+sim_data.gal_1+'/'+sim_data.gal_1+'_2P_mass_profiles_density_ratio.pdf')
plt.close()
