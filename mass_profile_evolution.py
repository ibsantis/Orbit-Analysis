#!/usr/bin/python3

"""

  ==========================
  = Mass Profile Evolution =
  ==========================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  Calculate what the mass profile is accounting for ALL particles
  within bins of spherical r, out to 500 kpc

  Do this every 100 Myr going back to 1 Gyr (11 snapshots total)

"""
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
print('Read in the tools')

### Set path and initial parameters
gal1 = 'm12r'
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

# Set up snapshot array to loop through
snaps = np.array([600, 587, 582, 578, 573, 569, 564, 560, 556, 551, 547])
times = np.array([13.8, 13.7, 13.6, 13.5, 13.4, 13.3, 13.2, 13.1, 13.0, 12.9, 12.8])
rs = np.logspace(np.log10(0.1), np.log10(500), 100)

# Set up dictionary to save the data
mass_dict = dict()
mass_dict['snapshot'] = snaps
mass_dict['time'] = times
mass_dict['mass.profile'] = np.zeros((len(snaps), len(rs)))

# Loop through the snapshots
for i in range(0, len(snaps)):
    #
    # Read in the data
    part = gizmo.io.Read.read_snapshots(['star','gas','dark'], 'snapshot', snaps[i], simulation_directory=simulation_dir, assign_hosts_rotation=True)
    print('Particles at snapshot {0} read in.'.format(snaps[i]))
    #
    # Find the enclosed mass of all particles within 0.1 < R < 500 kpc
    for j in range(0, len(rs)-1):
        star_inds = ut.array.get_indices(part['star'].prop('host.distance.total'), [rs[j], rs[j+1]])
        gas_inds = ut.array.get_indices(part['gas'].prop('host.distance.total'), [rs[j], rs[j+1]])
        dark_inds = ut.array.get_indices(part['dark'].prop('host.distance.total'), [rs[j], rs[j+1]])
        mass_dict['mass.profile'][i,j] = np.sum(part['star']['mass'][star_inds]) + np.sum(part['gas']['mass'][gas_inds]) + np.sum(part['dark']['mass'][dark_inds])
        print('Done with step', j)
    #
# Save this data to a file
ut.io.file_hdf5(file_name_base=home_dir+'/orbit_data/hdf5_files/mass_profiles/'+gal1+'_mass_profile_evolution', dict_or_array_to_write=mass_dict, verbose=True)

print('All done!')
