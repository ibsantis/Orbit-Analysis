#!/usr/bin/env python3
#SBATCH --job-name=halo_data_m12i_hr
##SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --partition=high2    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --ntasks=1    # processes total
#SBATCH --time=05:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/mass_profiles/halo_data_m12i_hr_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

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
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from scipy import special
import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i_hr', location='peloton')
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

ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting/halo/'+sim_data.galaxy+'_halo_fitting', dict_or_array_to_write=d1, verbose=True)

print('Done!')