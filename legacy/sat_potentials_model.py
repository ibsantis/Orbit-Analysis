#!/usr/bin/env python3
#SBATCH --job-name=model_potential
##SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --partition=high2    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
##SBATCH --partition=icx-normal
##SBATCH --mem=250G
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1    # processes total, used to be 4...
##SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/potential/model_potential_%j.txt
##SBATCH --output=/home1/05400/ibsantis/scripts/jobs/potential/model_potential_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin

"""
    ========================
    = Check E conservation =
    ========================

    Check if energy is conserved in the model for each host

"""

from galpy.orbit import Orbit
from orbit_analysis import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
from astropy import units as u
import pandas as pd
import sys
print('Read in the tools')


### Set path and initial parameters
loc = 'peloton'
#loc='stampede'
sim_data = orbit_io.OrbitRead(gal1=str(sys.argv[1]), location=loc)
aligned = True
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal, assign_hosts_rotation=aligned)

# Import the potentials and combine them for our model
fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import evaluatePotentials

if sim_data.num_gal == 1:
    #
    disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
    disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
    halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
    potential_two_power = disk_inner+disk_outer+halo_2p
    #
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, dmo=False)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.galaxy, location=loc, host=1, dmo=False)
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt)
    #
    # Integrate all of the orbits in both potentials
    ts = np.flip(snaps['time'] - snaps['time'][-1])*u.Gyr
    galpy_orbits.integrate(ts, potential_two_power, method='odeint')
    print('Done integrating in potential model')
    #
    model_pot = (-1)*np.ones(galpy_orbits.R(ts).shape)
    for i in range(0, galpy_orbits.R(ts).shape[0]):
        for j in range(0, galpy_orbits.R(ts).shape[1]):
            model_pot[i,j] = evaluatePotentials(potential_two_power, galpy_orbits.R(ts)[i,j]*u.kpc, galpy_orbits.z(ts)[i,j]*u.kpc, phi=galpy_orbits.phi(ts)[i,j]*u.deg, t=ts[j])
    #
    d = dict()
    d['model.potential'] = model_pot
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.galaxy+'_potentials_model', dict_or_array_to_write=d, verbose=True)

if sim_data.num_gal == 2:
    #
    ## Galaxy 1
    disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.gal_1]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.gal_1]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_1]*u.kpc)
    disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.gal_1]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.gal_1]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_1]*u.kpc)
    halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.gal_1]*u.solMass, a=fitting_data['a_halo'][sim_data.gal_1]*u.kpc, alpha=fitting_data['alpha'][sim_data.gal_1], beta=fitting_data['beta'][sim_data.gal_1])
    potential_two_power = disk_inner+disk_outer+halo_2p
    #
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, dmo=False)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location=loc, host=1, dmo=False)
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt)
    #
    # Integrate all of the orbits in both potentials
    ts = np.flip(snaps['time'] - snaps['time'][-1])*u.Gyr
    galpy_orbits.integrate(ts, potential_two_power, method='odeint')
    print('Done integrating in potential model')
    #
    model_pot = (-1)*np.ones(galpy_orbits.R(ts).shape)
    for i in range(0, galpy_orbits.R(ts).shape[0]):
        for j in range(0, galpy_orbits.R(ts).shape[1]):
            model_pot[i,j] = evaluatePotentials(potential_two_power, galpy_orbits.R(ts)[i,j]*u.kpc, galpy_orbits.z(ts)[i,j]*u.kpc, phi=galpy_orbits.phi(ts)[i,j]*u.deg, t=ts[j])
        print('Done with satellite {0} out of {1}'.format(i+1, galpy_orbits.R(ts).shape[0]+1))
    #
    d = dict()
    d['model.potential'] = model_pot
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.gal_1+'_potentials_model', dict_or_array_to_write=d, verbose=True)
    #
    #
    ## Galaxy 2
    disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.gal_2]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.gal_2]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_2]*u.kpc)
    disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.gal_2]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.gal_2]*u.kpc, hz=fitting_data['h_z'][sim_data.gal_2]*u.kpc)
    halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.gal_2]*u.solMass, a=fitting_data['a_halo'][sim_data.gal_2]*u.kpc, alpha=fitting_data['alpha'][sim_data.gal_2], beta=fitting_data['beta'][sim_data.gal_2])
    potential_two_power = disk_inner+disk_outer+halo_2p
    #
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, dmo=False)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location=loc, host=2, dmo=False)
    #
    # Initialize the orbits in Galpy
    galpy_orbits = orbit_gal.galpy_orbit_init(tree=halt, host=2)
    #
    # Integrate all of the orbits in both potentials
    ts = np.flip(snaps['time'] - snaps['time'][-1])*u.Gyr
    galpy_orbits.integrate(ts, potential_two_power, method='odeint')
    print('Done integrating in potential model')
    #
    model_pot = (-1)*np.ones(galpy_orbits.R(ts).shape)
    for i in range(0, galpy_orbits.R(ts).shape[0]):
        for j in range(0, galpy_orbits.R(ts).shape[1]):
            model_pot[i,j] = evaluatePotentials(potential_two_power, galpy_orbits.R(ts)[i,j]*u.kpc, galpy_orbits.z(ts)[i,j]*u.kpc, phi=galpy_orbits.phi(ts)[i,j]*u.deg, t=ts[j])
        print('Done with satellite {0} out of {1}'.format(i+1, galpy_orbits.R(ts).shape[0]+1))
    #
    d = dict()
    d['model.potential'] = model_pot
    ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/potentials/'+sim_data.gal_2+'_potentials_model', dict_or_array_to_write=d, verbose=True)
