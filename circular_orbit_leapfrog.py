#!/usr/bin/python3

"""

  ========================================
  = Correcting halo amplitude (leapfrog) =
  ========================================

  Written by Isaiah Santistevan (ibsantistevan@ucdavis.edu) during Winter Quarter, 2021

  - Set up halos in circular orbits
      - Integrate the halos in the 2P halo potential
        - Vary the amplitude until you get the most circular orbit

"""


from galpy.orbit import Orbit
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
from astropy import units as u
import pandas as pd
from scipy import special

print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='peloton')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir)

# Read in the fitting parameters
fitting_data_2p = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params.csv', index_col=0)
fitting_data_nfw = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params_nfw.csv', index_col=0)
fitting_data_nfw_v2 = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_params_nfw_v2.csv', index_col=0)
#
# Read in the subhalo initial conditions
ts = snaps['time']*(-1)*u.Gyr

# Define the 2P halo enclosed mass profile
def halo_mass(r, gal):
    A_halo = fitting_data_2p['A_halo'][gal]
    a_halo = fitting_data_2p['a_halo'][gal]
    alpha = fitting_data_2p['alpha'][gal]
    beta = fitting_data_2p['beta'][gal]
    #
    return ((A_halo/a_halo**3)*(r**(3-alpha))*(a_halo+r)**(alpha-beta)*(r/a_halo+1)**(beta-1)*a_halo**beta/(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a_halo)
#
# Define the circular velocity function
def vcirc(r,m):
    G = 6.67*10**(-11)*1000**(-3) # km^3 kg^(-1) s^(-2)
    mass = m*2*10**(30) # kg
    distance = r*10**3*3.086*10**(13) # km
    return np.sqrt(np.array(G*mass/distance, dtype=np.float64))


# Import the potentials
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import NFWPotential


# Find what the circular velocity should be for a given radius in 2P NFW
dists = np.array([5.0, 10.0, 15.0, 30.0, 50.0, 100.0, 150.0])
#
for k in range(0, len(dists)):
    vc = vcirc(dists[k], halo_mass(dists[k], sim_data.galaxy))
    #
    # Find out what is the best fix to the amplitude to ensure most circular orbit
    fixes = np.linspace(1.0, 2.0, 201)
    vars = np.zeros(len(fixes))
    for i in range(0, len(fixes)):
        potential_two_power = TwoPowerSphericalPotential(amp=fitting_data_2p['A_halo'][sim_data.galaxy]*fixes[i]*u.solMass, a=fitting_data_2p['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data_2p['alpha'][sim_data.galaxy], beta=fitting_data_2p['beta'][sim_data.galaxy])
        orb = Orbit([dists[k]*(u.kpc), 0.0*(u.km/u.s), (vc)*(u.km/u.s), 0.0*(u.kpc), 0.0*(u.km/u.s), 0.0*(u.deg)])
        orb.integrate(ts, potential_two_power, method='leapfrog')
        vars[i] = np.var(orb.r(ts))
    print(vars)
    print(np.where(np.min(vars) == vars)[0], np.min(vars), fixes[np.where(np.min(vars) == vars)[0]])
