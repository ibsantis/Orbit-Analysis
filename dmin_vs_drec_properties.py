

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
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Thelma', location='peloton')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)
#
orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1, dmo=False)
orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1, dmo=False)
orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.gal_1, location='peloton', host=1, dmo=False)
#
# Run the pipeline on the simulation data
halt_dists = orbits.halo_distances(tree=halt) # set host=1 for the first host, host=2 for the other
halt_dists_3d = orbits.halo_distances(tree=halt, dist_type='3d')
halt_vels = orbits.halo_velocities(halt)
host_radii = halt['radius'][halt.prop('progenitor.main.indices', halt['host.index'][0])]
halt_dists_norm = orbits.halo_distances_norm(halt_dists, host_radii)
infall_info = orbits.infall_times(halt_dists_norm, snaps)
infall_info_any = orbits.first_infall_any(halt, snaps)
peris = orbits.pericenter_interp(distances=halt_dists, velocities=halt_vels, virial_radii=host_radii, time_array=snaps, infall_array=infall_info)
apos = orbits.apocenter_interp(distances=halt_dists, velocities=halt_vels, time_array=snaps, infall_array=infall_info)
angs = orbits.angular_momentum(tree=halt)
#
times = snaps['time'][-1] - np.flip(snaps['time'])

index = 100
print(peris['pericenter.dist'][index])
print(peris['pericenter.time.lb'][index])
min_index = 5

# Recent pericenter
d_rec = peris['pericenter.dist'][index][0]
t_rec = peris['pericenter.time.lb'][index][0]
L_rec = angs['ang.mom.total'][index][np.where(np.min(np.abs(times - t_rec)) == np.abs(times - t_rec))[0][0]]
print('L at recent pericenter is {0}'.format(L_rec))
#
# Minimum pericenter
d_min = peris['pericenter.dist'][index][min_index]
t_min = peris['pericenter.time.lb'][index][min_index]
L_min = angs['ang.mom.total'][index][np.where(np.min(np.abs(times - t_min)) == np.abs(times - t_min))[0][0]]
print('L at minimum pericenter is {0}'.format(L_min))
#
# 200 Myr before minimim
t_200_before = peris['pericenter.time.lb'][index][min_index]+0.2
L_200_before = angs['ang.mom.total'][index][np.where(np.min(np.abs(times - t_200_before)) == np.abs(times - t_200_before))[0][0]]
mass_200_before = halt['mass'][orbits.sub_inds[index][np.where(np.min(np.abs(times - t_200_before)) == np.abs(times - t_200_before))[0][0]]]
print('L 200 Myr before minimum is {0}'.format(L_200_before))
print('Halo mass 200 Myr before minimum is {0}'.format(mass_200_before))
#
# 200 Myr after minimim
t_200_after = peris['pericenter.time.lb'][index][min_index]-0.2
L_200_after = angs['ang.mom.total'][index][np.where(np.min(np.abs(times - t_200_after)) == np.abs(times - t_200_after))[0][0]]
mass_200_after = halt['mass'][orbits.sub_inds[index][np.where(np.min(np.abs(times - t_200_after)) == np.abs(times - t_200_after))[0][0]]]
print('L 200 Myr after minimum is {0}'.format(L_200_after))
print('Halo mass 200 Myr after minimum is {0}'.format(mass_200_after))
print('time difference between pericenters is {0}'.format(t_min-t_rec))
