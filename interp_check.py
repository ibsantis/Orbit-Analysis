


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
sim_data = orbit_io.OrbitRead(gal1='m12b', location='peloton')
print('Set paths')

# Read in the snapshot dictionary and the entire tree
snaps = ut.simulation.read_snapshot_times(directory=sim_data.simulation_dir) # Saves snapshots, redshifts, lookback times, etc. to an array
halt = halo.io.IO.read_tree(simulation_directory=sim_data.simulation_dir, file_kind='hdf5', species='star', host_number=sim_data.num_gal)

if sim_data.num_gal == 1:
    # This initializes the classes and makes sure they inherit from the OrbitRead class
    orbits = orbit_io.OrbitAnalysis(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1, dmo=False)
    orbit_gal = orbit_io.OrbitGalpy(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1, dmo=False)
    orbit_plot = orbit_io.OrbitPlot(tree=halt, gal1=sim_data.galaxy, location='peloton', host=1, dmo=False)
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


# Calculate some of the pericenters
#
directory = '/home/ibsantis/scripts/orbit_data/plots/peri_interp_checks'
#sub = 35
for i in range(0, len(np.where(peris['pericenter.num'] > 1)[0])):
    sub = np.where(peris['pericenter.num'] > 1)[0][i]
    distances = halt_dists[sub]
    virial_radii = host_radii
    time_array = snaps
    reach = 20
    #
    temp_halo_d = distances # Now goes from z = 0 to z_form (un-normalized)
    peri_rad_list = []
    # Want initial element to be this because we check neighbors on each side
    temp_peri = temp_halo_d[reach]
    temp_check = np.zeros(len(temp_halo_d))
    temp_peri_spl = []
    temp_time_spl = []
    #
    # Loop through each subhalo
    start_ind = 4
    for i in range(start_ind, len(temp_halo_d)-reach):
        # These if-else statements allow for a "sliding", non-symmetric window to look for pericenters
        if (i-reach < 0):
            left_ind = 0
        else:
            left_ind = i-reach
        #
        # Check its neighbors and if it is within virial radius
        if (all(temp_peri < temp_halo_d[left_ind:i])) and (all(temp_peri < temp_halo_d[i+1:i+1+reach])) and (temp_peri/virial_radii[i] < 1):
            temp_check[i] = 1
            peri_rad_list.append(virial_radii[i])
            temp_peri_spl.append(temp_halo_d[left_ind:i+reach])
            temp_time_spl.append(np.flip(time_array['time'])[left_ind:i+reach])
            temp_peri = temp_halo_d[i+1]
        else:
            temp_peri = temp_halo_d[i+1]
    #
    temp_peri_new_spl = []
    temp_peri_vel_new_spl = []
    temp_time_new_spl = []
    # Loop over the number of pericenter events
    for j in range(0, len(temp_peri_spl)):
        temp_dist = temp_peri_spl[j]
        temp_time = temp_time_spl[j]
        # Work on distance
        f = interp1d(temp_time, temp_dist, kind='cubic')
        x_new = np.linspace(temp_time[0], temp_time[-1], 100)
        temp_peri_new_spl.append(np.min(f(x_new)))
        temp_time_new_spl.append(x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
        #
        fig, ax1 = plt.subplots(1, 1, figsize=(8,6))
        #
        # PLOTTING
        ax1.scatter(temp_time_spl[j], temp_peri_spl[j], s=30, color='b')
        ax1.plot(x_new, f(x_new), color='k')
        #
        ax1.set_xlabel('time [Gyr]', fontsize=22)
        ax1.set_ylabel('distance [kpc]', fontsize=22)
        #ax1.set_xlim(0.5, 2.1)
        #ax1.set_ylim(0, 1.75)
        plt.tight_layout()
        #plt.show()
        plt.savefig(directory+'/'+orbits.galaxy+'_sub_'+str(sub+1)+'_peri_'+str(j+1)+'.pdf')
        plt.close()
