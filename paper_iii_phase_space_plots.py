#!/usr/bin/python3

"""
    =========================================
    = Paper III Orbit History Distributions =
    =========================================

    Create the multi-panel orbit history PDFs for each satellite.

    This will create a figure that shows the differential and cumulative PDFs of:
        - Infall time
        - Apocenter time and distance (recent)
        - Pericenter time, distance, and velocity (recent and minimum)
        - Disance, radial velocity, and tangential velocity
"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import pandas as pd
import satellite_io
import matplotlib
from matplotlib import pyplot as plt
import time
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
sat_analysis = satellite_io.SatelliteAnalysis(gal1='m12i', location=loc)
#
print('Set paths')

# Read in the snapshot dictionary and the entire tree
lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)

#galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']
galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12n', 'm12q', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']

mw_sats_1Mpc =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
                    'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
                    'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
                    'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus III', 'Eridanus IV', \
                    'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
                    'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
                    'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
                    'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
                    'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sagittarius II', 'Sculptor', 'Segue 1', \
                    'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
                    'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
                    'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']

n_555 = [12, 330, 378, 83, 131, 3, 2, 305, 42, 213, 15, 13, 75, 89, 125, 5, 243, 153, 43, 13, 26, 57, 4, 125, 4, 99, 99, 111, 163, 301, 212, 49, 52, 0, 25, 167, 255, 347, 18, 67, 517, 362, 178, 13, 387, 127, 77, 363, 104, 954, 2, 107, 4, 103, 20, 12, 42, 17, 15, 151, 0, 183, 113, 193, 261, 20, 62, 87, 45, 53]

#galaxy = 'Sculptor'
for sat_idx, galaxy in enumerate(mw_sats_1Mpc):
    #
    satellite_name = galaxy.replace(' ', '_')
    if n_555[sat_idx] < 10:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_10_10_10/weights_{satellite_name}.txt'
    else:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
    #
    if len(gal_data['Host']) == 0:
        continue
    #
    orbit_dictionary = dict()
    orbit_dictionary['first.infall.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.num'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.vel'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.vel'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['apocenter.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['apocenter.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['halo.mass.peak'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['distance'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.rad'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.tan'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['L.tot.sim'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['v.tot.sim'] = np.zeros(gal_data.shape[0])
    #
    for sim_name in galaxies:
        if sim_name in np.array(gal_data['Host']):
            # Read in the mini data and snapshot information
            mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=True)
            snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)
            #
            orbit_history = sat_analysis.orbit_property_distribution(sim_name, mini_data, gal_data, snaps)
            mask = np.where(sim_name == gal_data['Host'])[0]
            for key in orbit_history.keys():
                orbit_dictionary[key][mask] = orbit_history[key]
    if len(orbit_history['distance']) == 0:
        continue
    #    
    #Plot the phase-space information
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(3, 3, figsize=(12,12))
    #
    axs[0,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
    axs[1,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['velocity.rad'], s=15, alpha=0.8, c='k')
    axs[2,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['distance'], s=15, alpha=0.8, c='k')
    #
    axs[0,1].scatter(orbit_dictionary['distance'], orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
    axs[1,1].scatter(orbit_dictionary['distance'], orbit_dictionary['velocity.rad'], s=15, alpha=0.8, c='k')
    #
    axs[0,2].scatter(orbit_dictionary['velocity.rad'], orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
    #
    axs[0,0].set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=17)
    axs[1,0].set_ylabel('Radial velocity [km s$^{-1}$]', fontsize=17)
    axs[2,0].set_ylabel('Distance [kpc]', fontsize=17)
    axs[2,0].set_xlabel('log $M_{\\rm halo, peak}$', fontsize=17)
    axs[1,1].set_xlabel('Distance [kpc]', fontsize=17)
    axs[0,2].set_xlabel('Radial velocity [km s$^{-1}$]', fontsize=17)
    #
    plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=20)
    axs[0,0].tick_params(axis='both', which='major', labelbottom=False, labelsize=14)
    axs[1,0].tick_params(axis='both', which='major', labelbottom=False, labelsize=14)
    axs[2,0].tick_params(axis='both', which='major', labelsize=14)
    axs[0,1].tick_params(axis='both', which='major', labelbottom=False, labelleft=False, labelsize=14)
    axs[1,1].tick_params(axis='both', which='major', labelsize=14, labelleft=False)
    axs[0,2].tick_params(axis='both', which='major', labelsize=14, labelleft=False)
    #
    axs[1,2].axison = False
    axs[2,1].axison = False
    axs[2,2].axison = False
    #
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/phase_space/'+satellite_name+'_phase_space_properties.pdf')
    plt.close()


galaxy = 'Canes Venatici II'
satellite_name = galaxy.replace(' ', '_')
#file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{satellite_name}.txt'
file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=galaxy, mass_err=0.35)
#
orbit_dictionary = dict()
orbit_dictionary['first.infall.time.lb'] = np.zeros(gal_data.shape[0])
orbit_dictionary['pericenter.num'] = np.zeros(gal_data.shape[0])
orbit_dictionary['pericenter.rec.time.lb'] = np.zeros(gal_data.shape[0])
orbit_dictionary['pericenter.rec.dist'] = np.zeros(gal_data.shape[0])
orbit_dictionary['pericenter.rec.vel'] = np.zeros(gal_data.shape[0])
orbit_dictionary['pericenter.min.time.lb'] = np.zeros(gal_data.shape[0])
orbit_dictionary['pericenter.min.dist'] = np.zeros(gal_data.shape[0])
orbit_dictionary['pericenter.min.vel'] = np.zeros(gal_data.shape[0])
orbit_dictionary['apocenter.time.lb'] = np.zeros(gal_data.shape[0])
orbit_dictionary['apocenter.dist'] = np.zeros(gal_data.shape[0])
orbit_dictionary['halo.mass.peak'] = np.zeros(gal_data.shape[0])
orbit_dictionary['distance'] = np.zeros(gal_data.shape[0])
orbit_dictionary['velocity.rad'] = np.zeros(gal_data.shape[0])
orbit_dictionary['velocity.tan'] = np.zeros(gal_data.shape[0])
orbit_dictionary['L.tot.sim'] = np.zeros(gal_data.shape[0])
orbit_dictionary['v.tot.sim'] = np.zeros(gal_data.shape[0])
#
for sim_name in galaxies:
    if sim_name in np.array(gal_data['Host']):
        # Read in the mini data and snapshot information
        mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=False)
        snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)
        #
        orbit_history = sat_analysis.orbit_property_distribution(sim_name, mini_data, gal_data, snaps)
        mask = np.where(sim_name == gal_data['Host'])[0]
        for key in orbit_history.keys():
            orbit_dictionary[key][mask] = orbit_history[key]
#
if match['host.distance.total.err'] < 5.0:
    match['host.distance.total.err'] = 5.0
if match['host.velocity.rad.err'] < 5.0:
    match['host.velocity.rad.err'] = 5.0
if match['host.velocity.tan.err'] < 5.0:
    match['host.velocity.tan.err'] = 5.0
#
#Plot the phase-space information
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(3, 3, figsize=(12,12))
#
axs[0,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
axs[1,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['velocity.rad'], s=15, alpha=0.8, c='k')
axs[2,0].scatter(np.log10(orbit_dictionary['halo.mass.peak']), orbit_dictionary['distance'], s=15, alpha=0.8, c='k')
#
axs[0,1].scatter(orbit_dictionary['distance'], orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
axs[1,1].scatter(orbit_dictionary['distance'], orbit_dictionary['velocity.rad'], s=15, alpha=0.8, c='k')
#
axs[0,2].scatter(orbit_dictionary['velocity.rad'], orbit_dictionary['velocity.tan'], s=15, alpha=0.8, c='k')
#
axs[0,0].axvspan(np.log10(match['mass.peak'])-match['mass.peak.err'], np.log10(match['mass.peak'])+match['mass.peak.err'], alpha=0.1, color='b', zorder=0)
axs[1,0].axvspan(np.log10(match['mass.peak'])-match['mass.peak.err'], np.log10(match['mass.peak'])+match['mass.peak.err'], alpha=0.1, color='b', zorder=0)
axs[2,0].axvspan(np.log10(match['mass.peak'])-match['mass.peak.err'], np.log10(match['mass.peak'])+match['mass.peak.err'], alpha=0.1, color='b', zorder=0)
axs[0,1].axvspan(match['host.distance.total']-match['host.distance.total.err'], match['host.distance.total']+match['host.distance.total.err'], alpha=0.1, color='b', zorder=0)
axs[1,1].axvspan(match['host.distance.total']-match['host.distance.total.err'], match['host.distance.total']+match['host.distance.total.err'], alpha=0.1, color='b', zorder=0)
axs[0,2].axvspan(match['host.velocity.rad']-match['host.velocity.rad.err'], match['host.velocity.rad']+match['host.velocity.rad.err'], alpha=0.1, color='b', zorder=0)
#
axs[0,0].axhspan(match['host.velocity.tan']-match['host.velocity.tan.err'], match['host.velocity.tan']+match['host.velocity.tan.err'], alpha=0.1, color='b', zorder=0)
axs[0,1].axhspan(match['host.velocity.tan']-match['host.velocity.tan.err'], match['host.velocity.tan']+match['host.velocity.tan.err'], alpha=0.1, color='b', zorder=0)
axs[0,2].axhspan(match['host.velocity.tan']-match['host.velocity.tan.err'], match['host.velocity.tan']+match['host.velocity.tan.err'], alpha=0.1, color='b', zorder=0)
axs[1,0].axhspan(match['host.velocity.rad']-match['host.velocity.rad.err'], match['host.velocity.rad']+match['host.velocity.rad.err'], alpha=0.1, color='b', zorder=0)
axs[1,1].axhspan(match['host.velocity.rad']-match['host.velocity.rad.err'], match['host.velocity.rad']+match['host.velocity.rad.err'], alpha=0.1, color='b', zorder=0)
axs[2,0].axhspan(match['host.distance.total']-match['host.distance.total.err'], match['host.distance.total']+match['host.distance.total.err'], alpha=0.1, color='b', zorder=0)
#
axs[0,0].set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=18)
axs[1,0].set_ylabel('Radial velocity [km s$^{-1}$]', fontsize=18)
axs[2,0].set_ylabel('Distance [kpc]', fontsize=20)
axs[2,0].set_xlabel('log $M_{\\rm halo, peak}$', fontsize=24)
axs[1,1].set_xlabel('Distance [kpc]', fontsize=24)
axs[0,2].set_xlabel('Radial velocity [km s$^{-1}$]', fontsize=20)
#
#plt.suptitle('{0} - Number of analogs = {1}'.format(galaxy, len(gal_data['Weight'])), fontsize=20)
axs[0,0].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
axs[1,0].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
axs[2,0].tick_params(axis='both', which='major', labelsize=18)
axs[0,1].tick_params(axis='both', which='major', labelbottom=False, labelleft=False, labelsize=18)
axs[1,1].tick_params(axis='both', which='major', labelsize=18, labelleft=False)
axs[0,2].tick_params(axis='both', which='major', labelsize=18, labelleft=False)
#
axs[1,2].axison = False
axs[2,1].axison = False
axs[2,2].axison = False
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/case_study_phase_space.pdf')
plt.close()
