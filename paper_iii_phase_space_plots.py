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

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12j', 'm12n']

mw_sats = ['HIZSS 3(A)', 'HIZSS 3B', 'NGC 55', 'LMC', 'SMC', 'IC 4662', 'IC 5152', 'NGC 6822', 'NGC 3109', 'IC 3104', \
           'Sextans B', 'DDO 190', 'DDO 125', 'Sextans A', 'NGC 4163', 'Sagittarius dSph', 'UGC 8508', 'Fornax', 'UGC 4879', \
           'UGC 9128', 'GR 8', 'Leo A', 'Leo 1', 'Sagittarius dIrr', 'ESO 294-G010', 'DDO 113', 'Sculptor', 'Antlia 2', 'Aquarius (DDO 210)',\
           'Phoenix', 'Leo 2', 'Antlia B', 'Tucana', 'KKR 3', 'Carina', 'Leo P', 'Crater 2', 'Ursa Minor', 'Sextans 1', \
           'Draco', 'Canes Venatici 1', 'Leo T', 'Eridanus 2', 'Bootes 1', 'Hercules', 'Bootes 3', 'Sagittarius 2', \
           'Canes Venatici 2', 'Ursa Major 1', 'Leo 4', 'Hydra 2', 'Hydrus 1', 'Carina 2', 'Ursa Major 2', 'Aquarius 2', \
           'Indus 2', 'Coma Berenices', 'Leo 5', 'Pisces 2', 'Columba 1', 'Tucana 5', 'Pegasus 3', 'Grus 2', 'Tucana 2', \
           'Reticulum 2', 'Horologium 1', 'Pictor 1', 'Tucana 4', 'Indus 1', 'Grus 1', 'Reticulum 3', 'Pictor 2', 'Bootes 2',\
           'Willman 1', 'Phoenix 2', 'Cetus 3', 'Carina 3', 'Eridanus 3', 'Segue 2', 'Triangulum 2', 'Horologium 2', 'Tucana 3',\
           'Segue 1', 'DES J0225+0304', 'Virgo 1', 'Draco 2', 'Cetus 2']


#galaxy = 'Sculptor'
for galaxy in mw_sats:
    #
    gal_data = sat_analysis.read_subhalo_matches(galaxy)
    satellite_name = galaxy.replace(' ', '_')
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
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/phase_space/'+satellite_name+'_phase_space_properties.pdf')
    plt.close()


galaxy = 'Carina'
gal_data = sat_analysis.read_subhalo_matches(galaxy)
satellite_name = galaxy.replace(' ', '_')
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
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/phase_space_example.pdf')
plt.close()
