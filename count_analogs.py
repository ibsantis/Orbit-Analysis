#!/usr/bin/python3

"""
    ==========================
    = Paper III Analog count =
    ==========================

    Count the number of analogs for each of the satellites in our sample.

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
import matplotlib.patches as mpatches
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

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12n']

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

# Loop through all of the satellites and create a figure for each of them
num_analogs = np.zeros(len(mw_sats_1Mpc))
#
for i in range(len(mw_sats_1Mpc)):
    count = 0
    satellite_name = mw_sats_1Mpc[i].replace(' ', '_')
    file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(mw_sats_1Mpc[i], file_path_read)
    mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+'m12i'+'_all_subhalos', verbose=True)
    sat_match = satellite_io.SatelliteMatch(tree=None, mini=mini_data, gal1='m12i', location=loc)
    match = sat_match.lg_satellite_properties(lg_data=lg_data, galaxy_name=mw_sats_1Mpc[i], mass_err=0.35)
    #
    for name in galaxies:
        mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+name+'_all_subhalos', verbose=True)
        snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+name)
        #
        # get the matches for a simulation host
        mask = (gal_data['Host'] == name)
        # get an array of the tree indices that are matches in m12m
        subhalo_inds = np.asarray(gal_data['Halo tree index'][mask])
        # Get indices within the mini data to select properties
        mini_data_match_inds = np.asarray([np.where(i == mini_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
        #
        count += len(mini_data_match_inds)
    num_analogs[i] += count
