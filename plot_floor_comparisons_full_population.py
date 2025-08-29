#!/usr/bin/python3

"""
    ======================================================
    = Plot the floor comparisons for the full population =
    ======================================================

    Read in the median and width files, and then create a scatter
    plot that shows how each of the selections affect the results
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

row_labels = ['infall', 'infall.68', 
              'n.peri', 'n.peri.68', 
              't.peri.rec', 't.peri.rec.68', 
              'd.peri.rec', 'd.peri.rec.68', 
              'v.peri.rec', 'v.peri.rec.68', 
              't.peri.min', 't.peri.min.68', 
              'd.peri.min', 'd.peri.min.68', 
              'v.peri.min', 'v.peri.min.68', 
              't.apo', 't.apo.68', 
              'd.apo', 'd.apo.68', 
              'ke', 'ke.68', 'ell', 'ell.68']

column_labels = ['Fiducial',
              '1, 1, 1',
              '1, 3, 3',
              '1, 3, 5',
              '1, 5, 5',
              '1, 7, 7',
              '1, 10, 10',
              '3, 1, 1',
              '3, 3, 3',
              '3, 3, 5',
              '3, 5, 5',
              '3, 7, 7',
              '3, 10, 10',
              '5, 1, 1',
              '5, 3, 3',
              '5, 3, 5',
              '5, 5, 5',
              '5, 7, 7',
              '5, 10, 10',
              '7, 1, 1',
              '7, 3, 3',
              '7, 3, 5',
              '7, 5, 5',
              '7, 7, 7',
              '7, 10, 10',
              '10, 1, 1',
              '10, 3, 3',
              '10, 3, 5',
              '10, 5, 5',
              '10, 7, 7',
              '10, 10, 10']

data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/mw_population_floor_test_medians.csv', index_col=0)
data_counts = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/mw_population_floor_test_counts.csv', index_col=0)

properties = ['infall', 
              'n.peri', 
              't.peri.rec',
              'd.peri.rec', 
              'v.peri.rec', 
              't.peri.min',
              'd.peri.min',
              'v.peri.min',
              't.apo',
              'd.apo', 
              'ke', 'ell']

property_titles =     {'infall':'Infall lookback time [Gyr]',
                       'n.peri':'Pericenter number',
                       't.peri.rec':'Recent pericenter lookback time [Gyr]',
                       'd.peri.rec':'Recent pericenter distance [kpc]',
                       'v.peri.rec':'Recent pericenter velocity [km/s]',
                       't.peri.min':'Minimum pericenter lookback time [Gyr]',
                       'd.peri.min':'Minimum pericenter distance [kpc]',
                       'v.peri.min':'Minimum pericenter velocity [km/s]',
                       't.apo':'Recent apocenter lookback time [Gyr]',
                       'd.apo':'Recent apocenter distance [kpc]',
                       'ke':'Specific kinetic energy [km$^2$/s$^2$]',
                       'ell':'Specific angular momentum [kpc km/s]'}

for prop in properties:
    med_array = data.transpose()[prop]
    width_array = data.transpose()[prop+'.68']
    count_array = data_counts.transpose()[prop]
    #
    # Create a scatter plot
    xtickArray = np.arange(len(med_array))
    f, axs = plt.subplots(3, 1, figsize=(12,16))
    #
    lim_med = 0.1*np.nanmax(med_array)
    lim_width = 0.1*np.nanmax(width_array)
    #
    axs[0].scatter(xtickArray, count_array, color='k', s=50)
    axs[1].scatter(xtickArray, med_array, color='#FF6F61', s=50)
    axs[2].scatter(xtickArray, width_array, color='#6A5ACD', s=50)
    xtickNames = column_labels
    #
    axs[0].set_xticks(xtickArray)
    axs[1].set_xticks(xtickArray)
    axs[2].set_xticks(xtickArray)
    #
    axs[0].set_xticklabels(np.asarray(xtickNames), rotation=90)
    axs[1].set_xticklabels(np.asarray(xtickNames), rotation=90)
    axs[2].set_xticklabels(np.asarray(xtickNames), rotation=90)
    axs[2].set_xlabel('Selection')
    axs[0].set_ylabel('Num analogs')
    axs[1].set_ylabel('Median')
    axs[2].set_ylabel('Width of 68%')
    plt.suptitle(f'{property_titles[prop]}')
    axs[0].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
    axs[1].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
    axs[2].tick_params(axis='both', which='major', labelsize=18)
    axs[0].tick_params(axis='x', which='minor', size=0)
    axs[1].tick_params(axis='x', which='minor', size=0)
    axs[2].tick_params(axis='x', which='minor', size=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/{prop}_floors.pdf')
    plt.close()
    #
    print(f'Finished property: {prop}')





for prop in properties:
    med_array = data.transpose()[prop]
    width_array = data.transpose()[prop+'.68']
    count_array = data_counts.transpose()[prop]
    mask = np.argsort(count_array)
    #
    # Create a scatter plot
    xtickArray = np.arange(len(med_array))
    f, axs = plt.subplots(3, 1, figsize=(12,16))
    #
    lim_med = 0.1*np.nanmax(med_array)
    lim_width = 0.1*np.nanmax(width_array)
    #
    axs[0].scatter(xtickArray, count_array[mask], color='k', s=50)
    axs[1].scatter(xtickArray, med_array[mask], color='#FF6F61', s=50)
    axs[2].scatter(xtickArray, width_array[mask], color='#6A5ACD', s=50)
    xtickNames = column_labels
    #
    axs[0].set_xticks(xtickArray)
    axs[1].set_xticks(xtickArray)
    axs[2].set_xticks(xtickArray)
    #
    axs[0].set_xticklabels(np.asarray(xtickNames)[mask], rotation=90)
    axs[1].set_xticklabels(np.asarray(xtickNames)[mask], rotation=90)
    axs[2].set_xticklabels(np.asarray(xtickNames)[mask], rotation=90)
    axs[2].set_xlabel('Selection')
    axs[0].set_ylabel('Num analogs')
    axs[1].set_ylabel('Median')
    axs[2].set_ylabel('Width of 68%')
    plt.suptitle(f'{property_titles[prop]}')
    axs[0].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
    axs[1].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
    axs[2].tick_params(axis='both', which='major', labelsize=18)
    axs[0].tick_params(axis='x', which='minor', size=0)
    axs[1].tick_params(axis='x', which='minor', size=0)
    axs[2].tick_params(axis='x', which='minor', size=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/{prop}_floors_sorted.pdf')
    plt.close()
    #
    print(f'Finished property: {prop}')




"""
    Try calculating the median across all selection values, and then sort the data by
    how close they are to the median-median?
"""
meds_tot = []
mask_tot = []
meds_all = []
for prop in properties:
    med_array = data.transpose()[prop]
    width_array = data.transpose()[prop+'.68']
    count_array = data_counts.transpose()[prop]
    meds_all.append(med_array)
    medmed = np.median(med_array)
    meds_tot.append(medmed)
    mask = np.argsort(np.abs(med_array.values - medmed))
    mask_tot.append(mask)
    #
    # Create a scatter plot
    xtickArray = np.arange(len(med_array))
    xminn = xtickArray[0]-0.5
    xmaxx = xtickArray[-1]+0.5
    f, axs = plt.subplots(3, 1, figsize=(12,16))
    #
    lim_count_max = (np.nanmax(count_array) + 0.05*np.nanmax(count_array))
    lim_count_min = (np.nanmin(count_array) - 0.5*np.nanmin(count_array))
    lim_med_max = (np.nanmax(med_array) + 0.05*np.nanmax(med_array))
    lim_med_min = (np.nanmin(med_array) - 0.05*np.nanmin(med_array))
    lim_width_max = (np.nanmax(width_array) + 0.05*np.nanmax(width_array))
    lim_width_min = (np.nanmin(width_array) - 0.05*np.nanmin(width_array))
    #
    axs[0].scatter(xtickArray, count_array[mask], color='k', s=50)
    axs[1].scatter(xtickArray, med_array[mask], color='#FF6F61', s=50)
    axs[2].scatter(xtickArray, width_array[mask], color='#6A5ACD', s=50)
    xtickNames = column_labels
    axs[1].axhline(medmed, xminn, xmaxx, color='k', alpha = 0.25)

    #
    axs[0].set_xticks(xtickArray)
    axs[1].set_xticks(xtickArray)
    axs[2].set_xticks(xtickArray)
    #
    # put some vlines in some noteable selections
    xfid = np.where('Fiducial' == np.asarray(xtickNames)[mask])[0][0]
    x_10_10_10 = np.where('10, 10, 10' == np.asarray(xtickNames)[mask])[0][0]
    x_10_5_5 = np.where('10, 5, 5' == np.asarray(xtickNames)[mask])[0][0]
    x_10_7_7 = np.where('10, 7, 7' == np.asarray(xtickNames)[mask])[0][0]
    x_5_5_5 = np.where('5, 5, 5' == np.asarray(xtickNames)[mask])[0][0]
    x_5_7_7 = np.where('5, 7, 7' == np.asarray(xtickNames)[mask])[0][0]
    axs[0].axvline(xfid, color='k', alpha=0.25)
    axs[0].axvline(x_10_10_10, color='k', alpha=0.25)
    axs[0].axvline(x_10_5_5, color='k', alpha=0.25)
    axs[0].axvline(x_10_7_7, color='k', alpha=0.25)
    axs[0].axvline(x_5_5_5, color='k', alpha=0.25)
    axs[0].axvline(x_5_7_7, color='k', alpha=0.25)
    axs[1].axvline(xfid, color='k', alpha=0.25)
    axs[1].axvline(x_10_10_10, color='k', alpha=0.25)
    axs[1].axvline(x_10_5_5, color='k', alpha=0.25)
    axs[1].axvline(x_10_7_7, color='k', alpha=0.25)
    axs[1].axvline(x_5_5_5, color='k', alpha=0.25)
    axs[1].axvline(x_5_7_7, color='k', alpha=0.25)
    axs[2].axvline(xfid, color='k', alpha=0.25)
    axs[2].axvline(x_10_10_10, color='k', alpha=0.25)
    axs[2].axvline(x_10_5_5, color='k', alpha=0.25)
    axs[2].axvline(x_10_7_7, color='k', alpha=0.25)
    axs[2].axvline(x_5_5_5, color='k', alpha=0.25)
    axs[2].axvline(x_5_7_7, color='k', alpha=0.25)
    #
    axs[0].set_xticklabels(np.asarray(xtickNames)[mask], rotation=90)
    axs[1].set_xticklabels(np.asarray(xtickNames)[mask], rotation=90)
    axs[2].set_xticklabels(np.asarray(xtickNames)[mask], rotation=90)
    axs[2].set_xlabel('Selection')
    axs[0].set_ylabel('Num analogs')
    axs[1].set_ylabel('Median')
    axs[2].set_ylabel('Width of 68%')
    axs[0].set_xlim(xminn, xmaxx)
    axs[1].set_xlim(xminn, xmaxx)
    axs[2].set_xlim(xminn, xmaxx)
    plt.suptitle(f'{property_titles[prop]}')
    axs[0].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
    axs[1].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
    axs[2].tick_params(axis='both', which='major', labelsize=18)
    axs[0].tick_params(axis='x', which='minor', size=0)
    axs[1].tick_params(axis='x', which='minor', size=0)
    axs[2].tick_params(axis='x', which='minor', size=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/{prop}_floors_best_median.pdf')
    plt.close()
    #
    print(f'Finished property: {prop}')







score = []
best = []
beta = 0.5
for prop in properties:
    med_array = data.transpose()[prop]
    width_array = data.transpose()[prop+'.68']
    count_array = data_counts.transpose()[prop]
    #
    #N_weight = np.log(count_array + 1)/np.log(np.max(count_array) + 1)
    #N_weight = count_array/np.max(count_array)
    #N_weight = (count_array/np.max(count_array))**beta
    #N_weight = count_array/np.median(count_array)
    #N_weight = count_array / (count_array + np.median(count_array))
    cap = np.percentile(count_array, 75)
    N_weight = np.minimum(count_array, cap)/cap
    W_weight = np.max(width_array)/width_array
    M_weight = np.exp((-1)*np.abs(med_array - np.median(med_array))/np.median(width_array))
    #
    score.append(N_weight * W_weight * M_weight)
    best.append((N_weight * W_weight * M_weight).idxmax())
print(best)
