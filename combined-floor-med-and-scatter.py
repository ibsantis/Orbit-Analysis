#!/usr/bin/python3

"""
    ==============================
    = Plot the floor comparisons =
    ==============================

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


orbit_properties = ['dperi_min', 'dperi_rec', 'tperi_min', 'tperi_rec', 'vperi_min', 'vperi_rec', 'nperi', 'dapo', 'tapo', 'infall', 'ke', 'ell']
for property in orbit_properties:
    file_name_median = f'{property}_median'
    file_name_width = f'{property}_width'
    data_median = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers/{file_name_median}.csv")
    data_width = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers/{file_name_width}.csv")
    #
    print(f'starting work for property: {property}')
    for galaxy in mw_sats_1Mpc:
        galaxy_name = galaxy.replace(' ', '_')

        # Select data for a single satellite by name
        satellite_data_med = data_median[data_median['Satellite'] == galaxy]
        satellite_data_width = data_width[data_width['Satellite'] == galaxy]

        # Transpose the DataFrame to make columns into rows
        # Drop the "Satellite" column to only include numerical data
        satellite_data_med_transposed = satellite_data_med.drop(columns=['Satellite']).transpose()
        satellite_data_width_transposed = satellite_data_width.drop(columns=['Satellite']).transpose()

        # Reset the index to use column names as x-axis labels
        satellite_data_med_transposed.reset_index(inplace=True)
        satellite_data_med_transposed.columns = ['Selection', 'Value']
        satellite_data_width_transposed.reset_index(inplace=True)
        satellite_data_width_transposed.columns = ['Selection', 'Value']

        # Create a scatter plot
        xtickArray = np.arange(len(satellite_data_width_transposed))
        mask_med = (satellite_data_med_transposed['Value'] != -1)
        mask_width = (satellite_data_width_transposed['Value'] != -1)
        f, axs = plt.subplots(2, 1, figsize=(10,8))
        #
        if len(satellite_data_med_transposed['Value'][mask_med]) == 0:
            continue
        lim_med = 0.1*np.nanmax(satellite_data_med_transposed['Value'][mask_med])
        lim_width = 0.1*np.nanmax(satellite_data_width_transposed['Value'][mask_width])
        #
        axs[0].scatter(xtickArray[mask_med], satellite_data_med_transposed['Value'][mask_med], color='#FF6F61', s=50)
        axs[1].scatter(xtickArray[mask_width], satellite_data_width_transposed['Value'][mask_width], color='#6A5ACD', s=50)
        #axs[0].set_ylim([np.nanmin(satellite_data_med_transposed['Value'][mask_med]) - lim_med, np.nanmax(satellite_data_med_transposed['Value'][mask_med]) + lim_med])
        #axs[1].set_ylim([np.nanmin(satellite_data_width_transposed['Value'][mask_width]) - lim_width, np.nanmax(satellite_data_width_transposed['Value'][mask_width]) + lim_width])
        xtickNames = [str(satellite_data_med_transposed['Selection'][i]) for i in range(len(satellite_data_med_transposed))]
        axs[0].set_xticks(xtickArray)
        axs[1].set_xticks(xtickArray)
        axs[0].set_xticklabels(np.asarray(xtickNames), rotation=90)
        axs[1].set_xticklabels(np.asarray(xtickNames), rotation=90)
        axs[1].set_xlabel('Selection')
        axs[0].set_ylabel('Median')
        axs[1].set_ylabel('Width of 68%')
        plt.suptitle(f'{galaxy} - {property}')
        axs[0].tick_params(axis='both', which='major', labelbottom=False, labelsize=18)
        axs[1].tick_params(axis='both', which='major', labelsize=18)
        axs[0].tick_params(axis='x', which='minor', size=0)
        axs[1].tick_params(axis='x', which='minor', size=0)
        plt.tight_layout()
        #plt.show()
        plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors/{galaxy_name}/{galaxy_name}_{property}_floors.pdf')
        plt.close()
    #
    print(f'Finished property: {property}')


# Count the number of analogs for the floor selections
data = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_analogs.csv")
for galaxy in mw_sats_1Mpc:
    galaxy_name = galaxy.replace(' ', '_')

    # Select data for a single satellite by name
    satellite_data = data[data['Satellite'] == galaxy]

    # Transpose the DataFrame to make columns into rows
    # Drop the "Satellite" column to only include numerical data
    satellite_data_transposed = satellite_data.drop(columns=['Satellite']).transpose()

    # Reset the index to use column names as x-axis labels
    satellite_data_transposed.reset_index(inplace=True)
    satellite_data_transposed.columns = ['Selection', 'Value']

    # Create a scatter plot
    xtickArray = np.arange(len(satellite_data_transposed))
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    #
    axs.scatter(xtickArray, satellite_data_transposed['Value'], color='#FF6F61', s=50)
    xtickNames = [str(satellite_data_transposed['Selection'][i]) for i in range(len(satellite_data_transposed))]
    axs.set_xticks(xtickArray)
    axs.set_xticklabels(np.asarray(xtickNames), rotation=90)
    axs.set_xlabel('Selection')
    axs.set_ylabel('Number of analogs')
    plt.suptitle(f'{galaxy}')
    axs.tick_params(axis='both', which='major', labelsize=18)
    axs.tick_params(axis='x', which='minor', size=0)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors/{galaxy_name}/{galaxy_name}_number_floors.pdf')
    plt.close()
#
