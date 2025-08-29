#!/usr/bin/python3

"""
    ===================================
    = Median/Scatter bias from floors =
    ===================================

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
import matplotlib.cm as cm
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

cols = ['Satellite', '1/1/1', '1/3/3', '1/3/5', '1/5/5', '1/7/7', '1/10/10',
       '3/1/1', '3/3/3', '3/3/5', '3/5/5', '3/7/7', '3/10/10', '5/1/1',
       '5/3/3', '5/3/5', '5/5/5', '5/7/7', '5/10/10', '7/1/1', '7/3/3',
       '7/3/5', '7/5/5', '7/7/7', '7/10/10', '10/1/1', '10/3/3', '10/3/5',
       '10/5/5', '10/7/7', '10/10/10']

orbit_properties = ['dperi_min', 'dperi_rec', 'tperi_min', 'tperi_rec', 'vperi_min', 'vperi_rec', 'nperi', 'dapo', 'tapo', 'infall', 'ke', 'ell']

finalists = ['5/5/5', '10/5/5', '10/7/7', '10/10/10']

results_dir = sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/with-5-5-5'

property = 'infall'
median_data = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{property}_median.csv", index_col=0, usecols=cols).replace(-1, np.nan)
median_data_finalists = median_data[finalists]

# """
#     Working case for one satellite, Antlia II
# """
# # Working on Antlia II right now
# #comparison = median_data['3/3/3'][0] # Comparing against a particular selection criteria
# infall_antlia = median_data.iloc[0]
# median_all = np.median(infall_antlia)

# # Compute the raw difference between the choices and the comparison array (median over all)
# median_diff = np.abs(infall_antlia - median_all)

# # Compute the mean absolute deviation
# #MAD = np.sum(np.abs(infall_antlia - median_all))/len(infall_antlia)

# # Compute the median absolute deviation
# MAD = np.median(median_diff)


"""
    Now scale up the previous code for all satellites for the infall time comparisons
"""
results = np.zeros((len(cols)+1, len(mw_sats_1Mpc))) # I really want len(cols)-1, but then also add the row for the MAD value, and another for the total median row
results_diff = np.zeros((len(cols)+1, len(mw_sats_1Mpc)))

# Loop over all satellites
for i in range(len(mw_sats_1Mpc)):
    #
    # Get the list of infall times for the current satellite
    infall_satellite = median_data.iloc[i]
    #
    # Calculate the median over all selections
    median_all_selection = np.nanmedian(infall_satellite)
    #
    # Calculate the median difference between each selection and the overall median
    median_diff = np.abs(infall_satellite - median_all_selection)
    #
    # Calculate the median absolute deviation
    MAD = np.nanmedian(median_diff)
    #
    # First save the data from the raw differences
    results_diff[0,i] = MAD
    results_diff[1, i] = median_all_selection
    results_diff[2:, i] = median_diff
    #
    if MAD == 0:
        #
        nonzero = median_diff[median_diff > 0]
        nonzeroMAD = np.nanmedian(nonzero)
        #
        denominator = np.where(median_diff > 0, median_diff, nonzeroMAD)
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/denominator
    else:
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/MAD

row_labels = [
    "MAD", "ALL",
    "1/1/1", "1/3/3", "1/3/5", "1/5/5", "1/7/7", "1/10/10",
    "3/1/1", "3/3/3", "3/3/5", "3/5/5", "3/7/7", "3/10/10",
    "5/1/1", "5/3/3", "5/3/5", "5/5/5", "5/7/7", "5/10/10",
    "7/1/1", "7/3/3", "7/3/5", "7/5/5", "7/7/7", "7/10/10",
    "10/1/1", "10/3/3", "10/3/5", "10/5/5", "10/7/7", "10/10/10"
]

# Save the data for the MAD table first
df = pd.DataFrame(results, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = "mad_infall_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)

# Now save the data for the raw differences
df = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = "raw_infall_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)




# Trying to make sense of all of the data
df_full = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df = df_full.loc[finalists]
df_other = df_full.drop(index=finalists)
#
tolerance = 0.25
verdict = (pd.DataFrame({
        'Δ_5/5/5'   : df.loc['5/5/5'],
        'Δ_10/5/5'   : df.loc['10/5/5'],
        'Δ_10/7/7'   : df.loc['10/7/7'],
        'Δ_10/10/10' : df.loc['10/10/10'],}).assign(
        n_fail = lambda d: (d.abs() > tolerance).sum(axis=1),
        pass_  = lambda d: d['n_fail'] == 0))

full_spread = np.nanmax(np.abs(df_other), axis=0)
verdict['max_Δ_other_selection'] = full_spread

verdict.to_csv(results_dir+"/per_satellite_infall_verdict.csv")




##################
"""
    Ran the following code for:
    - tperi,rec
    - tperi,min
    - tapo
        - ALL THREE OF THESE HAD A TOLERANCE OF 0.25 (~1 disk rotation)
    - dperi,rec
    - dperi,min
    - dapo
        - TOLERANCE OF 30 kpc (~10% Rvir)
    - vperi,rec
    - vperi,min
        - TOLERANCE OF 22 km/s (~10% v_circ)
    - Nperi
        - TOLERANCE OF 1 pericenter
"""
##################


property = 'nperi'
tolerance = 1
median_data = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{property}_median.csv", index_col=0, usecols=cols).replace(-1, np.nan)
median_data_finalists = median_data[finalists]

"""
    Check all satellites for the recent pericenter
"""
results = np.zeros((len(cols)+1, len(mw_sats_1Mpc))) # I really want len(cols)-1, but then also add the row for the MAD value
results_diff = np.zeros((len(cols)+1, len(mw_sats_1Mpc)))

# Loop over all satellites
for i in range(len(mw_sats_1Mpc)):
    #
    # Get the list of orbit properties for the current satellite
    satellite_prop = median_data.iloc[i]
    #
    # Calculate the median over all selections
    median_all_selection = np.nanmedian(satellite_prop)
    #
    # Calculate the median difference between each selection and the overall median
    median_diff = np.abs(satellite_prop - median_all_selection)
    #
    # Calculate the median absolute deviation
    MAD = np.nanmedian(median_diff)
    #
    # First save the data from the raw differences
    results_diff[0,i] = MAD
    results_diff[1, i] = median_all_selection
    results_diff[2:, i] = median_diff
    #
    if MAD == 0:
        #
        nonzero = median_diff[median_diff > 0]
        nonzeroMAD = np.nanmedian(nonzero)
        #
        denominator = np.where(median_diff > 0, median_diff, nonzeroMAD)
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/denominator
    else:
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/MAD

row_labels = [
    "MAD", "ALL",
    "1/1/1", "1/3/3", "1/3/5", "1/5/5", "1/7/7", "1/10/10",
    "3/1/1", "3/3/3", "3/3/5", "3/5/5", "3/7/7", "3/10/10",
    "5/1/1", "5/3/3", "5/3/5", "5/5/5", "5/7/7", "5/10/10",
    "7/1/1", "7/3/3", "7/3/5", "7/5/5", "7/7/7", "7/10/10",
    "10/1/1", "10/3/3", "10/3/5", "10/5/5", "10/7/7", "10/10/10"
]

# Save the data for the MAD table first
df = pd.DataFrame(results, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = f"mad_{property}_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)

# Now save the data for the raw differences
df = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = f"raw_{property}_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)




# Trying to make sense of all of the data
df_full = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df = df_full.loc[finalists]
df_other = df_full.drop(index=finalists)
#
verdict = (pd.DataFrame({
        'Δ_5/5/5'   : df.loc['5/5/5'],
        'Δ_10/5/5'   : df.loc['10/5/5'],
        'Δ_10/7/7'   : df.loc['10/7/7'],
        'Δ_10/10/10' : df.loc['10/10/10'],}).assign(
        n_fail = lambda d: (d.abs() > tolerance).sum(axis=1),
        pass_  = lambda d: d['n_fail'] == 0))

full_spread = np.nanmax(np.abs(df_other), axis=0)
verdict['max_Δ_other_selection'] = full_spread

verdict.to_csv(results_dir+f"/per_satellite_{property}_verdict.csv")








"""
    Now do the comparison for the 68th percentile widths
"""
property = 'infall'
scatter_data = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{property}_width.csv", index_col=0, usecols=cols).replace(-1, np.nan)
scatter_data_finalists = scatter_data[finalists]

"""
    Now scale up the previous code for all satellites for the infall time comparisons
"""
results = np.zeros((len(cols)+1, len(mw_sats_1Mpc))) # I really want len(cols)-1, but then also add the row for the MAD value
results_diff = np.zeros((len(cols)+1, len(mw_sats_1Mpc)))

# Loop over all satellites
for i in range(len(mw_sats_1Mpc)):
    #
    # Get the list of infall times for the current satellite
    infall_satellite = scatter_data.iloc[i]
    #
    # Calculate the median over all selections
    median_all_selection = np.nanmedian(infall_satellite)
    #
    # Calculate the median difference between each selection and the overall median
    median_diff = np.abs(infall_satellite - median_all_selection)
    #
    # Calculate the median absolute deviation
    MAD = np.nanmedian(median_diff)
    #
    # First save the data from the raw differences
    results_diff[0,i] = MAD
    results_diff[1, i] = median_all_selection
    results_diff[2:, i] = median_diff
    #
    if MAD == 0:
        #
        nonzero = median_diff[median_diff > 0]
        nonzeroMAD = np.nanmedian(nonzero)
        #
        denominator = np.where(median_diff > 0, median_diff, nonzeroMAD)
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/denominator
    else:
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/MAD

row_labels = [
    "MAD", "ALL",
    "1/1/1", "1/3/3", "1/3/5", "1/5/5", "1/7/7", "1/10/10",
    "3/1/1", "3/3/3", "3/3/5", "3/5/5", "3/7/7", "3/10/10",
    "5/1/1", "5/3/3", "5/3/5", "5/5/5", "5/7/7", "5/10/10",
    "7/1/1", "7/3/3", "7/3/5", "7/5/5", "7/7/7", "7/10/10",
    "10/1/1", "10/3/3", "10/3/5", "10/5/5", "10/7/7", "10/10/10"
]

# Save the data for the MAD table first
df = pd.DataFrame(results, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = "mad_infall_width_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)

# Now save the data for the raw differences
df = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = "raw_infall_width_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)




# Trying to make sense of all of the data
df_full = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df = df_full.loc[finalists]
df_other = df_full.drop(index=finalists)
#
tolerance = 0.25
verdict = (pd.DataFrame({
        'Δ_5/5/5'   : df.loc['5/5/5'],
        'Δ_10/5/5'   : df.loc['10/5/5'],
        'Δ_10/7/7'   : df.loc['10/7/7'],
        'Δ_10/10/10' : df.loc['10/10/10'],}).assign(
        n_fail = lambda d: (d.abs() > tolerance).sum(axis=1),
        pass_  = lambda d: d['n_fail'] == 0))

full_spread = np.nanmax(np.abs(df_other), axis=0)
verdict['max_Δ_other_selection'] = full_spread

verdict.to_csv(results_dir+"/per_satellite_infall_width_verdict.csv")





##################
"""
    Ran the following code for:
    - tperi,rec
    - tperi,min
    - tapo
        - ALL THREE OF THESE HAD A TOLERANCE OF 0.25 (~1 disk rotation)
    - dperi,rec
    - dperi,min
    - dapo
        - TOLERANCE OF 30 kpc (~10% Rvir)
    - vperi,rec
    - vperi,min
        - TOLERANCE OF 22 km/s (~10% v_circ)
    - Nperi
        - TOLERANCE OF 1 pericenter
"""
##################

property = 'nperi'
tolerance = 1
scatter_data = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{property}_width.csv", index_col=0, usecols=cols).replace(-1, np.nan)
scatter_data_finalists = scatter_data[finalists]

"""
    Check all satellites 
"""
results = np.zeros((len(cols)+1, len(mw_sats_1Mpc))) # I really want len(cols)-1, but then also add the row for the MAD value
results_diff = np.zeros((len(cols)+1, len(mw_sats_1Mpc)))

# Loop over all satellites
for i in range(len(mw_sats_1Mpc)):
    #
    # Get the list of orbit properties for the current satellite
    satellite_prop = scatter_data.iloc[i]
    #
    # Calculate the median over all selections
    median_all_selection = np.nanmedian(satellite_prop)
    #
    # Calculate the median difference between each selection and the overall median
    median_diff = np.abs(satellite_prop - median_all_selection)
    #
    # Calculate the median absolute deviation
    MAD = np.nanmedian(median_diff)
    #
    # First save the data from the raw differences
    results_diff[0,i] = MAD
    results_diff[1, i] = median_all_selection
    results_diff[2:, i] = median_diff
    #
    if MAD == 0:
        #
        nonzero = median_diff[median_diff > 0]
        nonzeroMAD = np.nanmedian(nonzero)
        #
        denominator = np.where(median_diff > 0, median_diff, nonzeroMAD)
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/denominator
    else:
        results[0, i] = MAD
        results[1, i] = median_all_selection
        results[2:, i] = median_diff/MAD

row_labels = [
    "MAD", "ALL",
    "1/1/1", "1/3/3", "1/3/5", "1/5/5", "1/7/7", "1/10/10",
    "3/1/1", "3/3/3", "3/3/5", "3/5/5", "3/7/7", "3/10/10",
    "5/1/1", "5/3/3", "5/3/5", "5/5/5", "5/7/7", "5/10/10",
    "7/1/1", "7/3/3", "7/3/5", "7/5/5", "7/7/7", "7/10/10",
    "10/1/1", "10/3/3", "10/3/5", "10/5/5", "10/7/7", "10/10/10"
]

# Save the data for the MAD table first
df = pd.DataFrame(results, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = f"mad_{property}_width_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)

# Now save the data for the raw differences
df = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = f"raw_{property}_width_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)




# Trying to make sense of all of the data
df_full = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df = df_full.loc[finalists]
df_other = df_full.drop(index=finalists)
#
verdict = (pd.DataFrame({
        'Δ_5/5/5'   : df.loc['5/5/5'],
        'Δ_10/5/5'   : df.loc['10/5/5'],
        'Δ_10/7/7'   : df.loc['10/7/7'],
        'Δ_10/10/10' : df.loc['10/10/10'],}).assign(
        n_fail = lambda d: (d.abs() > tolerance).sum(axis=1),
        pass_  = lambda d: d['n_fail'] == 0))

full_spread = np.nanmax(np.abs(df_other), axis=0)
verdict['max_Δ_other_selection'] = full_spread

verdict.to_csv(results_dir+f"/per_satellite_{property}_width_verdict.csv")

