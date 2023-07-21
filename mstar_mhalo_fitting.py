#!/usr/bin/python3

"""
    =========================
    = Mstar - Mhalo fitting =
    =========================

    Fit the stellar mass - halo mass relation from Paper I

    Plot the fits and save the best fit... somehow...
"""

from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from scipy import special
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import orbit_io
import summary_io
import model_io
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')

# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_z', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_z')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_z')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_z')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_3'

# Get the stellar mass and halo mass of satellites
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mhalo_peak_tot = summary.mhalo(data_total, masks_infall, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon')

bin_width = 0.5
bin_edges = (8,12)
limits=((7.99,11.3),(4,10))
#
x_array_type = 'M.halo.peak'
y_array_type = 'M.star.z0'

binss, half_binss = summary_plot.binning_scheme(Mhalo_peak_tot, x_array_type, bin_width, bin_edges)
median, upper_68, lower_68, upper_95, lower_95 = summary_plot.median_and_scatter(Mhalo_peak_tot, Mstar_z0_tot, x_array_type, y_array_type, binss)

f, ax1 = plt.subplots(1, 1, figsize=(10,8))
ax1.fill_between(10**(binss[:-1]+half_binss), 10**(upper_68), 10**(lower_68), color=summary_plot.colors[1], alpha=0.3)
ax1.fill_between(10**(binss[:-1]+half_binss), 10**(upper_95), 10**(lower_95), color=summary_plot.colors[1], alpha=0.15)
ax1.plot(10**(binss[:-1]+half_binss), 10**(median), color=summary_plot.colors[1], linewidth=3.5, alpha=0.9, label='Santistevan+23')
#
ax1.set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
ax1.set_ylim(10**(limits[1][0]), 10**(limits[1][1]))
#
plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
ax1.text(5*10**9,4*10**4, '$M_{\\rm star}$-limited selection', fontsize=24)
#
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$M_{\\rm halo,peak}$ [$M_{\\odot}$]', fontsize=30)
ax1.set_ylabel('$M_{\\rm star}$ [$M_{\\odot}$]', fontsize=30)
ax1.legend(prop={'size': 21}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
plt.savefig(directory+'/SMHMR_paper_i.pdf')


# Define the stellar mass halo mass relation
@custom_model
def power_law(ms, constant=5, slope=1):
    mstar = constant + slope*ms
    return mstar
#
# Initialize the model
model_init = power_law(bounds={'constant':(-100, 100), 'slope':(-10,10)})
fit = LevMarLSQFitter()
#
# Fit the model to the data and print it out
mstar_mhalo_1 = fit(model_init, binss[:-2]+half_binss, median[:-1], maxiter=10000)
mstar_mhalo_2 = fit(model_init, binss[1:-2]+half_binss, median[1:-1], maxiter=10000) 
mstar_mhalo_3 = fit(model_init, binss[2:-2]+half_binss, median[2:-1], maxiter=10000) 
print(mstar_mhalo_1)
print(mstar_mhalo_2)
print(mstar_mhalo_3)
#
x_model = np.linspace(8, 11.25, 1000)
y_model_1 = mstar_mhalo_1(x_model)
y_model_2 = mstar_mhalo_2(x_model)
y_model_3 = mstar_mhalo_3(x_model)


f, ax1 = plt.subplots(1, 1, figsize=(10,8))
ax1.fill_between(10**(binss[:-1]+half_binss), 10**(upper_68), 10**(lower_68), color=summary_plot.colors[1], alpha=0.3)
ax1.fill_between(10**(binss[:-1]+half_binss), 10**(upper_95), 10**(lower_95), color=summary_plot.colors[1], alpha=0.15)
ax1.plot(10**(binss[:-1]+half_binss), 10**(median), color=summary_plot.colors[1], linewidth=3.5, alpha=0.9, label='Santistevan+23')
#
ax1.plot(10**(x_model), 10**(y_model_1), color='r', label='Slope = '+str(np.around(mstar_mhalo_1.slope.value, 2)), alpha=0.5, linestyle='--')
ax1.plot(10**(x_model), 10**(y_model_2), color='b', label='Slope = '+str(np.around(mstar_mhalo_2.slope.value, 2)), alpha=0.5, linestyle='--')
ax1.plot(10**(x_model), 10**(y_model_3), color='k', label='Slope = '+str(np.around(mstar_mhalo_3.slope.value, 2)), alpha=0.5, linestyle='--')
#
ax1.set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
ax1.set_ylim(10**(limits[1][0]), 10**(limits[1][1]))
#
plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
ax1.text(5*10**9,4*10**4, '$M_{\\rm star}$-limited selection', fontsize=24)
#
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$M_{\\rm halo,peak}$ [$M_{\\odot}$]', fontsize=30)
ax1.set_ylabel('$M_{\\rm star}$ [$M_{\\odot}$]', fontsize=30)
ax1.legend(prop={'size': 21}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
plt.savefig(directory+'/SMHMR_with_fit.pdf')

