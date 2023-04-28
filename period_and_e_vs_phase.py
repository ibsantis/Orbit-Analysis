#!/usr/bin/python3

"""
    ====================================
    = Period and Eccentricity vs phase =
    ====================================

"""

import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import orbit_io
import summary_io
import model_io
from scipy import interpolate
from scipy import stats
import pandas as pd
from matplotlib import patches
from matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from astropy import units as u
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')


# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_no_r', new=True)
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2'

"""
    Period and e vs Lookback Nperi
"""
peri_d_colors = ['#337422']
peri_t_colors = ['#476258', '#624751']
#
x = []
y = []
# Loop through hosts
for name in summary.host_names['all_no_r']:
    # Loop through the subhalos
    for i in range(0, len(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]])):
        # Loop through the number of orbits that are in the sim AND model
        for j in range(0, np.min((len(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i]), len(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i])))):
            # Check to see if they both have values
            if (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i][j] != -1) & (data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i][j] != -1):
                # Save the orbit phase
                x.append(j+1)
                # Save the difference in the model and sim
                y.append(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i][j]-data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i][j])
x = np.asarray(x)
y = np.asarray(y)
#
onesigp = 84.13
onesigm = 15.87
twosigp = 100
twosigm = 0
#
meds = np.zeros(np.max(x))
upper = np.zeros(np.max(x))
lower = np.zeros(np.max(x))
highest = np.zeros(np.max(x))
lowest = np.zeros(np.max(x))
for i in range(0, np.max(x)):
    mask = (x == i+1)
    meds[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
    highest[i] = np.nanpercentile(y[mask], twosigp)
    lowest[i] = np.nanpercentile(y[mask], twosigm)
#
f, ax = plt.subplots(2, 1, figsize=(12,14))
#
ax[0].scatter(np.arange(np.max(x))+1, meds, s=75., marker='s', c=peri_t_colors[1], label='Model comparison')
ax[0].scatter(np.arange(np.max(x))+1, upper-lower, s=150., marker='*', c='k', label='Width of 68th percentile')
for j in range(0, np.max(x)):
    ax[0].errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lowest[j]],[highest[j]-meds[j]]]), alpha=0.3, color=peri_t_colors[1])
    ax[0].errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lower[j]],[upper[j]-meds[j]]]), alpha=0.7, color=peri_t_colors[1])
#
x = []
y = []
# Loop over hosts
for name in summary.host_names['all_no_r']:
    # Loop over satellites
    for i in range(0, len(data_total[name]['eccentricity.sim'][masks_infall[name]])):
        # Loop over the phase
        for j in range(0, np.min((len(data_total[name]['eccentricity.sim'][masks_infall[name]][i]), len(data_total[name]['eccentricity.model.apsis'][masks_infall[name]][i])))):
            # Make sure there is an event in both the sim and model
            if (data_total[name]['eccentricity.sim'][masks_infall[name]][i][j] != -1) & (data_total[name]['eccentricity.model.apsis'][masks_infall[name]][i][j] != -1):
                # Save the difference
                y.append(data_total[name]['eccentricity.model.apsis'][masks_infall[name]][i][j]-data_total[name]['eccentricity.sim'][masks_infall[name]][i][j])
                # Save the phase
                x.append(j+1)
x = np.asarray(x)
y = np.asarray(y)
#
# THINK MORE ABOUT WHAT TO REALLY PLOT AGAINST
onesigp = 84.13
onesigm = 15.87
twosigp = 100
twosigm = 0
meds = np.zeros(np.max(x))
upper = np.zeros(np.max(x))
lower = np.zeros(np.max(x))
highest = np.zeros(np.max(x))
lowest = np.zeros(np.max(x))
for i in range(0, np.max(x)):
    mask = (x == i+1)
    meds[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
    highest[i] = np.nanpercentile(y[mask], twosigp)
    lowest[i] = np.nanpercentile(y[mask], twosigm)
#
x_points = np.arange(1, x.max()/2+1, 0.5)
#
#
ax[1].scatter(x_points-0.5, meds, s=75., marker='s', c=peri_d_colors[0])
ax[1].scatter(x_points-0.5, upper-lower, s=150., marker='*', c='k')
for j in range(0, len(x_points)):
    ax[1].errorbar(x_points[j]-0.5, meds[j], yerr=np.array([[meds[j]-lowest[j]],[highest[j]-meds[j]]]), alpha=0.3, color=peri_d_colors[0])
    ax[1].errorbar(x_points[j]-0.5, meds[j], yerr=np.array([[meds[j]-lower[j]],[upper[j]-meds[j]]]), alpha=0.7, color=peri_d_colors[0])
#
ax[0].hlines(0, -0.5, np.max(np.arange(1, x.max()/2+1, 0.5))+1, linestyle='dotted', color='k', alpha=0.5)
ax[1].hlines(0, -0.5, np.max(np.arange(1, x.max()/2+1, 0.5))+1, linestyle='dotted', color='k', alpha=0.5)
#
ax[0].set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax[0].set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax[1].set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax[1].set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax[0].set_xlim(0,10)
ax[1].set_xlim(0,10)
ax[1].set_xlabel('Lookback $N_{\\rm peri}$', fontsize=28)
ax[0].set_ylabel('$T_{\\rm model} - T_{\\rm sim}$', fontsize=28)
ax[1].set_ylabel('$e_{\\rm model} - e_{\\rm sim}$', fontsize=28)
ax[0].get_yaxis().set_label_coords(-0.12,0.5)
ax[1].get_yaxis().set_label_coords(-0.12,0.5)
#
ax[0].tick_params(axis='both', which='both', bottom=True, top=True, labelbottom=False, labelsize=24)
ax[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#
plt.savefig(directory+'/period_vs_phase_both.pdf')
plt.close()
