#!/usr/bin/python3

"""
    ======================
    = Statistics =
    ======================

    Script to plot the summary statistics, rank ordered from best to worst.
        Plotting:
            - Median offset between sim and model
            - Width of the 68% scatter
            - RMS error
"""

import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import summary_io
import orbit_io
from matplotlib import patches
from matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')


labels = ['$d_{\\rm peri,rec}$', '$d_{\\rm peri,min}$', '$t_{\\rm peri,rec}$', '$N_{\\rm peri}$', '$N_{\\rm peri,fixed}$', '$v_{\\rm peri,rec}$', '$v_{\\rm peri,min}$', '$d_{\\rm apo,rec}$', '$t_{\\rm infall,lb}$', '$t_{\\rm infall,lb,fixed}$', '$e_{\\rm rec}$', '$T_{\\rm rec}$']
med_offset = np.array([-0.025, 0.066, -0.028, -1000, -1000, 0.030, 0.012, -0.013, -0.067, 0.441, 0.005, -0.071])
xs = np.arange(1, 13)

sort_mask = np.argsort(med_offset)

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.scatter(xs, med_offset[sort_mask], s=50, marker='o')
plt.xticks(xs, np.asarray(labels)[sort_mask], rotation=45)
plt.ylabel('Median fractional offset', fontsize=34)
plt.ylim(-0.1, 0.5)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/med_offset_sorted.pdf')

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.scatter(xs, med_offset, s=50, marker='o')
plt.xticks(xs, np.asarray(labels), rotation=45)
plt.ylabel('Median fractional offset', fontsize=34)
plt.ylim(-0.1, 0.5)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/med_offset.pdf')

# Width of the 68 percent scatter
width_68 = np.array([ 0.424, 1.067, 0.179, -1000, -1000, 0.202, 0.330, 0.116, 0.820, 1.088, 0.288, 0.267])
sort_mask = np.argsort(width_68)
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.scatter(xs, width_68[sort_mask], s=50, marker='o')
plt.xticks(xs, np.asarray(labels)[sort_mask], rotation=45)
plt.ylabel('Width of 68% scatter', fontsize=34)
plt.ylim(0, 1.15)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/width_of_68_sorted.pdf')


# RMS
rms = np.array([  1.644, 2.144, 6.807, -1000, -1000, 0.193, 0.210, 0.186, 10.631, 30.342, 3.381, 0.226])
sort_mask = np.argsort(rms)
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.scatter(xs, rms[sort_mask], s=50, marker='o')
plt.xticks(xs, np.asarray(labels)[sort_mask], rotation=45)
plt.ylabel('RMS', fontsize=34)
plt.ylim(0, 31)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/rms_sorted.pdf')
#
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.scatter(xs, rms[sort_mask], s=50, marker='o')
plt.xticks(xs, np.asarray(labels)[sort_mask], rotation=45)
plt.ylabel('RMS', fontsize=34)
plt.ylim(0, 11)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/rms_sorted_zoom1.pdf')
#
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1,1,1)
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.scatter(xs, rms[sort_mask], s=50, marker='o')
plt.xticks(xs, np.asarray(labels)[sort_mask], rotation=45)
plt.ylabel('RMS', fontsize=34)
plt.ylim(0, 7)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/rms_sorted_zoom2.pdf')
