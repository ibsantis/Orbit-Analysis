

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
data_dadr = summary.data_read_dadr(mass_profile=data_mp, hosts='all_no_r')
#
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False
#
# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2'


"""
    Calculate the maximum da/dr that a satellite will feel
"""
data = []
data_dist = []
data_time = []
#
# Loop through all of the hosts
for name in summary.host_names['all_no_r']:
    #
    # Loop through each of the satellites in a given host
    for i in range(0, len(data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]])):
        mask = (data_total[name]['d.tot.sim'][masks_infall_peri[name]][i] != -1)
        initial = (-1)*1e6
        d_initial = -1
        t_initial = -1
        #
        # Loop through the number of snapshots
        for j in range(0, len(data_total[name]['d.tot.sim'][masks_infall_peri[name]][i][mask])):
            d_ind = np.where(np.min(np.abs(data_total[name]['d.tot.sim'][masks_infall_peri[name]][i][mask][j] - data_mp['rs.interp'])) == np.abs(data_total[name]['d.tot.sim'][masks_infall_peri[name]][i][mask][j] - data_mp['rs.interp']))[0][0]
            if (np.flip(np.abs(data_dadr[name]), axis=0)[j][d_ind] > initial):
                initial = np.flip(np.abs(data_dadr[name]), axis=0)[j][d_ind]
                d_initial = data_mp['rs.interp'][d_ind]
                t_initial = np.flip(data_mp['time'], axis=0)[j]
        #
        data.append(np.repeat(initial, summary.oversample['baryon'][name]))
        data_dist.append(np.repeat(d_initial, summary.oversample['baryon'][name]))
        data_time.append(np.repeat(t_initial, summary.oversample['baryon'][name]))
#
d = dict()
d['dadr'] = np.hstack(data)
d['dadr.dist.interp'] =  np.hstack(data_dist)
d['dadr.time.interp'] =  np.hstack(data_time)
d['dadr.time.lb.interp'] =  np.hstack(data_mp['time'][-1] - data_time)


"""
    Calculate da/dr at d_peri,min
"""
data = []
data_dist = []
data_time = []
#
# Loop through all of the hosts
for name in summary.host_names['all_no_r']:
    #
    # Loop through each of the satellites in a given host
    for i in range(0, len(data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]])):
        mask = (data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]][i] != -1)
        #
        # Select the smallest pericenter
        dmin = np.min(data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]][i][mask])
        tmin = data_total[name]['pericenter.time.lb.sim'][masks_infall_peri[name]][i][mask][np.where(dmin == data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]][i][mask])[0][0]]
        #
        d_ind = np.where(np.min(np.abs(dmin - data_mp['rs.interp'])) == np.abs(dmin - data_mp['rs.interp']))[0][0]
        t_ind = np.where(np.min(np.abs(tmin - data_mp['time'])) == np.abs(tmin - data_mp['time']))[0][0]
        #
        initial = np.flip(np.abs(data_dadr[name]), axis=0)[t_ind][d_ind]
        #
        data.append(np.repeat(initial, summary.oversample['baryon'][name]))
        data_dist.append(np.repeat(d_initial, summary.oversample['baryon'][name]))
        data_time.append(np.repeat(t_initial, summary.oversample['baryon'][name]))
#
d2 = dict()
d2['dadr'] = np.hstack(data)
d2['dadr.dist.interp'] =  np.hstack(data_dist)
d2['dadr.time.interp'] =  np.hstack(data_time)
d2['dadr.time.lb.interp'] =  np.hstack(data_mp['time'][-1] - data_time)


summary_plot.plot_hist(x=(d2['dadr']-d['dadr'])/d['dadr'], xtype='dadr.frac', binsize=0.01, pdf=True, x_labels=('$(|da/dr|_{dperi,min}-|da/dr|_{max})/|da/dr|_{max}$'), file_path_and_name=directory+'/diagnostics/dadr_frac_hist.pdf')
