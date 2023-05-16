import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import ListedColormap
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
#
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False
#
data_total_rot = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', rotated=True)
masks_infall_rot = summary.data_mask(data_total_rot, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_rot = summary.data_mask(data_total_rot, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_rot = summary.data_mask_apo(data_total_rot, hosts='all_no_r')
masks_infall_rot['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_rot['m12f'][59] = False
masks_infall_apo_rot['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2/rotated_disk'


"""
    Comparing pericenter distances
"""
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_rot = summary.dperi_recent(data_total_rot, masks_infall_peri_rot, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(d_rec_mod_rot-d_rec_mod)/d_rec_mod, xtype='d.peri.text', binsize=0.01, xlimits=(-0.1,0.1), pdf=True, x_labels='$(d_{\\rm peri,rec,rot}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$', file_path_and_name=directory+'/d_peri_rec_hist.pdf')
summary_plot.median_plot(x=d_rec_mod, y=(d_rec_mod_rot-d_rec_mod), xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-2,5)), hl=True, axis_labels=['$d_{\\rm peri,rec,disk}$ [kpc]', '$d_{\\rm peri,rec,rot}-d_{\\rm peri,rec,disk}$ [kpc]'], file_path_and_name=directory+'/delta_d_peri_rec_vs_rec.pdf')
summary_plot.median_plot(x=d_rec_mod, y=(d_rec_mod_rot-d_rec_mod)/d_rec_mod, xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-0.01,0.03)), hl=True, axis_labels=['$d_{\\rm peri,rec,disk}$ [kpc]', '$(d_{\\rm peri,rec,rot}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$'], file_path_and_name=directory+'/d_peri_rec_frac_vs_rec.pdf')


d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod_rot = summary.dperi_min(data_total_rot, masks_infall_peri_rot, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(d_min_mod_rot-d_min_mod)/d_min_mod, xtype='d.peri.text', binsize=0.01, xlimits=(-0.1,0.1), pdf=True, x_labels='$(d_{\\rm peri,min,rot}-d_{\\rm peri,min,disk})/d_{\\rm peri,min,disk}$', file_path_and_name=directory+'/d_peri_min_hist.pdf')
summary_plot.median_plot(x=d_min_mod, y=(d_min_mod_rot-d_min_mod), xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-2,5)), hl=True, axis_labels=['$d_{\\rm peri,min,disk}$ [kpc]', '$d_{\\rm peri,min,rot}-d_{\\rm peri,min,disk}$ [kpc]'], file_path_and_name=directory+'/delta_d_peri_min_vs_min.pdf')
summary_plot.median_plot(x=d_min_mod, y=(d_min_mod_rot-d_min_mod)/d_min_mod, xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-0.01,0.03)), hl=True, axis_labels=['$d_{\\rm peri,min,disk}$ [kpc]', '$(d_{\\rm peri,min,rot}-d_{\\rm peri,min,disk})/d_{\\rm peri,min,disk}$'], file_path_and_name=directory+'/d_peri_min_frac_vs_min.pdf')