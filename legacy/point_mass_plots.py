#!/usr/bin/python3

"""
    ==========================
    = Point mass comparisons =
    ==========================

    Compare some of the key properties from Paper II with data from a point-mass
    potential.

    Will probably delete this file or move it somewhere else eventually.
    Or, more realistically, we will keep some of the plots and put them in an
    appendix in the paper...

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
from orbit_analysis import orbit_io
from orbit_analysis import summary_io
from orbit_analysis import model_io
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
#
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False
#
data_total_point = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', point_mass=True)
masks_infall_point = summary.data_mask(data_total_point, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_point = summary.data_mask(data_total_point, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_point = summary.data_mask_apo(data_total_point, hosts='all_no_r')
masks_infall_point['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_point['m12f'][59] = False
masks_infall_apo_point['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2/point_mass_checks'

"""
    Comapring infall times
"""
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_pm = summary.first_infall(data_total_point, masks_infall_point, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask_finite = np.isfinite(t_in_mod)*(t_in_mod != -1)
mask_finite_pm = np.isfinite(t_in_mod_pm)*(t_in_mod_pm != -1)
#
summary_plot.plot_hist(x=t_in_mod_pm[mask_finite*mask_finite_pm] - t_in_mod[mask_finite*mask_finite_pm], xtype='t.infall.text', binsize=0.1, pdf=True, xlimits=(-1,1), x_labels='$t_{\\rm in,lb,point}-t_{\\rm in,lb,disk}$ [Gyr]', file_path_and_name=directory+'/infall_hist.pdf')
summary_plot.median_plot(x=t_in_mod[mask_finite*mask_finite_pm], y=(t_in_mod_pm[mask_finite*mask_finite_pm] - t_in_mod[mask_finite*mask_finite_pm]), xtype='t.infall.text', ytype='delta_t', binsize=0.5, limits=((13.7,0),(-1, 0.5)), hl=True, axis_labels=['$t_{\\rm infall,lb,disk}$ [Gyr]', '$t_{\\rm infall,lb,point}-t_{\\rm infall,lb,disk}$ [Gyr]'], file_path_and_name=directory+'/delta_infall_vs_infall.pdf')
#
t_in_mod_fixed = summary.infall_fixed(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_fixed_pm = summary.infall_fixed(data_total_point, masks_infall_point, oversample=True, hosts='all_no_r', sim_type='baryon')
mask_finite_fixed = np.isfinite(t_in_mod_fixed)*(t_in_mod_fixed != -1)
mask_finite_fixed_pm = np.isfinite(t_in_mod_fixed_pm)*(t_in_mod_fixed_pm != -1)
#
summary_plot.plot_hist(x=(t_in_mod_fixed_pm[mask_finite_fixed*mask_finite_fixed_pm] - t_in_mod_fixed[mask_finite_fixed*mask_finite_fixed_pm]), xtype='t.infall.text', binsize=0.05, xlimits=(-0.5,0.5), pdf=True, x_labels='$t_{\\rm in,lb,point,fixed}-t_{\\rm in,lb,disk,fixed}$ [Gyr]', file_path_and_name=directory+'/infall_fixed_hist.pdf')
summary_plot.median_plot(x=t_in_mod_fixed[mask_finite_fixed*mask_finite_fixed_pm], y=(t_in_mod_fixed_pm[mask_finite_fixed*mask_finite_fixed_pm] - t_in_mod_fixed[mask_finite_fixed*mask_finite_fixed_pm]), xtype='t.infall.text', ytype='delta_t', binsize=0.5, limits=((12,0),(-0.5, 0.5)), hl=True, axis_labels=['$t_{\\rm infall,lb,disk}$ [Gyr]', '$t_{\\rm infall,lb,point}-t_{\\rm infall,lb,disk}$ [Gyr]'], file_path_and_name=directory+'/delta_infall_vs_infall_fixed.pdf')


"""
    Comparing pericenter distances
"""
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_pm = summary.dperi_recent(data_total_point, masks_infall_peri_point, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(d_rec_mod_pm-d_rec_mod)/d_rec_mod, xtype='d.peri.text', binsize=0.01, xlimits=(-0.1,0.1), pdf=True, x_labels='$(d_{\\rm peri,rec,point}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$', file_path_and_name=directory+'/d_peri_rec_hist.pdf')
summary_plot.median_plot(x=d_rec_mod, y=(d_rec_mod_pm-d_rec_mod), xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-2,5)), hl=True, axis_labels=['$d_{\\rm peri,rec,disk}$ [kpc]', '$d_{\\rm peri,rec,point}-d_{\\rm peri,rec,disk}$ [kpc]'], file_path_and_name=directory+'/delta_d_peri_rec_vs_rec.pdf')
summary_plot.median_plot(x=d_rec_mod, y=(d_rec_mod_pm-d_rec_mod)/d_rec_mod, xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-0.01,0.03)), hl=True, axis_labels=['$d_{\\rm peri,rec,disk}$ [kpc]', '$(d_{\\rm peri,rec,point}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$'], file_path_and_name=directory+'/d_peri_rec_frac_vs_rec.pdf')


d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod_pm = summary.dperi_min(data_total_point, masks_infall_peri_point, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(d_min_mod_pm-d_min_mod)/d_min_mod, xtype='d.peri.text', binsize=0.01, xlimits=(-0.1,0.1), pdf=True, x_labels='$(d_{\\rm peri,min,point}-d_{\\rm peri,min,disk})/d_{\\rm peri,min,disk}$', file_path_and_name=directory+'/d_peri_min_hist.pdf')
summary_plot.median_plot(x=d_min_mod, y=(d_min_mod_pm-d_min_mod), xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-2,5)), hl=True, axis_labels=['$d_{\\rm peri,min,disk}$ [kpc]', '$d_{\\rm peri,min,point}-d_{\\rm peri,min,disk}$ [kpc]'], file_path_and_name=directory+'/delta_d_peri_min_vs_min.pdf')
summary_plot.median_plot(x=d_min_mod, y=(d_min_mod_pm-d_min_mod)/d_min_mod, xtype='d.peri.text', ytype='d.peri.text', binsize=25, limits=((0,300),(-0.01,0.03)), hl=True, axis_labels=['$d_{\\rm peri,min,disk}$ [kpc]', '$(d_{\\rm peri,min,point}-d_{\\rm peri,min,disk})/d_{\\rm peri,min,disk}$'], file_path_and_name=directory+'/d_peri_min_frac_vs_min.pdf')


"""
    Comparing pericenter times
"""
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_pm = summary.tperi_recent(data_total_point, masks_infall_peri_point, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = np.isfinite(t_rec_mod)*np.isfinite(t_rec_mod_pm)*(t_rec_mod != -1)*(t_rec_mod_pm != -1)
#
summary_plot.plot_hist(x=(t_rec_mod_pm[mask]-t_rec_mod[mask]), xtype='t.peri.text', binsize=0.05, xlimits=(-0.2,0.2), pdf=True, x_labels='$(t_{\\rm peri,rec,point}-t_{\\rm peri,rec,disk})$ [Gyr]', file_path_and_name=directory+'/t_peri_rec_hist.pdf')
summary_plot.median_plot(x=t_rec_mod, y=(t_rec_mod_pm-t_rec_mod), xtype='t.peri.text', ytype='delta_t', binsize=0.5, limits=((9,0),(-0.1,0.2)), hl=True, axis_labels=['$t_{\\rm peri,rec,disk}$ [kpc]', '$t_{\\rm peri,rec,point}-t_{\\rm peri,rec,disk}$ [Gyr]'], file_path_and_name=directory+'/delta_t_peri_rec_vs_rec.pdf')


"""
    Comparing pericenter velocities
"""
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_pm = summary.vperi_recent(data_total_point, masks_infall_peri_point, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(v_rec_mod_pm-v_rec_mod), xtype='v.peri.text', binsize=1, xlimits=(-20,20), pdf=True, x_labels='$(v_{\\rm peri,rec,point}-v_{\\rm peri,rec,disk})$ [km s$^{-1}$]', file_path_and_name=directory+'/v_peri_rec_hist.pdf')
summary_plot.median_plot(x=v_rec_mod, y=(v_rec_mod_pm-v_rec_mod), xtype='v.peri.text', ytype='v.peri.text', binsize=25, limits=((0,650),(-10,20)), hl=True, axis_labels=['$v_{\\rm peri,rec,disk}$ [km s$^{-1}$]', '$v_{\\rm peri,rec,point}-v_{\\rm peri,rec,disk}$ [km s$^{-1}$]'], file_path_and_name=directory+'/delta_v_peri_rec_vs_rec.pdf')


"""
    Comparing pericenter numbers
"""
n_mod = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_pm = summary.nperi_model(data_total_point, masks_infall_point, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(n_mod_pm-n_mod), xtype='N.peri.text', binsize=1, xlimits=(-3,3), pdf=True, x_labels='$(N_{\\rm peri,point}-N_{\\rm peri,disk})$', file_path_and_name=directory+'/N_peri_hist.pdf')
summary_plot.median_plot(x=n_mod, y=(n_mod_pm-n_mod), xtype='N.peri.text', ytype='N.peri.text', binsize=1, limits=((0,13),(-1,0.1)), hl=True, axis_labels=['$N_{\\rm peri,disk}$', '$N_{\\rm peri,point}-N_{\\rm peri,disk}$'], file_path_and_name=directory+'/delta_N_peri.pdf')

n_mod_fixed = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_fixed_pm = summary.nperi_model(data_total_point, masks_infall_point, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(n_mod_fixed_pm-n_mod_fixed), xtype='N.peri.text', binsize=1, xlimits=(-3,3), pdf=True, x_labels='$(N_{\\rm peri,point}-N_{\\rm peri,disk})$', file_path_and_name=directory+'/N_peri_fixed_hist.pdf')
summary_plot.median_plot(x=n_mod_fixed, y=(n_mod_fixed_pm-n_mod_fixed), xtype='N.peri.text', ytype='N.peri.text', binsize=1, limits=((0,13),(-1,0.1)), hl=True, axis_labels=['$N_{\\rm peri,disk}$', '$N_{\\rm peri,point}-N_{\\rm peri,disk}$'], file_path_and_name=directory+'/delta_N_peri_fixed.pdf')


"""
    Comparing apocenter distances
"""
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_pm = summary.dapo_recent(data_total_point, masks_infall_apo_point, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(dapo_rec_mod_pm-dapo_rec_mod)/dapo_rec_mod, xtype='d.apo.text', binsize=0.01, xlimits=(-0.1,0.1), pdf=True, x_labels='$(d_{\\rm apo,rec,point}-d_{\\rm apo,rec,disk})/d_{\\rm apo,rec,disk}$', file_path_and_name=directory+'/d_apo_rec_hist.pdf')
summary_plot.median_plot(x=dapo_rec_mod, y=(dapo_rec_mod_pm-dapo_rec_mod)/dapo_rec_mod, xtype='d.apo.text', ytype='d.apo.text', binsize=25, limits=((0,500),(-0.01,0.02)), hl=True, axis_labels=['$d_{\\rm apo,rec,disk}$ [kpc]', '$(d_{\\rm apo,rec,point}-d_{\\rm apo,rec,disk})/d_{\\rm apo,rec,disk}$'], file_path_and_name=directory+'/d_apo_rec_frac_vs_rec.pdf')
