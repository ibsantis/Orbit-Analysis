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
#
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False
#
data_total_kep = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', point_mass=True)
masks_infall_kep = summary.data_mask(data_total_kep, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_kep = summary.data_mask(data_total_kep, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_kep = summary.data_mask_apo(data_total_kep, hosts='all_no_r')
masks_infall_kep['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_kep['m12f'][59] = False
masks_infall_apo_kep['m12f'][59] = False
#
data_total_90p = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', point_mass=True, percent=90)
masks_infall_90p = summary.data_mask(data_total_90p, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_90p = summary.data_mask(data_total_90p, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_90p = summary.data_mask_apo(data_total_90p, hosts='all_no_r')
masks_infall_90p['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_90p['m12f'][59] = False
masks_infall_apo_90p['m12f'][59] = False
#
data_total_50p = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', point_mass=True, percent=50)
masks_infall_50p = summary.data_mask(data_total_50p, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_50p = summary.data_mask(data_total_50p, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_50p = summary.data_mask_apo(data_total_50p, hosts='all_no_r')
masks_infall_50p['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_50p['m12f'][59] = False
masks_infall_apo_50p['m12f'][59] = False
#
data_total_10p = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', point_mass=True, percent=10)
masks_infall_10p = summary.data_mask(data_total_10p, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_10p = summary.data_mask(data_total_10p, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_10p = summary.data_mask_apo(data_total_10p, hosts='all_no_r')
masks_infall_10p['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_10p['m12f'][59] = False
masks_infall_apo_10p['m12f'][59] = False
#
data_total_1p = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', point_mass=True, percent=1)
masks_infall_1p = summary.data_mask(data_total_1p, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_1p = summary.data_mask(data_total_1p, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_1p = summary.data_mask_apo(data_total_1p, hosts='all_no_r')
masks_infall_1p['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_1p['m12f'][59] = False
masks_infall_apo_1p['m12f'][59] = False


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2/point_mass_checks'


def width_of_68(x_array):
    onesigp = 84.13
    onesigm = 15.87
    #
    upper = np.nanpercentile(x_array, onesigp)
    lower = np.nanpercentile(x_array, onesigm)
    #
    return upper-lower


# Recent pericenter distance
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_kep = summary.dperi_recent(data_total_kep, masks_infall_peri_kep, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_90p = summary.dperi_recent(data_total_90p, masks_infall_peri_90p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_50p = summary.dperi_recent(data_total_50p, masks_infall_peri_50p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_10p = summary.dperi_recent(data_total_10p, masks_infall_peri_10p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_1p = summary.dperi_recent(data_total_1p, masks_infall_peri_1p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')

mask = (d_rec_mod != 0)
print(np.nanmedian((d_rec_mod_kep[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(np.nanmedian((d_rec_mod_90p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(np.nanmedian((d_rec_mod_50p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(np.nanmedian((d_rec_mod_10p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(np.nanmedian((d_rec_mod_1p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_kep[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_90p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_50p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_10p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_1p[mask]-d_rec_mod[mask])/d_rec_mod[mask]))


# Recent pericenter time
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_kep = summary.tperi_recent(data_total_kep, masks_infall_peri_kep, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_90p = summary.tperi_recent(data_total_90p, masks_infall_peri_90p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_50p = summary.tperi_recent(data_total_50p, masks_infall_peri_50p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_10p = summary.tperi_recent(data_total_10p, masks_infall_peri_10p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_1p = summary.tperi_recent(data_total_1p, masks_infall_peri_1p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')

mask = (d_rec_mod != 0)*(t_rec_mod != 0)
print(np.nanmedian((t_rec_mod_kep[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(np.nanmedian((t_rec_mod_90p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(np.nanmedian((t_rec_mod_50p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(np.nanmedian((t_rec_mod_10p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(np.nanmedian((t_rec_mod_1p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_kep[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_90p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_50p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_10p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_1p[mask]-t_rec_mod[mask])/t_rec_mod[mask]))


# Pericenter number
n_mod = summary.nperi_model(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_kep = summary.nperi_model(data_total_kep, masks_infall_peri_kep, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_90p = summary.nperi_model(data_total_90p, masks_infall_peri_90p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_50p = summary.nperi_model(data_total_50p, masks_infall_peri_50p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_10p = summary.nperi_model(data_total_10p, masks_infall_peri_10p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_1p = summary.nperi_model(data_total_1p, masks_infall_peri_1p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')

mask = (d_rec_mod != 0)*(n_mod != 0)
print(np.nanmedian((n_mod_kep[mask]-n_mod[mask])/n_mod[mask]))
print(np.nanmedian((n_mod_90p[mask]-n_mod[mask])/n_mod[mask]))
print(np.nanmedian((n_mod_50p[mask]-n_mod[mask])/n_mod[mask]))
print(np.nanmedian((n_mod_10p[mask]-n_mod[mask])/n_mod[mask]))
print(np.nanmedian((n_mod_1p[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_kep[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_90p[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_50p[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_10p[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_1p[mask]-n_mod[mask])/n_mod[mask]))


n_mod_r200 = summary.nperi_model(data_total, masks_infall_peri, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_kep = summary.nperi_model(data_total_kep, masks_infall_peri_kep, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_90p = summary.nperi_model(data_total_90p, masks_infall_peri_90p, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_50p = summary.nperi_model(data_total_50p, masks_infall_peri_50p, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_10p = summary.nperi_model(data_total_10p, masks_infall_peri_10p, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_1p = summary.nperi_model(data_total_1p, masks_infall_peri_1p, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')

mask = (d_rec_mod != 0)*(n_mod_r200 != 0)
print(np.nanmedian((n_mod_r200_kep[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(np.nanmedian((n_mod_r200_90p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(np.nanmedian((n_mod_r200_50p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(np.nanmedian((n_mod_r200_10p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(np.nanmedian((n_mod_r200_1p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_kep[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_90p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_50p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_10p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_1p[mask]-n_mod_r200[mask])/n_mod_r200[mask]))


# Recent pericenter velocity
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_kep = summary.vperi_recent(data_total_kep, masks_infall_peri_kep, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_90p = summary.vperi_recent(data_total_90p, masks_infall_peri_90p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_50p = summary.vperi_recent(data_total_50p, masks_infall_peri_50p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_10p = summary.vperi_recent(data_total_10p, masks_infall_peri_10p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_1p = summary.vperi_recent(data_total_1p, masks_infall_peri_1p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')

mask = (d_rec_mod != 0)
print(np.nanmedian((v_rec_mod_kep[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(np.nanmedian((v_rec_mod_90p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(np.nanmedian((v_rec_mod_50p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(np.nanmedian((v_rec_mod_10p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(np.nanmedian((v_rec_mod_1p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_kep[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_90p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_50p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_10p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_1p[mask]-v_rec_mod[mask])/v_rec_mod[mask]))


# Revent apocenter distance
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_kep = summary.dapo_recent(data_total_kep, masks_infall_apo_kep, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_90p = summary.dapo_recent(data_total_90p, masks_infall_apo_90p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_50p = summary.dapo_recent(data_total_50p, masks_infall_apo_50p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_10p = summary.dapo_recent(data_total_10p, masks_infall_apo_10p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_1p = summary.dapo_recent(data_total_1p, masks_infall_apo_1p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')

mask = (dapo_rec_mod != 0)
print(np.nanmedian((dapo_rec_mod_kep[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(np.nanmedian((dapo_rec_mod_90p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(np.nanmedian((dapo_rec_mod_50p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(np.nanmedian((dapo_rec_mod_10p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(np.nanmedian((dapo_rec_mod_1p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_kep[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_90p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_50p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_10p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_1p[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))


# Infall time with evolving R200m
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_kep = summary.first_infall(data_total_kep, masks_infall_kep, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_90p = summary.first_infall(data_total_90p, masks_infall_90p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_50p = summary.first_infall(data_total_50p, masks_infall_50p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_10p = summary.first_infall(data_total_10p, masks_infall_10p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_1p = summary.first_infall(data_total_1p, masks_infall_1p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')

d_rec_mod = summary.dperi_recent(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (t_in_mod != 0)*(d_rec_mod != 0)
print(np.median((t_in_mod_kep[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(np.median((t_in_mod_90p[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(np.median((t_in_mod_50p[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(np.median((t_in_mod_10p[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(np.median((t_in_mod_1p[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_kep[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_90p[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_50p[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_10p[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_1p[mask]-t_in_mod[mask])/t_in_mod[mask]))


# Infall time with fixed R200m
t_in_mod_r200 = summary.infall_fixed(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_r200_kep = summary.infall_fixed(data_total_kep, masks_infall_kep, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_r200_90p = summary.infall_fixed(data_total_90p, masks_infall_90p, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_r200_50p = summary.infall_fixed(data_total_50p, masks_infall_50p, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_r200_10p = summary.infall_fixed(data_total_10p, masks_infall_10p, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_r200_1p = summary.infall_fixed(data_total_1p, masks_infall_1p, oversample=True, hosts='all_no_r', sim_type='baryon')

d_rec_mod = summary.dperi_recent(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (t_in_mod_r200 != 0)*(d_rec_mod != 0)
print(np.nanmedian((t_in_mod_r200_kep[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(np.nanmedian((t_in_mod_r200_90p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(np.nanmedian((t_in_mod_r200_50p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(np.nanmedian((t_in_mod_r200_10p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(np.nanmedian((t_in_mod_r200_1p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(width_of_68((t_in_mod_r200_kep[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(width_of_68((t_in_mod_r200_90p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(width_of_68((t_in_mod_r200_50p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(width_of_68((t_in_mod_r200_10p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))
print(width_of_68((t_in_mod_r200_1p[mask]-t_in_mod_r200[mask])/t_in_mod_r200[mask]))


# Eccentricity
ecc_model1 = []
ecc_model2 = []
ecc_model3 = []
ecc_model4 = []
ecc_model5 = []
d_rec_mod1 = []
d_rec_mod2 = []
d_rec_mod3 = []
d_rec_mod4 = []
d_rec_mod5 = []
ecc_model_kep = []
ecc_model_90p = []
ecc_model_50p = []
ecc_model_10p = []
ecc_model_1p = []

for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['infall.check'])):
        if data_total[name]['infall.check'][i]:
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_kep[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model1.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_kep.append(np.repeat(data_total_kep[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod1.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_90p[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model2.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_90p.append(np.repeat(data_total_90p[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod2.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_50p[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model3.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_50p.append(np.repeat(data_total_50p[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod3.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_10p[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model4.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_10p.append(np.repeat(data_total_10p[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod4.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_1p[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model5.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_1p.append(np.repeat(data_total_1p[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod5.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
ecc_model1 = np.hstack(ecc_model1)
ecc_model2 = np.hstack(ecc_model2)
ecc_model3 = np.hstack(ecc_model3)
ecc_model4 = np.hstack(ecc_model4)
ecc_model5 = np.hstack(ecc_model5)
ecc_model_kep = np.hstack(ecc_model_kep)
ecc_model_90p = np.hstack(ecc_model_90p)
ecc_model_50p = np.hstack(ecc_model_50p)
ecc_model_10p = np.hstack(ecc_model_10p)
ecc_model_1p = np.hstack(ecc_model_1p)
d_rec_mod1 = np.hstack(d_rec_mod1)
d_rec_mod2 = np.hstack(d_rec_mod2)
d_rec_mod3 = np.hstack(d_rec_mod3)
d_rec_mod4 = np.hstack(d_rec_mod4)
d_rec_mod5 = np.hstack(d_rec_mod5)

mask1 = (d_rec_mod1 != 0)#*(d_rec_mod1 < 50)
mask2 = (d_rec_mod2 != 0)#*(d_rec_mod2 < 50)
mask3 = (d_rec_mod3 != 0)#*(d_rec_mod3 < 50)
mask4 = (d_rec_mod4 != 0)
mask5 = (d_rec_mod5 != 0)
print(np.nanmedian((ecc_model_kep[mask1]-ecc_model1[mask1])/ecc_model1[mask1]))
print(np.nanmedian((ecc_model_90p[mask2]-ecc_model2[mask2])/ecc_model2[mask2]))
print(np.nanmedian((ecc_model_50p[mask3]-ecc_model3[mask3])/ecc_model3[mask3]))
print(np.nanmedian((ecc_model_10p[mask4]-ecc_model4[mask4])/ecc_model4[mask4]))
print(np.nanmedian((ecc_model_1p[mask5]-ecc_model5[mask5])/ecc_model5[mask5]))
print(width_of_68((ecc_model_kep[mask1]-ecc_model1[mask1])/ecc_model1[mask1]))
print(width_of_68((ecc_model_90p[mask2]-ecc_model2[mask2])/ecc_model2[mask2]))
print(width_of_68((ecc_model_50p[mask3]-ecc_model3[mask3])/ecc_model3[mask3]))
print(width_of_68((ecc_model_10p[mask4]-ecc_model4[mask4])/ecc_model4[mask4]))
print(width_of_68((ecc_model_1p[mask5]-ecc_model5[mask5])/ecc_model5[mask5]))


# Orbit Period
per_model1 = []
per_model2 = []
per_model3 = []
per_model4 = []
per_model5 = []
d_rec_mod1 = []
d_rec_mod2 = []
d_rec_mod3 = []
d_rec_mod4 = []
d_rec_mod5 = []
per_model_kep = []
per_model_90p = []
per_model_50p = []
per_model_10p = []
per_model_1p = []

for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['infall.check'])):
        if data_total[name]['infall.check'][i]:
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_kep[name]['orbit.period.peri.model'][i][0] != -1):
                per_model1.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_kep.append(np.repeat(data_total_kep[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod1.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_90p[name]['orbit.period.peri.model'][i][0] != -1):
                per_model2.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_90p.append(np.repeat(data_total_90p[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod2.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_50p[name]['orbit.period.peri.model'][i][0] != -1):
                per_model3.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_50p.append(np.repeat(data_total_50p[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod3.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_10p[name]['orbit.period.peri.model'][i][0] != -1):
                per_model4.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_10p.append(np.repeat(data_total_10p[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod4.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_1p[name]['orbit.period.peri.model'][i][0] != -1):
                per_model5.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_1p.append(np.repeat(data_total_1p[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod5.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
per_model1 = np.hstack(per_model1)
per_model2 = np.hstack(per_model2)
per_model3 = np.hstack(per_model3)
per_model4 = np.hstack(per_model4)
per_model5 = np.hstack(per_model5)
per_model_kep = np.hstack(per_model_kep)
per_model_90p = np.hstack(per_model_90p)
per_model_50p = np.hstack(per_model_50p)
per_model_10p = np.hstack(per_model_10p)
per_model_1p = np.hstack(per_model_1p)
d_rec_mod1 = np.hstack(d_rec_mod1)
d_rec_mod2 = np.hstack(d_rec_mod2)
d_rec_mod3 = np.hstack(d_rec_mod3)
d_rec_mod4 = np.hstack(d_rec_mod4)
d_rec_mod5 = np.hstack(d_rec_mod5)

mask1 = (d_rec_mod1 != 0)#*(d_rec_mod1 < 50)
mask2 = (d_rec_mod2 != 0)#*(d_rec_mod2 < 50)
mask3 = (d_rec_mod3 != 0)#*(d_rec_mod3 < 50)
mask4 = (d_rec_mod4 != 0)
mask5 = (d_rec_mod5 != 0)
print(np.nanmedian((per_model_kep[mask1]-per_model1[mask1])/per_model1[mask1]))
print(np.nanmedian((per_model_90p[mask2]-per_model2[mask2])/per_model2[mask2]))
print(np.nanmedian((per_model_50p[mask3]-per_model3[mask3])/per_model3[mask3]))
print(np.nanmedian((per_model_10p[mask4]-per_model4[mask4])/per_model4[mask4]))
print(np.nanmedian((per_model_1p[mask5]-per_model5[mask5])/per_model5[mask5]))
print(width_of_68((per_model_kep[mask1]-per_model1[mask1])/per_model1[mask1]))
print(width_of_68((per_model_90p[mask2]-per_model2[mask2])/per_model2[mask2]))
print(width_of_68((per_model_50p[mask3]-per_model3[mask3])/per_model3[mask3]))
print(width_of_68((per_model_10p[mask4]-per_model4[mask4])/per_model4[mask4]))
print(width_of_68((per_model_1p[mask5]-per_model5[mask5])/per_model5[mask5]))




# da/dr
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_no_r', new=True)
data_dadr = summary.data_read_dadr(mass_profile=data_mp, hosts='all_no_r')
#
dadr_max_mod = summary.da_dr(data_total, masks_infall_peri, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('Regular model done')
dadr_max_mod_kep = summary.da_dr(data_total_kep, masks_infall_peri_kep, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('Kepler model done')
dadr_max_mod_90p = summary.da_dr(data_total_90p, masks_infall_peri_90p, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('90% model done')
dadr_max_mod_50p = summary.da_dr(data_total_50p, masks_infall_peri_50p, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('50% model done')
dadr_max_mod_10p = summary.da_dr(data_total_10p, masks_infall_peri_10p, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('10% model done')
dadr_max_mod_1p = summary.da_dr(data_total_1p, masks_infall_peri_1p, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('1% model done')

d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (d_rec_mod != 0)#*(d_rec_mod < 50)

print(np.nanmedian((dadr_max_mod_kep['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(np.nanmedian((dadr_max_mod_90p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(np.nanmedian((dadr_max_mod_50p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(np.nanmedian((dadr_max_mod_10p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(np.nanmedian((dadr_max_mod_1p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_kep['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_90p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_50p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_10p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_1p['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))






"""
    Comparing pericenter distances
"""
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_kep = summary.dperi_recent(data_total_kep, masks_infall_peri_kep, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_90p = summary.dperi_recent(data_total_90p, masks_infall_peri_90p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_50p = summary.dperi_recent(data_total_50p, masks_infall_peri_50p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_10p = summary.dperi_recent(data_total_10p, masks_infall_peri_10p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_1p = summary.dperi_recent(data_total_1p, masks_infall_peri_1p, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[(d_rec_mod_kep-d_rec_mod)/d_rec_mod, (d_rec_mod_90p-d_rec_mod)/d_rec_mod, (d_rec_mod_50p-d_rec_mod)/d_rec_mod, (d_rec_mod_10p-d_rec_mod)/d_rec_mod, (d_rec_mod_1p-d_rec_mod)/d_rec_mod], xtype=['d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text'], labels=['Keplerian', '90% hr, hz', '50% hr, hz', '10% hr, hz', '1% hr, hz'], binsize=0.005, xlimits=(-0.05, 0.05), pdf=True, x_label='$(d_{\\rm peri,rec,point}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$', file_path_and_name=directory+'/d_peri_rec_hist.pdf')

summary_plot.plot_hist_mult(x=[(d_rec_mod_kep-d_rec_mod)/d_rec_mod, (d_rec_mod_1p-d_rec_mod)/d_rec_mod], xtype=['d.peri.text','d.peri.text'], labels=['Keplerian', 'Disk (1% $h_r, h_z$)'], leg_loc='upper left', binsize=0.005, xlimits=(-0.04, 0.04), pdf=True, x_label='$(d_{\\rm peri,rec,point}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$', file_path_and_name=directory+'/d_peri_rec_hist_lite.pdf')

