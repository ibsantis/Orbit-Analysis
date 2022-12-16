#!/usr/bin/python3

"""
    ==========================
    = Paper II Final Figures =
    ==========================

    Create the final plots for Paper II
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
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2'


"""
    Figure 1: Host M200m and R200m
"""

masses = (-1)*np.ones((len(data_total), len(data_total['m12b']['time.sim'])))
radii = (-1)*np.ones((len(data_total), len(data_total['m12b']['time.sim'])))
i = 0
for name in summary.host_names['all_no_r']:
    mask = (data_total[name]['host.mass'] != -1)*np.isfinite(data_total[name]['host.mass'])
    masses[i][:np.sum(mask)] = data_total[name]['host.mass'][mask]
    radii[i][:np.sum(mask)] = data_total[name]['host.radius'][mask]
    i += 1
#
lookback = data_total['m12b']['time.sim'][-1]-np.flip(data_total['m12b']['time.sim'])
#
masses_norm = (-1)*np.ones((len(data_total), len(data_total['m12b']['time.sim'])))
radii_norm = (-1)*np.ones((len(data_total), len(data_total['m12b']['time.sim'])))
for i in range(0, masses.shape[0]):
    mask = (masses[i] != -1)
    masses_norm[i][mask] = masses[i][mask]/masses[i][0]
    radii_norm[i][mask] = radii[i][mask]/radii[i][0]

binedges = None
binsize = 0.1
limits=((13.5, 0),None)
#
x = [lookback, lookback, lookback, lookback]
y = [masses, masses_norm, radii, radii_norm]
#
medians = []
lowers = []
uppers = []
lowests = []
highests = []
binss = []
half_bins = []
#
for j in range(0, len(x)):
    #
    if binedges:
        bin_num = int((binedges[1]-binedges[0])/binsize + 1)
        bins = np.linspace(binedges[0], binedges[1], bin_num)
        half_bin = (bins[1]-bins[0])/2
    else:
        minn = binsize*np.floor(np.min(x[j])/binsize)
        maxx = binsize*np.ceil(np.max(x[j])/binsize)
        if minn < 0:
            bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
        else:
            bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
        bins = np.linspace(minn, maxx, bin_num)
        half_bin = (bins[1]-bins[0])/2
    #
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        temp_med = []
        for k in range(0, y[j].shape[0]):
            mask2 = (y[j][k] != -1)
            temp_med.append(y[j][k][mask*mask2])
        med[i] = np.nanmedian(np.hstack(temp_med))
        upper[i] = np.nanpercentile(np.hstack(temp_med), onesigp)
        lower[i] = np.nanpercentile(np.hstack(temp_med), onesigm)
        highest[i] = np.nanpercentile(np.hstack(temp_med), twosigp)
        lowest[i] = np.nanpercentile(np.hstack(temp_med), twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(20,8))
# Plot the scatter for the recent and minimum pericenters
axs[0].fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color='g', alpha=0.3)
axs[0].fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color='g', alpha=0.15)
axs[0].plot(binss[1][:-1]+half_bins[1], medians[1], 'k', alpha=0.5, lw=4)
#
axis_y2_label = 'Median $M_{\\rm 200m}(t_{\\rm lb}) \ [10^{12} M_{\\odot}]$'
axis_y2_tick_labels = ['1.4', '1.2', '1.0', '0.8', '0.6', '0.4', '0.2', '0.0']
axis_y2_tick_values = [float(v) for v in axis_y2_tick_labels]
axis_y2_tick_locations = [1.061, 0.9094, 0.7578, 0.6063, 0.4547, 0.3031, 0.1516, 0.0]
axis_y2_minor_tick_locations = [1.06097825, 1.02308617, 0.98519409, 0.94730201, 0.90940993, 0.87151785, 0.83362577, 0.79573369, 0.7578416 , 0.71994952, 0.68205744, 0.64416536, 0.60627328, 0.5683812 , 0.53048912, 0.49259704, 0.45470496, 0.41681288, 0.3789208 , 0.34102872, 0.30313664, 0.26524456, 0.22735248, 0.1894604 , 0.15156832, 0.11367624, 0.07578416, 0.03789208, 0.]
#axis_y2_minor_tick_locations = np.arange(0, 1.4, 0.05)[1::2]
axy2 = axs[0].twinx()
axy2.set_xscale('linear')
axy2.set_yscale('linear')
axy2.set_yticks(axis_y2_tick_locations)
axy2.set_yticks(axis_y2_minor_tick_locations, minor=True)
axy2.set_yticklabels(axis_y2_tick_labels, fontsize=22)
axy2.set_ylabel(axis_y2_label, fontsize=30, labelpad=9)
axy2.tick_params(pad=3)
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=22)
axz.set_xlim(13.5,0)
axz.set_xlabel(axis_z_label, fontsize=30, labelpad=9)
axz.tick_params(pad=3)
#
axs[1].fill_between(binss[3][:-1]+half_bins[3], uppers[3], lowers[3], color='g', alpha=0.3)
axs[1].fill_between(binss[3][:-1]+half_bins[3], highests[3], lowests[3], color='g', alpha=0.15)
axs[1].plot(binss[3][:-1]+half_bins[3], medians[3], 'k', alpha=0.5, lw=4)
#
axis2_y2_label = 'Median $R_{\\rm 200m}(t_{\\rm lb}) \ [\\rm kpc]$'
axis2_y2_tick_labels = ['400', '300', '200', '100', '0']
axis2_y2_tick_locations = [1.022, 0.767, 0.511, 0.256, 0.0]
axis2_y2_minor_tick_locations = [1.02214915, 0.95826483, 0.89438051, 0.83049619, 0.76661186, 0.70272754, 0.63884322, 0.5749589 , 0.51107458, 0.44719025, 0.38330593, 0.31942161, 0.25553729, 0.19165297, 0.12776864, 0.06388432, 0.        ]
ax2y2 = axs[1].twinx()
ax2y2.set_xscale('linear')
ax2y2.set_yscale('linear')
ax2y2.set_yticks(axis2_y2_tick_locations)
ax2y2.set_yticks(axis2_y2_minor_tick_locations, minor=True)
ax2y2.set_yticklabels(axis2_y2_tick_labels, fontsize=22)
ax2y2.set_ylabel(axis2_y2_label, fontsize=30, labelpad=9)
ax2y2.tick_params(pad=3)
#
axis2_z_label = 'redshift'
axis2_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis2_z_tick_values = [float(v) for v in axis2_z_tick_labels]
axis2_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis2_z_tick_values)
ax2z = axs[1].twiny()
ax2z.set_xscale('linear')
ax2z.set_yscale('linear')
ax2z.set_xticks(axis2_z_tick_locations)
ax2z.set_xticklabels(axis2_z_tick_labels, fontsize=22)
ax2z.set_xlim(13.5,0)
ax2z.set_xlabel(axis2_z_label, fontsize=30, labelpad=9)
ax2z.tick_params(pad=3)
#
axs[0].set_xlim(13.5,0)
axs[0].set_ylim(0, 1.061)
axs[1].set_xlim(13.5,0)
axs[1].set_ylim(0, 1.022)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=22, labelbottom=True)
axs[1].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=22, labelbottom=True)
#
axs[0].set_ylabel('$M_{\\rm 200m}(t_{\\rm lb})\ /\ M_{\\rm 200m}(t_{\\rm lb}=0)$', fontsize=30)
axs[0].get_yaxis().set_label_coords(-0.1,0.5)
axs[1].set_ylabel('$R_{\\rm 200m}(t_{\\rm lb})\ /\ R_{\\rm 200m}(t_{\\rm lb}=0)$', fontsize=30)
axs[1].get_yaxis().set_label_coords(-0.1,0.5)
#
axs[0].set_xlabel('Lookback Time [Gyr]', fontsize=30)
axs[1].set_xlabel('Lookback Time [Gyr]', fontsize=30)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0)
plt.savefig(directory+'/host_properties.pdf')
plt.close()



"""
    Figure 2:
        Mass profile ratio vs lookback time
"""
# Read in the data
mass_m12b = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12b_full_mass_profile')
mass_m12c = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12c_full_mass_profile')
mass_m12f = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12f_full_mass_profile')
mass_m12i = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12i_full_mass_profile')
mass_m12m = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12m_full_mass_profile')
mass_m12w = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12w_full_mass_profile')
mass_m12z = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12z_full_mass_profile')
mass_Romeo = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romeo_full_mass_profile')
mass_Juliet = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Juliet_full_mass_profile')
mass_Thelma = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Thelma_full_mass_profile')
mass_Louise = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Louise_full_mass_profile')
mass_Romulus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romulus_full_mass_profile')
mass_Remus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Remus_full_mass_profile')
rs = np.logspace(np.log10(0.1), np.log10(500), 100)
#
times = np.around(13.8-mass_m12b['time'], decimals=2)
#
onesigp = 84.13
onesigm = 15.87
#
twosigp = 100
twosigm = 0
#
mass_prof_all_50 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:73]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:73],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:73]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:73]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:73]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:73]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:73]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:73]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:73],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:73])])

mass_prof_all_100 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:81]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:81],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:81]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:81]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:81]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:81]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:81]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:81]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:81],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:81])])

mass_prof_all_150 = np.array([np.cumsum(mass_m12b['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12b['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12c['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12c['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12f['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12f['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12i['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12i['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12m['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12m['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12w['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12w['mass.profile'][0][:86]),\
                          np.cumsum(mass_m12z['mass.profile'][:,:86],axis=1)/np.cumsum(mass_m12z['mass.profile'][0][:86]),\
                          np.cumsum(mass_Romeo['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0][:86]),\
                          np.cumsum(mass_Juliet['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0][:86]),\
                          np.cumsum(mass_Thelma['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0][:86]),\
                          np.cumsum(mass_Louise['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Louise['mass.profile'][0][:86]),\
                          np.cumsum(mass_Romulus['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0][:86]),\
                          np.cumsum(mass_Remus['mass.profile'][:,:86],axis=1)/np.cumsum(mass_Remus['mass.profile'][0][:86])])

upper_one_50 = np.percentile(mass_prof_all_50, onesigp, axis=0)[:,-1]
lower_one_50 = np.percentile(mass_prof_all_50, onesigm, axis=0)[:,-1]
upper_two_50 = np.percentile(mass_prof_all_50, twosigp, axis=0)[:,-1]
lower_two_50 = np.percentile(mass_prof_all_50, twosigm, axis=0)[:,-1]
#
med_50 = np.median(mass_prof_all_50, axis=0)
med_100 = np.median(mass_prof_all_100, axis=0)
med_150 = np.median(mass_prof_all_150, axis=0)
#
colorss = ['#006400', '#702f84', '#ab1d0f']
times = np.around(13.8-mass_m12b['time'], decimals=2)
#
plt.rcParams["font.family"] = "serif"
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
ax1.plot(times, med_50[:,-1], color=colorss[0], alpha=1, label='$r < 50$ kpc')
ax1.fill_between(times, upper_two_50, lower_two_50, color=colorss[0], alpha=0.15)
ax1.fill_between(times, upper_one_50, lower_one_50, color=colorss[0], alpha=0.3)
ax1.plot(times, med_100[:,-1], color=colorss[1], alpha=1, label='$r < 100$ kpc')
ax1.plot(times, med_150[:,-1], color=colorss[2], alpha=1, label='$r < 150$ kpc')
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_2_label = 'redshift'
axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
ax2 = ax1.twiny()
ax2.set_xscale('linear')
ax2.set_yscale('linear')
ax2.set_xticks(axis_2_tick_locations)
ax2.set_xticklabels(axis_2_tick_labels, fontsize=26)
ax2.set_xlim(13,0)
ax2.set_xlabel(axis_2_label, fontsize=32, labelpad=9)
ax2.tick_params(pad=3)
#
ax1.hlines(1, 13, 0, color='k', alpha=0.8, linestyles='dotted', zorder=100)
ax1.set_xlim(13, 0)
#plt.ylim(0.95, 1.05)
ax1.set_xscale('linear')
ax1.set_yscale('linear')
ax1.set_xlabel('Lookback time [Gyr]', fontsize=32)
ax1.set_ylabel('$M(t_{\\rm lb}, <r)$ / $M(t_{\\rm lb} = 0, <r)$', fontsize=32)
ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelsize=26)
ax1.legend(prop={'size': 24}, loc='lower right')
plt.tight_layout()
plt.savefig(directory+'/mass_profile_evolution_enclosed.pdf')
plt.close()



"""
    Figure 3:
        Mass profile short and long time evolution
"""

# Read in the data
mass_m12b = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12b_full_mass_profile')
mass_m12c = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12c_full_mass_profile')
mass_m12f = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12f_full_mass_profile')
mass_m12i = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12i_full_mass_profile')
mass_m12m = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12m_full_mass_profile')
mass_m12w = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12w_full_mass_profile')
mass_m12z = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/m12z_full_mass_profile')
mass_Romeo = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romeo_full_mass_profile')
mass_Juliet = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Juliet_full_mass_profile')
mass_Thelma = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Thelma_full_mass_profile')
mass_Louise = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Louise_full_mass_profile')
mass_Romulus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Romulus_full_mass_profile')
mass_Remus = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/Remus_full_mass_profile')
rs = np.logspace(np.log10(0.1), np.log10(500), 100)

# Cumulatively sum each and put them all in one array
mass_prof_all = np.array([np.cumsum(mass_m12b['mass.profile'],axis=1)/np.cumsum(mass_m12b['mass.profile'][0]),\
                          np.cumsum(mass_m12c['mass.profile'],axis=1)/np.cumsum(mass_m12c['mass.profile'][0]),\
                          np.cumsum(mass_m12f['mass.profile'],axis=1)/np.cumsum(mass_m12f['mass.profile'][0]),\
                          np.cumsum(mass_m12i['mass.profile'],axis=1)/np.cumsum(mass_m12i['mass.profile'][0]),\
                          np.cumsum(mass_m12m['mass.profile'],axis=1)/np.cumsum(mass_m12m['mass.profile'][0]),\
                          np.cumsum(mass_m12w['mass.profile'],axis=1)/np.cumsum(mass_m12w['mass.profile'][0]),\
                          np.cumsum(mass_m12z['mass.profile'],axis=1)/np.cumsum(mass_m12z['mass.profile'][0]),\
                          np.cumsum(mass_Romeo['mass.profile'],axis=1)/np.cumsum(mass_Romeo['mass.profile'][0]),\
                          np.cumsum(mass_Juliet['mass.profile'],axis=1)/np.cumsum(mass_Juliet['mass.profile'][0]),\
                          np.cumsum(mass_Thelma['mass.profile'],axis=1)/np.cumsum(mass_Thelma['mass.profile'][0]),\
                          np.cumsum(mass_Louise['mass.profile'],axis=1)/np.cumsum(mass_Louise['mass.profile'][0]),\
                          np.cumsum(mass_Romulus['mass.profile'],axis=1)/np.cumsum(mass_Romulus['mass.profile'][0]),\
                          np.cumsum(mass_Remus['mass.profile'],axis=1)/np.cumsum(mass_Remus['mass.profile'][0])])

mass_prof_all_med = np.median(mass_prof_all, axis=0)

inds = np.concatenate((np.arange(len(mass_m12b['time']))[:21][::5], np.arange(len(mass_m12b['time']))[21:39]))
times = np.around(13.8-np.concatenate((mass_m12b['time'][:21][::5], mass_m12b['time'][21:39])), decimals=2)

def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=1)


plt.rcParams["font.family"] = "serif"
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,12))
#colorss = ['#1d55a7']
limits_1 = ((5, 350),(0, 1.05))
limits_2 = ((5, 350),(0.8, 1.23))
#
for i in range(0, len(inds)):
    ax1.plot(rs[1:], mass_prof_all_med[inds[i]], color=colorss[i])
ax1.set_xlim(limits_1[0])
ax1.set_ylim(limits_1[1])
ax1.set_xscale('log')
#
#cbax = ax1.inset_axes([1.1, 0, 0.03, 1], transform=ax1.transAxes)
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax1, location='top', orientation='horizontal', pad=0.03)
cbar.set_label('Lookback time [Gyr]', fontsize=26)


# Cumulatively sum each and put them all in one array
mass_prof_avg_all = np.array([np.cumsum(mass_m12b['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12b['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12c['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12c['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12f['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12f['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12i['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12i['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12m['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12m['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12w['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12w['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_m12z['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_m12z['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Romeo['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Romeo['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Juliet['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Juliet['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Thelma['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Thelma['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Louise['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Louise['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Romulus['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Romulus['mass.profile'][:21], axis=1), axis=0),\
                              np.cumsum(mass_Remus['mass.profile'][:21],axis=1)/np.average(np.cumsum(mass_Remus['mass.profile'][:21], axis=1), axis=0)])

mass_prof_avg_all_med = np.median(mass_prof_avg_all, axis=0)

inds = np.arange(len(mass_m12b['time'][:21]))
times = np.around(13.8-mass_m12b['time'][:21], decimals=2)

# Plotting the 68%, 95%, and 100% scatters across all hosts and times for each distance
onesigp = 84.13
onesigm = 15.87
#
twosigp = 97.72
twosigm = 2.28
#
thrsigp = 100
thrsigm = 0
#
upper_one = np.percentile(mass_prof_avg_all, onesigp, axis=(0,1))
lower_one = np.percentile(mass_prof_avg_all, onesigm, axis=(0,1))
upper_two = np.percentile(mass_prof_avg_all, twosigp, axis=(0,1))
lower_two = np.percentile(mass_prof_avg_all, twosigm, axis=(0,1))
upper_thr = np.percentile(mass_prof_avg_all, thrsigp, axis=(0,1))
lower_thr = np.percentile(mass_prof_avg_all, thrsigm, axis=(0,1))

ax2.fill_between(rs[1:], upper_thr, lower_thr, color='#6fc4de', alpha=0.3)
#ax2.fill_between(rs[1:], upper_two, lower_two, color='#6fc4de', alpha=0.5)
ax2.fill_between(rs[1:], upper_one, lower_one, color='#6fc4de', alpha=1)
ax2.plot(rs[1:], np.median(mass_prof_avg_all_med, axis=0), color='k') # Comment this line out to get back to the old version
ax2.hlines(1, 0.1, 500, color='k', alpha=0.8, linestyles='dotted', zorder=100)
ax2.set_xlim(limits_2[0])
ax2.set_ylim(limits_2[1])
ax2.set_xscale('log')
#
ax2.set_xlabel('Host distance, $r$ [kpc]', fontsize=26)
ax1.set_ylabel('$M(t_{\\rm lb}, <r)$ / $M(t_{\\rm lb} = 0, <r)$', fontsize=24)
ax2.set_ylabel('$M(t_{\\rm lb}, <r)$ / $M_{\\rm 2\ Gyr\ avg}(<r)$', fontsize=24)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(directory+'/long_short_evolution.pdf')
#plt.show()
plt.close()



"""
    Figure 4:
        Enclosed mass fit
"""
mass_profs = []
mass_sims = []
rs = np.logspace(np.log10(0.1), np.log10(500), 81)
#
for name in summary.host_names['all_no_r']:
    # Loop through each galaxy, read in the data, and append the profiles
    data_disk_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+name+'_disk_radial_profile_fitting')
    density_disk_rad = data_disk_rad['density']
    mass_disk_rad = data_disk_rad['mass']
    rs_disk = data_disk_rad['rs']
    #
    data_halo = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+name+'_halo_fitting')
    density_halo = data_halo['density']
    mass_halo = data_halo['mass']
    rs_halo = data_halo['rs']
    #
    masses = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/full_profile/'+name+'_spherical_mass')
    #
    # Create the radial mass profile
    profiles = model_io.Profiles(sim_data.home_dir)
    disk_full_mass = profiles.disk_radial_mass(rs, name)
    halo_full_mass = profiles.halo_2p_nfw_mass(rs, name)
    total_mass = disk_full_mass+halo_full_mass
    #
    mass_sims.append(masses)
    mass_profs.append(total_mass)

mass_ratio = np.zeros((len(mass_profs), len(mass_profs[0][1:])))
for i in range(0, len(mass_profs)):
    for j in range(1, len(mass_profs[0])):
        mass_ratio[i,j-1] = mass_profs[i][j]/mass_sims[i]['mass.enclosed'][j-1]
#
onesigp = 84.13
onesigm = 15.87
#
thrsigp = 100
thrsigm = 0
#
upper_one = np.percentile(mass_ratio, onesigp, axis=0)
lower_one = np.percentile(mass_ratio, onesigm, axis=0)
upper_thr = np.percentile(mass_ratio, thrsigp, axis=0)
lower_thr = np.percentile(mass_ratio, thrsigm, axis=0)
#
mass_ratio_med = np.median(mass_ratio, axis=0)
#
plt.figure(figsize=(10,8))
plt.fill_between(rs[1:], upper_thr, lower_thr, color='#9966cc', alpha=0.15)
plt.fill_between(rs[1:], upper_one, lower_one, color='#9966cc', alpha=0.5)
plt.plot(rs[1:], mass_ratio_med, color='k', alpha=1)
plt.hlines(y=1, xmin=0, xmax=500, linestyles='dotted', colors='k', alpha=0.5)
plt.xscale('log')
plt.xlim(xmin=5, xmax=500)
plt.ylim(ymin=0.9, ymax=1.1)
plt.xlabel('Host distance, $r$ [kpc]', fontsize=34)
plt.ylabel('$M_{\\rm model}(<r)\ /\ M_{\\rm sim}(<r)$', fontsize=34)
plt.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.savefig(directory+'/mass_ratio_fit_median.pdf')
plt.close()



"""
    Figure 5:
        Satellite orbits
"""
snap_e = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_res7100')
tlb = snap_e['time'][-1] - np.flip(snap_e['time'])
data = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', selection='sim')
data_model = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', selection='model')
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_energy_new')
#
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=False, hosts='all_energy_new', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=False, hosts='all_energy_new', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=False, hosts='all_energy_new', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=False, hosts='all_energy_new', sim_type='baryon')
#
frac_d = np.abs((d_rec_mod-d_rec_sim)/d_rec_sim)
names = summary.halo_id(data_total, masks_infall_peri, hosts='all_energy_new')

#m1 = (frac_d > 0.75)#*(frac_d < 0.75)
#halo_index = np.random.randint(low=0, high=np.sum(m1))
#print(names['host'][m1][halo_index], names['ids'][m1][halo_index], frac_d[m1][halo_index])

from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import evaluatePotentials

peri_color, apo_color = '#337422', '#D994F8'

halo_ids = [28-1, 49-1, 49-1, 68-1] # 11%, 40%, 51%, 95%
#
hosts = ['m12w', 'm12f', 'Romeo', 'Louise']

# First column
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(4, 4, figsize=(32,22))
#
for i in range(0, len(halo_ids)):
    snaps = data_total[hosts[i]]['time.sim']
    #
    d_model = data_total[hosts[i]]['d.tot.model'][halo_ids[i]]
    v_model = data_total[hosts[i]]['v.tot.model'][halo_ids[i]]
    L_model = np.linalg.norm(data_total[hosts[i]]['L.model'], axis=2)[halo_ids[i]]/10000
    #
    # Set up the distances and times to plot
    d_sim = data_total[hosts[i]]['d.tot.sim'][halo_ids[i]]
    v_sim = data_total[hosts[i]]['v.tot.sim'][halo_ids[i]]
    L_sim = data_total[hosts[i]]['L.tot.sim'][halo_ids[i]]/10000
    #
    """
        Calculate the total energy for the satellites in the simulations
    """
    # Set the host potential at 100 kpc at z = 0
    phi_host_z0 = data[hosts[i]]['host.pot.100kpc'][-1] - data[hosts[i]]['host.pot.R200m'] - data[hosts[i]]['KE.at.Rvir']
    #
    # Set up the enclosed mass ratio within 100-ish kpc
    M_enc_100kpc_z0 = data_mp[hosts[i]][-1][16]
    M_enc_100kpc_z = data_mp[hosts[i]][:,16]
    #
    M_enc_ratio = M_enc_100kpc_z/M_enc_100kpc_z0
    #
    # Define the host potential at 100 kpc across all snapshots
    phi_host_z = np.flip(phi_host_z0*M_enc_ratio) # Goes from z = 0 to some other z
    #
    # Set up null array to save the normalized subhalo potentials to
    sub_pot = (-1)*np.ones(data[hosts[i]]['subhalo.pot'].shape)
    sub_energy = (-1)*np.ones(data[hosts[i]]['subhalo.pot'].shape)
    sub_pot_snaps = (-1)*np.ones(data[hosts[i]]['subhalo.pot'].shape, int)
    sub_pot_tlb = (-1)*np.ones(data[hosts[i]]['subhalo.pot'].shape)
    #
    # Loop through all of the satellites
    for j in range(0, sub_pot.shape[0]):
        # Create a mask for the subhalo data
        mask_sub = (np.flip(data[hosts[i]]['subhalo.pot'][j]) != -1)*np.isfinite(np.flip(data[hosts[i]]['subhalo.pot'][j]))*(data_total[hosts[i]]['v.tot.sim'][j][:len(data[hosts[i]]['subhalo.pot'][j])] != -1)*np.isfinite(data_total[hosts[i]]['v.tot.sim'][j][:len(data[hosts[i]]['subhalo.pot'][j])])*np.isfinite(np.flip(data[hosts[i]]['host.pot.100kpc']))
        #
        # Create a mask for the host data
        mask_host = (phi_host_z[:len(data[hosts[i]]['subhalo.pot'][j])] != 0)
        #
        # Calculate the normalized subhalo energy
        sub_pot[j][mask_sub*mask_host] = np.flip(data[hosts[i]]['subhalo.pot'][j])[mask_sub*mask_host] - np.flip(data[hosts[i]]['host.pot.100kpc'])[mask_sub*mask_host] + phi_host_z[:len(data[hosts[i]]['subhalo.pot'][j])][mask_sub*mask_host]
        # Keep which snapshots it has data for
        sub_pot_snaps[j][mask_sub*mask_host] = np.flip(snap_e['index'])[:len(data[hosts[i]]['subhalo.pot'][j])][mask_sub*mask_host]
        sub_pot_tlb[j][mask_sub*mask_host] = tlb[:len(data[hosts[i]]['subhalo.pot'][j])][mask_sub*mask_host]
        #
        # Calculate the total orbital energy
        sub_energy[j][mask_sub*mask_host] = sub_pot[j][mask_sub*mask_host] + 0.5*(data_total[hosts[i]]['v.tot.sim'][j][:len(data[hosts[i]]['subhalo.pot'][j])][mask_sub*mask_host])**2
    #
    """
        Calculate the total energy for the satellites in the model
    """
    # Set up the potential model
    fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
    #
    disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][hosts[i]]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][hosts[i]]*u.kpc, hz=fitting_data['h_z'][hosts[i]]*u.kpc)
    disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][hosts[i]]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][hosts[i]]*u.kpc, hz=fitting_data['h_z'][hosts[i]]*u.kpc)
    halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][hosts[i]]*u.solMass, a=fitting_data['a_halo'][hosts[i]]*u.kpc, alpha=fitting_data['alpha'][hosts[i]], beta=fitting_data['beta'][hosts[i]])
    potential_two_power = disk_inner+disk_outer+halo_2p
    #
    # Set the global potential at 100 kpc
    d100 = 100/np.sqrt(2)
    d200m = data_total[hosts[i]]['host.radius'][0]/np.sqrt(2)
    phi_model_100kpc = evaluatePotentials(potential_two_power, d100*u.kpc, d100*u.kpc)
    #
    # Set the potential at z = 0
    phi_model_z0 = evaluatePotentials(potential_two_power, d100*u.kpc, d100*u.kpc) - evaluatePotentials(potential_two_power, d200m*u.kpc, d200m*u.kpc) - data[hosts[i]]['KE.at.Rvir']
    # Set the potential across all time as just the potential at z = 0
    phi_model_z = phi_model_z0*np.ones(data_model[hosts[i]]['model.potential'].shape[1])
    #
    # Set up null array to save the normalized subhalo potentials to
    sub_pot_model = (-1)*np.ones(data_model[hosts[i]]['model.potential'].shape)
    sub_energy_model = (-1)*np.ones(data_model[hosts[i]]['model.potential'].shape)
    sub_pot_snaps_model = (-1)*np.ones(data_model[hosts[i]]['model.potential'].shape, int)
    sub_pot_tlb_model = (-1)*np.ones(data_model[hosts[i]]['model.potential'].shape)
    sub_kin_model = (-1)*np.ones(data_model[hosts[i]]['model.potential'].shape)
    #
    # Loop through all of the satellites and calculate the potential the same way that I do in the simulations
    for j in range(0, sub_pot_model.shape[0]):
        # Create a mask for the subhalo data
        # Calculate the normalized subhalo energy
        sub_pot_model[j] = data_model[hosts[i]]['model.potential'][j] - phi_model_100kpc + phi_model_z
        #
        # Keep which snapshots it has data for
        sub_pot_snaps_model[j] = np.flip(snap_e['index'])
        sub_pot_tlb_model[j] = tlb
        #
        # Calculate the total orbital energy
        sub_energy_model[j] = sub_pot_model[j] + 0.5*(data_total[hosts[i]]['v.tot.model'][j])**2
        sub_kin_model[j] =  0.5*(data_total[hosts[i]]['v.tot.model'][j])**2
    #
    """
        Mask the right data and plot
    """
    E_sim = sub_energy[halo_ids[i]]
    t_E = sub_pot_tlb[halo_ids[i]]
    E_model = sub_energy_model[halo_ids[i]]
    t_E_model = sub_pot_tlb_model[halo_ids[i]]
    #
    d_mask = (d_sim >= 0)
    d_sim = d_sim[d_mask]
    v_sim = v_sim[d_mask]
    L_sim = L_sim[d_mask]
    #
    E_sim = E_sim[E_sim != -1]
    t_E = t_E[:len(E_sim)][E_sim != -1]
    E_model = E_model[:len(E_sim)]
    t_E_model = t_E_model[:len(t_E)]
    #
    lookback_time = np.flip(snaps[-1] - snaps)
    times = lookback_time[:len(d_sim)]
    times_model = data_total[hosts[i]]['time.model']
    #
    # Plot the distances
    if i == 0:
        axs[0,i].plot(times, d_sim, 'k', label='Simulation')
        axs[0,i].plot(-1*times_model, d_model, label='Model', alpha=0.5)
        #axs[0,i].plot([], [], ' ', label='Recent $d_{\\rm peri,model}$: '+str(np.around(d_rec_mod[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2))+' kpc')
        #axs[0,i].plot([], [], ' ', label='Recent $d_{\\rm peri,sim}$: '+str(np.around(d_rec_sim[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2))+' kpc')
        axs[0,i].plot([], [], ' ', label='$\\Delta d_{\\rm peri}/d_{\\rm peri}$: '+str(np.around(frac_d[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2)))
    else:
        axs[0,i].plot(times, d_sim, 'k')
        axs[0,i].plot(-1*times_model, d_model, alpha=0.5)
        #axs[0,i].plot([], [], ' ', label='Recent $d_{\\rm peri,model}$: '+str(np.around(d_rec_mod[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2))+' kpc')
        #axs[0,i].plot([], [], ' ', label='Recent $d_{\\rm peri,sim}$: '+str(np.around(d_rec_sim[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2))+' kpc')
        axs[0,i].plot([], [], ' ', label='$\\Delta d_{\\rm peri}/d_{\\rm peri}$: '+str(np.around(frac_d[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2)))
    axs[0,i].plot(times, data_total[hosts[i]]['host.radius'][:len(times)], 'k', alpha=0.3)
    #axs[0,i].set_xlim(times[-1], times[0])
    #
    # Check to see if there were infall, pericenter, or apocenter events
    infall = data_total[hosts[i]]['infall.check'][halo_ids[i]]
    peri = data_total[hosts[i]]['pericenter.check.sim'][halo_ids[i]]
    #
    # If there were, plot when they occurred
    if infall:
        infall_time = data_total[hosts[i]]['first.infall.time.lb'][halo_ids[i]]
        axs[0,i].axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    if peri:
        for j in data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]][data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]] != -1]:
            axs[0,i].axvline(x=j, ymin=0, ymax=1, color=peri_color, linestyle=':', alpha=0.3)
    #
    # Set the labels and save the figure
    #axs[0,i].set_ylim(top=np.nanmax(d_sim)+50)
    axs[0,i].label_outer()
    #axs[0,i].legend(prop={'size': 24}, loc='upper right', framealpha=1)
    #
    # Plot the velocity
    axs[1,i].plot(times, v_sim, 'k')
    axs[1,i].plot(-1*times_model, v_model, alpha=0.5)
    #
    if infall:
        infall_time = data_total[hosts[i]]['first.infall.time.lb'][halo_ids[i]]
        axs[1,i].axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    if peri:
        for j in data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]][data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]] != -1]:
            axs[1,i].axvline(x=j, ymin=0, ymax=1, color=peri_color, linestyle=':', alpha=0.3)
    #
    # Set the labels and save the figure
    axs[1,i].set_ylim(top=np.nanmax(v_sim)+50)
    axs[1,i].label_outer()
    #
    # Plot the angular momentum
    axs[2,i].plot(times, L_sim, 'k')
    axs[2,i].plot(-1*times_model, L_model, alpha=0.5)
    #
    if infall:
        infall_time = data_total[hosts[i]]['first.infall.time.lb'][halo_ids[i]]
        axs[2,i].axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    if peri:
        for j in data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]][data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]] != -1]:
            axs[2,i].axvline(x=j, ymin=0, ymax=1, color=peri_color, linestyle=':', alpha=0.3)
    #
    # Set the labels and save the figure
    axs[2,i].set_ylim(top=np.nanmax(L_sim)+0.3)
    axs[2,i].label_outer()
    #
    # Plot the energy
    axs[3,i].plot(t_E, E_sim/1e4, 'k')
    axs[3,i].plot(t_E_model, E_model/1e4, alpha=0.5)
    #
    if infall:
        infall_time = data_total[hosts[i]]['first.infall.time.lb'][halo_ids[i]]
        axs[3,i].axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
    #
    if peri:
        for j in data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]][data_total[hosts[i]]['pericenter.time.lb.sim'][halo_ids[i]] != -1]:
            axs[3,i].axvline(x=j, ymin=0, ymax=1, color=peri_color, linestyle=':', alpha=0.3)
    #
    # Set the labels and save the figure
    axs[3,i].set_ylim(top=np.nanmax(L_sim)+0.3)
    axs[3,i].label_outer()
    #
axs[0,0].set_xlim(13.8,0)
axs[1,0].set_xlim(13.8,0)
axs[2,0].set_xlim(13.8,0)
axs[3,0].set_xlim(13.8,0)
axs[0,1].set_xlim(13.8,0)
axs[1,1].set_xlim(13.8,0)
axs[2,1].set_xlim(13.8,0)
axs[3,1].set_xlim(13.8,0)
axs[0,2].set_xlim(13.8,0)
axs[1,2].set_xlim(13.8,0)
axs[2,2].set_xlim(13.8,0)
axs[3,2].set_xlim(13.8,0)
axs[0,3].set_xlim(13.8,0)
axs[1,3].set_xlim(13.8,0)
axs[2,3].set_xlim(13.8,0)
axs[3,3].set_xlim(13.8,0)
#
axs[0,0].set_ylim(0,450)
axs[0,1].set_ylim(0,330)
axs[0,2].set_ylim(0,400)
axs[0,3].set_ylim(0,400)
axs[1,0].set_ylim(0,330)
axs[1,1].set_ylim(0,430)
axs[1,2].set_ylim(0,430)
axs[1,3].set_ylim(0,340)
axs[2,0].set_ylim(0,3.3)
axs[2,1].set_ylim(0,2.75)
axs[2,2].set_ylim(0,3.3)
axs[2,3].set_ylim(0,2.9)
axs[3,0].set_ylim(-10,10)
axs[3,1].set_ylim(-10,10)
axs[3,2].set_ylim(-10,10)
axs[3,3].set_ylim(-10,10)
#
axs[0,0].legend(prop={'size': 24}, loc='upper right', framealpha=1)
axs[0,1].legend(prop={'size': 24}, loc='upper right', framealpha=1)
axs[0,2].legend(prop={'size': 24}, loc='upper right', framealpha=1)
axs[0,3].legend(prop={'size': 24}, loc='upper right', framealpha=1)
#
axs[3,0].set_xlabel('Lookback time [Gyr]', fontsize=32)
axs[3,1].set_xlabel('Lookback time [Gyr]', fontsize=32)
axs[3,2].set_xlabel('Lookback time [Gyr]', fontsize=32)
axs[3,3].set_xlabel('Lookback time [Gyr]', fontsize=32)
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False, labelleft=True)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False, labelleft=True)
axs[0,2].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False, labelleft=True)
axs[0,3].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False, labelleft=True)
#
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
axs[1,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
axs[1,3].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
#
axs[2,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
axs[2,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
axs[2,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
axs[2,3].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False, labelleft=True)
#
axs[3,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=True)
axs[3,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=True)
axs[3,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=True)
axs[3,3].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=True)
#
#plt.tight_layout()
axs[0,0].set_ylabel('Host distance, r [kpc]', fontsize=26)
axs[0,0].get_yaxis().set_label_coords(-0.13,0.5)
axs[1,0].set_ylabel('Total Velocity [km s$^{-1}$]', fontsize=26)
axs[1,0].get_yaxis().set_label_coords(-0.13,0.5)
axs[2,0].set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=26)
axs[2,0].get_yaxis().set_label_coords(-0.13,0.5)
axs[3,0].set_ylabel('Energy [10$^4$ kpc km s$^{-1}$]', fontsize=26)
axs[3,0].get_yaxis().set_label_coords(-0.13,0.5)
#
r1 = mpatches.Rectangle(xy=(12.9,375),width=-1.35,height=55, color='k', alpha=0.2)
r2 = mpatches.Rectangle(xy=(12.9,275),width=-1.35,height=40, color='k', alpha=0.2)
r3 = mpatches.Rectangle(xy=(12.9,330),width=-1.35,height=50, color='k', alpha=0.2)
r4 = mpatches.Rectangle(xy=(12.9,330),width=-1.35,height=50, color='k', alpha=0.2)
axs[0,0].text(12.5,388,'A',fontsize=28)
axs[0,1].text(12.5,285,'B',fontsize=28)
axs[0,2].text(12.5,345,'C',fontsize=28)
axs[0,3].text(12.5,345,'D',fontsize=28)
axs[0,0].add_patch(r1)
axs[0,1].add_patch(r2)
axs[0,2].add_patch(r3)
axs[0,3].add_patch(r4)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0)
#
plt.savefig(directory+'/multiplot_orbit_properties.pdf')
plt.close()



"""
    Figure 6:
        Infall time comparisons
"""
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_diagnostics(data_total, masks_infall, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
mask_finite = np.isfinite(t_in_mod)*(t_in_mod != -1)
mask_finite_R200m = np.isfinite(t_in_mod_R200m)*(t_in_mod_R200m != -1)

#summary_plot.median_plot_mult(x=(t_in_sim[mask_finite_R200m], t_in_sim[mask_finite]), y=(t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m], t_in_mod[mask_finite]-t_in_sim[mask_finite]), xtype=['t.infall.text','t.infall.text'], ytype=['delta_t_infall','delta_t_infall'], binsize=1, binedges=(0,14), limits=((0,13.8),None), labels=['$R_{\\rm 200m}(t_{\\rm lb}=0)$','$R_{\\rm 200m}(t_{\\rm lb})$'], hl=True, file_path_and_name=directory+'/dt_infall_vs_t_infall.pdf')
#summary_plot.median_plot_mult(x=(Mstar_z0[mask_finite_R200m], Mstar_z0[mask_finite]), y=(t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m], t_in_mod[mask_finite]-t_in_sim[mask_finite]), xtype=['M.star.z0','M.star.z0'], ytype=['delta_t_infall','delta_t_infall'], binsize=0.5, limits=((4,9.5),None), labels=['$R_{\\rm 200m}(t_{\\rm lb}=0)$','$R_{\\rm 200m}(t_{\\rm lb})$'], hl=True, file_path_and_name=directory+'/dt_infall_vs_Mstar.pdf')

plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 2, figsize=(20,8.5))
#
xs = [t_in_sim[mask_finite_R200m], t_in_sim[mask_finite]]
ys = [t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m], t_in_mod[mask_finite]-t_in_sim[mask_finite]]
#
xtypes = ['t.infall.text','t.infall.text']
ytypes = ['t.infall.text','t.infall.text']
peri_d_colors = ['#457f44', '#6365ac']
labels = ['$R_{\\rm 200m}(t_{\\rm lb} = 0)$', '$R_{\\rm 200m}(t_{\\rm lb})$']
#
binsize = 1
limits = ((0,13),(-5,11))
#
axs[0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[0].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7, label=labels[i])
    #
    axs[0].set_xlim(limits[0])
    axs[0].set_ylim(limits[1])
axs[0].legend(prop={'size': 22}, loc='best')
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=26)
axz.set_xlim(0,13)
axz.set_xlabel(axis_z_label, fontsize=30, labelpad=9)
axz.tick_params(pad=3)
#
xs = [Mstar_z0[mask_finite_R200m], Mstar_z0[mask_finite]]
ys = [t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m], t_in_mod[mask_finite]-t_in_sim[mask_finite]]
#
xtypes = ['M.star.z0','M.star.z0']
ytypes = ['t.infall.text','t.infall.text']
#
binsize = 0.5
limits = ((4.5,9.5),(-3,12))
#
axs[1].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[1].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[1].plot(10**(binss[:-1]+half_binss), med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[1].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[1].set_ylim(limits[1])
    axs[1].set_xscale('log')
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=True)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
#
axs[0].set_ylabel('$t_{\\rm infall,lb,model}-t_{\\rm infall,lb,sim}$ [Gyr]', fontsize=30)
axs[0].get_yaxis().set_label_coords(-0.12,0.5)
axs[1].set_ylabel(' ', fontsize=26)
axs[1].get_yaxis().set_label_coords(-0.12,0.5)
#
axs[0].set_xlabel('Infall Lookback Time [Gyr]', fontsize=32)
axs[1].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=32)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)
#plt.show()
plt.savefig(directory+'/infall_vs_property.pdf')



"""
    Figure 7:
        Pericenter property plots
"""
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(4, 3, figsize=(24,24))
#
# Pericenter distances
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [t_in_sim, t_in_sim]
ys = [(d_rec_mod-d_rec_sim)/d_rec_sim, (d_min_mod-d_min_sim)/d_min_sim]
#
xtypes = ['t.infall.text', 't.infall.text']
ytypes = ['delta.d.frac', 'delta.d.frac']
labels = ['Recent', 'Minimum']
peri_d_colors = ['#457f44', '#6365ac']
#
binsize = 1
limits = ((0,13),(-0.5,3.5))
#
axs[0,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[0,0].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7, label=labels[i])
    #
    axs[0,0].set_xlim(limits[0])
    axs[0,0].set_ylim(limits[1])
axs[0,0].legend(prop={'size': 22}, loc='best')
#
# Pericenter velocities
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [t_in_sim, t_in_sim]
ys = [v_rec_mod-v_rec_sim, v_min_mod-v_min_sim]
xtypes = ['t.infall.text', 't.infall.text']
ytypes = ['delta.v', 'delta.v']
labels = ['Recent', 'Minimum']
#
peri_v_colors = ['#7c263e', '#263e7c']
#
binsize = 1
limits = ((0,13),(-70,90))
#
axs[1,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_v_colors[i], alpha=0.3)
    axs[1,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_v_colors[i], alpha=0.15)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[1,0].plot(binss[:-1]+half_binss, med, color=peri_v_colors[i], markersize=10, alpha=0.7, label=labels[i])
    #
    axs[1,0].set_xlim(limits[0])
    axs[1,0].set_ylim(limits[1])
axs[1,0].legend(prop={'size': 22}, loc='best')
#
# Pericenter number
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_diagnostics(data_total, masks_infall, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [t_in_sim, t_in_sim]
ys = [n_mod_mod_infall-n_sim, n_mod_r200-n_sim]
#
xtypes = ['t.infall.text', 't.infall.text']
ytypes = ['N.delta', 'N.delta']
labels = ['$R_{\\rm 200m}(t_{\\rm lb})$', '$R_{\\rm 200m}(t_{\\rm lb}=0)$']
peri_n_colors = ['#356b84', '#844e35']
#
binsize = 1
limits = ((0,13),(-1.1,2.5))
#
axs[2,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_n_colors[i], alpha=0.3)
    axs[2,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_n_colors[i], alpha=0.15)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[2,0].plot(binss[:-1]+half_binss, med, color=peri_n_colors[i], markersize=10, alpha=0.7, label=labels[i])
    #
    axs[2,0].set_xlim(limits[0])
    axs[2,0].set_ylim(limits[1])
axs[2,0].legend(prop={'size': 22}, loc='best')
#
# Pericenter times
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [t_in_sim]
ys = [t_rec_mod-t_rec_sim]
xtypes = ['t.infall.text']
ytypes = ['delta.t']
#
#peri_t_colors = ['#2c8ca9','#a36952']
peri_t_colors = ['#a36952']
#
binsize = 1
limits = ((0,13),(-1.5,0.5))
#
axs[3,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[3,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_t_colors[i], alpha=0.5)
    axs[3,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_t_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[3,0].plot(binss[:-1]+half_binss, med, color=peri_t_colors[i], markersize=10, alpha=0.7)
    #
    axs[3,0].set_xlim(limits[0])
    axs[3,0].set_ylim(limits[1])
#
# Pericenter distances vs d(z = 0)
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [dz0_tot, dz0_tot]
ys = [(d_rec_mod-d_rec_sim)/d_rec_sim, (d_min_mod-d_min_sim)/d_min_sim]
xtypes = ['d.z0', 'd.z0']
ytypes = ['delta.d.frac', 'delta.d.frac']
#
binsize = 50
limits = ((0,400),(-0.5,2))
#
axs[0,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[0,1].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[0,1].set_xlim(limits[0])
    axs[0,1].set_ylim(limits[1])
#
# Pericenter velocity vs d(z = 0)
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [dz0_tot, dz0_tot]
ys = [v_rec_mod-v_rec_sim, v_min_mod-v_min_sim]
xtypes = ['d.z0', 'd.z0']
ytypes = ['delta.v', 'delta.v']
#
binsize = 50
limits = ((0,400),(-70,90))
#
axs[1,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_v_colors[i], alpha=0.3)
    axs[1,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_v_colors[i], alpha=0.15)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[1,1].plot(binss[:-1]+half_binss, med, color=peri_v_colors[i], markersize=10, alpha=0.7)
    #
    axs[1,1].set_xlim(limits[0])
    axs[1,1].set_ylim(limits[1])
#
# Pericenter Number vs d(z = 0)
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [dz0_tot, dz0_tot]
ys = [n_mod_mod_infall-n_sim, n_mod_r200-n_sim]
#
xtypes = ['d.z0', 'd.z0']
ytypes = ['N.delta', 'N.delta']
colors = peri_n_colors
#
binsize = 50
limits = ((0,400),(-1.1,4.5))
#
axs[2,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_n_colors[i], alpha=0.3)
    axs[2,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_n_colors[i], alpha=0.15)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[2,1].plot(binss[:-1]+half_binss, med, color=peri_n_colors[i], markersize=10, alpha=0.7)
    #
    axs[2,1].set_xlim(limits[0])
    axs[2,1].set_ylim(limits[1])
#
# Pericenter times vs d(z = 0)
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [dz0_tot]
ys = [t_rec_mod-t_rec_sim]
xtypes = ['d.z0']
ytypes = ['delta.t']
#
binsize = 50
limits = ((0,400),(-1.5,0.3))
#
axs[3,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[3,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_t_colors[i], alpha=0.5)
    axs[3,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_t_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[3,1].plot(binss[:-1]+half_binss, med, color=peri_t_colors[i], markersize=10, alpha=0.7)
    #
    axs[3,1].set_xlim(limits[0])
    axs[3,1].set_ylim(limits[1])
#
# Pericenter distances vs Mstar(z=0)
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [Mstar_z0, Mstar_z0]
ys = [(d_rec_mod-d_rec_sim)/d_rec_sim, (d_min_mod-d_min_sim)/d_min_sim]
xtypes = ['M.star.z0', 'M.star.z0']
ytypes = ['delta.d.frac', 'delta.d.frac']
#
binsize = 0.5
limits = ((4.5,9.5),(-0.8,2))
#
axs[0,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[0,2].plot(10**(binss[:-1]+half_binss), med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[0,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[0,2].set_ylim(limits[1])
    axs[0,2].set_xscale('log')
#
# Pericenter velocities vs Mstar(z=0)
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [Mstar_z0, Mstar_z0]
ys = [v_rec_mod-v_rec_sim, v_min_mod-v_min_sim]
xtypes = ['M.star.z0', 'M.star.z0']
ytypes = ['delta.v', 'delta.v']
#
binsize = 0.5
limits = ((4.5,9.5),(-70,90))
#
axs[1,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_v_colors[i], alpha=0.3)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_v_colors[i], alpha=0.15)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[1,2].plot(10**(binss[:-1]+half_binss), med, color=peri_v_colors[i], markersize=10, alpha=0.7)
    #
    axs[1,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[1,2].set_ylim(limits[1])
    axs[1,2].set_xscale('log')
#
# Pericenter number vs Mstar(z=0)
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [Mstar_z0, Mstar_z0]
ys = [n_mod_mod_infall-n_sim, n_mod_r200-n_sim]
#
xtypes = ['M.star.z0', 'M.star.z0']
ytypes = ['N.delta', 'N.delta']
#
binsize = 0.5
limits = ((4.5,9.5),(-1.1,6.5))
#
axs[2,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_n_colors[i], alpha=0.3)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_n_colors[i], alpha=0.15)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[2,2].plot(10**(binss[:-1]+half_binss), med, color=peri_n_colors[i], markersize=10, alpha=0.7)
    #
    axs[2,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[2,2].set_ylim(limits[1])
    axs[2,2].set_xscale('log')
#
# Pericenter times vs Mstar(z=0)
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [Mstar_z0]
ys = [t_rec_mod-t_rec_sim]
xtypes = ['M.star.z0']
ytypes = ['delta.t']
#
binsize = 0.5
limits = ((4.5,9.5),(-0.7,0.3))
#
axs[3,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[3,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_t_colors[i], alpha=0.5)
    axs[3,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_t_colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[3,2].plot(10**(binss[:-1]+half_binss), med, color=peri_t_colors[i], markersize=10, alpha=0.7)
    #
    axs[3,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[3,2].set_ylim(limits[1])
    axs[3,2].set_xscale('log')
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[2,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[3,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=True)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[2,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[3,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=True)
axs[0,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[1,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[2,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[3,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=True)
#
axs[0,0].set_ylabel('$(d_{\\rm peri,model}-d_{\\rm peri,sim})/d_{\\rm peri,sim}$', fontsize=26)
axs[0,0].get_yaxis().set_label_coords(-0.15,0.5)
axs[1,0].set_ylabel('$v_{\\rm peri,model}-v_{\\rm peri,sim}$ [km s$^{-1}$]', fontsize=26)
axs[1,0].get_yaxis().set_label_coords(-0.15,0.5)
axs[2,0].set_ylabel('$N_{\\rm peri,model}-N_{\\rm peri,sim}$', fontsize=26)
axs[2,0].get_yaxis().set_label_coords(-0.15,0.5)
axs[3,0].set_ylabel('$t_{\\rm peri,model}-t_{\\rm peri,sim}$ [Gyr]', fontsize=26)
axs[3,0].get_yaxis().set_label_coords(-0.15,0.5)
#
axs[3,0].set_xlabel('Infall Lookback Time [Gyr]', fontsize=32)
axs[3,1].set_xlabel('Host distance, $r$ [kpc]', fontsize=32)
axs[3,2].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=32)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.17, hspace=0)
#plt.show()
plt.savefig(directory+'/pericenter_properties.pdf')



"""
    Figure 8:
        Apocenter + Period + eccentricity
"""
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(3, 3, figsize=(28,18))
#
# Apocenter distance vs t_infall
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [t_in_sim]
ys = [(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim]
#
xtypes = ['t.infall.text']
ytypes = ['delta.dapo.frac']
apo_color = ['#652782']
#
binsize = 1
limits = ((0,13),(-0.3,0.9))
#
axs[0,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,0].fill_between(binss[:-1]+half_binss, upper, lower, color=apo_color[i], alpha=0.5)
    axs[0,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=apo_color[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[0,0].plot(binss[:-1]+half_binss, med, color=apo_color[i], markersize=10, alpha=0.7)
    #
    axs[0,0].set_xlim(limits[0])
    axs[0,0].set_ylim(limits[1])
#
# Period vs t_infall
period_mod = []
period_sim = []
t_in_sim = []
for name in summary.host_names['all_no_r']:
    mask = (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][:,0] != -1)*(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][:,0] != -1)
    period_mod.append(np.repeat(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    period_sim.append(np.repeat(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    t_in_sim.append(np.repeat(data_total[name]['first.infall.time.lb'][masks_infall_peri[name]][mask], summary.oversample['baryon'][name]))
period_mod = np.hstack(period_mod)
period_sim = np.hstack(period_sim)
t_in_sim = np.hstack(t_in_sim)
#
xs = [t_in_sim]
ys = [period_mod-period_sim]
#
xtypes = ['t.infall.text']
ytypes = ['period.delta']
period_color = ['#ca2a21']
#
binsize = 1
limits = ((0,13),(-1.9,2.9))
#
axs[1,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,0].fill_between(binss[:-1]+half_binss, upper, lower, color=period_color[i], alpha=0.5)
    axs[1,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=period_color[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[1,0].plot(binss[:-1]+half_binss, med, color=period_color[i], markersize=10, alpha=0.7)
    #
    axs[1,0].set_xlim(limits[0])
    axs[1,0].set_ylim(limits[1])
#
# Eccentricity vs t_infall
ecc_mod = []
ecc_sim = []
t_in_sim = []
for name in summary.host_names['all_no_r']:
    mask = (data_total[name]['eccentricity.sim'][masks_infall_peri[name]][:,0] != -1)*(data_total[name]['eccentricity.model.apsis'][masks_infall_peri[name]][:,0] != -1)
    ecc_mod.append(np.repeat(data_total[name]['eccentricity.model.apsis'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    ecc_sim.append(np.repeat(data_total[name]['eccentricity.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    t_in_sim.append(np.repeat(data_total[name]['first.infall.time.lb'][masks_infall_peri[name]][mask], summary.oversample['baryon'][name]))
ecc_mod = np.hstack(ecc_mod)
ecc_sim = np.hstack(ecc_sim)
t_in_sim = np.hstack(t_in_sim)
#
xs = [t_in_sim]
ys = [ecc_mod-ecc_sim]
#
xtypes = ['t.infall.text']
ytypes = ['ecc.delta']
ecc_color = ['#1d5fee']
#
binsize = 1
limits = ((0,13),(-0.15,0.25))
#
axs[2,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,0].fill_between(binss[:-1]+half_binss, upper, lower, color=ecc_color[i], alpha=0.5)
    axs[2,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=ecc_color[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[2,0].plot(binss[:-1]+half_binss, med, color=ecc_color[i], markersize=10, alpha=0.7)
    #
    axs[2,0].set_xlim(limits[0])
    axs[2,0].set_ylim(limits[1])
#
# Apocenter distance vs d(z = 0)
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_apo, oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [dz0_tot]
ys = [(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim]
#
xtypes = ['d.z0']
ytypes = ['delta.dapo.frac']
colors = apo_color
#
binsize = 50
limits = ((0,400),(-0.2,0.5))
#
axs[0,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,1].fill_between(binss[:-1]+half_binss, upper, lower, color=colors[i], alpha=0.5)
    axs[0,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[0,1].plot(binss[:-1]+half_binss, med, color=colors[i], markersize=10, alpha=0.7)
    #
    axs[0,1].set_xlim(limits[0])
    axs[0,1].set_ylim(limits[1])
#
# Period vs d(z = 0)
period_mod = []
period_sim = []
dz0_tot = []
for name in summary.host_names['all_no_r']:
    mask = (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][:,0] != -1)*(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][:,0] != -1)
    period_mod.append(np.repeat(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    period_sim.append(np.repeat(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    dz0_tot.append(np.repeat(data_total[name]['d.tot.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
period_mod = np.hstack(period_mod)
period_sim = np.hstack(period_sim)
dz0_tot = np.hstack(dz0_tot)
#
xs = [dz0_tot]
ys = [period_mod-period_sim]
#
xtypes = ['d.z0']
ytypes = ['period.delta']
colors = period_color
#
binsize = 50
limits = ((0,400),(-1.5,2.4))
#
axs[1,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,1].fill_between(binss[:-1]+half_binss, upper, lower, color=colors[i], alpha=0.5)
    axs[1,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[1,1].plot(binss[:-1]+half_binss, med, color=colors[i], markersize=10, alpha=0.7)
    #
    axs[1,1].set_xlim(limits[0])
    axs[1,1].set_ylim(limits[1])
#
# Eccentricity vs d(z = 0)
ecc_mod = []
ecc_sim = []
dz0_tot = []
for name in summary.host_names['all_no_r']:
    mask = (data_total[name]['eccentricity.sim'][masks_infall_peri[name]][:,0] != -1)*(data_total[name]['eccentricity.model.apsis'][masks_infall_peri[name]][:,0] != -1)
    ecc_mod.append(np.repeat(data_total[name]['eccentricity.model.apsis'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    ecc_sim.append(np.repeat(data_total[name]['eccentricity.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    dz0_tot.append(np.repeat(data_total[name]['d.tot.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
ecc_mod = np.hstack(ecc_mod)
ecc_sim = np.hstack(ecc_sim)
dz0_tot = np.hstack(dz0_tot)
#
xs = [dz0_tot]
ys = [ecc_mod-ecc_sim]
#
xtypes = ['d.z0']
ytypes = ['ecc.delta']
colors = ecc_color
#
binsize = 50
limits = ((0,400),(-0.25,0.5))
#
axs[2,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,1].fill_between(binss[:-1]+half_binss, upper, lower, color=colors[i], alpha=0.5)
    axs[2,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[2,1].plot(binss[:-1]+half_binss, med, color=colors[i], markersize=10, alpha=0.7)
    #
    axs[2,1].set_xlim(limits[0])
    axs[2,1].set_ylim(limits[1])
#
# Apocenter distance vs Mstar(z=0)
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_apo, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [Mstar_z0]
ys = [(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim]
#
xtypes = ['M.star.z0']
ytypes = ['delta.dapo.frac']
colors = apo_color
#
binsize = 0.5
limits = ((4,9.5),(-0.25,0.1))
#
axs[0,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=colors[i], alpha=0.5)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[0,2].plot(10**(binss[:-1]+half_binss), med, color=colors[i], markersize=10, alpha=0.7)
    #
    axs[0,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[0,2].set_ylim(limits[1])
    axs[0,2].set_xscale('log')
#
# Period vs Mstar(z=0)
period_mod = []
period_sim = []
Mstar_z0 = []
for name in summary.host_names['all_no_r']:
    mask = (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][:,0] != -1)*(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][:,0] != -1)
    period_mod.append(np.repeat(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    period_sim.append(np.repeat(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    Mstar_z0.append(np.repeat(data_total[name]['M.star.z0'][masks_infall_peri[name]][mask], summary.oversample['baryon'][name]))
period_mod = np.hstack(period_mod)
period_sim = np.hstack(period_sim)
Mstar_z0 = np.hstack(Mstar_z0)
#
xs = [Mstar_z0]
ys = [period_mod-period_sim]
#
xtypes = ['M.star.z0']
ytypes = ['period.delta']
colors = period_color
#
binsize = 0.5
limits = ((4,9.5),(-1.9,1.4))
#
axs[1,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=colors[i], alpha=0.5)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[1,2].plot(10**(binss[:-1]+half_binss), med, color=colors[i], markersize=10, alpha=0.7)
    #
    axs[1,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[1,2].set_ylim(limits[1])
    axs[1,2].set_xscale('log')
#
# Eccentricity vs Mstar(z=0)
ecc_mod = []
ecc_sim = []
Mstar_z0 = []
for name in summary.host_names['all_no_r']:
    mask = (data_total[name]['eccentricity.sim'][masks_infall_peri[name]][:,0] != -1)*(data_total[name]['eccentricity.model.apsis'][masks_infall_peri[name]][:,0] != -1)
    ecc_mod.append(np.repeat(data_total[name]['eccentricity.model.apsis'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    ecc_sim.append(np.repeat(data_total[name]['eccentricity.sim'][masks_infall_peri[name]][:,0][mask], summary.oversample['baryon'][name]))
    Mstar_z0.append(np.repeat(data_total[name]['M.star.z0'][masks_infall_peri[name]][mask], summary.oversample['baryon'][name]))
ecc_mod = np.hstack(ecc_mod)
ecc_sim = np.hstack(ecc_sim)
Mstar_z0 = np.hstack(Mstar_z0)
#
xs = [Mstar_z0]
ys = [ecc_mod-ecc_sim]
#
xtypes = ['M.star.z0']
ytypes = ['ecc.delta']
colors = ecc_color
#
binsize = 0.5
limits = ((4,9.5),(-0.15,0.25))
#
axs[2,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=colors[i], alpha=0.5)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=colors[i], alpha=0.3)
    #
    # Plot the medians for the two mass bins (low-mass)
    axs[2,2].plot(10**(binss[:-1]+half_binss), med, color=colors[i], markersize=10, alpha=0.7)
    #
    axs[2,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[2,2].set_ylim(limits[1])
    axs[2,2].set_xscale('log')
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[2,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=True)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[2,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=True)
axs[0,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[1,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
axs[2,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=True)
#
axs[0,0].set_ylabel('$(d_{\\rm apo,model}-d_{\\rm apo,sim})/d_{\\rm apo,sim}$', fontsize=30)
axs[0,0].get_yaxis().set_label_coords(-0.15,0.5)
axs[1,0].set_ylabel('$T_{\\rm model}-T_{\\rm sim}$ [Gyr]', fontsize=30)
axs[1,0].get_yaxis().set_label_coords(-0.15,0.5)
axs[2,0].set_ylabel('$e_{\\rm model}-e_{\\rm sim}$', fontsize=34)
axs[2,0].get_yaxis().set_label_coords(-0.15,0.5)
#
axs[2,0].set_xlabel('Infall Lookback Time [Gyr]', fontsize=34)
axs[2,1].set_xlabel('Host distance, $r$ [kpc]', fontsize=34)
axs[2,2].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=34)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.18, hspace=0)
#plt.show()
plt.savefig(directory+'/other_orbit_properties.pdf')



"""
    Figure 9:
        Pericenter phase plots
"""
# Plot both of them on the same plot
peri_d_colors = ['#337422']
peri_t_colors = ['#476258', '#624751']
x = []
y = []
# Loop over hosts
for name in summary.host_names['all_no_r']:
    # Loop over satellites
    for i in range(0, len(data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]])):
        # Loop over the phase
        for j in range(0, np.min((len(data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]][i]), len(data_total[name]['pericenter.dist.model'][masks_infall_peri[name]][i])))):
            # Make sure there is an event in both the sim and model
            if (data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]][i][j] != -1) & (data_total[name]['pericenter.dist.model'][masks_infall_peri[name]][i][j] != -1):
                # Save the phase
                x.append(j+1)
                # Save the fractional difference
                y.append((data_total[name]['pericenter.dist.model'][masks_infall_peri[name]][i][j]-data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]][i][j])/data_total[name]['pericenter.dist.sim'][masks_infall_peri[name]][i][j])
x = np.asarray(x)
y = np.asarray(y)
#
# THINK MORE ABOUT WHAT TO REALLY PLOT AGAINST
onesigp = 84.13
onesigm = 15.87
twosigp = 97.72
twosigm = 2.28
thrsigp = 100
thrsigm = 0
meds = np.zeros(np.max(x))
upper = np.zeros(np.max(x))
lower = np.zeros(np.max(x))
highest = np.zeros(np.max(x))
lowest = np.zeros(np.max(x))
highests = np.zeros(np.max(x))
lowests = np.zeros(np.max(x))
for i in range(0, np.max(x)):
    mask = (x == i+1)
    meds[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
    highest[i] = np.nanpercentile(y[mask], twosigp)
    lowest[i] = np.nanpercentile(y[mask], twosigm)
    highests[i] = np.nanpercentile(y[mask], thrsigp)
    lowests[i] = np.nanpercentile(y[mask], thrsigm)
#
f, ax = plt.subplots(2, 1, figsize=(12,14))
gs = gridspec.GridSpec(3, 1, height_ratios=[1,4,4])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
#
xbins = summary_plot.binning_scheme(x, 'N.peri', binsize=1)[0]
ax0.hist(x, bins=xbins, density=False, color='k', alpha=0.35)
#
ax1.scatter(np.arange(np.max(x))+1, meds, s=75., marker='s', c=peri_d_colors[0])
ax1.scatter(np.arange(np.max(x))+1, upper-lower, s=150., marker='*', c='k')
for j in range(0, np.max(x)):
    ax1.errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lowests[j]],[highests[j]-meds[j]]]), alpha=0.3, color=peri_d_colors[0])
    #ax[0].errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lowest[j]],[highest[j]-meds[j]]]), alpha=0.3, color=peri_d_colors[0])
    ax1.errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lower[j]],[upper[j]-meds[j]]]), alpha=0.7, color=peri_d_colors[0])
ax1.hlines(0, -0.5, np.max(x)+1, linestyle='dotted', color='k', alpha=0.5)
#ax[0].hlines(0.69, -0.5, np.max(x)+1, linestyle='solid', color='k', linewidths=2, alpha=0.1)
#ax[0].hlines(1.42, -0.5, np.max(x)+1, linestyle='solid', color='k', linewidths=2, alpha=0.1)
#ax[0].fill_between(x=(-0.5,10), y1=0.69, y2=1.42, color='k', alpha=0.1)
#
x = []
y = []
# Loop over hosts
for name in summary.host_names['all_no_r']:
    # Loop over satellites
    for i in range(0, len(data_total[name]['pericenter.time.lb.sim'][masks_infall_peri[name]])):
        # Loop over the phase
        for j in range(0, np.min((len(data_total[name]['pericenter.time.lb.sim'][masks_infall_peri[name]][i]), len(data_total[name]['pericenter.time.lb.model'][masks_infall_peri[name]][i])))):
            # Make sure there is an event in both the sim and model
            if (data_total[name]['pericenter.time.lb.sim'][masks_infall_peri[name]][i][j] != -1) & (data_total[name]['pericenter.time.lb.model'][masks_infall_peri[name]][i][j] != -1):
                # Save the phase
                x.append(j+1)
                # Save the fractional difference
                y.append(data_total[name]['pericenter.time.lb.model'][masks_infall_peri[name]][i][j]-data_total[name]['pericenter.time.lb.sim'][masks_infall_peri[name]][i][j])
x = np.asarray(x)
y = np.asarray(y)
#
# THINK MORE ABOUT WHAT TO REALLY PLOT AGAINST
onesigp = 84.13
onesigm = 15.87
twosigp = 97.72
twosigm = 2.28
thrsigp = 100
thrsigm = 0
meds = np.zeros(np.max(x))
upper = np.zeros(np.max(x))
lower = np.zeros(np.max(x))
highest = np.zeros(np.max(x))
lowest = np.zeros(np.max(x))
highests = np.zeros(np.max(x))
lowests = np.zeros(np.max(x))
for i in range(0, np.max(x)):
    mask = (x == i+1)
    meds[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
    highest[i] = np.nanpercentile(y[mask], twosigp)
    lowest[i] = np.nanpercentile(y[mask], twosigm)
    highests[i] = np.nanpercentile(y[mask], thrsigp)
    lowests[i] = np.nanpercentile(y[mask], thrsigm)
#
ax2.scatter(np.arange(np.max(x))+1, meds, s=75., marker='s', c=peri_t_colors[1])
ax2.scatter(np.arange(np.max(x))+1, upper-lower, s=150., marker='*', c='k')
for j in range(0, np.max(x)):
    ax2.errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lowests[j]],[highests[j]-meds[j]]]), alpha=0.3, color=peri_t_colors[1])
    #ax[1].errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lowest[j]],[highest[j]-meds[j]]]), alpha=0.3, color=peri_t_colors[1])
    ax2.errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lower[j]],[upper[j]-meds[j]]]), alpha=0.7, color=peri_t_colors[1])
ax2.hlines(0, -0.5, np.max(x)+1, linestyle='dotted', color='k', alpha=0.5)
#ax[1].fill_between(x=(-0.5,10), y1=0.87, y2=8.65, color='k', alpha=0.1)
#
ax0.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax1.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax2.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax0.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax1.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax2.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax0.set_xlim(0, np.max(x)+0.5)
ax1.set_xlim(0, np.max(x)+0.5)
ax2.set_xlim(0, np.max(x)+0.5)
ax1.set_ylim(-1.1, 2.9)
ax2.set_ylim(-3, 3.8)
ax0.set_yscale('log')
ax2.set_xlabel('Lookback $N_{\\rm peri}$', fontsize=28)
ax0.set_ylabel('N', fontsize=22)
ax1.set_ylabel('($d_{\\rm peri,model}-d_{\\rm peri,sim}$)/$d_{\\rm peri,sim}$', fontsize=22)
ax2.set_ylabel('$t_{\\rm peri,model}-t_{\\rm peri,sim}$ [Gyr]', fontsize=22)
ax1.get_yaxis().set_label_coords(-0.1,0.5)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
ax0.tick_params(axis='both', which='both', bottom=True, top=True, labelbottom=False, labelsize=24)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelbottom=False, labelsize=24)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(directory+'/peri_info_vs_phase_zoom.pdf')
plt.close()



"""
    Figure 10:
        Other phase plots
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
gs = gridspec.GridSpec(3, 1, height_ratios=[1,4,4])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
#
ax1.scatter(np.arange(np.max(x))+1, meds, s=75., marker='s', c=peri_t_colors[1])
ax1.scatter(np.arange(np.max(x))+1, upper-lower, s=150., marker='*', c='k')
for j in range(0, np.max(x)):
    ax1.errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lowest[j]],[highest[j]-meds[j]]]), alpha=0.3, color=peri_t_colors[1])
    ax1.errorbar(np.arange(np.max(x))[j]+1, meds[j], yerr=np.array([[meds[j]-lower[j]],[upper[j]-meds[j]]]), alpha=0.7, color=peri_t_colors[1])
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
#xbins = summary_plot.binning_scheme(x/2, 'dperi', binsize=0.25)[0]
ax0.hist(x/2, bins=np.arange(0.25, 9.8, 0.5), density=False, color='k', alpha=0.35)
#
ax2.scatter(x_points-0.5, meds, s=75., marker='s', c=peri_d_colors[0])
ax2.scatter(x_points-0.5, upper-lower, s=150., marker='*', c='k')
for j in range(0, len(x_points)):
    ax2.errorbar(x_points[j]-0.5, meds[j], yerr=np.array([[meds[j]-lowest[j]],[highest[j]-meds[j]]]), alpha=0.3, color=peri_d_colors[0])
    ax2.errorbar(x_points[j]-0.5, meds[j], yerr=np.array([[meds[j]-lower[j]],[upper[j]-meds[j]]]), alpha=0.7, color=peri_d_colors[0])
#
ax1.hlines(0, -0.5, np.max(np.arange(1, x.max()/2+1, 0.5))+1, linestyle='dotted', color='k', alpha=0.5)
ax2.hlines(0, -0.5, np.max(np.arange(1, x.max()/2+1, 0.5))+1, linestyle='dotted', color='k', alpha=0.5)
#
ax0.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax0.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax1.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax1.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax2.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax2.set_xticks(np.arange(0.5, 10.5, 1), minor=True)
ax0.set_xlim(0,10)
ax1.set_xlim(0,10)
ax2.set_xlim(0,10)
ax0.set_yscale('log')
ax2.set_xlabel('Lookback $N_{\\rm peri}$', fontsize=28)
ax0.set_ylabel('N', fontsize=22)
ax1.set_ylabel('$T_{\\rm model} - T_{\\rm sim}$', fontsize=28)
ax2.set_ylabel('$e_{\\rm model} - e_{\\rm sim}$', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.12,0.5)
ax2.get_yaxis().set_label_coords(-0.12,0.5)
#
ax0.tick_params(axis='both', which='both', bottom=True, top=True, labelbottom=False, labelsize=24)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelbottom=False, labelsize=24)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#
plt.savefig(directory+'/period_vs_phase_both.pdf')
plt.close()


"""
    da/dr plots
        - NEED TO FIGURE OUT HOW TO ORGANIZE THESE
          I DON'T WANT 3 INDIVIDUAL PLOTS...
"""
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_no_r')
dadr = summary.da_dr(mass_profile=data_mp, hosts='all_no_r')
#
dadr_max = summary.da_dr_max(data_total, masks_infall_peri, data_mp, dadr, selection='sim', oversample=True, hosts='all_no_r')
dadr_max_mod = summary.da_dr_max(data_total, masks_infall_peri, data_mp, dadr, selection='model', oversample=True, hosts='all_no_r')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# Plotting vs t_infall
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#006400', '#000080']
binedges = None
binsize = 0.5
limits=((0,13.8),(-1,3))
#
x = [t_in_sim]
y = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['t.infall.text']
ytype = ['dadr.frac']
#
medians = []
lowers = []
uppers = []
lowests = []
highests = []
binss = []
half_bins = []
#
for j in range(0, len(x)):
    #
    if binedges:
        bin_num = int((binedges[1]-binedges[0])/binsize + 1)
        bins = np.linspace(binedges[0], binedges[1], bin_num)
        half_bin = (bins[1]-bins[0])/2
    else:
        minn = binsize*np.floor(np.min(x[j])/binsize)
        maxx = binsize*np.ceil(np.max(x[j])/binsize)
        if minn < 0:
            bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
        else:
            bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
        bins = np.linspace(minn, maxx, bin_num)
        half_bin = (bins[1]-bins[0])/2
    #
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
plt.hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], colors='k', linestyles='dotted', alpha=0.5)
#
ax1.set_xlabel('Lookback Infall Time [Gyr]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm model} - |da/dr|_{\\rm sim})/ |da/dr|_{\\rm sim}$', fontsize=20)
#ax1.legend(prop={'size': 21}, loc='best')
ax1.set_xlim(limits[0])
ax1.set_ylim(limits[1])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/dadr_frac_vs_t_infall.pdf')
plt.close()


# Plotting vs t_infall
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#006400', '#000080']
binedges = None
binsize = 50
limits=((0,400),(-1,2.5))
#
x = [dz0_tot]
y = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['d.z0']
ytype = ['dadr.frac']
#
medians = []
lowers = []
uppers = []
lowests = []
highests = []
binss = []
half_bins = []
#
for j in range(0, len(x)):
    #
    if binedges:
        bin_num = int((binedges[1]-binedges[0])/binsize + 1)
        bins = np.linspace(binedges[0], binedges[1], bin_num)
        half_bin = (bins[1]-bins[0])/2
    else:
        minn = binsize*np.floor(np.min(x[j])/binsize)
        maxx = binsize*np.ceil(np.max(x[j])/binsize)
        if minn < 0:
            bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
        else:
            bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
        bins = np.linspace(minn, maxx, bin_num)
        half_bin = (bins[1]-bins[0])/2
    #
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
plt.hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], colors='k', linestyles='dotted', alpha=0.5)
#
ax1.set_xlabel('Host Distance, r [kpc]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm model} - |da/dr|_{\\rm sim})/ |da/dr|_{\\rm sim}$', fontsize=20)
#ax1.legend(prop={'size': 21}, loc='best')
ax1.set_xlim(limits[0])
ax1.set_ylim(limits[1])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/dadr_frac_vs_dz0.pdf')
plt.close()



# Plotting vs t_infall
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#006400', '#000080']
binedges = None
binsize = 0.5
limits=((4,9.5),(-1,1.5))
#
x = [np.log10(Mstar_z0)]
y = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['M.star.z0']
ytype = ['dadr.frac']
#
medians = []
lowers = []
uppers = []
lowests = []
highests = []
binss = []
half_bins = []
#
for j in range(0, len(x)):
    #
    if binedges:
        bin_num = int((binedges[1]-binedges[0])/binsize + 1)
        bins = np.linspace(binedges[0], binedges[1], bin_num)
        half_bin = (bins[1]-bins[0])/2
    else:
        minn = binsize*np.floor(np.min(x[j])/binsize)
        maxx = binsize*np.ceil(np.max(x[j])/binsize)
        if minn < 0:
            bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
        else:
            bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
        bins = np.linspace(minn, maxx, bin_num)
        half_bin = (bins[1]-bins[0])/2
    #
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
plt.hlines(y=0, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
ax1.set_xlabel('$M_{\\rm star}$ [$M_{\\odot}$]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm model} - |da/dr|_{\\rm sim})/ |da/dr|_{\\rm sim}$', fontsize=20)
ax1.set_xscale('log')
ax1.set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
ax1.set_ylim(limits[1])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/dadr_frac_vs_Mstar.pdf')
plt.close()
