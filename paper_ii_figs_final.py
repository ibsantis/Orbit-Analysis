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
    Figure 1:
        Host M200m and R200m
"""
t90_values = np.array([1.27, 0.98, 1.19, 1.45, 1.49, 0.92, 0.55, 2.13, 1.57, 1.08, 1.52, 1.38, 1.89])
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')

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
    onesigp = summary_plot.onesigp
    onesigm = summary_plot.onesigm
    twosigp = summary_plot.twosigp
    twosigm = summary_plot.twosigm
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
#
axs[0].fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color='g', alpha=0.3, zorder=2)
axs[0].fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color='g', alpha=0.15, zorder=1)
axs[0].plot(binss[1][:-1]+half_bins[1], medians[1], 'k', alpha=0.5, lw=4, zorder=3)
#
sigma_one_op = np.nanpercentile(t_in_sim, onesigp)
sigma_one_om = np.nanpercentile(t_in_sim, onesigm)
#
axs[0].vlines(np.median(t_in_sim), 0, 1.06, color='k', alpha=0.2, zorder=4)
axs[0].axvspan(sigma_one_om, sigma_one_op, alpha=0.1, color='k', zorder=0)
#
axis_y2_label = 'Median $M_{\\rm 200m}(t_{\\rm lb}) \ [10^{11} M_{\\odot}]$'
axis_y2_tick_labels = ['14', '12', '10', '8', '6', '4', '2', '0']
axis_y2_tick_values = [float(v) for v in axis_y2_tick_labels]
axis_y2_tick_locations = [1.061, 0.9094, 0.7578, 0.6063, 0.4547, 0.3031, 0.1516, 0.0]
axis_y2_minor_tick_locations = [1.06097825, 1.02308617, 0.98519409, 0.94730201, 0.90940993, 0.87151785, 0.83362577, 0.79573369, 0.7578416 , 0.71994952, 0.68205744, 0.64416536, 0.60627328, 0.5683812 , 0.53048912, 0.49259704, 0.45470496, 0.41681288, 0.3789208 , 0.34102872, 0.30313664, 0.26524456, 0.22735248, 0.1894604 , 0.15156832, 0.11367624, 0.07578416, 0.03789208, 0.]
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
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
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
axs[1].fill_between(binss[3][:-1]+half_bins[3], uppers[3], lowers[3], color='g', alpha=0.3, zorder=2)
axs[1].fill_between(binss[3][:-1]+half_bins[3], highests[3], lowests[3], color='g', alpha=0.15, zorder=1)
axs[1].plot(binss[3][:-1]+half_bins[3], medians[3], 'k', alpha=0.5, lw=4, zorder=3)
axs[1].vlines(np.median(t_in_sim), 0, 1.02, color='k', alpha=0.2, label='Satellite infall times', zorder=4)
axs[1].axvspan(sigma_one_om, sigma_one_op, alpha=0.1, color='k', zorder=0)
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
axis2_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
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
axs[0].text(9.3, 0.97, 'Satellite infall times', fontsize=20)
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
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
#
times = np.around(13.8-mass_m12b['time'], decimals=2)
#
onesigp = summary_plot.onesigp
onesigm = summary_plot.onesigm
#
twosigp = summary_plot.twosigp
twosigm = summary_plot.twosigm
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
#
sigma_one_op = np.nanpercentile(t_in_sim, onesigp)
sigma_one_om = np.nanpercentile(t_in_sim, onesigm)
ax1.vlines(np.median(t_in_sim), -0.05, 1.3, color='k', alpha=0.2, zorder=100)
ax1.axvspan(sigma_one_om, sigma_one_op, alpha=0.1, color='k', zorder=-100)
#
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
axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
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
plt.ylim(0,1.25)
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

inds = np.concatenate((np.arange(len(mass_m12b['time']))[:21][::10], np.arange(len(mass_m12b['time']))[22:39][::2]))
times = np.around(13.8-np.concatenate((mass_m12b['time'][:21][::10], mass_m12b['time'][22:39][::2])), decimals=2)

def color_cycle(cycle_length=len(inds), cmap_name='plasma', low=0, high=0.75):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))
    return colors
colorss = color_cycle(len(inds), cmap_name='plasma', low=0, high=0.75)


plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(8,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
limits_1 = ((5, 500),(0, 1.05))
limits_2 = ((5, 500),(0, 0.099))
#
for i in range(0, len(inds)):
    ax1.plot(rs[1:], mass_prof_all_med[inds[i]], color=colorss[i])
ax1.set_xlim(limits_1[0])
ax1.set_ylim(limits_1[1])
ax1.set_xscale('log')
#
cmap = plt.get_cmap('plasma', len(mass_m12b['time']))
newcmap = ListedColormap(cmap(np.linspace(0, 0.75, len(mass_m12b['time']))))
norm = matplotlib.colors.Normalize(vmin=times[-1], vmax=times[0])
sm = plt.cm.ScalarMappable(cmap=newcmap, norm=norm)
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

ax2.plot(rs[1:], (upper_one-lower_one)/2, color='#6fc4de', linewidth=5, linestyle='solid', alpha=0.8, label='1 $\\sigma$')
ax2.plot(rs[1:], (upper_two-lower_two)/2, color='#6fc4de', linewidth=2, linestyle='dashed', alpha=1, label='2 $\\sigma$')
ax2.set_xlim(limits_2[0])
ax2.set_ylim(limits_2[1])
ax2.set_xscale('log')
ax2.legend(prop={'size': 14}, loc='best', framealpha=1)
#
ax2.set_xlabel('Host distance, $r$ [kpc]', fontsize=26)
ax1.set_ylabel('$M(t_{\\rm lb}, <r)$ / $M(t_{\\rm lb} = 0, <r)$', fontsize=24)
ax2.set_ylabel('$\\sigma(M(t_{\\rm lb} < \\rm 2 Gyr))$', fontsize=20)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.13)
plt.savefig(directory+'/long_short_evolution_sigma.pdf')
plt.close()



"""
    Figure 4:
        Energy trends
"""
data = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', new=True)
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100')
#
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 3, figsize=(30,12))
#
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_energy_new', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy_new', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_energy_new', sim_type='baryon')
sub_energy = summary.energies_new(data_total, masks_infall, data, data_mp, snaps, oversample=True, hosts='all_energy_new')
#
"""
    Components vs infall time
"""
xs = [t_in_sim]
ys = [(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir'])]
#
xtypes = ['t.infall.text']
ytypes = ['E.tot']
peri_d_colors = ['#01033C']
#
binsize = 1
limits = ((0,13.5),(-2.8,0.8))
#
axs[0,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[0,0].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[0,0].set_xlim(limits[0])
    axs[0,0].set_ylim(limits[1])
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0,0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=28)
axz.set_xlim(0,13)
axz.set_xlabel(axis_z_label, fontsize=30, labelpad=9)
axz.tick_params(pad=3)
#
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
L_in = summary.L_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [t_in_tot]
ys = [(L_tot-L_in)/L_tot]
#
xtypes = ['t.infall.text']
ytypes = ['L.frac']
peri_d_colors = ['#6769A8']
#
binsize = 1
limits = ((0,13.5),(-0.8,0.8))
#
axs[1,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[1,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[1,0].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[1,0].set_xlim(limits[0])
    axs[1,0].set_ylim(limits[1])
#
"""
    Components vs d(z = 0)
"""
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_energy_new', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy_new', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_energy_new', sim_type='baryon')
sub_energy = summary.energies_new(data_total, masks_infall, data, data_mp, snaps, oversample=True, hosts='all_energy_new')
#
xs = [dz0_tot]
ys = [(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir'])]
#
xtypes = ['d.z0']
ytypes = ['E.tot']
#
binsize = 50
limits = ((0,400),(-2.8,0.8))
peri_d_colors = ['#01033C']
#
axs[0,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[0,1].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[0,1].set_xlim(limits[0][0], limits[0][1])
    axs[0,1].set_ylim(limits[1])
#
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
L_in = summary.L_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [dz0_tot]
ys = [(L_tot-L_in)/L_tot]
#
xtypes = ['d.z0']
ytypes = ['L.frac']
peri_d_colors = ['#6769A8']
#
binsize = 50
limits = ((0,400),(-0.8,0.8))
#
axs[1,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[1,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[1,1].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[1,1].set_xlim(limits[0][0], limits[0][1])
    axs[1,1].set_ylim(limits[1])
#
"""
    Components vs stellar mass
"""
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_energy_new', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy_new', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_energy_new', sim_type='baryon')
sub_energy = summary.energies_new(data_total, masks_infall, data, data_mp, snaps, oversample=True, hosts='all_energy_new')
#
xs = [Mstar_z0]
ys = [(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir'])]
#
xtypes = ['M.star.z0']
ytypes = ['E.tot']
peri_d_colors = ['#01033C']
#
binsize = 0.5
limits = ((4,9.5),(-2.8,0.8))
#
axs[0,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[0,2].plot(10**(binss[:-1]+half_binss), med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[0,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[0,2].set_ylim(limits[1])
    axs[0,2].set_xscale('log')
#
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
L_in = summary.L_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
#
xs = [Mstar_z0_tot]
ys = [(L_tot-L_in)/L_tot]
#
xtypes = ['M.star.z0']
ytypes = ['L.frac']
peri_d_colors = ['#6769A8']
#
binsize = 0.5
limits = ((4,9.5),(-0.8,0.8))
#
axs[1,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[1,2].plot(10**(binss[:-1]+half_binss), med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[1,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[1,2].set_ylim(limits[1])
    axs[1,2].set_xscale('log')
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
axs[0,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False)
axs[1,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
#
axs[0,0].set_ylabel('($E_0$ -  $E_{\\rm infall}$) / $U_{\\rm 200m,0}$', fontsize=30)
axs[1,0].set_ylabel('($\\ell_0 - \\ell_{\\rm infall}$)/$\\ell_0$', fontsize=32)
#
axs[0,0].get_yaxis().set_label_coords(-0.12,0.5)
axs[0,1].set_ylabel(' ', fontsize=26)
axs[0,1].get_yaxis().set_label_coords(-0.12,0.5)
axs[0,2].set_ylabel(' ', fontsize=26)
axs[0,2].get_yaxis().set_label_coords(-0.12,0.5)
axs[1,0].get_yaxis().set_label_coords(-0.12,0.5)
axs[1,1].set_ylabel(' ', fontsize=26)
axs[1,1].get_yaxis().set_label_coords(-0.12,0.5)
axs[1,2].set_ylabel(' ', fontsize=26)
axs[1,2].get_yaxis().set_label_coords(-0.12,0.5)
#
axs[1,0].set_xlabel('Infall Lookback Time [Gyr]', fontsize=36)
axs[1,1].set_xlabel('Host distance, $r$ [kpc]', fontsize=36)
axs[1,2].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=36)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.04)
plt.savefig(directory+'/energy_and_ell_vs_property.pdf')
plt.close()



"""
    Figure 5:
        E and ell conservation
"""
data = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', new=True)
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100')
tlb = snaps['time'][-1] - np.flip(snaps['time'])
#
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_energy_new', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy_new', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_energy_new', sim_type='baryon')
sub_energy = summary.energies_new(data_total, masks_infall, data, data_mp, snaps, oversample=True, hosts='all_energy_new')

count = 0
lengths = []
for name in summary.host_names['all_energy_new']:
    count += np.sum(masks_infall[name])
    lengths.append(len(sub_energy['energy.all'][name][masks_infall[name]][0]))

all_e_norm = (-1)*np.ones((count, np.max(lengths)))
n = 0
for name in summary.host_names['all_energy_new']:
    for i in range(0, len(sub_energy['energy.all'][name][masks_infall[name]])):
        mask = (sub_energy['energy.all'][name][masks_infall[name]][i] != -1)*np.isfinite(sub_energy['energy.all'][name][masks_infall[name]][i])*(sub_energy['energy.all.tlb'][name][masks_infall[name]][i] < data_total[name]['first.infall.time.lb'][masks_infall[name]][i])
        temp = (sub_energy['energy.all'][name][masks_infall[name]][i][mask] - sub_energy['energy.all'][name][masks_infall[name]][i][mask][0])/sub_energy['E.vir'][i]
        all_e_norm[n][:np.sum(mask)] = temp
        n += 1

all_e_norm_med = (-1)*np.ones(all_e_norm.shape[1])
all_e_norm_mean = (-1)*np.ones(all_e_norm.shape[1])
all_e_norm_upper = (-1)*np.ones(all_e_norm.shape[1])
all_e_norm_lower = (-1)*np.ones(all_e_norm.shape[1])
#
for j in range(0, all_e_norm.shape[1]):
    mask = (all_e_norm[:,j] != -1)*np.isfinite(all_e_norm[:,j])
    if (np.sum(mask) == 0):
        all_e_norm_med[j] = np.nan
        all_e_norm_mean[j] = np.nan
        all_e_norm_upper[j] = np.nan
        all_e_norm_lower[j] = np.nan
    else:
        all_e_norm_med[j] = np.nanmedian(all_e_norm[:,j][mask])
        all_e_norm_mean[j] = np.nanmean(all_e_norm[:,j][mask])
        all_e_norm_upper[j] = np.nanpercentile(all_e_norm[:,j][mask], 84.13)
        all_e_norm_lower[j] = np.nanpercentile(all_e_norm[:,j][mask], 15.87)

count = 0
lengths = []
for name in summary.host_names['all_no_r']:
    count += np.sum(masks_infall[name])
    lengths.append(len(data_total[name]['L.tot.sim'][masks_infall[name]][0]))

all_l_norm = (-1)*np.ones((count, np.max(lengths)))
n = 0
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['L.tot.sim'][masks_infall[name]])):
        time_mask = (tlb[:len(data_total[name]['L.tot.sim'][masks_infall[name]][i])] < data_total[name]['first.infall.time.lb'][masks_infall[name]][i])
        mask = (data_total[name]['L.tot.sim'][masks_infall[name]][i] != -1)*np.isfinite(data_total[name]['L.tot.sim'][masks_infall[name]][i])*(time_mask)
        temp = (data_total[name]['L.tot.sim'][masks_infall[name]][i][mask]-data_total[name]['L.tot.sim'][masks_infall[name]][i][mask][0])/data_total[name]['L.tot.sim'][masks_infall[name]][i][mask][0]
        all_l_norm[n][:np.sum(mask)] = temp
        n += 1

all_l_norm_med = (-1)*np.ones(all_l_norm.shape[1])
all_l_norm_mean = (-1)*np.ones(all_l_norm.shape[1])
all_l_norm_upper = (-1)*np.ones(all_l_norm.shape[1])
all_l_norm_lower = (-1)*np.ones(all_l_norm.shape[1])
#
for j in range(0, all_l_norm.shape[1]):
    mask = (all_l_norm[:,j] != -1)*np.isfinite(all_l_norm[:,j])
    if (np.sum(mask) == 0):
        all_l_norm_med[j] = np.nan
        all_l_norm_mean[j] = np.nan
        all_l_norm_upper[j] = np.nan
        all_l_norm_lower[j] = np.nan
    else:
        all_l_norm_med[j] = np.nanmedian(all_l_norm[:,j][mask])
        all_l_norm_mean[j] = np.nanmean(all_l_norm[:,j][mask])
        all_l_norm_upper[j] = np.nanpercentile(all_l_norm[:,j][mask], 84.13)
        all_l_norm_lower[j] = np.nanpercentile(all_l_norm[:,j][mask], 15.87)

plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 1, figsize=(10,12))
#
tlb = snaps['time'][-1] - np.flip(snaps['time'])
temp_t1 = tlb[:len(all_e_norm_med)]
temp_t1[~np.isfinite(all_e_norm_med)] = np.nan
temp_t2 = tlb[:len(all_e_norm_mean)]
temp_t2[~np.isfinite(all_e_norm_mean)] = np.nan
#
diff = (all_e_norm_upper-all_e_norm_lower)
temp_t3 = tlb[:len(diff)]
temp_t3[~np.isfinite(diff)] = np.nan
#
axs[0].plot(temp_t1, all_e_norm_med, c='#01033C', linewidth=4, alpha=0.7, label='median')
axs[0].plot(temp_t3, diff, c='#01033C', linewidth=2.5, linestyle='dashed', alpha=0.6, label='width of 68%-ile')
axs[0].hlines(0, 0, 13.5, linestyle='dotted', color='k', alpha=0.5)
axs[0].legend(prop={'size': 22}, loc='best')
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=24)
axz.set_xlim(0,13.5)
axz.set_xlabel(axis_z_label, fontsize=24, labelpad=9)
axz.tick_params(pad=3)
#
tlb = snaps['time'][-1] - np.flip(snaps['time'])
temp_t1 = tlb[:len(all_l_norm_med)]
temp_t1[~np.isfinite(all_l_norm_med)] = np.nan
temp_t2 = tlb[:len(all_l_norm_mean)]
temp_t2[~np.isfinite(all_l_norm_mean)] = np.nan
#
diff = (all_l_norm_upper-all_l_norm_lower)
temp_t3 = tlb[:len(diff)]
temp_t3[~np.isfinite(diff)] = np.nan
#
axs[1].plot(temp_t1, all_l_norm_med, c='#6769A8', linewidth=4, alpha=0.7, label='median')
axs[1].plot(temp_t3, diff, c='#6769A8', linewidth=2.5, linestyle='dashed', alpha=0.6, label='width of 68% scatter')
axs[1].hlines(0, 0, 13.8, linestyle='dotted', color='k', alpha=0.5)
#
axs[0].set_ylabel('$(E(t) - E_0) \ / \ U_{\\rm 200m, 0}$', fontsize=30)
axs[0].get_yaxis().set_label_coords(-0.12,0.5)
axs[0].set_ylim(-0.05, 1.5)
axs[0].set_xlim(0,13.5)
#
axs[1].set_ylabel('$(\\ell(t) - \\ell_0) \ / \ \\ell_0$', fontsize=30)
axs[1].get_yaxis().set_label_coords(-0.12,0.5)
axs[1].set_ylim(-0.6, 1.2)
axs[1].set_xlim(0,13.5)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=24, labelbottom=False)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=True)
axs[1].set_xlabel('Lookback time [Gyr]', fontsize=30)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.05)
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/energy_and_ell_conservation.pdf')
plt.close()



"""
    Figure 6:
        Satellite orbits
"""
snap_e = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100')
tlb = snap_e['time'][-1] - np.flip(snap_e['time'])
data = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', selection='sim', new=True)
data_model = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', selection='model')
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_energy_new', new=True)
#
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=False, hosts='all_energy_new', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=False, hosts='all_energy_new', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=False, hosts='all_energy_new', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=False, hosts='all_energy_new', sim_type='baryon')
sub_energy = summary.energies_new(data_total, masks_infall_peri, data, data_mp, snap_e, oversample=False, hosts='all_energy_new', sim_type='baryon')
sub_energy_model = summary.energies_model(data_total, masks_infall_peri, data, data_model, snap_e, oversample=False, hosts='all_energy_new', sim_type='baryon')
#
frac_d = np.abs((d_rec_mod-d_rec_sim)/d_rec_sim)
names = summary.halo_id(data_total, masks_infall_peri, hosts='all_energy_new')
#
#m1 = (frac_d > 0.25)*(frac_d < 0.5)
#halo_index = np.random.randint(low=0, high=np.sum(m1))
#print(names['host'][m1][halo_index], names['ids'][m1][halo_index], frac_d[m1][halo_index])

from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import evaluatePotentials

peri_color, apo_color = '#337422', '#D994F8'

#halo_ids = [28-1, 49-1, 49-1, 68-1] # 11%, 40%, 51%, 95%
halo_ids = [28-1, 34-1, 56-1, 68-1] # 11%, 40%, 55%, 95%
#
#hosts = ['m12w', 'm12f', 'Romeo', 'Louise']
hosts = ['m12w', 'm12b', 'm12f', 'Louise']

# First column
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(4, 4, figsize=(34,22))
#
for i in range(0, len(halo_ids)):
    snaps = data_total[hosts[i]]['time.sim']
    #
    d_model = data_total[hosts[i]]['d.tot.model'][halo_ids[i]]
    v_model = data_total[hosts[i]]['v.tot.model'][halo_ids[i]]
    L_model = np.linalg.norm(data_total[hosts[i]]['L.model'], axis=2)[halo_ids[i]]
    #
    # Set up the distances and times to plot
    d_sim = data_total[hosts[i]]['d.tot.sim'][halo_ids[i]]
    v_sim = data_total[hosts[i]]['v.tot.sim'][halo_ids[i]]
    L_sim = data_total[hosts[i]]['L.tot.sim'][halo_ids[i]]
    #
    """
        Mask the right data and plot
    """
    E_sim = sub_energy['energy.all'][hosts[i]][halo_ids[i]]
    t_E = sub_energy['energy.all.tlb'][hosts[i]][halo_ids[i]]
    E_model = sub_energy_model['energy.all'][hosts[i]][halo_ids[i]]
    t_E_model = sub_energy_model['energy.all.tlb'][hosts[i]][halo_ids[i]]
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
        axs[0,i].plot([], [], ' ', label='$\\Delta d_{\\rm peri}/d_{\\rm peri}$: '+str(np.around(frac_d[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2)))
    else:
        axs[0,i].plot(times, d_sim, 'k')
        axs[0,i].plot(-1*times_model, d_model, alpha=0.5)
        axs[0,i].plot([], [], ' ', label='$\\Delta d_{\\rm peri}/d_{\\rm peri}$: '+str(np.around(frac_d[np.where((halo_ids[i]+1 == names['ids'])&(hosts[i] == names['host']))[0][0]], decimals=2)))
    axs[0,i].plot(times, data_total[hosts[i]]['host.radius'][:len(times)], 'k', alpha=0.3)
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
    axs[0,i].label_outer()
    #
    axis_z_label = 'redshift'
    axis_z_tick_labels = ['0', '0.1', '0.3', '0.6', '1', '2', '3', '6']
    axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
    axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
    axz = axs[0,i].twiny()
    axz.set_xscale('linear')
    axz.set_yscale('linear')
    axz.set_xticks(axis_z_tick_locations)
    axz.set_xticklabels(axis_z_tick_labels, fontsize=28)
    axz.set_xlim(0,13.7)
    axz.set_xlabel(axis_z_label, fontsize=28, labelpad=9)
    axz.tick_params(pad=3)
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
    axs[1,i].label_outer()
    #
    # Plot the angular momentum
    E_vir = (6.67*10**(-11)*2*10**(30))/(1000**3*3.086*10**(16))*data_total[hosts[i]]['host.mass'][0]/data_total[hosts[i]]['host.radius'][0]
    #L_vir = np.sqrt(E_vir)*data_total[hosts[i]]['host.radius'][0]
    axs[2,i].plot(times, (L_sim-L_sim[0])/L_sim[0], 'k')
    axs[2,i].plot(-1*times_model, (L_model-L_model[0])/L_model[0], alpha=0.5)
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
    axs[2,i].label_outer()
    #
    # Plot the energy
    E_vir = (6.67*10**(-11)*2*10**(30))/(1000**3*3.086*10**(16))*data_total[hosts[i]]['host.mass'][0]/data_total[hosts[i]]['host.radius'][0]
    axs[3,i].plot(t_E, (E_sim-E_sim[0])/E_vir, 'k')
    axs[3,i].plot(t_E_model, (E_model-E_model[0])/E_vir, alpha=0.5)
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
    axs[3,i].label_outer()
    #
axs[0,0].set_xlim(0,13.7)
axs[1,0].set_xlim(0,13.7)
axs[2,0].set_xlim(0,13.7)
axs[3,0].set_xlim(0,13.7)
axs[0,1].set_xlim(0,13.7)
axs[1,1].set_xlim(0,13.7)
axs[2,1].set_xlim(0,13.7)
axs[3,1].set_xlim(0,13.7)
axs[0,2].set_xlim(0,13.7)
axs[1,2].set_xlim(0,13.7)
axs[2,2].set_xlim(0,13.7)
axs[3,2].set_xlim(0,13.7)
axs[0,3].set_xlim(0,13.7)
axs[1,3].set_xlim(0,13.7)
axs[2,3].set_xlim(0,13.7)
axs[3,3].set_xlim(0,13.7)
#
axs[0,0].set_ylim(0,450)
axs[0,1].set_ylim(0,400)
axs[0,2].set_ylim(0,300)
axs[0,3].set_ylim(0,400)
axs[1,0].set_ylim(0,330)
axs[1,1].set_ylim(0,290)
axs[1,2].set_ylim(0,440)
axs[1,3].set_ylim(0,299)
#
axs[0,0].legend(prop={'size': 25}, loc='upper right', framealpha=1)
axs[0,1].legend(prop={'size': 25}, loc='upper right', framealpha=1)
axs[0,2].legend(prop={'size': 25}, loc='upper right', framealpha=1)
axs[0,3].legend(prop={'size': 25}, loc='upper right', framealpha=1)
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
axs[0,0].set_ylabel('Host distance, r [kpc]', fontsize=28)
axs[0,0].get_yaxis().set_label_coords(-0.14,0.5)
axs[1,0].set_ylabel('Total Velocity [km s$^{-1}$]', fontsize=28)
axs[1,0].get_yaxis().set_label_coords(-0.14,0.5)
axs[2,0].set_ylabel('$(\\ell(t) - \\ell_0) \ / \ \\ell_0$', fontsize=28)
axs[2,0].get_yaxis().set_label_coords(-0.14,0.5)
axs[3,0].set_ylabel('$(E(t) - E_0) \ / \ U_{\\rm R200m,z=0}$', fontsize=28)
axs[3,0].get_yaxis().set_label_coords(-0.14,0.5)
#
r1 = mpatches.Rectangle(xy=(0.8,375),width=2.64,height=55, facecolor='#D3D3D3', alpha=1, zorder=10)
r2 = mpatches.Rectangle(xy=(0.8,330),width=2.85,height=50, facecolor='#D3D3D3', alpha=1, zorder=10)
r3 = mpatches.Rectangle(xy=(0.8,245),width=2.4,height=40, facecolor='#D3D3D3', alpha=1, zorder=10)
r4 = mpatches.Rectangle(xy=(0.8,330),width=3.12,height=50, facecolor='#D3D3D3', alpha=1, zorder=10)
axs[0,0].text(1.2,388,'Best',fontsize=28, zorder=11)
axs[0,1].text(1.2,345,'Good',fontsize=28, zorder=11)
axs[0,2].text(1.2,255,'Bad',fontsize=28, zorder=11)
axs[0,3].text(1.2,345,'Worst',fontsize=28, zorder=11)
axs[0,0].add_patch(r1)
axs[0,1].add_patch(r2)
axs[0,2].add_patch(r3)
axs[0,3].add_patch(r4)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0)
#
plt.savefig(directory+'/multiplot_orbit_properties_virial.pdf')
plt.close()



"""
    Figure 7:
        Orbit conservation
"""
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100')
tlb = snaps['time'][-1] - np.flip(snaps['time'])
#
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')

count = 0
lengths = []
for name in summary.host_names['all_no_r']:
    count += np.sum(masks_infall[name])
    lengths.append(len(data_total[name]['d.tot.sim'][masks_infall[name]][0]))

all_d_frac = (-1)*np.ones((count, np.max(lengths)))
n = 0
for name in summary.host_names['all_no_r']:
    for i in range(0, len(masks_infall[name])):
        if masks_infall[name][i]:
            mask = (data_total[name]['d.tot.sim'][i] != -1)*np.isfinite(data_total[name]['d.tot.sim'][i] != -1)*(tlb[:len(data_total[name]['d.tot.sim'][i])] < data_total[name]['first.infall.time.lb'][i])
            r_sim = data_total[name]['d.tot.sim'][i][mask]
            r_model = data_total[name]['d.tot.model'][i][:len(r_sim)]
            all_d_frac[n][:np.sum(mask)] = (r_model-r_sim)/r_sim
            n += 1

all_d_frac_med = (-1)*np.ones(all_d_frac.shape[1])
all_d_frac_mean = (-1)*np.ones(all_d_frac.shape[1])
all_d_frac_upper = (-1)*np.ones(all_d_frac.shape[1])
all_d_frac_lower = (-1)*np.ones(all_d_frac.shape[1])
#
for j in range(0, all_d_frac.shape[1]):
    mask = (all_d_frac[:,j] != -1)*np.isfinite(all_d_frac[:,j])
    if (np.sum(mask) == 0):
        all_d_frac_med[j] = np.nan
        all_d_frac_mean[j] = np.nan
        all_d_frac_upper[j] = np.nan
        all_d_frac_lower[j] = np.nan
    else:
        all_d_frac_med[j] = np.nanmedian(all_d_frac[:,j][mask])
        all_d_frac_mean[j] = np.nanmean(all_d_frac[:,j][mask])
        all_d_frac_upper[j] = np.nanpercentile(all_d_frac[:,j][mask], 84.13)
        all_d_frac_lower[j] = np.nanpercentile(all_d_frac[:,j][mask], 15.87)

plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
tlb = snaps['time'][-1] - np.flip(snaps['time'])
temp_t1 = tlb[:len(all_d_frac_med)]
temp_t1[~np.isfinite(all_d_frac_med)] = np.nan
#
diff = (all_d_frac_upper-all_d_frac_lower)
temp_t2 = tlb[:len(diff)]
temp_t2[~np.isfinite(diff)] = np.nan
#
axs.plot(temp_t1, np.abs(all_d_frac_med), c='#2C0E2C', linewidth=4, alpha=0.7, label='median')
axs.plot(temp_t2, diff, c='#2C0E2C', linewidth=2.5, linestyle='dashed', alpha=0.6, label='width of 68%-ile')
axs.hlines(0, 0, 13.5, linestyle='dotted', color='k', alpha=0.5)
axs.legend(prop={'size': 22}, loc='best')
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs.twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=26)
axz.set_xlim(0,13.5)
axz.set_xlabel(axis_z_label, fontsize=28, labelpad=9)
axz.tick_params(pad=3)
#
axs.set_ylabel('$|d_{\\rm model} - d_{\\rm sim}| \ / \ d_{\\rm sim}$', fontsize=30)
axs.set_xlabel('Lookback time [Gyr]', fontsize=30)
axs.get_yaxis().set_label_coords(-0.12,0.5)
axs.set_ylim(-0.1, 1.5)
axs.set_xlim(0,13.5)
#
axs.tick_params(axis='both', which='both', bottom=True, top=False, labelsize=26, labelbottom=True)
#
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/orbit_conservation.pdf')
plt.close()



"""
    Figure 8:
        Infall time comparisons
"""
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_fixed(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
mask_finite = np.isfinite(t_in_mod)*(t_in_mod != -1)
mask_finite_R200m = np.isfinite(t_in_mod_R200m)*(t_in_mod_R200m != -1)

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
limits = ((0,13),(-2,11))
#
axs[0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.3)
    #axs[0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.15)
    #
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
limits = ((4.5,9.5),(-2,12))
#
axs[1].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_d_colors[i], alpha=0.3)
    #axs[1].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_d_colors[i], alpha=0.15)
    #
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
    Figure 9:
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
limits = ((0,13),(-0.5,1.2))
#
axs[0,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
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
limits = ((0,13),(-40,75))
#
axs[1,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_v_colors[i], alpha=0.3)
    axs[1,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_v_colors[i], alpha=0.15)
    #
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
t_in_mod_R200m = summary.infall_fixed(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
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
limits = ((0,13),(-1.1,2.25))
#
axs[2,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_n_colors[i], alpha=0.3)
    axs[2,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_n_colors[i], alpha=0.15)
    #
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
peri_t_colors = ['#a36952']
#
binsize = 1
limits = ((0,13),(-1,0.24))
#
axs[3,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[3,0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_t_colors[i], alpha=0.5)
    axs[3,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_t_colors[i], alpha=0.3)
    #
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
limits = ((0,400),(-0.49,1))
#
axs[0,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
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
limits = ((0,400),(-50,65))
#
axs[1,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_v_colors[i], alpha=0.3)
    axs[1,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_v_colors[i], alpha=0.15)
    #
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
limits = ((0,400),(-1.1,3.5))
#
axs[2,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_n_colors[i], alpha=0.3)
    axs[2,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_n_colors[i], alpha=0.15)
    #
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
limits = ((0,400),(-1,0.3))
#
axs[3,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[3,1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_t_colors[i], alpha=0.5)
    axs[3,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_t_colors[i], alpha=0.3)
    #
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
limits = ((4.5,9.5),(-0.8,1))
#
axs[0,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
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
limits = ((4.5,9.5),(-50,60))
#
axs[1,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_v_colors[i], alpha=0.3)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_v_colors[i], alpha=0.15)
    #
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
limits = ((4.5,9.5),(-1.1,3.5))
#
axs[2,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_n_colors[i], alpha=0.3)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_n_colors[i], alpha=0.15)
    #
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
axs[0,0].get_yaxis().set_label_coords(-0.18,0.5)
axs[1,0].set_ylabel('$v_{\\rm peri,model}-v_{\\rm peri,sim}$ [km s$^{-1}$]', fontsize=26)
axs[1,0].get_yaxis().set_label_coords(-0.18,0.5)
axs[2,0].set_ylabel('$N_{\\rm peri,model}-N_{\\rm peri,sim}$', fontsize=26)
axs[2,0].get_yaxis().set_label_coords(-0.18,0.5)
axs[3,0].set_ylabel('$t_{\\rm peri,model}-t_{\\rm peri,sim}$ [Gyr]', fontsize=26)
axs[3,0].get_yaxis().set_label_coords(-0.18,0.5)
#
axs[3,0].set_xlabel('Infall Lookback Time [Gyr]', fontsize=32)
axs[3,1].set_xlabel('Host distance, $r$ [kpc]', fontsize=32)
axs[3,2].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=32)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.17, hspace=0)
plt.savefig(directory+'/pericenter_properties.pdf')



"""
    Figure 10:
        da/dr plots
            - Takes a long fucking time
"""
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_no_r', new=True)
data_dadr = summary.data_read_dadr(mass_profile=data_mp, hosts='all_no_r')
#
dadr_max = summary.da_dr(data_total, masks_infall_peri, data_mp, data_dadr, selection='sim', oversample=True, hosts='all_no_r')
dadr_max_mod = summary.da_dr(data_total, masks_infall_peri, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 3, figsize=(30,10))
binedges = None
binsize = 1
limits=((0,13.7),(-1,1))
#
xs = [t_in_sim]
ys = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtypes = ['t.infall.text']
ytypes = ['dadr.frac']
#
peri_d_colors = ['#610000']
#
axs[0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[0].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[0].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[0].set_xlim(limits[0])
    axs[0].set_ylim(limits[1])
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.6', '0.3', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=28)
axz.set_xlim(0,13)
axz.set_xlabel(axis_z_label, fontsize=36, labelpad=9)
axz.tick_params(pad=3)
#
binedges = None
binsize = 50
limits=((0,400),(-1,1.5))
#
xs = [dz0_tot]
ys = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtypes = ['d.z0']
ytypes = ['dadr.frac']
#
axs[1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1].fill_between(binss[:-1]+half_binss, upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[1].fill_between(binss[:-1]+half_binss, highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[1].plot(binss[:-1]+half_binss, med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[1].set_xlim(limits[0][0], limits[0][1])
    axs[1].set_ylim(limits[1])
#
binedges = None
binsize = 0.5
limits=((4,9.5),(-1,1.5))
#
xs = [Mstar_z0]
ys = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtypes = ['M.star.z0']
ytypes = ['dadr.frac']
#
axs[2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=peri_d_colors[i], alpha=0.5)
    axs[2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=peri_d_colors[i], alpha=0.3)
    #
    axs[2].plot(10**(binss[:-1]+half_binss), med, color=peri_d_colors[i], markersize=10, alpha=0.7)
    #
    axs[2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[2].set_ylim(limits[1])
    axs[2].set_xscale('log')
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=True)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
axs[2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
#
axs[0].set_ylabel('($|da/dr|_{\\rm model} - |da/dr|_{\\rm sim}) \ / \ |da/dr|_{\\rm sim}$', fontsize=36)
axs[0].get_yaxis().set_label_coords(-0.18,0.5)
axs[1].set_ylabel(' ', fontsize=26)
axs[1].get_yaxis().set_label_coords(-0.12,0.5)
axs[2].set_ylabel(' ', fontsize=26)
axs[2].get_yaxis().set_label_coords(-0.12,0.5)
#
axs[0].set_xlabel('Infall Lookback Time [Gyr]', fontsize=36)
axs[1].set_xlabel('Host distance, $r$ [kpc]', fontsize=36)
axs[2].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=36)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)
plt.savefig(directory+'/dadr_vs_property.pdf')
plt.close()



"""
    Figure 11:
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
onesigp = 84.13
onesigm = 15.87
meds = np.zeros(np.max(x))
upper = np.zeros(np.max(x))
lower = np.zeros(np.max(x))
for i in range(0, np.max(x)):
    mask = (x == i+1)
    meds[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
#
f, ax = plt.subplots(2, 1, figsize=(12,10))
#
ax[0].scatter(np.arange(6)+1, meds[:6], s=100., marker='s', c=peri_d_colors[0])
ax[0].scatter(-100,-100, s=100., marker='s', c='k', label='Model comparison')
ax[0].scatter(np.arange(6)+1, (upper-lower)[:6], s=200., marker='*', c='k', label='Width of 68th percentile')
ax[0].hlines(0, -0.5, np.max(x)+1, linestyle='dotted', color='k', alpha=0.5)
ax[0].legend(prop={'size': 24}, loc='best')
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
onesigp = 84.13
onesigm = 15.87
meds = np.zeros(np.max(x))
upper = np.zeros(np.max(x))
lower = np.zeros(np.max(x))
for i in range(0, np.max(x)):
    mask = (x == i+1)
    meds[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
#
ax[1].scatter(np.arange(6)+1, meds[:6], s=100., marker='s', c=peri_t_colors[1])
ax[1].scatter(np.arange(6)+1, (upper-lower)[:6], s=200., marker='*', c='k')
ax[1].hlines(0, -0.5, np.max(x)+1, linestyle='dotted', color='k', alpha=0.5)
#
ax[0].set_xticks([0,1,2,3,4,5,6])
ax[1].set_xticks([0,1,2,3,4,5,6])
#ax[0].set_xticks(np.arange(0.5, 8.5, 1), minor=True)
#ax[1].set_xticks(np.arange(0.5, 8.5, 1), minor=True)
ax[0].set_xlim(0, 6.5)
ax[1].set_xlim(0, 6.5)
ax[0].set_ylim(-0.25, 1.1)
ax[1].set_ylim(-1.5, 2.5)
ax[1].set_xlabel('Lookback Pericenter Event', fontsize=30)
ax[0].set_ylabel('($d_{\\rm peri,model}-d_{\\rm peri,sim}$)/$d_{\\rm peri,sim}$', fontsize=22)
ax[1].set_ylabel('$t_{\\rm peri,model}-t_{\\rm peri,sim}$ [Gyr]', fontsize=22)
ax[0].get_yaxis().set_label_coords(-0.12,0.5)
ax[1].get_yaxis().set_label_coords(-0.12,0.5)
ax[0].tick_params(axis='both', which='major', bottom=True, top=True, labelbottom=False, labelsize=24)
ax[1].tick_params(axis='both', which='major', bottom=True, top=True, labelsize=24)
ax[0].tick_params(axis='x', which='minor', bottom=False, top=False)
ax[1].tick_params(axis='x', which='minor', bottom=False, top=False)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(directory+'/peri_info_vs_phase_zoom.pdf')
plt.close()



"""
    Figure 12:
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
limits = ((0,13),(-0.2,0.6))
#
axs[0,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,0].fill_between(binss[:-1]+half_binss, upper, lower, color=apo_color[i], alpha=0.5)
    axs[0,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=apo_color[i], alpha=0.3)
    #
    axs[0,0].plot(binss[:-1]+half_binss, med, color=apo_color[i], markersize=10, alpha=0.7)
    #
    axs[0,0].set_xlim(limits[0])
    axs[0,0].set_ylim(limits[1])
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0,0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=26)
axz.set_xlim(0,13)
axz.set_xlabel(axis_z_label, fontsize=30, labelpad=9)
axz.tick_params(pad=3)
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
limits = ((0,13),(-1.8,1.8))
#
axs[1,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,0].fill_between(binss[:-1]+half_binss, upper, lower, color=period_color[i], alpha=0.5)
    axs[1,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=period_color[i], alpha=0.3)
    #
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
limits = ((0,13),(-0.1,0.19))
#
axs[2,0].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,0].fill_between(binss[:-1]+half_binss, upper, lower, color=ecc_color[i], alpha=0.5)
    axs[2,0].fill_between(binss[:-1]+half_binss, highest, lowest, color=ecc_color[i], alpha=0.3)
    #
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
limits = ((0,400),(-0.15,0.25))
#
axs[0,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,1].fill_between(binss[:-1]+half_binss, upper, lower, color=colors[i], alpha=0.5)
    axs[0,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=colors[i], alpha=0.3)
    #
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
limits = ((0,400),(-1.5,1.49))
#
axs[1,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,1].fill_between(binss[:-1]+half_binss, upper, lower, color=colors[i], alpha=0.5)
    axs[1,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=colors[i], alpha=0.3)
    #
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
limits = ((0,400),(-0.19,0.29))
#
axs[2,1].hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,1].fill_between(binss[:-1]+half_binss, upper, lower, color=colors[i], alpha=0.5)
    axs[2,1].fill_between(binss[:-1]+half_binss, highest, lowest, color=colors[i], alpha=0.3)
    #
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
limits = ((4,9.5),(-0.24,0.1))
#
axs[0,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=colors[i], alpha=0.5)
    axs[0,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=colors[i], alpha=0.3)
    #
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
limits = ((4,9.5),(-1.8,1.4))
#
axs[1,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=colors[i], alpha=0.5)
    axs[1,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=colors[i], alpha=0.3)
    #
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
limits = ((4,9.5),(-0.13,0.19))
#
axs[2,2].hlines(y=0, xmin=10**limits[0][0], xmax=10**limits[0][1], linestyle='dotted', color='k', alpha=0.5)
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=binsize)
    med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), upper, lower, color=colors[i], alpha=0.5)
    axs[2,2].fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=colors[i], alpha=0.3)
    #
    axs[2,2].plot(10**(binss[:-1]+half_binss), med, color=colors[i], markersize=10, alpha=0.7)
    #
    axs[2,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
    axs[2,2].set_ylim(limits[1])
    axs[2,2].set_xscale('log')
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=26, labelbottom=False)
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
plt.savefig(directory+'/other_orbit_properties.pdf')


"""
    Figrue 13:
        Summary plots
"""
# Plotting the fractional median offset (values from Table 3 in the paper)
labels = ['$d_{\\rm peri,rec}$', '$d_{\\rm peri,min}$', '$t_{\\rm peri,rec}$', '$N_{\\rm peri}$', '$N_{\\rm peri,fixed}$', '$v_{\\rm peri,rec}$', '$v_{\\rm peri,min}$', '$d_{\\rm apo,rec}$', '$t_{\\rm inf}$', '$t_{\\rm inf,fixed}$', '$e_{\\rm rec}$', '$T_{\\rm rec}$', '$|da/dr|$', '$E_{\\rm inf}$', '$\\ell_{\\rm inf}$']
sum_color = '#D36F5C'
#
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 3, figsize=(30,12))
#
med_offset = np.array([-0.025, 0.066, -0.028, 0.246, 0.259, 0.030, 0.012, -0.013, -0.067, 0.441, 0.005, -0.071, -0.017, np.abs(-0.537), -0.015])
xs = np.arange(1, 16)
sort_mask = np.argsort(med_offset)
axs[0].scatter(xs, med_offset[sort_mask], s=50, c=sum_color, marker='o')
axs[0].hlines(y=0, xmin=xs[0]-0.5, xmax=xs[-1]+0.5, linestyle='dotted', color='k', alpha=0.5)
axs[0].set_xticks(xs, minor=False)
axs[0].set_xticklabels(np.asarray(labels)[sort_mask], rotation=90)
#
# Width of the 68 percent scatter
width_68 = np.array([ 0.424, 1.067, 0.179, 0.486, 0.486, 0.202, 0.330, 0.116, 0.820, 1.088, 0.288, 0.267, 1.320, 1.628, 0.838])
sort_mask = np.argsort(width_68)
axs[1].scatter(xs, width_68[sort_mask], s=50, c=sum_color, marker='o')
axs[1].hlines(y=0, xmin=xs[0]-0.5, xmax=xs[-1]+0.5, linestyle='dotted', color='k', alpha=0.5)
axs[1].set_xticks(xs, minor=False)
axs[1].set_xticklabels(np.asarray(labels)[sort_mask], rotation=90)
#
# Width of the 95 percent scatter
width_95 = np.array([ 2.383, 7.112, 1.290, 1.944, 1.944, 0.747, 0.861, 0.754, 5.338, 5.842, 1.082, 0.820, 8.400, 3.648, 3.359])
sort_mask = np.argsort(width_95)
axs[2].scatter(xs, width_95[sort_mask], s=50, c=sum_color, marker='o')
axs[2].hlines(y=0, xmin=xs[0]-0.5, xmax=xs[-1]+0.5, linestyle='dotted', color='k', alpha=0.5)
axs[2].set_xticks(xs, minor=False)
axs[2].set_xticklabels(np.asarray(labels)[sort_mask], rotation=90)
#
axs[0].tick_params(axis='x', which='minor', bottom=False, top=False)
axs[1].tick_params(axis='x', which='minor', bottom=False, top=False)
axs[2].tick_params(axis='x', which='minor', bottom=False, top=False)
axs[0].tick_params(axis='both', which='major', bottom=True, top=False, labelsize=28)
axs[1].tick_params(axis='both', which='major', bottom=True, top=False, labelsize=28)
axs[2].tick_params(axis='both', which='major', bottom=True, top=False, labelsize=28)
#
axs[0].set_ylabel('Median fractional offset', fontsize=38)
axs[1].set_ylabel('Width of 68% scatter', fontsize=38)
axs[2].set_ylabel('Width of 95% scatter', fontsize=38)
#
axs[0].set_ylim(-0.1, 0.6)
axs[1].set_ylim(bottom=-0.1)
axs[2].set_ylim(bottom=-0.1)
axs[0].set_xlim(xs[0]-0.5, xs[-1]+0.5)
axs[1].set_xlim(xs[0]-0.5, xs[-1]+0.5)
axs[2].set_xlim(xs[0]-0.5, xs[-1]+0.5)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/summary.pdf')
plt.close()



"""
    Figure A1:
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
onesigp = summary_plot.onesigp
onesigm = summary_plot.onesigm
#
twosigp = summary_plot.twosigp
twosigm = summary_plot.twosigm
#
upper_one = np.percentile(mass_ratio, onesigp, axis=0)
lower_one = np.percentile(mass_ratio, onesigm, axis=0)
upper_two = np.percentile(mass_ratio, twosigp, axis=0)
lower_two = np.percentile(mass_ratio, twosigm, axis=0)
#
mass_ratio_med = np.median(mass_ratio, axis=0)
#
plt.figure(figsize=(10,8))
plt.fill_between(rs[1:], upper_two, lower_two, color='#9966cc', alpha=0.15)
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
