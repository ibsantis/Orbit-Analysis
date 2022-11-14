#!/usr/bin/python3

"""
    ============================================
    = Paper plots with orbits from aligned ICs =
    ============================================

    Similar figures as in paper_ii_figs_final.py and other checks

"""

## Import all of the tools for analysis
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt
from matplotlib import gridspec
import orbit_io
import summary_io
from scipy import interpolate
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')


# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
#
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', aligned=False)
#
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
#
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False
#
data_total_aligned = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', aligned=True)
#
masks_infall_aligned = summary.data_mask(data_total_aligned, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_aligned = summary.data_mask(data_total_aligned, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_aligned = summary.data_mask_apo(data_total_aligned, hosts='all_no_r')
#
masks_infall_aligned['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_aligned['m12f'][59] = False
masks_infall_apo_aligned['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2_fix/aligned'

"""
    Make a plot of the raw and fractional difference in the recent model pericenter
"""

d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_aligned = summary.dperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(d_rec_mod-d_rec_mod_aligned), xtype='d.model', binsize=0.05, pdf=True, title='Recent model pericenters', x_labels='$d_{\\rm peri,unaligned} - d_{\\rm peri,aligned}$ [kpc]', file_path_and_name=directory+'/model_peri_comp_raw_pdf.pdf')
summary_plot.plot_hist(x=(d_rec_mod-d_rec_mod_aligned), xtype='d.model', binsize=0.05, pdf=True, xlimits=(-0.6,0.6), title='Recent model pericenters', x_labels='$d_{\\rm peri,unaligned} - d_{\\rm peri,aligned}$ [kpc]', file_path_and_name=directory+'/model_peri_comp_raw_pdf_zoom.pdf')
summary_plot.plot_hist(x=(d_rec_mod-d_rec_mod_aligned)/d_rec_mod, xtype='d.model', binsize=0.005, pdf=True, title='Recent model pericenters', x_labels='$(d_{\\rm peri,unaligned} - d_{\\rm peri,aligned})/d_{\\rm peri,unaligned}$', file_path_and_name=directory+'/model_peri_comp_frac_pdf.pdf')
summary_plot.plot_hist(x=(d_rec_mod-d_rec_mod_aligned)/d_rec_mod, xtype='d.model', binsize=0.005, pdf=True, xlimits=(-0.06,0.06), title='Recent model pericenters', x_labels='$(d_{\\rm peri,unaligned} - d_{\\rm peri,aligned})/d_{\\rm peri,unaligned}$', file_path_and_name=directory+'/model_peri_comp_frac_pdf_zoom.pdf')

"""
    Find the min, max, med of the raw and fractional differences between the two models
"""

onesigp = 84.13
onesigm = 15.87

print('The signed minimum between unaligned vs aligned is {0}'.format(np.min(d_rec_mod - d_rec_mod_aligned)))
print('The signed maximum between unaligned vs aligned is {0}'.format(np.max(d_rec_mod - d_rec_mod_aligned)))
print('The signed median between unaligned vs aligned is {0}'.format(np.median(d_rec_mod - d_rec_mod_aligned)))
pos = np.nanpercentile((d_rec_mod - d_rec_mod_aligned), onesigp)
neg = np.nanpercentile((d_rec_mod - d_rec_mod_aligned), onesigm)
print('The signed width of the 68% scatter is {0}'.format(pos-neg))
#
print('The unsigned minimum between unaligned vs aligned is {0}'.format(np.min(np.abs(d_rec_mod - d_rec_mod_aligned))))
print('The unsigned maximum between unaligned vs aligned is {0}'.format(np.max(np.abs(d_rec_mod - d_rec_mod_aligned))))
print('The unsigned median between unaligned vs aligned is {0}'.format(np.median(np.abs(d_rec_mod - d_rec_mod_aligned))))
pos = np.nanpercentile(np.abs(d_rec_mod - d_rec_mod_aligned), onesigp)
neg = np.nanpercentile(np.abs(d_rec_mod - d_rec_mod_aligned), onesigm)
print('The signed width of the 68% scatter is {0}'.format(pos-neg))
#
print('The signed fractional minimum between unaligned vs aligned is {0}'.format(np.min((d_rec_mod - d_rec_mod_aligned)/d_rec_mod)))
print('The signed fractional maximum between unaligned vs aligned is {0}'.format(np.max((d_rec_mod - d_rec_mod_aligned)/d_rec_mod)))
print('The signed fractional median between unaligned vs aligned is {0}'.format(np.median((d_rec_mod - d_rec_mod_aligned)/d_rec_mod)))
pos = np.nanpercentile((d_rec_mod - d_rec_mod_aligned)/d_rec_mod, onesigp)
neg = np.nanpercentile((d_rec_mod - d_rec_mod_aligned)/d_rec_mod, onesigm)
print('The signed width of the 68% scatter is {0}'.format(pos-neg))
#
print('The unsigned fractional minimum between unaligned vs aligned is {0}'.format(np.min(np.abs((d_rec_mod - d_rec_mod_aligned)/d_rec_mod))))
print('The unsigned fractional maximum between unaligned vs aligned is {0}'.format(np.max(np.abs((d_rec_mod - d_rec_mod_aligned)/d_rec_mod))))
print('The unsigned fractional median between unaligned vs aligned is {0}'.format(np.median(np.abs((d_rec_mod - d_rec_mod_aligned)/d_rec_mod))))
pos = np.nanpercentile(np.abs((d_rec_mod - d_rec_mod_aligned)/d_rec_mod), onesigp)
neg = np.nanpercentile(np.abs((d_rec_mod - d_rec_mod_aligned)/d_rec_mod), onesigm)
print('The signed width of the 68% scatter is {0}'.format(pos-neg))



"""
    Re-make some of the same plots in paper_ii_figs_final.py
"""

"""
    Figure 6:
        Infall time comparisons
"""
t_in_sim = summary.first_infall(data_total_aligned, masks_infall_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total_aligned, masks_infall_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_diagnostics(data_total_aligned, masks_infall_aligned, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total_aligned, masks_infall_aligned, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
mask_finite = np.isfinite(t_in_mod)*(t_in_mod != -1)
mask_finite_R200m = np.isfinite(t_in_mod_R200m)*(t_in_mod_R200m != -1)
#
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
axs[0].set_xlabel('A Infall Lookback Time [Gyr]', fontsize=32)
axs[1].set_xlabel('A $M_{\\rm star} [M_{\\odot}]$', fontsize=32)
#plt.suptitle('Aligned')
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)
#plt.show()
plt.savefig(directory+'/infall_vs_property_aligned.pdf')


"""
    Figure 7:
        Pericenter property plots
"""
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(4, 3, figsize=(24,24))
#
# Pericenter distances
d_rec_sim = summary.dperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
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
v_rec_sim = summary.vperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
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
n_sim = summary.nperi(data_total_aligned, masks_infall_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total_aligned, masks_infall_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total_aligned, masks_infall_aligned, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total_aligned, masks_infall_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total_aligned, masks_infall_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_diagnostics(data_total_aligned, masks_infall_aligned, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
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
t_rec_sim = summary.tperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
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
d_rec_sim = summary.dperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total_aligned, masks_infall_peri_aligned, oversample=True, hosts='all_no_r', sim_type='baryon')
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
v_rec_sim = summary.vperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total_aligned, masks_infall_peri_aligned, oversample=True, hosts='all_no_r', sim_type='baryon')
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
n_sim = summary.nperi(data_total_aligned, masks_infall_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total_aligned, masks_infall_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total_aligned, masks_infall_aligned, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total_aligned, masks_infall_aligned, oversample=True, hosts='all_no_r', sim_type='baryon')
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
t_rec_sim = summary.tperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total_aligned, masks_infall_peri_aligned, oversample=True, hosts='all_no_r', sim_type='baryon')
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
d_rec_sim = summary.dperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total_aligned, masks_infall_peri_aligned, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
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
v_rec_sim = summary.vperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total_aligned, masks_infall_peri_aligned, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
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
n_sim = summary.nperi(data_total_aligned, masks_infall_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total_aligned, masks_infall_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total_aligned, masks_infall_aligned, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total_aligned, masks_infall_aligned, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
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
t_rec_sim = summary.tperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total_aligned, masks_infall_peri_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total_aligned, masks_infall_peri_aligned, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
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
axs[3,0].set_xlabel('A Infall Lookback Time [Gyr]', fontsize=32)
axs[3,1].set_xlabel('A Host distance, $r$ [kpc]', fontsize=32)
axs[3,2].set_xlabel('A $M_{\\rm star} [M_{\\odot}]$', fontsize=32)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.17, hspace=0)
#plt.show()
plt.savefig(directory+'/pericenter_properties_aligned.pdf')



"""
    Figure 8:
        Apocenter + Period + eccentricity
"""
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(3, 3, figsize=(28,18))
#
# Apocenter distance vs t_infall
dapo_rec_sim = summary.dapo_recent(data_total_aligned, masks_infall_apo_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total_aligned, masks_infall_apo_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total_aligned, masks_infall_apo_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total_aligned, masks_infall_apo_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
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
    mask = (data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][:,0] != -1)*(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][:,0] != -1)
    period_mod.append(np.repeat(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    period_sim.append(np.repeat(data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    t_in_sim.append(np.repeat(data_total_aligned[name]['first.infall.time.lb'][masks_infall_peri_aligned[name]][mask], summary.oversample['baryon'][name]))
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
    mask = (data_total_aligned[name]['eccentricity.sim'][masks_infall_peri_aligned[name]][:,0] != -1)*(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_peri_aligned[name]][:,0] != -1)
    ecc_mod.append(np.repeat(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    ecc_sim.append(np.repeat(data_total_aligned[name]['eccentricity.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    t_in_sim.append(np.repeat(data_total_aligned[name]['first.infall.time.lb'][masks_infall_peri_aligned[name]][mask], summary.oversample['baryon'][name]))
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
dapo_rec_sim = summary.dapo_recent(data_total_aligned, masks_infall_apo_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total_aligned, masks_infall_apo_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total_aligned, masks_infall_apo_aligned, oversample=True, hosts='all_no_r', sim_type='baryon')
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
    mask = (data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][:,0] != -1)*(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][:,0] != -1)
    period_mod.append(np.repeat(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    period_sim.append(np.repeat(data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    dz0_tot.append(np.repeat(data_total_aligned[name]['d.tot.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
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
    mask = (data_total_aligned[name]['eccentricity.sim'][masks_infall_peri_aligned[name]][:,0] != -1)*(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_peri_aligned[name]][:,0] != -1)
    ecc_mod.append(np.repeat(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    ecc_sim.append(np.repeat(data_total_aligned[name]['eccentricity.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    dz0_tot.append(np.repeat(data_total_aligned[name]['d.tot.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
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
dapo_rec_sim = summary.dapo_recent(data_total_aligned, masks_infall_apo_aligned, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total_aligned, masks_infall_apo_aligned, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total_aligned, masks_infall_apo_aligned, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
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
    mask = (data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][:,0] != -1)*(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][:,0] != -1)
    period_mod.append(np.repeat(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    period_sim.append(np.repeat(data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    Mstar_z0.append(np.repeat(data_total_aligned[name]['M.star.z0'][masks_infall_peri_aligned[name]][mask], summary.oversample['baryon'][name]))
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
    mask = (data_total_aligned[name]['eccentricity.sim'][masks_infall_peri_aligned[name]][:,0] != -1)*(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_peri_aligned[name]][:,0] != -1)
    ecc_mod.append(np.repeat(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    ecc_sim.append(np.repeat(data_total_aligned[name]['eccentricity.sim'][masks_infall_peri_aligned[name]][:,0][mask], summary.oversample['baryon'][name]))
    Mstar_z0.append(np.repeat(data_total_aligned[name]['M.star.z0'][masks_infall_peri_aligned[name]][mask], summary.oversample['baryon'][name]))
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
axs[2,0].set_xlabel('A Infall Lookback Time [Gyr]', fontsize=34)
axs[2,1].set_xlabel('A Host distance, $r$ [kpc]', fontsize=34)
axs[2,2].set_xlabel('A $M_{\\rm star} [M_{\\odot}]$', fontsize=34)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.18, hspace=0)
#plt.show()
plt.savefig(directory+'/other_orbit_properties_aligned.pdf')



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
    for i in range(0, len(data_total_aligned[name]['pericenter.dist.sim'][masks_infall_peri_aligned[name]])):
        # Loop over the phase
        for j in range(0, np.min((len(data_total_aligned[name]['pericenter.dist.sim'][masks_infall_peri_aligned[name]][i]), len(data_total_aligned[name]['pericenter.dist.model'][masks_infall_peri_aligned[name]][i])))):
            # Make sure there is an event in both the sim and model
            if (data_total_aligned[name]['pericenter.dist.sim'][masks_infall_peri_aligned[name]][i][j] != -1) & (data_total_aligned[name]['pericenter.dist.model'][masks_infall_peri_aligned[name]][i][j] != -1):
                # Save the phase
                x.append(j+1)
                # Save the fractional difference
                y.append((data_total_aligned[name]['pericenter.dist.model'][masks_infall_peri_aligned[name]][i][j]-data_total_aligned[name]['pericenter.dist.sim'][masks_infall_peri_aligned[name]][i][j])/data_total_aligned[name]['pericenter.dist.sim'][masks_infall_peri_aligned[name]][i][j])
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
    for i in range(0, len(data_total_aligned[name]['pericenter.time.lb.sim'][masks_infall_peri_aligned[name]])):
        # Loop over the phase
        for j in range(0, np.min((len(data_total_aligned[name]['pericenter.time.lb.sim'][masks_infall_peri_aligned[name]][i]), len(data_total_aligned[name]['pericenter.time.lb.model'][masks_infall_peri_aligned[name]][i])))):
            # Make sure there is an event in both the sim and model
            if (data_total_aligned[name]['pericenter.time.lb.sim'][masks_infall_peri_aligned[name]][i][j] != -1) & (data_total_aligned[name]['pericenter.time.lb.model'][masks_infall_peri_aligned[name]][i][j] != -1):
                # Save the phase
                x.append(j+1)
                # Save the fractional difference
                y.append(data_total_aligned[name]['pericenter.time.lb.model'][masks_infall_peri_aligned[name]][i][j]-data_total_aligned[name]['pericenter.time.lb.sim'][masks_infall_peri_aligned[name]][i][j])
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
ax2.set_xlabel('A Lookback $N_{\\rm peri}$', fontsize=28)
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
plt.savefig(directory+'/peri_info_vs_phase_zoom_aligned.pdf')
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
    for i in range(0, len(data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]])):
        # Loop through the number of orbits that are in the sim AND model
        for j in range(0, np.min((len(data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][i]), len(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][i])))):
            # Check to see if they both have values
            if (data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][i][j] != -1) & (data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][i][j] != -1):
                # Save the orbit phase
                x.append(j+1)
                # Save the difference in the model and sim
                y.append(data_total_aligned[name]['orbit.period.peri.model'][masks_infall_peri_aligned[name]][i][j]-data_total_aligned[name]['orbit.period.peri.sim'][masks_infall_peri_aligned[name]][i][j])
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
ax1.scatter(np.arange(np.max(x))+1, meds, s=75., marker='s', c=peri_t_colors[0])
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
    for i in range(0, len(data_total_aligned[name]['eccentricity.sim'][masks_infall_aligned[name]])):
        # Loop over the phase
        for j in range(0, np.min((len(data_total_aligned[name]['eccentricity.sim'][masks_infall_aligned[name]][i]), len(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_aligned[name]][i])))):
            # Make sure there is an event in both the sim and model
            if (data_total_aligned[name]['eccentricity.sim'][masks_infall_aligned[name]][i][j] != -1) & (data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_aligned[name]][i][j] != -1):
                # Save the difference
                y.append(data_total_aligned[name]['eccentricity.model.apsis'][masks_infall_aligned[name]][i][j]-data_total_aligned[name]['eccentricity.sim'][masks_infall_aligned[name]][i][j])
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
ax2.set_xlabel('A Lookback $N_{\\rm peri}$', fontsize=28)
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
plt.savefig(directory+'/period_vs_phase_both_aligned.pdf')
plt.close()
