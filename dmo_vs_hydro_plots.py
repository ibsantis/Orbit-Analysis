#!/usr/bin/python3

"""
    =========================
    = Summary Plots Paper I =
    =========================

    Compare distributions of properties for DMO vs hydro sims

    NOTE: For right now, I'm only comparing the isolated runs, still need
          to run "summary_data_dmo.py" on the LG runs when they finish
          running their trees.

"""

## Import all of the tools for analysis
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import orbit_io
import summary_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')


# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
#data_total = summary.data_read(directory=sim_data.home_dir, hosts='iso', selection='baryon')
data_total = summary.data_read(directory=sim_data.home_dir, hosts='iso', sim_type='all_baryon')
data_total_dmo = summary.data_read(directory=sim_data.home_dir, hosts='iso', sim_type='dmo')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='iso')
masks_infall_dmo = summary.data_mask(data_total_dmo, peri_sim=False, peri_model=False, hosts='iso')
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1'


### Generate all of the data for the plots below
# Hydro
mask_selection = masks_infall
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='iso', sim_type='baryon_all')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='baryon_all')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='baryon_all')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='iso', sim_type='baryon_all')
#
# DMO
mask_selection = masks_infall_dmo
N_sim_tot_dmo = summary.nperi(data_total_dmo, mask_selection, oversample=True, selection='sim', hosts='iso', sim_type='dmo')
d_sim_tot_dmo = summary.dperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='dmo')
dz0_tot_dmo = summary.d_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
t_sim_tot_dmo = summary.tperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='dmo')
t_in_tot_dmo = summary.first_infall(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, mask_selection, selection='peak', oversample=True, hosts='iso', sim_type='dmo')


### Histograms
# Recent pericenter distances
summary_plot.plot_hist_mult(x=[d_sim_tot, d_sim_tot_dmo], xtype=['d.sim', 'd.sim'], labels=['Hydro', 'DMO'], binsize=50, pdf=True, file_path_and_name=directory+'/dperi_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[d_sim_tot, d_sim_tot_dmo], xtype=['d.sim', 'd.sim'], labels=['Hydro', 'DMO'], binsize=50, pdf=True, xlimits=[-5,400], file_path_and_name=directory+'/dperi_comare_iso_zoom.pdf')

# Recent pericenter times
summary_plot.plot_hist_mult(x=[t_sim_tot, t_sim_tot_dmo], xtype=['t.sim', 't.sim'], labels=['Hydro', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/tperi_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[t_sim_tot, t_sim_tot_dmo], xtype=['t.sim', 't.sim'], labels=['Hydro', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,11], file_path_and_name=directory+'/tperi_comare_iso_zoom.pdf')

# Pericenter number
summary_plot.plot_hist_mult(x=[N_sim_tot, N_sim_tot_dmo], xtype=['N.sim', 'N.sim'], labels=['Hydro', 'DMO'], binsize=1, pdf=True, file_path_and_name=directory+'/Nperi_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[N_sim_tot, N_sim_tot_dmo], xtype=['N.sim', 'N.sim'], labels=['Hydro', 'DMO'], binsize=1, pdf=True, xlimits=[0,11], file_path_and_name=directory+'/Nperi_comare_iso_zoom.pdf')

# Infall times
summary_plot.plot_hist_mult(x=[t_in_tot, t_in_tot_dmo], xtype=['t.infall', 't.infall'], labels=['Hydro', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/t_infall_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[t_in_tot, t_in_tot_dmo], xtype=['t.infall', 't.infall'], labels=['Hydro', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,13.8], file_path_and_name=directory+'/t_infall_comare_iso_zoom.pdf')

# d(z = 0)
summary_plot.plot_hist_mult(x=[dz0_tot, dz0_tot_dmo], xtype=['d.z0', 'd.z0'], labels=['Hydro', 'DMO'], binsize=50, pdf=True, file_path_and_name=directory+'/dz0_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[dz0_tot, dz0_tot_dmo], xtype=['d.z0', 'd.z0'], labels=['Hydro', 'DMO'], binsize=50, pdf=True, xlimits=[0,1000], file_path_and_name=directory+'/dz0_comare_iso_zoom.pdf')

# Mhalo (peak)
summary_plot.plot_hist_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], labels=['Hydro', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/Mhalo_peak_comare_iso.pdf')



### Median plots
# Infall vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall', 't.infall'], labels=['Hydro', 'DMO'], binsize=0.5, file_path_and_name=directory+'/infall_vs_mhalo_peak.pdf')


# d(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[dz0_tot, dz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.z0', 'd.z0'], labels=['Hydro', 'DMO'], binsize=0.5, file_path_and_name=directory+'/dz0_vs_mhalo_peak.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[dz0_tot, dz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.z0', 'd.z0'], labels=['Hydro', 'DMO'], binsize=0.5, limits=(None,(-5,350)), file_path_and_name=directory+'/dz0_vs_mhalo_peak_zoom.pdf')




# Infall vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['Hydro', 'DMO'], binsize=50, file_path_and_name=directory+'/infall_vs_dz0.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['Hydro', 'DMO'], binsize=50, limits=((-5,350),None), file_path_and_name=directory+'/infall_vs_dz0_zoom.pdf')

# Splitting into mass bins
t_in_mask = summary.mass_masking_property(data_total, masks_infall, prop='t.infall', mass_array=[1e8,1e9,1e10], mass_type='Mhalo.peak', oversample=True, hosts='iso', sim_type='baryon_all')
t_in_mask_dmo = summary.mass_masking_property(data_total_dmo, masks_infall_dmo, prop='t.infall', mass_array=[1e8,1e9,1e10], mass_type='Mhalo.peak', oversample=True, hosts='iso', sim_type='dmo')
#
dz0_mask = summary.mass_masking_property(data_total, masks_infall, prop='dz0', mass_array=[1e8,1e9,1e10], mass_type='Mhalo.peak', oversample=True, hosts='iso', sim_type='baryon_all')
dz0_mask_dmo = summary.mass_masking_property(data_total_dmo, masks_infall_dmo, prop='dz0', mass_array=[1e8,1e9,1e10], mass_type='Mhalo.peak', oversample=True, hosts='iso', sim_type='dmo')
#
# Baryon plots
y = t_in_tot_dmo
x = dz0_tot_dmo
binsize = 50
#
minn = int(binsize*np.floor(np.min(x)/binsize))
maxx = int(binsize*np.ceil(np.max(x)/binsize))
bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
bins = np.linspace(minn, maxx, bin_num)
#
half_bin = (bins[1]-bins[0])/2
onesigp = 84.13
onesigm = 15.87
#
med = np.zeros(len(bins)-1)
lower = np.zeros(len(bins)-1)
upper = np.zeros(len(bins)-1)
#
med_low = np.zeros(len(bins)-1)
med_mid = np.zeros(len(bins)-1)
med_high = np.zeros(len(bins)-1)
#
for i in range(0, len(bins)-1):
    mask = (x >= bins[i]) & (x <= bins[i+1])
    med[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
#
for i in range(0, len(bins)-1):
    mask_low = (dz0_mask_dmo['low'] >= bins[i]) & (dz0_mask_dmo['low'] <= bins[i+1])
    med_low[i] = np.nanmedian(t_in_mask_dmo['low'][mask_low])
    mask_mid = (dz0_mask_dmo['mid'] >= bins[i]) & (dz0_mask_dmo['mid'] <= bins[i+1])
    med_mid[i] = np.nanmedian(t_in_mask_dmo['mid'][mask_mid])
    mask_high = (dz0_mask_dmo['high'] >= bins[i]) & (dz0_mask_dmo['high'] <= bins[i+1])
    med_high[i] = np.nanmedian(t_in_mask_dmo['high'][mask_high])
#
f, ax = plt.subplots(figsize=(10, 8))
plt.plot(bins[:-1]+half_bin, med, color='k', alpha=0.5)
plt.fill_between(bins[:-1]+half_bin, upper, lower, color='k', alpha=0.3)
#
plt.plot(bins[:-1]+half_bin, med_low, color=summary_plot.colors[1], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm halo}$ = [8, 9]')
plt.plot(bins[:-1]+half_bin, med_mid, color=summary_plot.colors[2], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm halo}$ = [9, 10]')
plt.plot(bins[:-1]+half_bin, med_high, color=summary_plot.colors[3], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm halo}$ > 11')
#
plt.xlim(-5, 350)
plt.xlabel('d(z = 0) [kpc]', fontsize=28)
plt.ylabel('t$_{\\rm infall,lb}$ [Gyr]', fontsize=28)
plt.legend(prop={'size': 16})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(directory+'/infall_vs_d_z0_mass_bins_dmo_zoom.pdf')
plt.close()
