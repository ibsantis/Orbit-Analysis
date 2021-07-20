#!/usr/bin/python3

"""
    =================
    = Summary Plots =
    =================

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
data_total = summary.data_read(directory=sim_data.home_dir)
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False) # For cases where no satellite is required to have experienced a pericenter
masks_outliers = summary.data_mask(data_total, outliers=True)
masks_either = summary.data_mask(data_total, either=True)
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
mask_selection = masks_either
directory = sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri'


### Generate all of the data for the plots below
# No oversample
delta_N = summary.delta_nperi(data_total, mask_selection, oversample=False)
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=False, selection='sim')
N_model_tot = summary.nperi(data_total, mask_selection, oversample=False, selection='model')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=False)
d_sim_min_tot = summary.dperi_min(data_total, mask_selection, oversample=False)
d_model_tot = summary.dperi_recent(data_total, mask_selection, selection='model', oversample=False)
delta_df_tot = summary.delta_dperi(data_total, mask_selection, fraction=True, oversample=False)
delta_d_tot = summary.delta_dperi(data_total, mask_selection, fraction=False, oversample=False)
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=False)
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=False)
t_sim_min_tot = summary.tperi_min(data_total, mask_selection, oversample=False)
t_model_tot = summary.tperi_recent(data_total, mask_selection, selection='model', oversample=False)
delta_tf_tot = summary.delta_tperi(data_total, mask_selection, fraction=True, oversample=False)
delta_t_tot = summary.delta_tperi(data_total, mask_selection, fraction=False, oversample=False)
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=False)
Mstar_peak_tot = summary.mstar(data_total, mask_selection, selection='peak', oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, mask_selection, selection='z0', oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=False)


# Outliers
delta_N_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
N_sim_tot_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
N_model_tot_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
d_sim_tot_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_sim_min_tot_out = summary.dperi_min(data_total, masks_outliers, oversample=False)
d_model_tot_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
t_sim_tot_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_sim_min_tot_out = summary.tperi_min(data_total, masks_outliers, oversample=False)
t_model_tot_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)


# Oversample
delta_No = summary.delta_nperi(data_total, mask_selection, oversample=True)
N_sim_o_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim')
N_model_o_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='model')
d_sim_o_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True)
d_sim_min_o_tot = summary.dperi_min(data_total, mask_selection, oversample=True)
d_model_o_tot = summary.dperi_recent(data_total, mask_selection, selection='model', oversample=True)
delta_dfo_tot = summary.delta_dperi(data_total, mask_selection, fraction=True, oversample=True)
delta_do_tot = summary.delta_dperi(data_total, mask_selection, fraction=False, oversample=True)
dz0_o_tot = summary.d_z0(data_total, mask_selection, oversample=True)
t_sim_o_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True)
t_sim_min_o_tot = summary.tperi_min(data_total, mask_selection, oversample=True)
t_model_o_tot = summary.tperi_recent(data_total, mask_selection, selection='model', oversample=True)
delta_tfo_tot = summary.delta_tperi(data_total, mask_selection, fraction=True, oversample=True)
delta_to_tot = summary.delta_tperi(data_total, mask_selection, fraction=False, oversample=True)
t_in_o_tot = summary.first_infall(data_total, mask_selection, oversample=True)
Mstar_z0_o_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True)
Mstar_peak_o_tot = summary.mstar(data_total, mask_selection, selection='peak', oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, mask_selection, selection='z0', oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True)


# Plots with minimum pericenter distance instead
delta_df_min_tot = (d_model_tot - d_sim_min_tot)/d_sim_min_tot
delta_df_min_tot_out = (d_model_tot_out - d_sim_min_tot_out)/d_sim_min_tot_out
delta_dfo_min_tot = (d_model_o_tot - d_sim_min_o_tot)/d_sim_min_o_tot
#
# Recent pericenter distance comparison
# no oversample, cases with peris in sim and model, but outliers in red
summary_plot.scatter_plot(x=d_sim_min_tot, y=d_model_tot, x_out=d_sim_min_tot_out, y_out=d_model_tot_out, xtype='d.sim.min', ytype='d.model', limits=(-10,350), file_path_and_name=directory+'/scatter/recent_min_peri_comparison.pdf')
#
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=d_sim_min_o_tot, y=d_model_o_tot, binsize=20, xtype='d.sim.min', ytype='d.model', file_path_and_name=directory+'/median/recent_min_peri_comparison.pdf')



# Minimum vs recent pericenter distance
# no oversample, cases with peris in sim and model, but outliers in red
summary_plot.scatter_plot(x=d_sim_tot, y=d_sim_min_tot, x_out=d_sim_tot_out, y_out=d_sim_min_tot_out, xtype='d.sim', ytype='d.sim.min', limits=(-10,350), file_path_and_name=directory+'/scatter/recent_vs_min_peri_comparison.pdf')
#
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=d_sim_o_tot, y=d_sim_min_o_tot, binsize=20, xtype='d.sim', ytype='d.sim.min', file_path_and_name=directory+'/median/recent_vs_min_peri_comparison.pdf')



# d_peri histograms
summary_plot.plot_hist(x=d_sim_min_o_tot, binsize=10, pdf=True, xtype='d.sim.min', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_peri_sim_min_histogram_pdf.pdf')
summary_plot.plot_hist(x=d_sim_min_tot, binsize=10, pdf=False, xtype='d.sim.min', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_peri_sim_min_histogram.pdf')



# delta d_peri fraction histogram
# oversample, cases with pericenters in sim, but not required in model
summary_plot.plot_hist(delta_dfo_min_tot, binsize=0.1, pdf=True, xtype='delta.d.frac', file_path_and_name=directory+'/histogram/peri_diff_frac_min_histogram.pdf')
summary_plot.plot_hist(delta_dfo_min_tot, binsize=0.1, pdf=True, xlimits=(-1,2), xtype='delta.d.frac', file_path_and_name=directory+'/histogram/peri_diff_frac_min_histogram_zoom.pdf')



# delta d_peri fraction vs d_peri
# no oversample, cases with peris in sim and model, but outliers in red
# Scatter plots
summary_plot.scatter_plot(x=d_sim_min_tot, y=delta_df_min_tot, x_out=d_sim_min_tot_out, y_out=delta_df_min_tot_out, xtype='d.sim.min', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_sim_min.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_df_min_tot, x_out=d_model_tot_out, y_out=delta_df_min_tot_out, xtype='d.model', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_model_min.pdf')
summary_plot.scatter_plot(x=d_sim_min_tot, y=delta_df_min_tot, x_out=d_sim_min_tot_out, y_out=delta_df_min_tot_out, xtype='d.sim.min', ytype='delta.d.frac', limits=((-5,350),(-1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_sim_min_zoom.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_df_min_tot, x_out=d_model_tot_out, y_out=delta_df_min_tot_out, xtype='d.model', ytype='delta.d.frac', limits=((-5,350),(-1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_model_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=d_sim_min_o_tot, y=delta_dfo_min_tot, binsize=50, xtype='d.sim.min', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_d_sim_min.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_dfo_min_tot, binsize=50, xtype='d.model', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_d_model_min.pdf')
summary_plot.median_plot(x=d_sim_min_o_tot, y=delta_dfo_min_tot, binsize=50, xtype='d.sim.min', ytype='delta.d.frac', limits=((0,350),(-1,2.5)), file_path_and_name=directory+'/median/delta_d_frac_vs_d_sim_min_zoom.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_dfo_min_tot, binsize=50, xtype='d.model', ytype='delta.d.frac', limits=((0,350),(-1,2.5)), file_path_and_name=directory+'/median/delta_d_frac_vs_d_model_min_zoom.pdf')



# delta d_peri fraction vs d(z = 0)
# Scatter plots
summary_plot.scatter_plot(x=dz0_tot, y=delta_df_min_tot, x_out=dz0_tot_out, y_out=delta_df_min_tot_out, xtype='d.z0', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_z0_min.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=delta_df_min_tot, x_out=dz0_tot_out, y_out=delta_df_min_tot_out, xtype='d.z0', ytype='delta.d.frac', limits=((-5,350),(-1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_z0_min_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=dz0_o_tot, y=delta_dfo_min_tot, binsize=50, xtype='d.z0', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_d_z0_min.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=delta_dfo_min_tot, binsize=50, xtype='d.z0', ytype='delta.d.frac', limits=((-5,350),(-1,2.5)), file_path_and_name=directory+'/median/delta_d_frac_vs_d_z0_min_zoom.pdf')



# delta_d fraction vs t_peri
# Scatter plots
summary_plot.scatter_plot(x=t_sim_min_tot, y=delta_df_min_tot, x_out=t_sim_min_tot_out, y_out=delta_df_min_tot_out, xtype='t.sim.min', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_sim_min.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_df_min_tot, x_out=t_model_tot_out, y_out=delta_df_min_tot_out, xtype='t.model', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_model_min.pdf')
summary_plot.scatter_plot(x=t_sim_min_tot, y=delta_df_min_tot, x_out=t_sim_min_tot_out, y_out=delta_df_min_tot_out, xtype='t.sim.min', ytype='delta.d.frac', limits=((None),(-1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_sim_min_zoom.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_df_min_tot, x_out=t_model_tot_out, y_out=delta_df_min_tot_out, xtype='t.model', ytype='delta.d.frac', limits=((None),(-1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_model_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=t_sim_min_o_tot, y=delta_dfo_min_tot, binsize=1, xtype='t.sim.min', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_t_sim_min.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_dfo_min_tot, binsize=1, xtype='t.model', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_t_model_min.pdf')



# delta_d fraction vs t_infall
# Scatter plots
summary_plot.scatter_plot(x=t_in_tot, y=delta_df_min_tot, x_out=t_in_tot_out, y_out=delta_df_min_tot_out, xtype='t.infall', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_infall_min.pdf')
summary_plot.scatter_plot(x=t_in_tot, y=delta_df_min_tot, x_out=t_in_tot_out, y_out=delta_df_min_tot_out, xtype='t.infall', ytype='delta.d.frac', limits=(None,(-1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_infall_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=t_in_o_tot, y=delta_dfo_min_tot, binsize=1, xtype='t.infall', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_t_infall_min.pdf')



# delta_d fraction vs N
# Scatter plots
summary_plot.scatter_plot(x=N_sim_tot, y=delta_df_min_tot, x_out=N_sim_tot_out, y_out=delta_df_min_tot_out, xtype='N.sim', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_sim_min.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_df_min_tot, x_out=N_model_tot_out, y_out=delta_df_min_tot_out, xtype='N.model', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_model_min.pdf')
summary_plot.scatter_plot(x=N_sim_tot, y=delta_df_tot, x_out=N_sim_tot_out, y_out=delta_df_tot_out, xtype='N.sim', ytype='delta.d.frac', limits=((-0.5,13.5),(-1,4)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_sim_min_zoom.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_df_tot, x_out=N_model_tot_out, y_out=delta_df_tot_out, xtype='N.model', ytype='delta.d.frac', limits=((-0.5,13.5),(-1,4)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_model_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=N_sim_o_tot, y=delta_dfo_min_tot, binsize=1, xtype='N.sim', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_N_sim_min.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_dfo_min_tot, binsize=1, xtype='N.model', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_N_model_min.pdf')



# delta_d fraction vs Mstar (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_df_min_tot, x_out=Mstar_z0_tot_out, y_out=delta_df_min_tot_out, xtype='M.star.z0', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_z0_min.pdf')
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_df_min_tot, x_out=Mstar_z0_tot_out, y_out=delta_df_min_tot_out, xtype='M.star.z0', ytype='delta.d.frac', limits=(None,(-1.1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_z0_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_dfo_min_tot, binsize=0.5, xtype='M.star.z0', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mstar_z0_min.pdf')



# delta_d fraction vs Mstar (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_df_min_tot, x_out=Mstar_peak_tot_out, y_out=delta_df_min_tot_out, xtype='M.star.peak', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_peak_min.pdf')
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_df_min_tot, x_out=Mstar_peak_tot_out, y_out=delta_df_min_tot_out, xtype='M.star.peak', ytype='delta.d.frac', limits=(None,(-1.1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_peak_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mstar_peak_o_tot, y=delta_dfo_min_tot, binsize=0.5, xtype='M.star.peak', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mstar_peak_min.pdf')



# delta_d fraction vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_df_min_tot, x_out=Mhalo_z0_tot_out, y_out=delta_df_min_tot_out, xtype='M.halo.z0', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_z0_min.pdf')
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_df_min_tot, x_out=Mhalo_z0_tot_out, y_out=delta_df_min_tot_out, xtype='M.halo.z0', ytype='delta.d.frac', limits=(None,(-1.1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_z0_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=delta_dfo_min_tot, binsize=0.5, xtype='M.halo.z0', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mhalo_z0_min.pdf')



# delta_d fraction vs Mhalo (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_df_min_tot, x_out=Mhalo_peak_tot_out, y_out=delta_df_min_tot_out, xtype='M.halo.peak', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_peak_min.pdf')
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_df_min_tot, x_out=Mhalo_peak_tot_out, y_out=delta_df_min_tot_out, xtype='M.halo.peak', ytype='delta.d.frac', limits=(None,(-1.1,5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_peak_min_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=delta_dfo_min_tot, binsize=0.5, xtype='M.halo.peak', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mhalo_peak_min.pdf')
