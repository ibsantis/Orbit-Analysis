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
masks_1 = summary.data_mask(data_total) # for cases where there are pericenters in sim, but not required in model
masks_2 = summary.data_mask(data_total, peri_model=True) # for cases where there are pericenters in both sim AND model
masks_3 = summary.data_mask(data_total, peri_sim=False, peri_model=False) # For cases where no satellite is required to have experienced a pericenter
masks_outliers = summary.data_mask(data_total, outliers=True)
masks_either = summary.data_mask(data_total, either=True)
summary_plot = summary_io.SummaryDataPlot()



# N histogram
N_sim_o_tot = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
summary_plot.plot_hist(x=N_sim_o_tot, binsize=1, xtype='N.sim', pdf=True, xlimits=(0,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/N_peri_sim_histogram_pdf.pdf')
N_sim_tot = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
summary_plot.plot_hist(x=N_sim_tot, binsize=1, xtype='N.sim', pdf=False, xlimits=(0,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/N_peri_sim_histogram.pdf')
#
N_model_o_tot = summary.nperi(data_total, masks_either, oversample=True, selection='model')
summary_plot.plot_hist(x=N_model_o_tot, binsize=1, xtype='N.model', pdf=True, xlimits=(0,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/N_peri_model_histogram_pdf.pdf')
N_model_tot = summary.nperi(data_total, masks_either, oversample=False, selection='model')
summary_plot.plot_hist(x=N_model_tot, binsize=1, xtype='N.model', pdf=False, xlimits=(0,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/N_peri_model_histogram.pdf')



# Delta N histogram
delta_No = summary.delta_nperi(data_total, masks_either)
delta_N = summary.delta_nperi(data_total, masks_either, oversample=False)
#
summary_plot.plot_hist(x=delta_No, binsize=1, xtype='N.delta', pdf=True, xlimits=(-5,5), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/delta_N_peri_histogram_pdf.pdf')
summary_plot.plot_hist(x=delta_N, binsize=1, xtype='N.delta', pdf=False, xlimits=(-5,5), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/delta_N_peri_histogram.pdf')




# Delta N vs N
#
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_either, oversample=False)
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
#
summary_plot.delta_nperi_vs_prop_scatter(x=n_peri_sim2, y=delta_N2, x_out=n_peri_sim2_out, y_out=delta_N2_out, versus='N.sim', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_N_sim.pdf')
summary_plot.delta_nperi_vs_prop_scatter(x=n_peri_galpy2, y=delta_N2, x_out=n_peri_galpy2_out, y_out=delta_N2_out, versus='N.model', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_N_model.pdf')
#
# oversample, cases with peris in sim, but not required in model
delta_N1 = summary.delta_nperi(data_total, masks_either, oversample=True)
n_peri_o_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_o_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
#
summary_plot.delta_nperi_vs_prop_median(x=n_peri_o_sim1, y=delta_N1, versus='N.sim', binsize=None, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_N_sim.pdf')
summary_plot.delta_nperi_vs_prop_median(x=n_peri_o_galpy1, y=delta_N1, versus='N.model', binsize=None, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_N_model.pdf')



# delta_N vs d_peri
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_either, oversample=False)
d_tot_sim2 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
d_tot_galpy2 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
d_tot_sim2_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_tot_galpy2_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=d_tot_sim2, y=delta_N2, x_out=d_tot_sim2_out, y_out=delta_N2_out, versus='d.sim', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_d_sim.pdf')
summary_plot.delta_nperi_vs_prop_scatter(x=d_tot_galpy2, y=delta_N2, x_out=d_tot_galpy2_out, y_out=delta_N2_out, versus='d.model', limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_d_model_zoom.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_either, oversample=True)
d_tot_sim1 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
d_tot_galpy1 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=d_tot_sim1, y=delta_N1, binsize=50, versus='d.sim', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_d_sim.pdf')
summary_plot.delta_nperi_vs_prop_median(x=d_tot_galpy1, y=delta_N1, binsize=50, versus='d.model', limits=((-5,350), None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_d_model_zoom.pdf')



# delta_N vs d(z = 0)
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_either, oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=dz0_tot, y=delta_N2, x_out=dz0_tot_out, y_out=delta_N2_out, versus='d.z0', limits=((-5,350), None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_dz0_zoom.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_either, oversample=True)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=dz0_tot, y=delta_N1, binsize=50, versus='d.z0', limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_dz0_zoom.pdf')



# delta_N vs t_peri
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_either, oversample=False)
t_tot_sim2 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
t_tot_galpy2 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
t_tot_sim2_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_tot_galpy2_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=t_tot_sim2, y=delta_N2, x_out=t_tot_sim2_out, y_out=delta_N2_out, versus='t.sim', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_t_sim.pdf')
summary_plot.delta_nperi_vs_prop_scatter(x=t_tot_galpy2, y=delta_N2, x_out=t_tot_galpy2_out, y_out=delta_N2_out, versus='t.model', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_t_model.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_either, oversample=True)
t_tot_sim1 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
t_tot_galpy1 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=t_tot_sim1, y=delta_N1, binsize=1, versus='t.sim', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_t_sim.pdf')
summary_plot.delta_nperi_vs_prop_median(x=t_tot_galpy1, y=delta_N1, binsize=1, versus='t.model', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_t_model.pdf')



# delta_N vs t_infall
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_either, oversample=False)
t_in_tot = summary.first_infall(data_total, masks_either,oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=t_in_tot, y=delta_N2, x_out=t_in_tot_out, y_out=delta_N2_out, versus='t.infall', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_t_infall.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_either, oversample=True)
t_in_tot = summary.first_infall(data_total, masks_either, oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=t_in_tot, y=delta_N1, binsize=1, versus='t.infall', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_t_infall.pdf')



# delta_N vs Mstar (z = 0)
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_either, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=Mstar_z0_tot, y=delta_N2, x_out=Mstar_z0_tot_out, y_out=delta_N2_out, versus='M.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_mstar_z0.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_either, oversample=True)
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=Mstar_z0_tot, y=delta_N1, binsize=0.5, versus='M.star.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_mstar_z0.pdf')



# delta_N vs Mstar (peak)
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_either, oversample=False)
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=Mstar_peak_tot, y=delta_N2, x_out=Mstar_peak_tot_out, y_out=delta_N2_out, versus='M.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_N_vs_mstar_peak.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_either, oversample=True)
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=Mstar_peak_tot, y=delta_N1, binsize=0.5, versus='M.star.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_N_vs_mstar_peak.pdf')



# delta_N vs Mhalo (z = 0)
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_1, oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, masks_1, selection='z0', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=Mhalo_z0_tot, y=delta_N2, x_out=Mhalo_z0_tot_out, y_out=delta_N2_out, versus='M.halo.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_peri_sim/scatter/delta_N_vs_mhalo_z0.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_1, oversample=True)
Mhalo_z0_tot = summary.mhalo(data_total, masks_1, selection='z0', oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=Mhalo_z0_tot, y=delta_N1, binsize=0.5, versus='M.halo.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_peri_sim/median/delta_N_vs_mhalo_z0.pdf')



# delta_N vs Mhalo (peak)
# Scatter plots
delta_N2 = summary.delta_nperi(data_total, masks_3, oversample=False)
delta_N2_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_nperi_vs_prop_scatter(x=Mhalo_peak_tot, y=delta_N2, x_out=Mhalo_peak_tot_out, y_out=delta_N2_out, versus='M.halo.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/delta_N_vs_mhalo_peak.pdf')
#
# Median plots
delta_N1 = summary.delta_nperi(data_total, masks_3, oversample=True)
Mhalo_peak_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=True)
#
summary_plot.delta_nperi_vs_prop_median(x=Mhalo_peak_tot, y=delta_N1, binsize=0.5, versus='M.halo.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/delta_N_vs_mhalo_peak.pdf')



# N vs d_peri
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
d_tot_sim2 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
d_tot_galpy2 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
d_tot_sim2_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_tot_galpy2_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=d_tot_sim2, y=n_peri_sim2, x_out=d_tot_sim2_out, y_out=n_peri_sim2_out, xtype='d.sim', ytype='N.sim', limits=((-5,350),(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_d_sim_zoom.pdf')
summary_plot.nperi_vs_prop_scatter(x=d_tot_galpy2, y=n_peri_galpy2, x_out=d_tot_galpy2_out, y_out=n_peri_galpy2_out, xtype='d.model', ytype='N.model', limits=((-5,350),(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_d_model_zoom.pdf')
summary_plot.nperi_vs_prop_scatter(x=d_tot_sim2, y=n_peri_galpy2, x_out=d_tot_sim2_out, y_out=n_peri_galpy2_out, xtype='d.sim', ytype='N.model', limits=((-5,350),(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_d_sim_zoom.pdf')
summary_plot.nperi_vs_prop_scatter(x=d_tot_galpy2, y=n_peri_sim2, x_out=d_tot_galpy2_out, y_out=n_peri_sim2_out, xtype='d.model', ytype='N.sim', limits=((-5,350),(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_d_model_zoom.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
d_tot_sim1 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
d_tot_galpy1 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.nperi_vs_prop_median(x=d_tot_sim1, y=n_peri_sim1, xtype='d.sim', ytype='N.sim', binsize=50, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_d_sim_zoom.pdf')
summary_plot.nperi_vs_prop_median(x=d_tot_galpy1, y=n_peri_galpy1, xtype='d.model', ytype='N.model', binsize=50, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_d_model_zoom.pdf')
summary_plot.nperi_vs_prop_median(x=d_tot_sim1, y=n_peri_galpy1, xtype='d.sim', ytype='N.model', binsize=50, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_d_sim_zoom.pdf')
summary_plot.nperi_vs_prop_median(x=d_tot_galpy1, y=n_peri_sim1, xtype='d.model', ytype='N.sim', binsize=50, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_d_model_zoom.pdf')



# N vs d(z = 0)
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=dz0_tot, y=n_peri_sim2, x_out=dz0_tot_out, y_out=n_peri_sim2_out, xtype='d.z0', ytype='N.sim', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_dz0.pdf')
summary_plot.nperi_vs_prop_scatter(x=dz0_tot, y=n_peri_galpy2, x_out=dz0_tot_out, y_out=n_peri_galpy2_out, xtype='d.z0', ytype='N.model', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_dz0.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.nperi_vs_prop_median(x=dz0_o_tot, y=n_peri_sim1, xtype='d.z0', ytype='N.sim', binsize=50, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_dz0_zoom.pdf')
summary_plot.nperi_vs_prop_median(x=dz0_o_tot, y=n_peri_galpy1, xtype='d.z0', ytype='N.model', binsize=50, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_dz0_zoom.pdf')



# N vs t_peri
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
t_tot_sim2 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
t_tot_galpy2 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
t_tot_sim2_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_tot_galpy2_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=t_tot_sim2, y=n_peri_sim2, x_out=t_tot_sim2_out, y_out=n_peri_sim2_out, xtype='t.sim', ytype='N.sim', limits=((None), (-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_t_sim.pdf')
summary_plot.nperi_vs_prop_scatter(x=t_tot_galpy2, y=n_peri_galpy2, x_out=t_tot_galpy2_out, y_out=n_peri_galpy2_out, xtype='t.model', ytype='N.model', limits=((None), (-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_t_model.pdf')
summary_plot.nperi_vs_prop_scatter(x=t_tot_sim2, y=n_peri_galpy2, x_out=t_tot_sim2_out, y_out=n_peri_galpy2_out, xtype='t.sim', ytype='N.model', limits=((None), (-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_t_sim.pdf')
summary_plot.nperi_vs_prop_scatter(x=t_tot_galpy2, y=n_peri_sim2, x_out=t_tot_galpy2_out, y_out=n_peri_sim2_out, xtype='t.model', ytype='N.sim', limits=((None), (-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_t_model.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
t_tot_sim1 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
t_tot_galpy1 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.nperi_vs_prop_median(x=t_tot_sim1, y=n_peri_sim1, xtype='t.sim', ytype='N.sim', binsize=1, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_t_sim.pdf')
summary_plot.nperi_vs_prop_median(x=t_tot_galpy1, y=n_peri_galpy1, xtype='t.model', ytype='N.model', binsize=1, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_t_model.pdf')
summary_plot.nperi_vs_prop_median(x=t_tot_sim1, y=n_peri_galpy1, xtype='t.sim', ytype='N.model', binsize=1, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_t_sim.pdf')
summary_plot.nperi_vs_prop_median(x=t_tot_galpy1, y=n_peri_sim1, xtype='t.model', ytype='N.sim', binsize=1, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_t_model.pdf')



# N vs t_infall
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=t_in_tot, y=n_peri_sim2, x_out=t_in_tot_out, y_out=n_peri_sim2_out, xtype='t.infall', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_t_infall.pdf')
summary_plot.nperi_vs_prop_scatter(x=t_in_tot, y=n_peri_galpy2, x_out=t_in_tot_out, y_out=n_peri_galpy2_out, xtype='t.infall', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_t_infall.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
t_in_tot = summary.first_infall(data_total, masks_either, oversample=True)
#
summary_plot.nperi_vs_prop_median(x=t_in_tot, y=n_peri_sim1, xtype='t.infall', ytype='N.sim', binsize=1, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_t_infall.pdf')
summary_plot.nperi_vs_prop_median(x=t_in_tot, y=n_peri_galpy1, xtype='t.infall', ytype='N.model', binsize=1, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_t_infall.pdf')



# N vs Mstar (z = 0)
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=Mstar_z0_tot, y=n_peri_sim2, x_out=Mstar_z0_tot_out, y_out=n_peri_sim2_out, xtype='M.z0', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_mstar_z0.pdf')
summary_plot.nperi_vs_prop_scatter(x=Mstar_z0_tot, y=n_peri_galpy2, x_out=Mstar_z0_tot_out, y_out=n_peri_galpy2_out, xtype='M.z0', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_mstar_z0.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
#
summary_plot.nperi_vs_prop_median(x=Mstar_z0_tot, y=n_peri_sim1, xtype='M.star.z0', ytype='N.sim', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_mstar_z0.pdf')
summary_plot.nperi_vs_prop_median(x=Mstar_z0_tot, y=n_peri_galpy1, xtype='M.star.z0', ytype='N.model', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_mstar_z0.pdf')



# N vs Mstar (peak)
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=Mstar_peak_tot, y=n_peri_sim2, x_out=Mstar_peak_tot_out, y_out=n_peri_sim2_out, xtype='M.peak', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_sim_vs_mstar_peak.pdf')
summary_plot.nperi_vs_prop_scatter(x=Mstar_peak_tot, y=n_peri_galpy2, x_out=Mstar_peak_tot_out, y_out=n_peri_galpy2_out, xtype='M.peak', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/N_model_vs_mstar_peak.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.nperi_vs_prop_median(x=Mstar_peak_tot, y=n_peri_sim1, xtype='M.star.peak', ytype='N.sim', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_sim_vs_mstar_peak.pdf')
summary_plot.nperi_vs_prop_median(x=Mstar_peak_tot, y=n_peri_galpy1, xtype='M.star.peak', ytype='N.model', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/N_model_vs_mstar_peak.pdf')



# N vs Mhalo (z = 0)
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_3, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_3, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=Mhalo_z0_tot, y=n_peri_sim2, x_out=Mhalo_z0_tot_out, y_out=n_peri_sim2_out, xtype='M.halo.z0', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/N_sim_vs_mhalo_z0.pdf')
summary_plot.nperi_vs_prop_scatter(x=Mhalo_z0_tot, y=n_peri_galpy2, x_out=Mhalo_z0_tot_out, y_out=n_peri_galpy2_out, xtype='M.halo.z0', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/N_model_vs_mhalo_z0.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_3, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_3, oversample=True, selection='model')
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=True)
#
summary_plot.nperi_vs_prop_median(x=Mhalo_z0_tot, y=n_peri_sim1, xtype='M.halo.z0', ytype='N.sim', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/N_sim_vs_mhalo_z0.pdf')
summary_plot.nperi_vs_prop_median(x=Mhalo_z0_tot, y=n_peri_galpy1, xtype='M.halo.z0', ytype='N.model', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/N_model_vs_mhalo_z0.pdf')



# N vs Mhalo (peak)
# Scatter plots
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
Mhalo_peak_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.nperi_vs_prop_scatter(x=Mhalo_peak_tot, y=n_peri_sim2, x_out=Mhalo_peak_tot_out, y_out=n_peri_sim2_out, xtype='M.halo.peak', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/scatter/N_sim_vs_mhalo_peak.pdf')
summary_plot.nperi_vs_prop_scatter(x=Mhalo_peak_tot, y=n_peri_galpy2, x_out=Mhalo_peak_tot_out, y_out=n_peri_galpy2_out, xtype='M.halo.peak', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/scatter/N_model_vs_mhalo_peak.pdf')
#
# Median plots
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
Mhalo_peak_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.nperi_vs_prop_median(x=Mhalo_peak_tot, y=n_peri_sim1, xtype='M.halo.peak', ytype='N.sim', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/median/N_sim_vs_mhalo_peak.pdf')
summary_plot.nperi_vs_prop_median(x=Mhalo_peak_tot, y=n_peri_galpy1, xtype='M.halo.peak', ytype='N.model', binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/median/N_model_vs_mhalo_peak.pdf')



# Recent pericenter distance comparison
# no oversample, cases with peris in sim and model, but outliers in red
d_tot_sim2 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
d_tot_sim2_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_tot_galpy2 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
d_tot_galpy2_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.dperi_comparison_scatter(x=d_tot_sim2, y=d_tot_galpy2, x_out=d_tot_sim2_out, y_out=d_tot_galpy2_out, limits=(-10,350), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/recent_peri_comparison.pdf')
#
# oversample, cases with peris in sim, but not required in model
d_tot_sim1 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
d_tot_galpy1 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.dperi_comparison_median(x=d_tot_sim1, y=d_tot_galpy1, binsize=20, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/recent_peri_comparison.pdf')



# d_peri histograms
d_tot_o_sim = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
summary_plot.plot_hist(x=d_tot_o_sim, binsize=10, pdf=True, xtype='d.sim', xlimits=(-5,350), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_peri_sim_histogram_pdf.pdf')
d_tot_sim = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
summary_plot.plot_hist(x=d_tot_sim, binsize=10, pdf=False, xtype='d.sim', xlimits=(-5,350), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_peri_sim_histogram.pdf')
#
d_tot_o_model = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
summary_plot.plot_hist(x=d_tot_o_model, binsize=10, pdf=True, xtype='d.model', xlimits=(-5,350), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_peri_model_histogram_pdf.pdf')
d_tot_model = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
summary_plot.plot_hist(x=d_tot_model, binsize=10, pdf=False, xtype='d.model', xlimits=(-5,350), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_peri_model_histogram.pdf')



# delta d_peri fraction histogram
# oversample, cases with pericenters in sim, but not required in model
delta_dfo_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
summary_plot.delta_dperi_hist(delta_dfo_tot, binsize=0.1, fraction=True, pdf=True, xlimits=(-1,2), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/peri_diff_frac_histogram_zoom.pdf')



# delta d_peri fraction vs d_peri
# no oversample, cases with peris in sim and model, but outliers in red
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
d_tot_sim2 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
d_tot_galpy2 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
d_tot_sim2_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_tot_galpy2_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=d_tot_sim2, y=delta_df_tot, x_out=d_tot_sim2_out, y_out=delta_df_tot_out, versus='d.sim', limits=((-5,350),(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_d_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_scatter(x=d_tot_galpy2, y=delta_df_tot, x_out=d_tot_galpy2_out, y_out=delta_df_tot_out, versus='d.model', limits=((-5,350),(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_d_model_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
d_tot_sim1 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
d_tot_galpy1 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=d_tot_sim1, y=delta_dfo_tot1, binsize=50, versus='d.sim', fraction=True, limits=((0,350),(-1,1.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_d_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_median(x=d_tot_galpy1, y=delta_dfo_tot1, binsize=50, versus='d.model', fraction=True, limits=((0,350),(-1,1.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_d_model_zoom.pdf')



# delta d_peri fraction vs d(z = 0)
# Scatter plots
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=dz0_tot, y=delta_df_tot, x_out=dz0_tot_out, y_out=delta_df_tot_out, versus='d.z0', limits=((-5,350),(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_d_z0_zoom.pdf')
#
# Median plots
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=dz0_o_tot, y=delta_dfo_tot1, binsize=50, versus='d.z0', fraction=True, limits=((-5,350),(-1.1,1.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_d_z0_zoom.pdf')



# delta_d fraction vs t_peri
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
t_tot_sim2 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
t_tot_galpy2 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
t_tot_sim2_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_tot_galpy2_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=t_tot_sim2, y=delta_df_tot, x_out=t_tot_sim2_out, y_out=delta_df_tot_out, versus='t.sim', limits=((None),(-1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_t_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_scatter(x=t_tot_galpy2, y=delta_df_tot, x_out=t_tot_galpy2_out, y_out=delta_df_tot_out, versus='t.model', limits=((None),(-1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_t_model_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
t_tot_sim1 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
t_tot_galpy1 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=t_tot_sim1, y=delta_dfo_tot1, binsize=1, versus='t.sim', fraction=True, limits=((None),(-1,4)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_t_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_median(x=t_tot_galpy1, y=delta_dfo_tot1, binsize=1, versus='t.model', fraction=True, limits=((None),(-1,4)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_t_model_zoom.pdf')



# delta_d fraction vs t_infall
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=t_in_tot, y=delta_df_tot, x_out=t_in_tot_out, y_out=delta_df_tot_out, versus='t.infall', limits=(None,(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_t_infall_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=t_in_o_tot, y=delta_dfo_tot1, binsize=1, versus='t.infall', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_t_infall.pdf')



# delta_d fraction vs N
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=n_peri_sim2, y=delta_df_tot, x_out=n_peri_sim2_out, y_out=delta_df_tot_out, versus='N.sim', limits=((-0.5,13.5),(-1,3)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_N_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_scatter(x=n_peri_galpy2, y=delta_df_tot, x_out=n_peri_galpy2_out, y_out=delta_df_tot_out, versus='N.model', limits=((-0.5,13.5),(-1,3)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_N_model_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
n_peri_sim1 = summary.nperi(data_total, masks_either, oversample=True, selection='sim')
n_peri_galpy1 = summary.nperi(data_total, masks_either, oversample=True, selection='model')
#
summary_plot.delta_dperi_vs_prop_median(x=n_peri_sim1, y=delta_dfo_tot1, binsize=1, versus='N.sim', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_N_sim.pdf')
summary_plot.delta_dperi_vs_prop_median(x=n_peri_galpy1, y=delta_dfo_tot1, binsize=1, versus='N.model', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_N_model.pdf')



# delta_d fraction vs Mstar (z = 0)
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=Mstar_z0_tot, y=delta_df_tot, x_out=Mstar_z0_tot_out, y_out=delta_df_tot_out, versus='M.z0', limits=(None,(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_mstar_z0_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
Mstar_z0_o_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mstar_z0_o_tot, y=delta_dfo_tot1, binsize=0.5, versus='M.z0', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_mstar_z0.pdf')



# delta_d fraction vs Mstar (peak)
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=Mstar_peak_tot, y=delta_df_tot, x_out=Mstar_peak_tot_out, y_out=delta_df_tot_out, versus='M.star.peak', limits=(None,(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_frac_vs_mstar_peak_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
Mstar_peak_o_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mstar_peak_o_tot, y=delta_dfo_tot1, binsize=0.5, versus='M.star.peak', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_frac_vs_mstar_peak.pdf')



# delta_d fraction vs Mhalo (z = 0)
delta_df_tot = summary.delta_dperi(data_total, masks_3, fraction=True, oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=Mhalo_z0_tot, y=delta_df_tot, x_out=Mhalo_z0_tot_out, y_out=delta_df_tot_out, versus='M.halo.z0', limits=(None,(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/delta_d_frac_vs_mhalo_z0_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_3, fraction=True, oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mhalo_z0_o_tot, y=delta_dfo_tot1, binsize=0.5, versus='M.halo.z0', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/delta_d_frac_vs_mhalo_z0.pdf')



# delta_d fraction vs Mhalo (peak)
delta_df_tot = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
# Scatter plots
summary_plot.delta_dperi_vs_prop_scatter(x=Mhalo_peak_tot, y=delta_df_tot, x_out=Mhalo_peak_tot_out, y_out=delta_df_tot_out, versus='M.halo.peak', limits=(None,(-1.1,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/scatter/delta_d_frac_vs_mhalo_peak_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
delta_dfo_tot1 = summary.delta_dperi(data_total, masks_either, fraction=True, oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mhalo_peak_o_tot, y=delta_dfo_tot1, binsize=0.5, versus='M.halo.peak', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/median/delta_d_frac_vs_mhalo_peak.pdf')



# delta d_peri histogram
# Histogram
delta_do_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
summary_plot.delta_dperi_hist(delta_do_tot, binsize=20, fraction=False, pdf=True, xlimits=(-100,150), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/peri_diff_histogram_zoom.pdf')



# delta d_peri vs d_peri
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=False)
d_tot_sim2 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
d_tot_galpy2 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
d_tot_sim2_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_tot_galpy2_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=d_tot_sim2, y=delta_d_tot, x_out=d_tot_sim2_out, y_out=delta_d_tot_out, versus='d.sim', fraction=False, limits=((-5,350), (-300,400)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_d_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_scatter(x=d_tot_galpy2, y=delta_d_tot, x_out=d_tot_galpy2_out, y_out=delta_d_tot_out, versus='d.model', fraction=False, limits=((-5,350), (-300,400)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_d_model_zoom.pdf')
#
# Median Plots, oversample, with outliers
delta_do_tot1 = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
d_tot_sim1 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
d_tot_galpy1 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=d_tot_sim1, y=delta_do_tot1, binsize=50, versus='d.sim', fraction=False, limits=((-5,350), (-100,300)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_d_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_median(x=d_tot_galpy1, y=delta_do_tot1, binsize=50, versus='d.model', fraction=False, limits=((-5,350), (-100,300)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_d_model_zoom.pdf')



# delta d_peri vs d(z = 0)
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=dz0_tot, y=delta_d_tot, x_out=dz0_tot_out, y_out=delta_d_tot_out, versus='d.z0', fraction=False, limits=((-5,350),(-200,200)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_d_z0_zoom.pdf')
#
# Median Plots, oversample, with outliers
delta_do_tot1 = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=dz0_o_tot, y=delta_do_tot1, binsize=50, versus='d.z0', fraction=False, limits=((-5,350),(-50,100)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_d_z0_zoom.pdf')



# delta d_peri vs t_peri
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=False)
t_tot_sim2 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
t_tot_galpy2 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
t_tot_sim2_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_tot_galpy2_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=t_tot_sim2, y=delta_d_tot, x_out=t_tot_sim2_out, y_out=delta_d_tot_out, versus='t.sim', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_t_sim.pdf')
summary_plot.delta_dperi_vs_prop_scatter(x=t_tot_galpy2, y=delta_d_tot, x_out=t_tot_galpy2_out, y_out=delta_d_tot_out, versus='t.model', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_t_model.pdf')
#
# Median plots
delta_do_tot1 = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
t_tot_sim1 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
t_tot_galpy1 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=t_tot_sim1, y=delta_do_tot1, binsize=1, versus='t.sim', fraction=False, limits=(None, (-150,150)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_t_sim_zoom.pdf')
summary_plot.delta_dperi_vs_prop_median(x=t_tot_galpy1, y=delta_do_tot1, binsize=1, versus='t.model', fraction=False, limits=(None, (-150,150)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_t_model_zoom.pdf')



# delta d_peri vs t_infall
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=False)
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=t_in_tot, y=delta_d_tot, x_out=t_in_tot_out, y_out=delta_d_tot_out, versus='t.infall', fraction=False, limits=(None,(-100,200)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_t_infall_zoom.pdf')
#
# Median plots
delta_do_tot1 = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=t_in_o_tot, y=delta_do_tot1, binsize=1, versus='t.infall', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_t_infall.pdf')



# delta d_peri vs N
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=False)
n_peri_sim2 = summary.nperi(data_total, masks_either, oversample=False, selection='sim')
n_peri_galpy2 = summary.nperi(data_total, masks_either, oversample=False, selection='model')
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
#
summary_plot.delta_dperi_vs_prop_scatter(x=n_peri_sim2, y=delta_d_tot, x_out=n_peri_sim2_out, y_out=delta_d_tot_out, versus='N.sim', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_N_sim.pdf')
summary_plot.delta_dperi_vs_prop_scatter(x=n_peri_galpy2, y=delta_d_tot, x_out=n_peri_galpy2_out, y_out=delta_d_tot_out, versus='N.model', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_N_model.pdf')
#
# Median plots
delta_do_tot1 = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
n_peri_sim1 = summary.nperi(data_total, masks_either, selection='sim', oversample=True)
n_peri_galpy1 = summary.nperi(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=n_peri_sim1, y=delta_do_tot1, binsize=1, versus='N.sim', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_N_sim.pdf')
summary_plot.delta_dperi_vs_prop_median(x=n_peri_galpy1, y=delta_do_tot1, binsize=1, versus='N.model', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_N_model.pdf')



# delta d_peri vs Mstar (z = 0)
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=Mstar_z0_tot, y=delta_d_tot, x_out=Mstar_z0_tot_out, y_out=delta_d_tot_out, versus='M.z0', fraction=False, limits=(None,(-100,200)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_mstar_z0_zoom.pdf')
#
# Median plots
delta_do_tot1 = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
Mstar_z0_o_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mstar_z0_o_tot, y=delta_do_tot1, binsize=0.5, versus='M.z0', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_mstar_z0.pdf')



# delta d_peri vs Mstar (peak)
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=False)
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=Mstar_peak_tot, y=delta_d_tot, x_out=Mstar_peak_tot_out, y_out=delta_d_tot_out, versus='M.star.peak', fraction=False, limits=(None,(-100,200)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_d_vs_mstar_peak_zoom.pdf')
#
# Median plots
delta_do_tot1 = summary.delta_dperi(data_total, masks_either, fraction=False, oversample=True)
Mstar_peak_o_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mstar_peak_o_tot, y=delta_do_tot1, binsize=0.5, versus='M.star.peak', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_d_vs_mstar_peak.pdf')



# delta d_peri vs Mhalo (z = 0)
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_3, fraction=False, oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=Mhalo_z0_tot, y=delta_d_tot, x_out=Mhalo_z0_tot_out, y_out=delta_d_tot_out, versus='M.halo.z0', fraction=False, limits=(None,(-100,200)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/delta_d_vs_mhalo_z0_zoom.pdf')
#
# Median plots
delta_do_tot1 = summary.delta_dperi(data_total, masks_3, fraction=False, oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mhalo_z0_o_tot, y=delta_do_tot1, binsize=0.5, versus='M.halo.z0', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/delta_d_vs_mhalo_z0.pdf')



# delta d_peri vs Mhalo (peak)
# Scatter plots
delta_d_tot = summary.delta_dperi(data_total, masks_3, fraction=False, oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_dperi_vs_prop_scatter(x=Mhalo_peak_tot, y=delta_d_tot, x_out=Mhalo_peak_tot_out, y_out=delta_d_tot_out, versus='M.halo.peak', fraction=False, limits=(None,(-100,200)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/scatter/delta_d_vs_mhalo_peak_zoom.pdf')
#
# Median plots
delta_do_tot1 = summary.delta_dperi(data_total, masks_3, fraction=False, oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=True)
#
summary_plot.delta_dperi_vs_prop_median(x=Mhalo_peak_o_tot, y=delta_do_tot1, binsize=0.5, versus='M.halo.peak', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/median/delta_d_vs_mhalo_peak.pdf')



# Recent pericenter time comparison
# no oversample, cases with peris in sim and model, but outliers in red
t_tot_sim2 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
t_tot_sim2_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_tot_galpy2 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
t_tot_galpy2_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.tperi_comparison_scatter(x=t_tot_sim2, y=t_tot_galpy2, x_out=t_tot_sim2_out, y_out=t_tot_galpy2_out, limits=((-0.5, 13.8)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/recent_tperi_comparison.pdf')
#
# Median plot
t_tot_sim1 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
t_tot_galpy1 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.tperi_comparison_median(x=t_tot_sim1, y=t_tot_galpy1, binsize=0.5, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/recent_tperi_comparison.pdf')



# t_peri histograms
t_tot_o_sim = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
summary_plot.plot_hist(x=t_tot_o_sim, binsize=0.5, pdf=True, xtype='t.sim', xlimits=(-0.5,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/t_peri_sim_histogram_pdf.pdf')
t_tot_sim = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
summary_plot.plot_hist(x=t_tot_sim, binsize=0.5, pdf=False, xtype='t.sim', xlimits=(-0.5,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/t_peri_sim_histogram.pdf')
#
t_tot_o_model = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
summary_plot.plot_hist(x=t_tot_o_model, binsize=0.5, pdf=True, xtype='t.model', xlimits=(-0.5,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/t_peri_model_histogram_pdf.pdf')
t_tot_model = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
summary_plot.plot_hist(x=t_tot_model, binsize=0.5, pdf=False, xtype='t.model', xlimits=(-0.5,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/t_peri_model_histogram.pdf')



# delta t_peri fractions
# Histogram
# oversample, cases with pericenters in sim, but not required in model
delta_tfo_tot = summary.delta_tperi(data_total, mask3, fraction=True, oversample=True)
summary_plot.delta_tperi_hist(delta_tfo_tot, binsize=0.1, xlimits=(-1,2), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/peri_tlb_diff_frac_histogram_zoom.pdf')



# delta t_peri fraction vs t_peri
# no oversample, cases with peris in sim and model, but outliers in red
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
t_tot_sim2 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
t_tot_galpy2 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
t_tot_sim2_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_tot_galpy2_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
# Scatter plots
summary_plot.delta_tperi_vs_prop_scatter(x=t_tot_sim2, y=delta_tf_tot, x_out=t_tot_sim2_out, y_out=delta_tf_tot_out, versus='t.sim', limits=((0, 10.5),(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_t_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_scatter(x=t_tot_galpy2, y=delta_tf_tot, x_out=t_tot_galpy2_out, y_out=delta_tf_tot_out, versus='t.model', limits=((0, 10.5),(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_t_model_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
t_tot_sim1 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
t_tot_galpy1 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=t_tot_sim1, y=delta_tfo_tot1, binsize=0.5, versus='t.sim', fraction=True, limits=((0, 11),(-1.1, 1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_t_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_median(x=t_tot_galpy1, y=delta_tfo_tot1, binsize=0.5, versus='t.model', fraction=True, limits=((0, 14),(-1.1, 5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_t_model_zoom.pdf')



# delta t_peri fraction vs t_infall
# no oversample, cases with peris in sim and model, but outliers in red
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
#
# Scatter plots
summary_plot.delta_tperi_vs_prop_scatter(x=t_in_tot, y=delta_tf_tot, x_out=t_in_tot_out, y_out=delta_tf_tot_out, versus='t.infall', limits=(None,(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_t_infall_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=t_in_o_tot, y=delta_tfo_tot1, binsize=0.5, versus='t.infall', fraction=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_t_infall.pdf')



# delta t_peri fraction vs d_peri
# Scatter plots
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
d_tot_sim2 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
d_tot_galpy2 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
d_tot_sim2_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_tot_galpy2_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=d_tot_sim2, y=delta_tf_tot, x_out=d_tot_sim2_out, y_out=delta_tf_tot_out, versus='d.sim', limits=((-5,350),(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_d_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_scatter(x=d_tot_galpy2, y=delta_tf_tot, x_out=d_tot_galpy2_out, y_out=delta_tf_tot_out, versus='d.model', limits=((-5,350),(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_d_model_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
d_tot_sim1 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
d_tot_galpy1 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=d_tot_sim1, y=delta_tfo_tot1, binsize=50, versus='d.sim', fraction=True, limits=((-5,350),(-1,1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_d_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_median(x=d_tot_galpy1, y=delta_tfo_tot1, binsize=50, versus='d.model', fraction=True, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_d_model_zoom.pdf')



# delta t_peri fraction vs d(z = 0)
# no oversample, cases with peris in sim and model, but outliers in red
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
# Scatter plots
summary_plot.delta_tperi_vs_prop_scatter(x=dz0_tot, y=delta_tf_tot, x_out=dz0_tot_out, y_out=delta_tf_tot_out, versus='d.z0', limits=((-5,350),(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_d_z0_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=dz0_o_tot, y=delta_tfo_tot1, binsize=50, versus='d.z0', fraction=True, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_d_z0_zoom.pdf')



# delta t_peri fraction vs Nperi
# Scatter plots
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
n_peri_sim2 = summary.nperi(data_total, masks_either, selection='sim', oversample=False)
n_peri_galpy2 = summary.nperi(data_total, masks_either, selection='model', oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, selection='sim', oversample=False)
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=n_peri_sim2, y=delta_tf_tot, x_out=n_peri_sim2_out, y_out=delta_tf_tot_out, versus='N.sim', limits=((-0.5,13.5),(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_N_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_scatter(x=n_peri_galpy2, y=delta_tf_tot, x_out=n_peri_galpy2_out, y_out=delta_tf_tot_out, versus='N.model', limits=((-0.5,13.5),(-1.1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_N_model_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
n_peri_sim1 = summary.nperi(data_total, masks_either, selection='sim', oversample=True)
n_peri_galpy1 = summary.nperi(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=n_peri_sim1, y=delta_tfo_tot1, binsize=1, versus='N.sim', fraction=True, limits=((-0.5,13.5),(-1,1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_N_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_median(x=n_peri_galpy1, y=delta_tfo_tot1, binsize=1, versus='N.model', fraction=True, limits=((-0.5,13.5),(-1,1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_N_model_zoom.pdf')



# delta t_peri fraction vs Mstar (z = 0)
# Scatter plots
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mstar_z0_tot, y=delta_tf_tot, x_out=Mstar_z0_tot_out, y_out=delta_tf_tot_out, versus='M.z0', limits=(None,(-1.1,4.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_mstar_z0_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
Mstar_z0_o_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mstar_z0_o_tot, y=delta_tfo_tot1, binsize=0.5, versus='M.z0', fraction=True, limits=(None,(-1,1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_mstar_z0_zoom.pdf')



# delta t_peri fraction vs Mstar (peak)
# Scatter plots
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mstar_peak_tot, y=delta_tf_tot, x_out=Mstar_peak_tot_out, y_out=delta_tf_tot_out, versus='M.peak', limits=(None,(-1.1,4.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_frac_vs_mstar_peak_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
Mstar_peak_o_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mstar_peak_o_tot, y=delta_tfo_tot1, binsize=0.5, versus='M.peak', fraction=True, limits=(None,(-1,1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_frac_vs_mstar_peak_zoom.pdf')



# delta t_peri fraction vs Mhalo (z = 0)
# Scatter plots
delta_tf_tot = summary.delta_tperi(data_total, masks_3, fraction=True, oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mhalo_z0_tot, y=delta_tf_tot, x_out=Mhalo_z0_tot_out, y_out=delta_tf_tot_out, versus='M.halo.z0', limits=(None,(-1.1,4.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/delta_t_frac_vs_mhalo_z0_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_3, fraction=True, oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mhalo_z0_o_tot, y=delta_tfo_tot1, binsize=0.5, versus='M.halo.z0', fraction=True, limits=(None,(-1,1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/delta_t_frac_vs_mhalo_z0_zoom.pdf')



# delta t_peri fraction vs Mhalo (z = 0)
# Scatter plots
delta_tf_tot = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mhalo_peak_tot, y=delta_tf_tot, x_out=Mhalo_peak_tot_out, y_out=delta_tf_tot_out, versus='M.halo.peak', limits=(None,(-1.1,4.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/scatter/delta_t_frac_vs_mhalo_peak_zoom.pdf')
#
# Median plots
delta_tfo_tot1 = summary.delta_tperi(data_total, masks_either, fraction=True, oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mhalo_peak_o_tot, y=delta_tfo_tot1, binsize=0.5, versus='M.halo.peak', fraction=True, limits=(None,(-1,1)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/median/delta_t_frac_vs_mhalo_peak_zoom.pdf')



# delta t_peri histogram
delta_to_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
summary_plot.delta_tperi_hist(delta_to_tot, binsize=0.5, xlimits=(-3,4), fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/peri_tlb_diff_histogram_zoom.pdf')



# delta t_peri vs t_peri
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
t_tot_sim2 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=False)
t_tot_galpy2 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
t_tot_sim2_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_tot_galpy2_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=t_tot_sim2, y=delta_t_tot, x_out=t_tot_sim2_out, y_out=delta_t_tot_out, versus='t.sim', fraction=False, limits=((-0.1, 10.5),(-5, 10)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_t_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_scatter(x=t_tot_galpy2, y=delta_t_tot, x_out=t_tot_galpy2_out, y_out=delta_t_tot_out, versus='t.model', fraction=False, limits=((-0.1, 13.8),(-5, 10)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_t_model_zoom.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
t_tot_sim1 = summary.tperi_recent(data_total, masks_either, selection='sim', oversample=True)
t_tot_galpy1 = summary.tperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=t_tot_sim1, y=delta_to_tot1, binsize=1, versus='t.sim', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_t_sim.pdf')
summary_plot.delta_tperi_vs_prop_median(x=t_tot_galpy1, y=delta_to_tot1, binsize=1, versus='t.model', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_t_model.pdf')



# delta t_peri vs t_infall
# no oversample, cases with peris in sim and model, but outliers in red
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
#
# Scatter plots
summary_plot.delta_tperi_vs_prop_scatter(x=t_in_tot, y=delta_t_tot, x_out=t_in_tot_out, y_out=delta_t_tot_out, versus='t.infall', fraction=False, limits=(None, (-3,3)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_t_infall_zoom.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=t_in_o_tot, y=delta_to_tot1, binsize=1, versus='t.infall', fraction=False, limits=(None, (-1,3)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_t_infall_zoom.pdf')



# delta t_peri vs d_peri
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
d_tot_sim2 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=False)
d_tot_galpy2 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
d_tot_sim2_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_tot_galpy2_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=d_tot_sim2, y=delta_t_tot, x_out=d_tot_sim2_out, y_out=delta_t_tot_out, versus='d.sim', fraction=False, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_d_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_scatter(x=d_tot_galpy2, y=delta_t_tot, x_out=d_tot_galpy2_out, y_out=delta_t_tot_out, versus='d.model', fraction=False, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_d_model_zoom.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
d_tot_sim1 = summary.dperi_recent(data_total, masks_either, selection='sim', oversample=True)
d_tot_galpy1 = summary.dperi_recent(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=d_tot_sim1, y=delta_to_tot1, binsize=50, versus='d.sim', fraction=False, limits=((-5,350),None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_d_sim.pdf')
summary_plot.delta_tperi_vs_prop_median(x=d_tot_galpy1, y=delta_to_tot1, binsize=50, versus='d.model', fraction=False, limits=((-5,350),(-2,4)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_d_model_zoom.pdf')



# delta t_peri vs d(z = 0)
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=dz0_tot, y=delta_t_tot, x_out=dz0_tot_out, y_out=delta_t_tot_out, versus='d.z0', fraction=False, limits=((-5,350),(-5,5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_d_z0_zoom.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=dz0_o_tot, y=delta_to_tot1, binsize=50, versus='d.z0', fraction=False, limits=((-5,350),(-1,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_d_z0_zoom.pdf')



# delta t_peri vs N_peri
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
n_peri_sim2 = summary.nperi(data_total, masks_either, selection='sim', oversample=False)
n_peri_galpy2 = summary.nperi(data_total, masks_either, selection='model', oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
n_peri_sim2_out = summary.nperi(data_total, masks_outliers, selection='sim', oversample=False)
n_peri_galpy2_out = summary.nperi(data_total, masks_outliers, selection='model', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=n_peri_sim2, y=delta_t_tot, x_out=n_peri_sim2_out, y_out=delta_t_tot_out, versus='N.sim', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_N_sim.pdf')
summary_plot.delta_tperi_vs_prop_scatter(x=n_peri_galpy2, y=delta_t_tot, x_out=n_peri_galpy2_out, y_out=delta_t_tot_out, versus='N.model', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_N_model.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
n_peri_sim1 = summary.nperi(data_total, masks_either, selection='sim', oversample=True)
n_peri_galpy1 = summary.nperi(data_total, masks_either, selection='model', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=n_peri_sim1, y=delta_to_tot1, binsize=1, versus='N.sim', fraction=False, limits=((-0.5,13.5),(-2,2.5)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_N_sim_zoom.pdf')
summary_plot.delta_tperi_vs_prop_median(x=n_peri_galpy1, y=delta_to_tot1, binsize=1, versus='N.model', fraction=False, limits=((-0.5,13.5),(-2,2)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_N_model_zoom.pdf')



# delta t_peri vs Mstar (z = 0)
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mstar_z0_tot, y=delta_t_tot, x_out=Mstar_z0_tot_out, y_out=delta_t_tot_out, versus='M.z0', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_mstar_z0.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
Mstar_z0_o_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mstar_z0_o_tot, y=delta_to_tot1, binsize=0.5, versus='M.z0', fraction=False, limits=((4,9.5),(-1,3)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_mstar_z0_zoom.pdf')



# delta t_peri vs Mstar (peak)
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mstar_peak_tot, y=delta_t_tot, x_out=Mstar_peak_tot_out, y_out=delta_t_tot_out, versus='M.peak', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/delta_t_vs_mstar_peak.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
Mstar_peak_o_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mstar_peak_o_tot, y=delta_to_tot1, binsize=0.5, versus='M.peak', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/delta_t_vs_mstar_peak.pdf')



# delta t_peri vs Mhalo (z = 0)
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_3, fraction=False, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mhalo_z0_tot, y=delta_t_tot, x_out=Mhalo_z0_tot_out, y_out=delta_t_tot_out, versus='M.halo.z0', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/delta_t_vs_mhalo_z0.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_3, fraction=False, oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mhalo_z0_o_tot, y=delta_to_tot1, binsize=0.5, versus='M.halo.z0', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/delta_t_vs_mhalo_z0.pdf')



# delta t_peri vs Mhalo (peak)
# Scatter plots
delta_t_tot = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.delta_tperi_vs_prop_scatter(x=Mhalo_peak_tot, y=delta_t_tot, x_out=Mhalo_peak_tot_out, y_out=delta_t_tot_out, versus='M.halo.peak', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/scatter/delta_t_vs_mhalo_peak.pdf')
#
# Median plots
delta_to_tot1 = summary.delta_tperi(data_total, masks_either, fraction=False, oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.delta_tperi_vs_prop_median(x=Mhalo_peak_o_tot, y=delta_to_tot1, binsize=0.5, versus='M.halo.peak', fraction=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/median/delta_t_vs_mhalo_peak.pdf')



# t_infall histogram
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
summary_plot.plot_hist(t_in_o_tot, binsize=0.5, pdf=True, xtype='t.infall', xlimits=(-0.5,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/infall_histogram_pdf.pdf')
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
summary_plot.plot_hist(t_in_tot, binsize=0.5, pdf=False, xtype='t.infall', xlimits=(-0.5,14), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/infall_histogram.pdf')



# t_infall vs Mstar (z = 0)
# Scatter plot
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.infall_vs_prop_scatter(x=Mstar_z0_tot, y=t_in_tot, x_out=Mstar_z0_tot_out, y_out=t_in_tot_out, xtype='M.star.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/infall_vs_mstar_z0.pdf')
#
# Median plot
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
Mstar_z0_o_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
#
summary_plot.infall_vs_prop_median(x=Mstar_z0_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.star.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/infall_vs_mstar_z0.pdf')



# t_infall vs Mstar (peak)
# Scatter plot
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.infall_vs_prop_scatter(x=Mstar_peak_tot, y=t_in_tot, x_out=Mstar_peak_tot_out, y_out=t_in_tot_out, xtype='M.star.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/infall_vs_mstar_peak.pdf')
#
# Median plot
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
Mstar_peak_o_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.infall_vs_prop_median(x=Mstar_peak_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.star.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/infall_vs_mstar_peak.pdf')



# t_infall vs Mhalo (z = 0)
# Scatter plot
t_in_tot = summary.first_infall(data_total, masks_3, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.infall_vs_prop_scatter(x=Mhalo_z0_tot, y=t_in_tot, x_out=Mhalo_z0_tot_out, y_out=t_in_tot_out, xtype='M.halo.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/infall_vs_mhalo_z0.pdf')
#
# Median plot
t_in_o_tot = summary.first_infall(data_total, masks_3, oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=True)
#
summary_plot.infall_vs_prop_median(x=Mhalo_z0_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.halo.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/infall_vs_mhalo_z0.pdf')



# t_infall vs Mhalo (peak)
# Scatter plot
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.infall_vs_prop_scatter(x=Mhalo_peak_tot, y=t_in_tot, x_out=Mhalo_peak_tot_out, y_out=t_in_tot_out, xtype='M.halo.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/scatter/infall_vs_mhalo_peak.pdf')
#
# Median plot
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, masks_either, selection='peak', oversample=True)
#
summary_plot.infall_vs_prop_median(x=Mhalo_peak_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.halo.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_either_peri/median/infall_vs_mhalo_peak.pdf')



# t_infall vs d(z = 0)
# Scatter plot
t_in_tot = summary.first_infall(data_total, masks_either, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.infall_vs_prop_scatter(x=dz0_tot, y=t_in_tot, x_out=dz0_tot_out, y_out=t_in_tot_out, xtype='d.z0', limits=((-5,350), None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/infall_vs_d_z0_zoom.pdf')
#
# Median plot
t_in_o_tot = summary.first_infall(data_total, masks_either, oversample=True)
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.infall_vs_prop_median(x=dz0_o_tot, y=t_in_o_tot, binsize=50, xtype='d.z0', limits=((-5,350), None), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/infall_vs_d_z0_zoom.pdf')
#
#
#
# Split everything up into mass_bins
t_ins = summary.mass_masking_property(data_total, masks_3, prop='t.infall', mass_type='Mstar.z0', oversample=True)
d_z0s = summary.mass_masking_property(data_total, masks_3, prop='dz0', mass_type='Mstar.z0', oversample=True)
y = summary.first_infall(data_total, masks_either, oversample=True)
x = summary.d_z0(data_total, masks_either, oversample=True)
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
    mask_low = (d_z0s['low'] >= bins[i]) & (d_z0s['low'] <= bins[i+1])
    med_low[i] = np.nanmedian(t_ins['low'][mask_low])
    mask_mid = (d_z0s['mid'] >= bins[i]) & (d_z0s['mid'] <= bins[i+1])
    med_mid[i] = np.nanmedian(t_ins['mid'][mask_mid])
    mask_high = (d_z0s['high'] >= bins[i]) & (d_z0s['high'] <= bins[i+1])
    med_high[i] = np.nanmedian(t_ins['high'][mask_high])
#
f, ax = plt.subplots(figsize=(10, 8))
plt.plot(bins[:-1]+half_bin, med, color='k', alpha=0.5)
plt.fill_between(bins[:-1]+half_bin, upper, lower, color='k', alpha=0.3)
#
plt.plot(bins[:-1]+half_bin, med_low, color=summary_plot.colors[1], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm star}$ < 5')
plt.plot(bins[:-1]+half_bin, med_mid, color=summary_plot.colors[2], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm star}$ = [5, 7]')
plt.plot(bins[:-1]+half_bin, med_high, color=summary_plot.colors[3], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm star}$ > 7')
#
plt.xlim(-5, 350)
plt.xlabel('d(z = 0) [kpc]', fontsize=28)
plt.ylabel('t$_{\\rm infall,lb}$ [Gyr]', fontsize=28)
plt.legend(prop={'size': 16})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/infall_vs_d_z0_mass_bins.pdf')
plt.close()












# M_star histogram (z = 0)
Mstar_z0_tot = summary.mstar(data_total, masks_3, selection='z0', oversample=False)
Mstar_z0_o_tot = summary.mstar(data_total, masks_3, selection='z0', oversample=True)
summary_plot.mstar_hist(Mstar_z0_tot, binsize=0.1, log=True, pdf=False, selection='z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/histogram/mstar_z0_histogram.pdf')
summary_plot.mstar_hist(Mstar_z0_o_tot, binsize=0.1, log=True, pdf=True, selection='z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/histogram/mstar_z0_histogram_pdf.pdf')



# Mstar(z = 0) vs d(z = 0)
# Scatter plot
Mstar_z0_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.mstar_vs_prop_scatter(x=Mstar_z0_tot, y=dz0_tot, x_out=Mstar_z0_tot_out, y_out=dz0_tot_out, xtype='M.z0', ytype='d.z0', limits=(None,(-5,350)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/mstar_z0_vs_d_z0_zoom.pdf')
#
# Median plot
Mstar_z0_o_tot = summary.mstar(data_total, masks_either, selection='z0', oversample=True)
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.mstar_vs_prop_median(x=Mstar_z0_o_tot, y=dz0_o_tot, binsize=0.5, xtype='M.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/mstar_z0_vs_d_z0.pdf')



# Mstar(z = 0) vs Mhalo(z = 0)
# Scatter plot
Mstar_z0_tot = summary.mstar(data_total, masks_3, selection='z0', oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
#
summary_plot.mstar_mhalo_scatter(x=Mhalo_z0_tot, y=Mstar_z0_tot, x_out=Mhalo_z0_tot_out, y_out=Mstar_z0_tot_out, masstype='z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/mstar_mhalo_z0.pdf')
#
# Median plot
Mstar_z0_o_tot = summary.mstar(data_total, masks_3, selection='z0', oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, masks_3, selection='z0', oversample=True)
#
summary_plot.mstar_mhalo_median(x=Mhalo_z0_o_tot, y=Mstar_z0_o_tot, binsize=0.5, masstype='z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/mstar_mhalo_z0.pdf')



# Mstar(peak) vs Mhalo(peak)
# Scatter plot
Mstar_peak_tot = summary.mstar(data_total, masks_3, selection='peak', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
#
summary_plot.mstar_mhalo_scatter(x=Mhalo_peak_tot, y=Mstar_peak_tot, x_out=Mhalo_peak_tot_out, y_out=Mstar_peak_tot_out, masstype='peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/scatter/mstar_mhalo_peak.pdf')
#
# Median plot
Mstar_peak_o_tot = summary.mstar(data_total, masks_3, selection='peak', oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=True)
#
summary_plot.mstar_mhalo_median(x=Mhalo_peak_o_tot, y=Mstar_peak_o_tot, binsize=0.5, masstype='peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/median/mstar_mhalo_peak.pdf')



# Mstar histogram (peak)
Mstar_peak_tot = summary.mstar(data_total, masks_1, selection='peak', oversample=False)
Mstar_peak_o_tot = summary.mstar(data_total, masks_1, selection='peak', oversample=True)
summary_plot.mstar_hist(Mstar_peak_tot, binsize=0.1, log=True, pdf=False, selection='peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_peri_sim/histogram/mstar_peak_histogram.pdf')
summary_plot.mstar_hist(Mstar_peak_o_tot, binsize=0.1, log=True, pdf=True, selection='peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_peri_sim/histogram/mstar_peak_histogram_pdf.pdf')



# Mstar(peak) vs d(z = 0)
# Scatter plot
Mstar_peak_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
#
summary_plot.mstar_vs_prop_scatter(x=Mstar_peak_tot, y=dz0_tot, x_out=Mstar_peak_tot_out, y_out=dz0_tot_out, xtype='M.peak', ytype='d.z0', limits=(None,(-5,350)), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/scatter/mstar_peak_vs_d_z0_zoom.pdf')
#
# Median plot
Mstar_peak_o_tot = summary.mstar(data_total, masks_either, selection='peak', oversample=True)
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
#
summary_plot.mstar_vs_prop_median(x=Mstar_peak_o_tot, y=dz0_o_tot, binsize=0.5, xtype='M.peak', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/median/mstar_peak_vs_d_z0.pdf')



# d_z0
dz0_o_tot = summary.d_z0(data_total, masks_either, oversample=True)
summary_plot.plot_hist(x=dz0_o_tot, binsize=10, pdf=True, xtype='d.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_z0_histogram_pdf.pdf')
summary_plot.plot_hist(x=dz0_o_tot, binsize=10, pdf=True, xtype='d.z0', xlimits=(-5,350), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_z0_histogram_pdf_zoom.pdf')
dz0_tot = summary.d_z0(data_total, masks_either, oversample=False)
summary_plot.plot_hist(x=dz0_tot, binsize=10, pdf=False, xtype='d.z0', file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_z0_histogram.pdf')
summary_plot.plot_hist(x=dz0_tot, binsize=10, pdf=False, xtype='d.z0', xlimits=(-5,350), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/histogram/d_z0_histogram_zoom.pdf')



# M_halo histogram (z = 0)
Mhalo_z0_tot = summary.mhalo(data_total, masks_1, selection='z0', oversample=False)
Mhalo_z0_o_tot = summary.mhalo(data_total, masks_1, selection='z0', oversample=True)
summary_plot.mhalo_hist(Mhalo_z0_tot, binsize=0.1, log=True, selection='z0', pdf=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_peri_sim/histogram/mhalo_z0_histogram.pdf')
summary_plot.mhalo_hist(Mhalo_z0_o_tot, binsize=0.1, log=True, selection='z0', pdf=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/sats_w_peri_sim/histogram/mhalo_z0_histogram_pdf.pdf')



# M_halo histogram (peak)
Mhalo_peak_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=False)
Mhalo_peak_o_tot = summary.mhalo(data_total, masks_3, selection='peak', oversample=True)
summary_plot.mhalo_hist(Mhalo_peak_tot, binsize=0.1, log=True, selection='peak', pdf=False, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/histogram/mhalo_peak_histogram.pdf')
summary_plot.mhalo_hist(Mhalo_peak_o_tot, binsize=0.1, log=True, selection='peak', pdf=True, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/all_infall_sats/histogram/mhalo_peak_histogram_pdf.pdf')
