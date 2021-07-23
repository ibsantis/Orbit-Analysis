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
d_model_tot = summary.dperi_recent(data_total, mask_selection, selection='model', oversample=False)
delta_df_tot = summary.delta_dperi(data_total, mask_selection, fraction=True, oversample=False)
delta_d_tot = summary.delta_dperi(data_total, mask_selection, fraction=False, oversample=False)
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=False)
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=False)
t_model_tot = summary.tperi_recent(data_total, mask_selection, selection='model', oversample=False)
delta_tf_tot = summary.delta_tperi(data_total, mask_selection, fraction=True, oversample=False)
delta_t_tot = summary.delta_tperi(data_total, mask_selection, fraction=False, oversample=False)
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=False)
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=False)
Mstar_peak_tot = summary.mstar(data_total, mask_selection, selection='peak', oversample=False)
Mhalo_z0_tot = summary.mhalo(data_total, mask_selection, selection='z0', oversample=False)
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=False)
ke_max_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='max', oversample=False)
ke_peri_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='peri', oversample=False)


# Outliers
delta_N_out = summary.delta_nperi(data_total, masks_outliers, oversample=False)
N_sim_tot_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='sim')
N_model_tot_out = summary.nperi(data_total, masks_outliers, oversample=False, selection='model')
d_sim_tot_out = summary.dperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
d_model_tot_out = summary.dperi_recent(data_total, masks_outliers, selection='model', oversample=False)
delta_df_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=True, oversample=False)
delta_d_tot_out = summary.delta_dperi(data_total, masks_outliers, fraction=False, oversample=False)
dz0_tot_out = summary.d_z0(data_total, masks_outliers, oversample=False)
t_sim_tot_out = summary.tperi_recent(data_total, masks_outliers, selection='sim', oversample=False)
t_model_tot_out = summary.tperi_recent(data_total, masks_outliers, selection='model', oversample=False)
delta_tf_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=True, oversample=False)
delta_t_tot_out = summary.delta_tperi(data_total, masks_outliers, fraction=False, oversample=False)
t_in_tot_out = summary.first_infall(data_total, masks_outliers, oversample=False)
Mstar_z0_tot_out = summary.mstar(data_total, masks_outliers, selection='z0', oversample=False)
Mstar_peak_tot_out = summary.mstar(data_total, masks_outliers, selection='peak', oversample=False)
Mhalo_z0_tot_out = summary.mhalo(data_total, masks_outliers, selection='z0', oversample=False)
Mhalo_peak_tot_out = summary.mhalo(data_total, masks_outliers, selection='peak', oversample=False)
ke_max_tot_out = summary.kinetic_energy(data_total, masks_outliers, ke_type='max', oversample=False)
ke_peri_tot_out = summary.kinetic_energy(data_total, masks_outliers, ke_type='peri', oversample=False)


# Oversample
delta_No = summary.delta_nperi(data_total, mask_selection, oversample=True)
N_sim_o_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim')
N_model_o_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='model')
d_sim_o_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True)
d_model_o_tot = summary.dperi_recent(data_total, mask_selection, selection='model', oversample=True)
delta_dfo_tot = summary.delta_dperi(data_total, mask_selection, fraction=True, oversample=True)
delta_do_tot = summary.delta_dperi(data_total, mask_selection, fraction=False, oversample=True)
dz0_o_tot = summary.d_z0(data_total, mask_selection, oversample=True)
t_sim_o_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True)
t_model_o_tot = summary.tperi_recent(data_total, mask_selection, selection='model', oversample=True)
delta_tfo_tot = summary.delta_tperi(data_total, mask_selection, fraction=True, oversample=True)
delta_to_tot = summary.delta_tperi(data_total, mask_selection, fraction=False, oversample=True)
t_in_o_tot = summary.first_infall(data_total, mask_selection, oversample=True)
Mstar_z0_o_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True)
Mstar_peak_o_tot = summary.mstar(data_total, mask_selection, selection='peak', oversample=True)
Mhalo_z0_o_tot = summary.mhalo(data_total, mask_selection, selection='z0', oversample=True)
Mhalo_peak_o_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True)
ke_max_o_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='max', oversample=True)
ke_peri_o_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='peri', oversample=True)




### Summary plots
# N histogram
summary_plot.plot_hist(x=N_sim_o_tot, binsize=1, xtype='N.sim', pdf=True, xlimits=(0,14), file_path_and_name=directory+'/histogram/N_peri_sim_histogram_pdf.pdf')
summary_plot.plot_hist(x=N_sim_tot, binsize=1, xtype='N.sim', pdf=False, xlimits=(0,14), file_path_and_name=directory+'/histogram/N_peri_sim_histogram.pdf')
#
summary_plot.plot_hist(x=N_model_o_tot, binsize=1, xtype='N.model', pdf=True, xlimits=(0,14), file_path_and_name=directory+'/histogram/N_peri_model_histogram_pdf.pdf')
summary_plot.plot_hist(x=N_model_tot, binsize=1, xtype='N.model', pdf=False, xlimits=(0,14), file_path_and_name=directory+'/histogram/N_peri_model_histogram.pdf')



# Delta N histogram
summary_plot.plot_hist(x=delta_No, binsize=1, xtype='N.delta', pdf=True, xlimits=(-5,5), file_path_and_name=directory+'/histogram/delta_N_peri_histogram_pdf.pdf')
summary_plot.plot_hist(x=delta_N, binsize=1, xtype='N.delta', pdf=False, xlimits=(-5,5), file_path_and_name=directory+'/histogram/delta_N_peri_histogram.pdf')




# Delta N vs N
#
# Scatter plots
summary_plot.scatter_plot(x=N_sim_tot, y=delta_N, x_out=N_sim_tot_out, y_out=delta_N_out, xtype='N.sim', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_N_sim.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_N, x_out=N_model_tot_out, y_out=delta_N_out, xtype='N.model', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_N_model.pdf')
#
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=N_sim_o_tot, y=delta_No, xtype='N.sim', ytype='N.delta', binsize=1, file_path_and_name=directory+'/median/delta_N_vs_N_sim.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_No, xtype='N.model', ytype='N.delta', binsize=1, file_path_and_name=directory+'/median/delta_N_vs_N_model.pdf')



# delta_N vs d_peri
# Scatter plots
summary_plot.scatter_plot(x=d_sim_tot, y=delta_N, x_out=d_sim_tot_out, y_out=delta_N_out, xtype='d.sim', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_d_sim.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_N, x_out=d_model_tot_out, y_out=delta_N_out, xtype='d.model', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_d_model.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_N, x_out=d_model_tot_out, y_out=delta_N_out, xtype='d.model', ytype='N.delta', limits=((-5,350),None), file_path_and_name=directory+'/scatter/delta_N_vs_d_model_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=d_sim_o_tot, y=delta_No, binsize=50, xtype='d.sim', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_d_sim.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_No, binsize=50, xtype='d.model', ytype='N.delta',file_path_and_name=directory+'/median/delta_N_vs_d_model.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_No, binsize=50, xtype='d.model', ytype='N.delta', limits=((-5,350), None), file_path_and_name=directory+'/median/delta_N_vs_d_model_zoom.pdf')



# delta_N vs d(z = 0)
# Scatter plots
summary_plot.scatter_plot(x=dz0_tot, y=delta_N, x_out=dz0_tot_out, y_out=delta_N_out, xtype='d.z0', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_dz0.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=delta_N, x_out=dz0_tot_out, y_out=delta_N_out, xtype='d.z0', ytype='N.delta', limits=((-5,350), None), file_path_and_name=directory+'/scatter/delta_N_vs_dz0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=dz0_o_tot, y=delta_No, binsize=50, xtype='d.z0', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=delta_No, binsize=50, xtype='d.z0', ytype='N.delta', limits=((-5,350),None), file_path_and_name=directory+'/median/delta_N_vs_dz0_zoom.pdf')



# delta_N vs t_peri
# Scatter plots
summary_plot.scatter_plot(x=t_sim_tot, y=delta_N, x_out=t_sim_tot_out, y_out=delta_N_out, xtype='t.sim', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_t_sim.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_N, x_out=t_model_tot_out, y_out=delta_N_out, xtype='t.model', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_t_model.pdf')
#
# Median plots
summary_plot.median_plot(x=t_sim_o_tot, y=delta_No, binsize=1, xtype='t.sim', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_t_sim.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_No, binsize=1, xtype='t.model', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_t_model.pdf')



# delta_N vs t_infall
# Scatter plots
summary_plot.scatter_plot(x=t_in_tot, y=delta_N, x_out=t_in_tot_out, y_out=delta_N_out, xtype='t.infall', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_t_infall.pdf')
#
# Median plots
summary_plot.median_plot(x=t_in_o_tot, y=delta_No, binsize=1, xtype='t.infall', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_t_infall.pdf')



# delta_N vs Mstar (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_N, x_out=Mstar_z0_tot_out, y_out=delta_N_out, xtype='M.star.z0', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_mstar_z0.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_No, binsize=0.5, xtype='M.star.z0', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_mstar_z0.pdf')



# delta_N vs Mstar (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_N, x_out=Mstar_peak_tot_out, y_out=delta_N_out, xtype='M.star.peak', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_mstar_peak.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_peak_o_tot, y=delta_No, binsize=0.5, xtype='M.star.peak', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_mstar_peak.pdf')



# delta_N vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_N, x_out=Mhalo_z0_tot_out, y_out=delta_N_out, xtype='M.halo.z0', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_mhalo_z0.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=delta_No, binsize=0.5, xtype='M.halo.z0', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_mhalo_z0.pdf')



# delta_N vs Mhalo (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_N, x_out=Mhalo_peak_tot_out, y_out=delta_N_out, xtype='M.halo.peak', ytype='N.delta', file_path_and_name=directory+'/scatter/delta_N_vs_mhalo_peak.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=delta_No, binsize=0.5, xtype='M.halo.peak', ytype='N.delta', file_path_and_name=directory+'/median/delta_N_vs_mhalo_peak.pdf')



# N vs d_peri
# Scatter plots
summary_plot.scatter_plot(x=d_sim_tot, y=N_sim_tot, x_out=d_sim_tot_out, y_out=N_sim_tot_out, xtype='d.sim', ytype='N.sim', file_path_and_name=directory+'/scatter/N_sim_vs_d_sim.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=N_model_tot, x_out=d_model_tot_out, y_out=N_model_tot_out, xtype='d.model', ytype='N.model', file_path_and_name=directory+'/scatter/N_model_vs_d_model.pdf')
summary_plot.scatter_plot(x=d_sim_tot, y=N_model_tot, x_out=d_sim_tot_out, y_out=N_model_tot_out, xtype='d.sim', ytype='N.model', file_path_and_name=directory+'/scatter/N_model_vs_d_sim.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=N_sim_tot, x_out=d_model_tot_out, y_out=N_sim_tot_out, xtype='d.model', ytype='N.sim', file_path_and_name=directory+'/scatter/N_sim_vs_d_model.pdf')
summary_plot.scatter_plot(x=d_sim_tot, y=N_sim_tot, x_out=d_sim_tot_out, y_out=N_sim_tot_out, xtype='d.sim', ytype='N.sim', limits=((-5,350),(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_d_sim_zoom.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=N_model_tot, x_out=d_model_tot_out, y_out=N_model_tot_out, xtype='d.model', ytype='N.model', limits=((-5,350),(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_d_model_zoom.pdf')
summary_plot.scatter_plot(x=d_sim_tot, y=N_model_tot, x_out=d_sim_tot_out, y_out=N_model_tot_out, xtype='d.sim', ytype='N.model', limits=((-5,350),(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_d_sim_zoom.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=N_sim_tot, x_out=d_model_tot_out, y_out=N_sim_tot_out, xtype='d.model', ytype='N.sim', limits=((-5,350),(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_d_model_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=d_sim_o_tot, y=N_sim_o_tot, xtype='d.sim', ytype='N.sim', binsize=50, file_path_and_name=directory+'/median/N_sim_vs_d_sim.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=N_model_o_tot, xtype='d.model', ytype='N.model', binsize=50, file_path_and_name=directory+'/median/N_model_vs_d_model.pdf')
summary_plot.median_plot(x=d_sim_o_tot, y=N_model_o_tot, xtype='d.sim', ytype='N.model', binsize=50, file_path_and_name=directory+'/median/N_model_vs_d_sim.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=N_sim_o_tot, xtype='d.model', ytype='N.sim', binsize=50, file_path_and_name=directory+'/median/N_sim_vs_d_model.pdf')
summary_plot.median_plot(x=d_sim_o_tot, y=N_sim_o_tot, xtype='d.sim', ytype='N.sim', binsize=50, limits=((-5,350),None), file_path_and_name=directory+'/median/N_sim_vs_d_sim_zoom.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=N_model_o_tot, xtype='d.model', ytype='N.model', binsize=50, limits=((-5,350),None), file_path_and_name=directory+'/median/N_model_vs_d_model_zoom.pdf')
summary_plot.median_plot(x=d_sim_o_tot, y=N_model_o_tot, xtype='d.sim', ytype='N.model', binsize=50, limits=((-5,350),None), file_path_and_name=directory+'/median/N_model_vs_d_sim_zoom.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=N_sim_o_tot, xtype='d.model', ytype='N.sim', binsize=50, limits=((-5,350),None), file_path_and_name=directory+'/median/N_sim_vs_d_model_zoom.pdf')



# N vs d(z = 0)
# Scatter plots
summary_plot.scatter_plot(x=dz0_tot, y=N_sim_tot, x_out=dz0_tot_out, y_out=N_sim_tot_out, xtype='d.z0', ytype='N.sim', file_path_and_name=directory+'/scatter/N_sim_vs_dz0.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=N_model_tot, x_out=dz0_tot_out, y_out=N_model_tot_out, xtype='d.z0', ytype='N.model', file_path_and_name=directory+'/scatter/N_model_vs_dz0.pdf')
#
# Median plots
summary_plot.median_plot(x=dz0_o_tot, y=N_sim_o_tot, xtype='d.z0', ytype='N.sim', binsize=50, file_path_and_name=directory+'/median/N_sim_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=N_model_o_tot, xtype='d.z0', ytype='N.model', binsize=50, file_path_and_name=directory+'/median/N_model_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=N_sim_o_tot, xtype='d.z0', ytype='N.sim', binsize=50, limits=((-5,350),None), file_path_and_name=directory+'/median/N_sim_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=N_model_o_tot, xtype='d.z0', ytype='N.model', binsize=50, limits=((-5,350),None), file_path_and_name=directory+'/median/N_model_vs_dz0_zoom.pdf')



# N vs t_peri
# Scatter plots
summary_plot.scatter_plot(x=t_sim_tot, y=N_sim_tot, x_out=t_sim_tot_out, y_out=N_sim_tot_out, xtype='t.sim', ytype='N.sim', limits=((None), (-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_t_sim.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=N_model_tot, x_out=t_model_tot_out, y_out=N_model_tot_out, xtype='t.model', ytype='N.model', limits=((None), (-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_t_model.pdf')
summary_plot.scatter_plot(x=t_sim_tot, y=N_model_tot, x_out=t_sim_tot_out, y_out=N_model_tot_out, xtype='t.sim', ytype='N.model', limits=((None), (-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_t_sim.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=N_sim_tot, x_out=t_model_tot_out, y_out=N_sim_tot_out, xtype='t.model', ytype='N.sim', limits=((None), (-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_t_model.pdf')
#
# Median plots
summary_plot.median_plot(x=t_sim_o_tot, y=N_sim_o_tot, xtype='t.sim', ytype='N.sim', binsize=1, file_path_and_name=directory+'/median/N_sim_vs_t_sim.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=N_model_o_tot, xtype='t.model', ytype='N.model', binsize=1, file_path_and_name=directory+'/median/N_model_vs_t_model.pdf')
summary_plot.median_plot(x=t_sim_o_tot, y=N_model_o_tot, xtype='t.sim', ytype='N.model', binsize=1, file_path_and_name=directory+'/median/N_model_vs_t_sim.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=N_sim_o_tot, xtype='t.model', ytype='N.sim', binsize=1, file_path_and_name=directory+'/median/N_sim_vs_t_model.pdf')



# N vs t_infall
# Scatter plots
summary_plot.scatter_plot(x=t_in_tot, y=N_sim_tot, x_out=t_in_tot_out, y_out=N_sim_tot_out, xtype='t.infall', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_t_infall.pdf')
summary_plot.scatter_plot(x=t_in_tot, y=N_model_tot, x_out=t_in_tot_out, y_out=N_model_tot_out, xtype='t.infall', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_t_infall.pdf')
#
# Median plots
summary_plot.median_plot(x=t_in_o_tot, y=N_sim_o_tot, xtype='t.infall', ytype='N.sim', binsize=1, file_path_and_name=directory+'/median/N_sim_vs_t_infall.pdf')
summary_plot.median_plot(x=t_in_o_tot, y=N_model_o_tot, xtype='t.infall', ytype='N.model', binsize=1, file_path_and_name=directory+'/median/N_model_vs_t_infall.pdf')



# N vs Mstar (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_z0_tot, y=N_sim_tot, x_out=Mstar_z0_tot_out, y_out=N_sim_tot_out, xtype='M.star.z0', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_mstar_z0.pdf')
summary_plot.scatter_plot(x=Mstar_z0_tot, y=N_model_tot, x_out=Mstar_z0_tot_out, y_out=N_model_tot_out, xtype='M.star.z0', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_mstar_z0.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_z0_o_tot, y=N_sim_o_tot, xtype='M.star.z0', ytype='N.sim', binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_mstar_z0.pdf')
summary_plot.median_plot(x=Mstar_z0_o_tot, y=N_model_o_tot, xtype='M.star.z0', ytype='N.model', binsize=0.5, file_path_and_name=directory+'/median/N_model_vs_mstar_z0.pdf')



# N vs Mstar (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_peak_tot, y=N_sim_tot, x_out=Mstar_peak_tot_out, y_out=N_sim_tot_out, xtype='M.star.peak', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_mstar_peak.pdf')
summary_plot.scatter_plot(x=Mstar_peak_tot, y=N_model_tot, x_out=Mstar_peak_tot_out, y_out=N_model_tot_out, xtype='M.star.peak', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_mstar_peak.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_peak_o_tot, y=N_sim_o_tot, xtype='M.star.peak', ytype='N.sim', binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_mstar_peak.pdf')
summary_plot.median_plot(x=Mstar_peak_o_tot, y=N_model_o_tot, xtype='M.star.peak', ytype='N.model', binsize=0.5, file_path_and_name=directory+'/median/N_model_vs_mstar_peak.pdf')



# N vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=N_sim_tot, x_out=Mhalo_z0_tot_out, y_out=N_sim_tot_out, xtype='M.halo.z0', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_mhalo_z0.pdf')
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=N_model_tot, x_out=Mhalo_z0_tot_out, y_out=N_model_tot_out, xtype='M.halo.z0', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_mhalo_z0.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=N_sim_o_tot, xtype='M.halo.z0', ytype='N.sim', binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_mhalo_z0.pdf')
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=N_model_o_tot, xtype='M.halo.z0', ytype='N.model', binsize=0.5, file_path_and_name=directory+'/median/N_model_vs_mhalo_z0.pdf')



# N vs Mhalo (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=N_sim_tot, x_out=Mhalo_peak_tot_out, y_out=N_sim_tot_out, xtype='M.halo.peak', ytype='N.sim', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_sim_vs_mhalo_peak.pdf')
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=N_model_tot, x_out=Mhalo_peak_tot_out, y_out=N_model_tot_out, xtype='M.halo.peak', ytype='N.model', limits=(None,(-0.5,13.5)), file_path_and_name=directory+'/scatter/N_model_vs_mhalo_peak.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=N_sim_o_tot, xtype='M.halo.peak', ytype='N.sim', binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_mhalo_peak.pdf')
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=N_model_o_tot, xtype='M.halo.peak', ytype='N.model', binsize=0.5, file_path_and_name=directory+'/median/N_model_vs_mhalo_peak.pdf')



# Recent pericenter distance comparison
# no oversample, cases with peris in sim and model, but outliers in red
summary_plot.scatter_plot(x=d_sim_tot, y=d_model_tot, x_out=d_sim_tot_out, y_out=d_model_tot_out, xtype='d.sim', ytype='d.model', limits=(-10,350), file_path_and_name=directory+'/scatter/recent_peri_comparison.pdf')
#
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=d_sim_o_tot, y=d_model_o_tot, binsize=20, xtype='d.sim', ytype='d.model', file_path_and_name=directory+'/median/recent_peri_comparison.pdf')



# d_peri histograms
summary_plot.plot_hist(x=d_sim_o_tot, binsize=10, pdf=True, xtype='d.sim', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_peri_sim_histogram_pdf.pdf')
summary_plot.plot_hist(x=d_sim_tot, binsize=10, pdf=False, xtype='d.sim', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_peri_sim_histogram.pdf')
#
summary_plot.plot_hist(x=d_model_o_tot, binsize=10, pdf=True, xtype='d.model', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_peri_model_histogram_pdf.pdf')
summary_plot.plot_hist(x=d_model_tot, binsize=10, pdf=False, xtype='d.model', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_peri_model_histogram.pdf')



# delta d_peri fraction histogram
# oversample, cases with pericenters in sim, but not required in model
summary_plot.plot_hist(delta_dfo_tot, binsize=0.1, pdf=True, xtype='delta.d.frac', file_path_and_name=directory+'/histogram/peri_diff_frac_histogram.pdf')
summary_plot.plot_hist(delta_dfo_tot, binsize=0.1, pdf=True, xlimits=(-1,2), xtype='delta.d.frac', file_path_and_name=directory+'/histogram/peri_diff_frac_histogram_zoom.pdf')



# delta d_peri fraction vs d_peri
# no oversample, cases with peris in sim and model, but outliers in red
# Scatter plots
summary_plot.scatter_plot(x=d_sim_tot, y=delta_df_tot, x_out=d_sim_tot_out, y_out=delta_df_tot_out, xtype='d.sim', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_sim.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_df_tot, x_out=d_model_tot_out, y_out=delta_df_tot_out, xtype='d.model', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_model.pdf')
summary_plot.scatter_plot(x=d_sim_tot, y=delta_df_tot, x_out=d_sim_tot_out, y_out=delta_df_tot_out, xtype='d.sim', ytype='delta.d.frac', limits=((-5,350),(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_sim_zoom.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_df_tot, x_out=d_model_tot_out, y_out=delta_df_tot_out, xtype='d.model', ytype='delta.d.frac', limits=((-5,350),(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_model_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=d_sim_o_tot, y=delta_dfo_tot, binsize=50, xtype='d.sim', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_d_sim.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_dfo_tot, binsize=50, xtype='d.model', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_d_model.pdf')
summary_plot.median_plot(x=d_sim_o_tot, y=delta_dfo_tot, binsize=50, xtype='d.sim', ytype='delta.d.frac', limits=((0,350),(-1,1.5)), file_path_and_name=directory+'/median/delta_d_frac_vs_d_sim_zoom.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_dfo_tot, binsize=50, xtype='d.model', ytype='delta.d.frac', limits=((0,350),(-1,1.5)), file_path_and_name=directory+'/median/delta_d_frac_vs_d_model_zoom.pdf')



# delta d_peri fraction vs d(z = 0)
# Scatter plots
summary_plot.scatter_plot(x=dz0_tot, y=delta_df_tot, x_out=dz0_tot_out, y_out=delta_df_tot_out, xtype='d.z0', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_z0.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=delta_df_tot, x_out=dz0_tot_out, y_out=delta_df_tot_out, xtype='d.z0', ytype='delta.d.frac', limits=((-5,350),(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_d_z0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=dz0_o_tot, y=delta_dfo_tot, binsize=50, xtype='d.z0', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_d_z0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=delta_dfo_tot, binsize=50, xtype='d.z0', ytype='delta.d.frac', limits=((-5,350),(-1.1,1.5)), file_path_and_name=directory+'/median/delta_d_frac_vs_d_z0_zoom.pdf')



# delta_d fraction vs t_peri
# Scatter plots
summary_plot.scatter_plot(x=t_sim_tot, y=delta_df_tot, x_out=t_sim_tot_out, y_out=delta_df_tot_out, xtype='t.sim', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_sim.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_df_tot, x_out=t_model_tot_out, y_out=delta_df_tot_out, xtype='t.model', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_model.pdf')
summary_plot.scatter_plot(x=t_sim_tot, y=delta_df_tot, x_out=t_sim_tot_out, y_out=delta_df_tot_out, xtype='t.sim', ytype='delta.d.frac', limits=((None),(-1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_sim_zoom.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_df_tot, x_out=t_model_tot_out, y_out=delta_df_tot_out, xtype='t.model', ytype='delta.d.frac', limits=((None),(-1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_model_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=t_sim_o_tot, y=delta_dfo_tot, binsize=1, xtype='t.sim', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_t_sim.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_dfo_tot, binsize=1, xtype='t.model', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_t_model.pdf')
summary_plot.median_plot(x=t_sim_o_tot, y=delta_dfo_tot, binsize=1, xtype='t.sim', ytype='delta.d.frac', limits=((None),(-1,4)), file_path_and_name=directory+'/median/delta_d_frac_vs_t_sim_zoom.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_dfo_tot, binsize=1, xtype='t.model', ytype='delta.d.frac', limits=((None),(-1,4)), file_path_and_name=directory+'/median/delta_d_frac_vs_t_model_zoom.pdf')



# delta_d fraction vs t_infall
# Scatter plots
summary_plot.scatter_plot(x=t_in_tot, y=delta_df_tot, x_out=t_in_tot_out, y_out=delta_df_tot_out, xtype='t.infall', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_infall.pdf')
summary_plot.scatter_plot(x=t_in_tot, y=delta_df_tot, x_out=t_in_tot_out, y_out=delta_df_tot_out, xtype='t.infall', ytype='delta.d.frac', limits=(None,(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_t_infall_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=t_in_o_tot, y=delta_dfo_tot, binsize=1, xtype='t.infall', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_t_infall.pdf')



# delta_d fraction vs N
# Scatter plots
summary_plot.scatter_plot(x=N_sim_tot, y=delta_df_tot, x_out=N_sim_tot_out, y_out=delta_df_tot_out, xtype='N.sim', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_sim.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_df_tot, x_out=N_model_tot_out, y_out=delta_df_tot_out, xtype='N.model', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_model.pdf')
summary_plot.scatter_plot(x=N_sim_tot, y=delta_df_tot, x_out=N_sim_tot_out, y_out=delta_df_tot_out, xtype='N.sim', ytype='delta.d.frac', limits=((-0.5,13.5),(-1,3)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_sim_zoom.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_df_tot, x_out=N_model_tot_out, y_out=delta_df_tot_out, xtype='N.model', ytype='delta.d.frac', limits=((-0.5,13.5),(-1,3)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_N_model_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=N_sim_o_tot, y=delta_dfo_tot, binsize=1, xtype='N.sim', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_N_sim.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_dfo_tot, binsize=1, xtype='N.model', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_N_model.pdf')



# delta_d fraction vs Mstar (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_df_tot, x_out=Mstar_z0_tot_out, y_out=delta_df_tot_out, xtype='M.star.z0', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_z0.pdf')
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_df_tot, x_out=Mstar_z0_tot_out, y_out=delta_df_tot_out, xtype='M.star.z0', ytype='delta.d.frac', limits=(None,(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_z0_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_dfo_tot, binsize=0.5, xtype='M.star.z0', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mstar_z0.pdf')



# delta_d fraction vs Mstar (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_df_tot, x_out=Mstar_peak_tot_out, y_out=delta_df_tot_out, xtype='M.star.peak', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_peak.pdf')
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_df_tot, x_out=Mstar_peak_tot_out, y_out=delta_df_tot_out, xtype='M.star.peak', ytype='delta.d.frac', limits=(None,(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mstar_peak_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mstar_peak_o_tot, y=delta_dfo_tot, binsize=0.5, xtype='M.star.peak', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mstar_peak.pdf')



# delta_d fraction vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_df_tot, x_out=Mhalo_z0_tot_out, y_out=delta_df_tot_out, xtype='M.halo.z0', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_z0.pdf')
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_df_tot, x_out=Mhalo_z0_tot_out, y_out=delta_df_tot_out, xtype='M.halo.z0', ytype='delta.d.frac', limits=(None,(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_z0_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=delta_dfo_tot, binsize=0.5, xtype='M.halo.z0', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mhalo_z0.pdf')



# delta_d fraction vs Mhalo (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_df_tot, x_out=Mhalo_peak_tot_out, y_out=delta_df_tot_out, xtype='M.halo.peak', ytype='delta.d.frac', file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_peak.pdf')
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_df_tot, x_out=Mhalo_peak_tot_out, y_out=delta_df_tot_out, xtype='M.halo.peak', ytype='delta.d.frac', limits=(None,(-1.1,2.5)), file_path_and_name=directory+'/scatter/delta_d_frac_vs_mhalo_peak_zoom.pdf')
#
# Median plots
# oversample, cases with peris in sim, but not required in model
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=delta_dfo_tot, binsize=0.5, xtype='M.halo.peak', ytype='delta.d.frac', file_path_and_name=directory+'/median/delta_d_frac_vs_mhalo_peak.pdf')



# delta d_peri histogram
summary_plot.plot_hist(delta_do_tot, binsize=20, pdf=True, xtype='delta.d', file_path_and_name=directory+'/histogram/peri_diff_histogram.pdf')
summary_plot.plot_hist(delta_do_tot, binsize=20, pdf=True, xlimits=(-100,150), xtype='delta.d', file_path_and_name=directory+'/histogram/peri_diff_histogram_zoom.pdf')



# delta d_peri vs d_peri
# Scatter plots
summary_plot.scatter_plot(x=d_sim_tot, y=delta_d_tot, x_out=d_sim_tot_out, y_out=delta_d_tot_out, xtype='d.sim', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_d_sim.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_d_tot, x_out=d_model_tot_out, y_out=delta_d_tot_out, xtype='d.model', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_d_model.pdf')
summary_plot.scatter_plot(x=d_sim_tot, y=delta_d_tot, x_out=d_sim_tot_out, y_out=delta_d_tot_out, xtype='d.sim', ytype='delta.d', limits=((-5,350), (-300,400)), file_path_and_name=directory+'/scatter/delta_d_vs_d_sim_zoom.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_d_tot, x_out=d_model_tot_out, y_out=delta_d_tot_out, xtype='d.model', ytype='delta.d', limits=((-5,350), (-300,400)), file_path_and_name=directory+'/scatter/delta_d_vs_d_model_zoom.pdf')
#
# Median Plots, oversample, with outliers
summary_plot.median_plot(x=d_sim_o_tot, y=delta_do_tot, binsize=50, xtype='d.sim', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_d_sim.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_do_tot, binsize=50, xtype='d.model', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_d_model.pdf')
summary_plot.median_plot(x=d_sim_o_tot, y=delta_do_tot, binsize=50, xtype='d.sim', ytype='delta.d', limits=((-5,350), (-100,300)), file_path_and_name=directory+'/median/delta_d_vs_d_sim_zoom.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_do_tot, binsize=50, xtype='d.model', ytype='delta.d', limits=((-5,350), (-100,300)), file_path_and_name=directory+'/median/delta_d_vs_d_model_zoom.pdf')



# delta d_peri vs d(z = 0)
# Scatter plots
summary_plot.scatter_plot(x=dz0_tot, y=delta_d_tot, x_out=dz0_tot_out, y_out=delta_d_tot_out, xtype='d.z0', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_d_z0.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=delta_d_tot, x_out=dz0_tot_out, y_out=delta_d_tot_out, xtype='d.z0', ytype='delta.d', limits=((-5,350),(-200,200)), file_path_and_name=directory+'/scatter/delta_d_vs_d_z0_zoom.pdf')
#
# Median Plots, oversample, with outliers
summary_plot.median_plot(x=dz0_o_tot, y=delta_do_tot, binsize=50, xtype='d.z0', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_d_z0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=delta_do_tot, binsize=50, xtype='d.z0', ytype='delta.d', limits=((-5,350),(-50,100)), file_path_and_name=directory+'/median/delta_d_vs_d_z0_zoom.pdf')



# delta d_peri vs t_peri
# Scatter plots
summary_plot.scatter_plot(x=t_sim_tot, y=delta_d_tot, x_out=t_sim_tot_out, y_out=delta_d_tot_out, xtype='t.sim', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_t_sim.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_d_tot, x_out=t_model_tot_out, y_out=delta_d_tot_out, xtype='t.model', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_t_model.pdf')
#
# Median plots
summary_plot.median_plot(x=t_sim_o_tot, y=delta_do_tot, binsize=1, xtype='t.sim', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_t_sim.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_do_tot, binsize=1, xtype='t.model', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_t_model.pdf')
summary_plot.median_plot(x=t_sim_o_tot, y=delta_do_tot, binsize=1, xtype='t.sim', ytype='delta.d', limits=(None, (-150,150)), file_path_and_name=directory+'/median/delta_d_vs_t_sim_zoom.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_do_tot, binsize=1, xtype='t.model', ytype='delta.d', limits=(None, (-150,150)), file_path_and_name=directory+'/median/delta_d_vs_t_model_zoom.pdf')



# delta d_peri vs t_infall
# Scatter plots
summary_plot.scatter_plot(x=t_in_tot, y=delta_d_tot, x_out=t_in_tot_out, y_out=delta_d_tot_out, xtype='t.infall', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_t_infall.pdf')
summary_plot.scatter_plot(x=t_in_tot, y=delta_d_tot, x_out=t_in_tot_out, y_out=delta_d_tot_out, xtype='t.infall', ytype='delta.d', limits=(None,(-100,200)), file_path_and_name=directory+'/scatter/delta_d_vs_t_infall_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=t_in_o_tot, y=delta_do_tot, binsize=1, xtype='t.infall', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_t_infall.pdf')



# delta d_peri vs N
# Scatter plots
summary_plot.scatter_plot(x=N_sim_tot, y=delta_d_tot, x_out=N_sim_tot_out, y_out=delta_d_tot_out, xtype='N.sim', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_N_sim.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_d_tot, x_out=N_model_tot_out, y_out=delta_d_tot_out, xtype='N.model', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_N_model.pdf')
#
# Median plots
summary_plot.median_plot(x=N_sim_o_tot, y=delta_do_tot, binsize=1, xtype='N.sim', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_N_sim.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_do_tot, binsize=1, xtype='N.model', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_N_model.pdf')



# delta d_peri vs Mstar (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_d_tot, x_out=Mstar_z0_tot_out, y_out=delta_d_tot_out, xtype='M.star.z0', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_mstar_z0.pdf')
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_d_tot, x_out=Mstar_z0_tot_out, y_out=delta_d_tot_out, xtype='M.star.z0', ytype='delta.d', limits=(None,(-100,200)), file_path_and_name=directory+'/scatter/delta_d_vs_mstar_z0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_do_tot, binsize=0.5, xtype='M.star.z0', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_mstar_z0.pdf')



# delta d_peri vs Mstar (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_d_tot, x_out=Mstar_peak_tot_out, y_out=delta_d_tot_out, xtype='M.star.peak', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_mstar_peak.pdf')
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_d_tot, x_out=Mstar_peak_tot_out, y_out=delta_d_tot_out, xtype='M.star.peak', ytype='delta.d', limits=(None,(-100,200)), file_path_and_name=directory+'/scatter/delta_d_vs_mstar_peak_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_peak_o_tot, y=delta_do_tot, binsize=0.5, xtype='M.star.peak', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_mstar_peak.pdf')



# delta d_peri vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_d_tot, x_out=Mhalo_z0_tot_out, y_out=delta_d_tot_out, xtype='M.halo.z0', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_mhalo_z0.pdf')
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_d_tot, x_out=Mhalo_z0_tot_out, y_out=delta_d_tot_out, xtype='M.halo.z0', ytype='delta.d', limits=(None,(-100,200)), file_path_and_name=directory+'/scatter/delta_d_vs_mhalo_z0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=delta_do_tot, binsize=0.5, xtype='M.halo.z0', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_mhalo_z0.pdf')



# delta d_peri vs Mhalo (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_d_tot, x_out=Mhalo_peak_tot_out, y_out=delta_d_tot_out, xtype='M.halo.peak', ytype='delta.d', file_path_and_name=directory+'/scatter/delta_d_vs_mhalo_peak.pdf')
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_d_tot, x_out=Mhalo_peak_tot_out, y_out=delta_d_tot_out, xtype='M.halo.peak', ytype='delta.d', limits=(None,(-100,200)), file_path_and_name=directory+'/scatter/delta_d_vs_mhalo_peak_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=delta_do_tot, binsize=0.5, xtype='M.halo.peak', ytype='delta.d', file_path_and_name=directory+'/median/delta_d_vs_mhalo_peak.pdf')



# Recent pericenter time comparison
# no oversample, cases with peris in sim and model, but outliers in red
summary_plot.scatter_plot(x=t_sim_tot, y=t_model_tot, x_out=t_sim_tot_out, y_out=t_model_tot_out, xtype='t.sim', ytype='t.model', limits=((-0.5, 13.8)), file_path_and_name=directory+'/scatter/recent_tperi_comparison.pdf')
#
# Median plot
summary_plot.median_plot(x=t_sim_o_tot, y=t_model_o_tot, xtype='t.sim', ytype='t.model', binsize=0.5, file_path_and_name=directory+'/median/recent_tperi_comparison.pdf')



# t_peri histograms
summary_plot.plot_hist(x=t_sim_o_tot, binsize=0.5, pdf=True, xtype='t.sim', xlimits=(-0.5,14), file_path_and_name=directory+'/histogram/t_peri_sim_histogram_pdf.pdf')
summary_plot.plot_hist(x=t_sim_tot, binsize=0.5, pdf=False, xtype='t.sim', xlimits=(-0.5,14), file_path_and_name=directory+'/histogram/t_peri_sim_histogram.pdf')
#
summary_plot.plot_hist(x=t_model_o_tot, binsize=0.5, pdf=True, xtype='t.model', xlimits=(-0.5,14), file_path_and_name=directory+'/histogram/t_peri_model_histogram_pdf.pdf')
summary_plot.plot_hist(x=t_model_tot, binsize=0.5, pdf=False, xtype='t.model', xlimits=(-0.5,14), file_path_and_name=directory+'/histogram/t_peri_model_histogram.pdf')



# delta t_peri fractions
# Histogram
# oversample, cases with pericenters in sim, but not required in model
summary_plot.plot_hist(delta_tfo_tot, binsize=0.1, pdf=True, xtype='delta.t.frac', file_path_and_name=directory+'/histogram/peri_tlb_diff_frac_histogram.pdf')
summary_plot.plot_hist(delta_tfo_tot, binsize=0.1, pdf=True, xlimits=(-1,2), xtype='delta.t.frac', file_path_and_name=directory+'/histogram/peri_tlb_diff_frac_histogram_zoom.pdf')



# delta t_peri fraction vs t_peri
# no oversample, cases with peris in sim and model, but outliers in red
# Scatter plots
summary_plot.scatter_plot(x=t_sim_tot, y=delta_tf_tot, x_out=t_sim_tot_out, y_out=delta_tf_tot_out, xtype='t.sim', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_t_sim.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_tf_tot, x_out=t_model_tot_out, y_out=delta_tf_tot_out, xtype='t.model', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_t_model.pdf')
summary_plot.scatter_plot(x=t_sim_tot, y=delta_tf_tot, x_out=t_sim_tot_out, y_out=delta_tf_tot_out, xtype='t.sim', ytype='delta.t.frac', limits=((0, 10.5),(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_t_sim_zoom.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_tf_tot, x_out=t_model_tot_out, y_out=delta_tf_tot_out, xtype='t.model', ytype='delta.t.frac', limits=((0, 10.5),(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_t_model_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=t_sim_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='t.sim', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_t_sim.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='t.model', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_t_model.pdf')
summary_plot.median_plot(x=t_sim_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='t.sim', ytype='delta.t.frac', limits=((0, 11),(-1.1, 1)), file_path_and_name=directory+'/median/delta_t_frac_vs_t_sim_zoom.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='t.model', ytype='delta.t.frac', limits=((0, 14),(-1.1, 5)), file_path_and_name=directory+'/median/delta_t_frac_vs_t_model_zoom.pdf')



# delta t_peri fraction vs t_infall
# no oversample, cases with peris in sim and model, but outliers in red
# Scatter plots
summary_plot.scatter_plot(x=t_in_tot, y=delta_tf_tot, x_out=t_in_tot_out, y_out=delta_tf_tot_out, xtype='t.infall', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_t_infall.pdf')
summary_plot.scatter_plot(x=t_in_tot, y=delta_tf_tot, x_out=t_in_tot_out, y_out=delta_tf_tot_out, xtype='t.infall', ytype='delta.t.frac', limits=(None,(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_t_infall_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=t_in_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='t.infall', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_t_infall.pdf')



# delta t_peri fraction vs d_peri
# Scatter plots
summary_plot.scatter_plot(x=d_sim_tot, y=delta_tf_tot, x_out=d_sim_tot_out, y_out=delta_tf_tot_out, xtype='d.sim', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_d_sim.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_tf_tot, x_out=d_model_tot_out, y_out=delta_tf_tot_out, xtype='d.model', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_d_model.pdf')
summary_plot.scatter_plot(x=d_sim_tot, y=delta_tf_tot, x_out=d_sim_tot_out, y_out=delta_tf_tot_out, xtype='d.sim', ytype='delta.t.frac', limits=((-5,350),(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_d_sim_zoom.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_tf_tot, x_out=d_model_tot_out, y_out=delta_tf_tot_out, xtype='d.model', ytype='delta.t.frac', limits=((-5,350),(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_d_model_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=d_sim_o_tot, y=delta_tfo_tot, binsize=50, xtype='d.sim', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_d_sim.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_tfo_tot, binsize=50, xtype='d.model', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_d_model.pdf')
summary_plot.median_plot(x=d_sim_o_tot, y=delta_tfo_tot, binsize=50, xtype='d.sim', ytype='delta.t.frac', limits=((-5,350),(-1,1)), file_path_and_name=directory+'/median/delta_t_frac_vs_d_sim_zoom.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_tfo_tot, binsize=50, xtype='d.model', ytype='delta.t.frac', limits=((-5,350),None), file_path_and_name=directory+'/median/delta_t_frac_vs_d_model_zoom.pdf')



# delta t_peri fraction vs d(z = 0)
# no oversample, cases with peris in sim and model, but outliers in red
# Scatter plots
summary_plot.scatter_plot(x=dz0_tot, y=delta_tf_tot, x_out=dz0_tot_out, y_out=delta_tf_tot_out, xtype='d.z0', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_d_z0.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=delta_tf_tot, x_out=dz0_tot_out, y_out=delta_tf_tot_out, xtype='d.z0', ytype='delta.t.frac', limits=((-5,350),(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_d_z0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=dz0_o_tot, y=delta_tfo_tot, binsize=50, xtype='d.z0', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_d_z0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=delta_tfo_tot, binsize=50, xtype='d.z0', ytype='delta.t.frac', limits=((-5,350),None), file_path_and_name=directory+'/median/delta_t_frac_vs_d_z0_zoom.pdf')



# delta t_peri fraction vs Nperi
# Scatter plots
summary_plot.scatter_plot(x=N_sim_tot, y=delta_tf_tot, x_out=N_sim_tot_out, y_out=delta_tf_tot_out, xtype='N.sim', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_N_sim.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_tf_tot, x_out=N_model_tot_out, y_out=delta_tf_tot_out, xtype='N.model', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_N_model.pdf')
summary_plot.scatter_plot(x=N_sim_tot, y=delta_tf_tot, x_out=N_sim_tot_out, y_out=delta_tf_tot_out, xtype='N.sim', ytype='delta.t.frac', limits=((-0.5,13.5),(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_N_sim_zoom.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_tf_tot, x_out=N_model_tot_out, y_out=delta_tf_tot_out, xtype='N.model', ytype='delta.t.frac', limits=((-0.5,13.5),(-1.1,2)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_N_model_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=N_sim_o_tot, y=delta_tfo_tot, binsize=1, xtype='N.sim', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_N_sim.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_tfo_tot, binsize=1, xtype='N.model', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_N_model.pdf')
summary_plot.median_plot(x=N_sim_o_tot, y=delta_tfo_tot, binsize=1, xtype='N.sim', ytype='delta.t.frac', limits=((-0.5,13.5),(-1,1)), file_path_and_name=directory+'/median/delta_t_frac_vs_N_sim_zoom.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_tfo_tot, binsize=1, xtype='N.model', ytype='delta.t.frac', limits=((-0.5,13.5),(-1,1)), file_path_and_name=directory+'/median/delta_t_frac_vs_N_model_zoom.pdf')



# delta t_peri fraction vs Mstar (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_tf_tot, x_out=Mstar_z0_tot_out, y_out=delta_tf_tot_out, xtype='M.star.z0', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_mstar_z0.pdf')
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_tf_tot, x_out=Mstar_z0_tot_out, y_out=delta_tf_tot_out, xtype='M.star.z0', ytype='delta.t.frac', limits=(None,(-1.1,4.5)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_mstar_z0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.star.z0', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_mstar_z0.pdf')
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.star.z0', ytype='delta.t.frac', limits=(None,(-1,1)), file_path_and_name=directory+'/median/delta_t_frac_vs_mstar_z0_zoom.pdf')



# delta t_peri fraction vs Mstar (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_tf_tot, x_out=Mstar_peak_tot_out, y_out=delta_tf_tot_out, xtype='M.star.peak', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_mstar_peak.pdf')
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_tf_tot, x_out=Mstar_peak_tot_out, y_out=delta_tf_tot_out, xtype='M.star.peak', ytype='delta.t.frac', limits=(None,(-1.1,4.5)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_mstar_peak_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_peak_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.star.peak', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_mstar_peak.pdf')
summary_plot.median_plot(x=Mstar_peak_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.star.peak', ytype='delta.t.frac', limits=(None,(-1,1)), file_path_and_name=directory+'/median/delta_t_frac_vs_mstar_peak_zoom.pdf')



# delta t_peri fraction vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_tf_tot, x_out=Mhalo_z0_tot_out, y_out=delta_tf_tot_out, xtype='M.halo.z0', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_mhalo_z0.pdf')
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_tf_tot, x_out=Mhalo_z0_tot_out, y_out=delta_tf_tot_out, xtype='M.halo.z0', ytype='delta.t.frac', limits=(None,(-1.1,4.5)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_mhalo_z0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.halo.z0', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_mhalo_z0.pdf')
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.halo.z0', ytype='delta.t.frac', limits=(None,(-1,1)), file_path_and_name=directory+'/median/delta_t_frac_vs_mhalo_z0_zoom.pdf')



# delta t_peri fraction vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_tf_tot, x_out=Mhalo_peak_tot_out, y_out=delta_tf_tot_out, xtype='M.halo.peak', ytype='delta.t.frac', file_path_and_name=directory+'/scatter/delta_t_frac_vs_mhalo_peak.pdf')
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_tf_tot, x_out=Mhalo_peak_tot_out, y_out=delta_tf_tot_out, xtype='M.halo.peak', ytype='delta.t.frac', limits=(None,(-1.1,4.5)), file_path_and_name=directory+'/scatter/delta_t_frac_vs_mhalo_peak_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.halo.peak', ytype='delta.t.frac', file_path_and_name=directory+'/median/delta_t_frac_vs_mhalo_peak.pdf')
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=delta_tfo_tot, binsize=0.5, xtype='M.halo.peak', ytype='delta.t.frac', limits=(None,(-1,1)), file_path_and_name=directory+'/median/delta_t_frac_vs_mhalo_peak_zoom.pdf')



# delta t_peri histogram
summary_plot.plot_hist(delta_to_tot, pdf=True, binsize=0.5, xtype='delta.t', file_path_and_name=directory+'/histogram/peri_tlb_diff_histogram.pdf')
summary_plot.plot_hist(delta_to_tot, pdf=True, binsize=0.5, xlimits=(-3,4), xtype='delta.t', file_path_and_name=directory+'/histogram/peri_tlb_diff_histogram_zoom.pdf')



# delta t_peri vs t_peri
# Scatter plots
summary_plot.scatter_plot(x=t_sim_tot, y=delta_t_tot, x_out=t_sim_tot_out, y_out=delta_t_tot_out, xtype='t.sim', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_t_sim.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_t_tot, x_out=t_model_tot_out, y_out=delta_t_tot_out, xtype='t.model', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_t_model.pdf')
summary_plot.scatter_plot(x=t_sim_tot, y=delta_t_tot, x_out=t_sim_tot_out, y_out=delta_t_tot_out, xtype='t.sim', ytype='delta.t', limits=((-0.1, 10.5),(-5, 10)), file_path_and_name=directory+'/scatter/delta_t_vs_t_sim_zoom.pdf')
summary_plot.scatter_plot(x=t_model_tot, y=delta_t_tot, x_out=t_model_tot_out, y_out=delta_t_tot_out, xtype='t.model', ytype='delta.t', limits=((-0.1, 13.8),(-5, 10)), file_path_and_name=directory+'/scatter/delta_t_vs_t_model_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=t_sim_o_tot, y=delta_to_tot, binsize=1, xtype='t.sim', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_t_sim.pdf')
summary_plot.median_plot(x=t_model_o_tot, y=delta_to_tot, binsize=1, xtype='t.model', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_t_model.pdf')



# delta t_peri vs t_infall
# no oversample, cases with peris in sim and model, but outliers in red
# Scatter plots
summary_plot.scatter_plot(x=t_in_tot, y=delta_t_tot, x_out=t_in_tot_out, y_out=delta_t_tot_out, xtype='t.infall', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_t_infall.pdf')
summary_plot.scatter_plot(x=t_in_tot, y=delta_t_tot, x_out=t_in_tot_out, y_out=delta_t_tot_out, xtype='t.infall', ytype='delta.t', limits=(None, (-3,3)), file_path_and_name=directory+'/scatter/delta_t_vs_t_infall_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=t_in_o_tot, y=delta_to_tot, binsize=1, xtype='t.infall', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_t_infall.pdf')
summary_plot.median_plot(x=t_in_o_tot, y=delta_to_tot, binsize=1, xtype='t.infall', ytype='delta.t', limits=(None, (-1,3)), file_path_and_name=directory+'/median/delta_t_vs_t_infall_zoom.pdf')



# delta t_peri vs d_peri
# Scatter plots
summary_plot.scatter_plot(x=d_sim_tot, y=delta_t_tot, x_out=d_sim_tot_out, y_out=delta_t_tot_out, xtype='d.sim', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_d_sim.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_t_tot, x_out=d_model_tot_out, y_out=delta_t_tot_out, xtype='d.model', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_d_model.pdf')
summary_plot.scatter_plot(x=d_sim_tot, y=delta_t_tot, x_out=d_sim_tot_out, y_out=delta_t_tot_out, xtype='d.sim', ytype='delta.t', limits=((-5,350),None), file_path_and_name=directory+'/scatter/delta_t_vs_d_sim_zoom.pdf')
summary_plot.scatter_plot(x=d_model_tot, y=delta_t_tot, x_out=d_model_tot_out, y_out=delta_t_tot_out, xtype='d.model', ytype='delta.t', limits=((-5,350),None), file_path_and_name=directory+'/scatter/delta_t_vs_d_model_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=d_sim_o_tot, y=delta_to_tot, binsize=50, xtype='d.sim', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_d_sim.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_to_tot, binsize=50, xtype='d.model', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_d_model.pdf')
summary_plot.median_plot(x=d_sim_o_tot, y=delta_to_tot, binsize=50, xtype='d.sim', ytype='delta.t', limits=((-5,350),None), file_path_and_name=directory+'/median/delta_t_vs_d_sim_zoom.pdf')
summary_plot.median_plot(x=d_model_o_tot, y=delta_to_tot, binsize=50, xtype='d.model', ytype='delta.t', limits=((-5,350),(-2,4)), file_path_and_name=directory+'/median/delta_t_vs_d_model_zoom.pdf')



# delta t_peri vs d(z = 0)
# Scatter plots
summary_plot.scatter_plot(x=dz0_tot, y=delta_t_tot, x_out=dz0_tot_out, y_out=delta_t_tot_out, xtype='d.z0', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_d_z0.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=delta_t_tot, x_out=dz0_tot_out, y_out=delta_t_tot_out, xtype='d.z0', ytype='delta.t', limits=((-5,350),(-5,5)), file_path_and_name=directory+'/scatter/delta_t_vs_d_z0_zoom.pdf')
#
# Median plots
summary_plot.median_plot(x=dz0_o_tot, y=delta_to_tot, binsize=50, xtype='d.z0', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_d_z0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=delta_to_tot, binsize=50, xtype='d.z0', ytype='delta.t', limits=((-5,350),(-1,2)), file_path_and_name=directory+'/median/delta_t_vs_d_z0_zoom.pdf')



# delta t_peri vs N_peri
# Scatter plots
summary_plot.scatter_plot(x=N_sim_tot, y=delta_t_tot, x_out=N_sim_tot_out, y_out=delta_t_tot_out, xtype='N.sim', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_N_sim.pdf')
summary_plot.scatter_plot(x=N_model_tot, y=delta_t_tot, x_out=N_model_tot_out, y_out=delta_t_tot_out, xtype='N.model', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_N_model.pdf')
#
# Median plots
summary_plot.median_plot(x=N_sim_o_tot, y=delta_to_tot, binsize=1, xtype='N.sim', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_N_sim.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_to_tot, binsize=1, xtype='N.model', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_N_model.pdf')
summary_plot.median_plot(x=N_sim_o_tot, y=delta_to_tot, binsize=1, xtype='N.sim', ytype='delta.t', limits=((-0.5,13.5),(-2,2.5)), file_path_and_name=directory+'/median/delta_t_vs_N_sim_zoom.pdf')
summary_plot.median_plot(x=N_model_o_tot, y=delta_to_tot, binsize=1, xtype='N.model', ytype='delta.t', limits=((-0.5,13.5),(-2,2)), file_path_and_name=directory+'/median/delta_t_vs_N_model_zoom.pdf')



# delta t_peri vs Mstar (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_z0_tot, y=delta_t_tot, x_out=Mstar_z0_tot_out, y_out=delta_t_tot_out, xtype='M.star.z0', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_mstar_z0.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_to_tot, binsize=0.5, xtype='M.star.z0', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_mstar_z0.pdf')
summary_plot.median_plot(x=Mstar_z0_o_tot, y=delta_to_tot, binsize=0.5, xtype='M.star.z0', ytype='delta.t', limits=((4,9.5),(-1,3)), file_path_and_name=directory+'/median/delta_t_vs_mstar_z0_zoom.pdf')



# delta t_peri vs Mstar (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mstar_peak_tot, y=delta_t_tot, x_out=Mstar_peak_tot_out, y_out=delta_t_tot_out, xtype='M.star.peak', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_mstar_peak.pdf')
#
# Median plots
summary_plot.median_plot(x=Mstar_peak_o_tot, y=delta_to_tot, binsize=0.5, xtype='M.star.peak', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_mstar_peak.pdf')



# delta t_peri vs Mhalo (z = 0)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=delta_t_tot, x_out=Mhalo_z0_tot_out, y_out=delta_t_tot_out, xtype='M.halo.z0', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_mhalo_z0.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=delta_to_tot, binsize=0.5, xtype='M.halo.z0', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_mhalo_z0.pdf')



# delta t_peri vs Mhalo (peak)
# Scatter plots
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=delta_t_tot, x_out=Mhalo_peak_tot_out, y_out=delta_t_tot_out, xtype='M.halo.peak', ytype='delta.t', file_path_and_name=directory+'/scatter/delta_t_vs_mhalo_peak.pdf')
#
# Median plots
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=delta_to_tot, binsize=0.5, xtype='M.halo.peak', ytype='delta.t', file_path_and_name=directory+'/median/delta_t_vs_mhalo_peak.pdf')



# M_star histogram (z = 0)
summary_plot.plot_hist(Mstar_z0_tot, binsize=0.1, pdf=False, xtype='M.star.z0', file_path_and_name=directory+'/histogram/mstar_z0_histogram.pdf')
summary_plot.plot_hist(Mstar_z0_o_tot, binsize=0.1, pdf=True, xtype='M.star.z0', file_path_and_name=directory+'/histogram/mstar_z0_histogram_pdf.pdf')



# Mstar(z = 0) vs d(z = 0)
# Scatter plot
summary_plot.scatter_plot(x=Mstar_z0_tot, y=dz0_tot, x_out=Mstar_z0_tot_out, y_out=dz0_tot_out, xtype='M.star.z0', ytype='d.z0', file_path_and_name=directory+'/scatter/mstar_z0_vs_d_z0.pdf')
summary_plot.scatter_plot(x=Mstar_z0_tot, y=dz0_tot, x_out=Mstar_z0_tot_out, y_out=dz0_tot_out, xtype='M.star.z0', ytype='d.z0', limits=(None,(-5,350)), file_path_and_name=directory+'/scatter/mstar_z0_vs_d_z0_zoom.pdf')
#
# Median plot
summary_plot.median_plot(x=Mstar_z0_o_tot, y=dz0_o_tot, binsize=0.5, xtype='M.star.z0', ytype='d.z0', file_path_and_name=directory+'/median/mstar_z0_vs_d_z0.pdf')



# Mstar(z = 0) vs Mhalo(z = 0)
# Scatter plot
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=Mstar_z0_tot, x_out=Mhalo_z0_tot_out, y_out=Mstar_z0_tot_out, xtype='M.halo.z0', ytype='M.star.z0', file_path_and_name=directory+'/scatter/mstar_mhalo_z0.pdf')
#
# Median plot
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=Mstar_z0_o_tot, binsize=0.5, xtype='M.halo.z0', ytype='M.star.z0', file_path_and_name=directory+'/median/mstar_mhalo_z0.pdf')



# Mstar(peak) vs Mhalo(peak)
# Scatter plot
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=Mstar_peak_tot, x_out=Mhalo_peak_tot_out, y_out=Mstar_peak_tot_out, xtype='M.halo.peak', ytype='M.star.peak', file_path_and_name=directory+'/scatter/mstar_mhalo_peak.pdf')
#
# Median plot
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=Mstar_peak_o_tot, binsize=0.5, xtype='M.halo.peak', ytype='M.star.peak', file_path_and_name=directory+'/median/mstar_mhalo_peak.pdf')



# Mstar histogram (peak)
summary_plot.plot_hist(Mstar_peak_tot, binsize=0.1, pdf=False, xtype='M.star.peak', file_path_and_name=directory+'/histogram/mstar_peak_histogram.pdf')
summary_plot.plot_hist(Mstar_peak_o_tot, binsize=0.1, pdf=True, xtype='M.star.peak', file_path_and_name=directory+'/histogram/mstar_peak_histogram_pdf.pdf')



# Mstar(peak) vs d(z = 0)
# Scatter plot
summary_plot.scatter_plot(x=Mstar_peak_tot, y=dz0_tot, x_out=Mstar_peak_tot_out, y_out=dz0_tot_out, xtype='M.star.peak', ytype='d.z0', file_path_and_name=directory+'/scatter/mstar_peak_vs_d_z0.pdf')
summary_plot.scatter_plot(x=Mstar_peak_tot, y=dz0_tot, x_out=Mstar_peak_tot_out, y_out=dz0_tot_out, xtype='M.star.peak', ytype='d.z0', limits=(None,(-5,350)), file_path_and_name=directory+'/scatter/mstar_peak_vs_d_z0_zoom.pdf')
#
# Median plot
summary_plot.median_plot(x=Mstar_peak_o_tot, y=dz0_o_tot, binsize=0.5, xtype='M.star.peak', ytype='d.z0', file_path_and_name=directory+'/median/mstar_peak_vs_d_z0.pdf')



# d_z0
summary_plot.plot_hist(x=dz0_o_tot, binsize=10, pdf=True, xtype='d.z0', file_path_and_name=directory+'/histogram/d_z0_histogram_pdf.pdf')
summary_plot.plot_hist(x=dz0_o_tot, binsize=10, pdf=True, xtype='d.z0', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_z0_histogram_pdf_zoom.pdf')
#
summary_plot.plot_hist(x=dz0_tot, binsize=10, pdf=False, xtype='d.z0', file_path_and_name=directory+'/histogram/d_z0_histogram.pdf')
summary_plot.plot_hist(x=dz0_tot, binsize=10, pdf=False, xtype='d.z0', xlimits=(-5,350), file_path_and_name=directory+'/histogram/d_z0_histogram_zoom.pdf')



# M_halo histogram (z = 0)
summary_plot.plot_hist(Mhalo_z0_tot, binsize=0.1, xtype='M.halo.z0', pdf=False, file_path_and_name=directory+'/histogram/mhalo_z0_histogram.pdf')
summary_plot.plot_hist(Mhalo_z0_o_tot, binsize=0.1, xtype='M.halo.z0', pdf=True, file_path_and_name=directory+'/histogram/mhalo_z0_histogram_pdf.pdf')



# M_halo histogram (peak)
summary_plot.plot_hist(Mhalo_peak_tot, binsize=0.1, xtype='M.halo.peak', pdf=False, file_path_and_name=directory+'/histogram/mhalo_peak_histogram.pdf')
summary_plot.plot_hist(Mhalo_peak_o_tot, binsize=0.1, xtype='M.halo.peak', pdf=True, file_path_and_name=directory+'/histogram/mhalo_peak_histogram_pdf.pdf')



# t_infall histogram
summary_plot.plot_hist(t_in_o_tot, binsize=0.5, pdf=True, xtype='t.infall', xlimits=(-0.5,14), file_path_and_name=directory+'/histogram/infall_histogram_pdf.pdf')
summary_plot.plot_hist(t_in_tot, binsize=0.5, pdf=False, xtype='t.infall', xlimits=(-0.5,14), file_path_and_name=directory+'/histogram/infall_histogram.pdf')



# t_infall vs Mstar (z = 0)
# Scatter plot
summary_plot.scatter_plot(x=Mstar_z0_tot, y=t_in_tot, x_out=Mstar_z0_tot_out, y_out=t_in_tot_out, xtype='M.star.z0', ytype='t.infall', file_path_and_name=directory+'/scatter/infall_vs_mstar_z0.pdf')
#
# Median plot
summary_plot.median_plot(x=Mstar_z0_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.star.z0', ytype='t.infall', file_path_and_name=directory+'/median/infall_vs_mstar_z0.pdf')



# t_infall vs Mstar (peak)
# Scatter plot
summary_plot.scatter_plot(x=Mstar_peak_tot, y=t_in_tot, x_out=Mstar_peak_tot_out, y_out=t_in_tot_out, xtype='M.star.peak', ytype='t.infall', file_path_and_name=directory+'/scatter/infall_vs_mstar_peak.pdf')
#
# Median plot
summary_plot.median_plot(x=Mstar_peak_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.star.peak', ytype='t.infall', file_path_and_name=directory+'/median/infall_vs_mstar_peak.pdf')



# t_infall vs Mhalo (z = 0)
# Scatter plot
summary_plot.scatter_plot(x=Mhalo_z0_tot, y=t_in_tot, x_out=Mhalo_z0_tot_out, y_out=t_in_tot_out, xtype='M.halo.z0', ytype='t.infall', file_path_and_name=directory+'/scatter/infall_vs_mhalo_z0.pdf')
#
# Median plot
summary_plot.median_plot(x=Mhalo_z0_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.halo.z0', ytype='t.infall', file_path_and_name=directory+'/median/infall_vs_mhalo_z0.pdf')



# t_infall vs Mhalo (peak)
# Scatter plot
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=t_in_tot, x_out=Mhalo_peak_tot_out, y_out=t_in_tot_out, xtype='M.halo.peak', ytype='t.infall', file_path_and_name=directory+'/scatter/infall_vs_mhalo_peak.pdf')
#
# Median plot
summary_plot.median_plot(x=Mhalo_peak_o_tot, y=t_in_o_tot, binsize=0.5, xtype='M.halo.peak', ytype='t.infall', file_path_and_name=directory+'/median/infall_vs_mhalo_peak.pdf')



# t_infall vs d(z = 0)
# Scatter plot
summary_plot.scatter_plot(x=dz0_tot, y=t_in_tot, x_out=dz0_tot_out, y_out=t_in_tot_out, xtype='d.z0', ytype='t.infall', file_path_and_name=directory+'/scatter/infall_vs_d_z0.pdf')
summary_plot.scatter_plot(x=dz0_tot, y=t_in_tot, x_out=dz0_tot_out, y_out=t_in_tot_out, xtype='d.z0', ytype='t.infall', limits=((-5,350), None), file_path_and_name=directory+'/scatter/infall_vs_d_z0_zoom.pdf')
#
# Median plot
summary_plot.median_plot(x=dz0_o_tot, y=t_in_o_tot, binsize=50, xtype='d.z0', ytype='t.infall', file_path_and_name=directory+'/median/infall_vs_d_z0.pdf')
summary_plot.median_plot(x=dz0_o_tot, y=t_in_o_tot, binsize=50, xtype='d.z0', ytype='t.infall', limits=((-5,350), None), file_path_and_name=directory+'/median/infall_vs_d_z0_zoom.pdf')
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
plt.savefig(directory+'/median/infall_vs_d_z0_mass_bins.pdf')
plt.close()


# Compare the KE at pericenter and the maximum KE
summary_plot.scatter_plot(x=ke_max_tot/1e4, y=ke_peri_tot/1e4, x_out=ke_max_tot_out/1e4, y_out=ke_peri_tot_out/1e4, xtype='KE.max.sim', ytype='KE.peri.sim', limits=(0,17), file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/ke_test.pdf')
summary_plot.median_plot(x=ke_max_o_tot/1e4, y=ke_peri_o_tot/1e4, xtype='KE.max.sim', ytype='KE.peri.sim', binsize=1, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/ke_test_med.pdf')
