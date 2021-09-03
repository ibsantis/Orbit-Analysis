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
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1/baryon_vs_dmo'


### Generate all of the data for the plots below
# Hydro
mask_selection = masks_infall
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='iso', sim_type='baryon_all')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='baryon_all')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='baryon_all')
t_min_tot = summary.tperi_min(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
t_in_any_tot = summary.first_infall_any(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='iso', sim_type='baryon_all')
vtan_tot = summary.velocities(data_total, mask_selection, selection='tan', oversample=True, hosts='iso', sim_type='baryon_all')
vrad_tot = summary.velocities(data_total, mask_selection, selection='rad', oversample=True, hosts='iso', sim_type='baryon_all')
vz0_tot = summary.v_z0(data_total, mask_selection, oversample=True, hosts='iso', sim_type='baryon_all')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='baryon_all')
#
# DMO
mask_selection = masks_infall_dmo
N_sim_tot_dmo = summary.nperi(data_total_dmo, mask_selection, oversample=True, selection='sim', hosts='iso', sim_type='dmo')
d_sim_tot_dmo = summary.dperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='dmo')
d_min_tot_dmo = summary.dperi_min(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
dz0_tot_dmo = summary.d_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
t_sim_tot_dmo = summary.tperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='dmo')
t_min_tot_dmo = summary.tperi_min(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
t_in_tot_dmo = summary.first_infall(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
t_in_any_tot_dmo = summary.first_infall_any(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, mask_selection, selection='peak', oversample=True, hosts='iso', sim_type='dmo')
vtan_tot_dmo = summary.velocities(data_total_dmo, mask_selection, selection='tan', oversample=True, hosts='iso', sim_type='dmo')
vrad_tot_dmo = summary.velocities(data_total_dmo, mask_selection, selection='rad', oversample=True, hosts='iso', sim_type='dmo')
vz0_tot_dmo = summary.v_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso', sim_type='dmo')
L_tot_dmo = summary.L_z0(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso', sim_type='dmo')


### Histograms
# Recent pericenter distances
summary_plot.plot_hist_mult(x=[d_sim_tot, d_sim_tot_dmo], xtype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, file_path_and_name=directory+'/histogram/d_sim_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[d_sim_tot, d_sim_tot_dmo], xtype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, xlimits=[-5,400], file_path_and_name=directory+'/histogram/d_sim_comare_iso_zoom.pdf')

# Minimum pericenter distances
summary_plot.plot_hist_mult(x=[d_min_tot, d_min_tot_dmo], xtype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, file_path_and_name=directory+'/histogram/d_min_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[d_min_tot, d_min_tot_dmo], xtype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, xlimits=[-5,400], file_path_and_name=directory+'/histogram/d_min_comare_iso_zoom.pdf')

# Recent pericenter times
summary_plot.plot_hist_mult(x=[t_sim_tot, t_sim_tot_dmo], xtype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_sim_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[t_sim_tot, t_sim_tot_dmo], xtype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,11], file_path_and_name=directory+'/histogram/t_sim_comare_iso_zoom.pdf')

# Time of minimum pericenter
summary_plot.plot_hist_mult(x=[t_min_tot, t_min_tot_dmo], xtype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_min_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[t_min_tot, t_min_tot_dmo], xtype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,11], file_path_and_name=directory+'/histogram/t_min_comare_iso_zoom.pdf')

# Pericenter number
summary_plot.plot_hist_mult(x=[N_sim_tot, N_sim_tot_dmo], xtype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=1, pdf=True, file_path_and_name=directory+'/histogram/N_sim_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[N_sim_tot, N_sim_tot_dmo], xtype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=1, pdf=True, xlimits=[0,11], file_path_and_name=directory+'/histogram/N_sim_comare_iso_zoom.pdf')

# Infall times
summary_plot.plot_hist_mult(x=[t_in_tot, t_in_tot_dmo], xtype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_infall_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[t_in_tot, t_in_tot_dmo], xtype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,13.8], file_path_and_name=directory+'/histogram/t_infall_comare_iso_zoom.pdf')

# Infall times (any)
summary_plot.plot_hist_mult(x=[t_in_any_tot, t_in_any_tot_dmo], xtype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_infall_any_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[t_in_any_tot, t_in_any_tot_dmo], xtype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,13.8], file_path_and_name=directory+'/histogram/t_infall_any_comare_iso_zoom.pdf')

# d(z = 0)
summary_plot.plot_hist_mult(x=[dz0_tot, dz0_tot_dmo], xtype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, file_path_and_name=directory+'/histogram/dz0_comare_iso.pdf')
summary_plot.plot_hist_mult(x=[dz0_tot, dz0_tot_dmo], xtype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, xlimits=[0,400], file_path_and_name=directory+'/histogram/dz0_comare_iso_zoom.pdf')

# Mhalo (peak)
summary_plot.plot_hist_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/Mhalo_peak_comare_iso.pdf')

# Tangential velocity
summary_plot.plot_hist_mult(x=[vtan_tot, vtan_tot_dmo], xtype=['v.tan','v.tan'], labels=['Baryon', 'DMO'], binsize=10, pdf=True, file_path_and_name=directory+'/histogram/vtan_z0_comare_iso.pdf')

# Radial velocity
summary_plot.plot_hist_mult(x=[vrad_tot, vrad_tot_dmo], xtype=['v.rad','v.rad'], labels=['Baryon', 'DMO'], binsize=10, pdf=True, file_path_and_name=directory+'/histogram/vrad_z0_comare_iso.pdf')

# Total velocity
summary_plot.plot_hist_mult(x=[vz0_tot, vz0_tot_dmo], xtype=['v.tot','v.tot'], labels=['Baryon', 'DMO'], binsize=10, pdf=True, file_path_and_name=directory+'/histogram/vtot_z0_comare_iso.pdf')

# Total angular momentum
summary_plot.plot_hist_mult(x=[L_tot/1e4, L_tot_dmo/1e4], xtype=['L.tot','L.tot'], labels=['Baryon', 'DMO'], binsize=0.1, pdf=True, file_path_and_name=directory+'/histogram/Ltot_z0_comare_iso.pdf')


### Median plots
# N_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/N_sim_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,9)), file_path_and_name=directory+'/median/N_sim_vs_dz0_compare_iso_zoom.pdf')

# d_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/d_sim_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,200)), file_path_and_name=directory+'/median/d_sim_vs_dz0_compare_iso_zoom.pdf')

# d_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/d_min_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,200)), file_path_and_name=directory+'/median/d_min_vs_dz0_compare_iso_zoom.pdf')




# d_sim and d_recent vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot, dz0_tot_dmo, dz0_tot_dmo], y=[d_sim_tot, d_min_tot, d_sim_tot_dmo, d_min_tot_dmo], xtype=['d.z0', 'd.z0', 'd.z0', 'd.z0'], ytype=['d.sim', 'd.sim', 'd.sim', 'd.sim'], labels=['$d_{\\rm peri,recent}$, Baryon', '$d_{\\rm peri,min}$, Baryon', '$d_{\\rm peri,recent}$, DMO', '$d_{\\rm peri,min}$, DMO'], binsize=50, file_path_and_name=directory+'/median/d_peri_both_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot, dz0_tot_dmo, dz0_tot_dmo], y=[d_sim_tot, d_min_tot, d_sim_tot_dmo, d_min_tot_dmo], xtype=['d.z0', 'd.z0', 'd.z0', 'd.z0'], ytype=['d.sim', 'd.sim', 'd.sim', 'd.sim'], labels=['$d_{\\rm peri,recent}$, Baryon', '$d_{\\rm peri,min}$, Baryon', '$d_{\\rm peri,recent}$, DMO', '$d_{\\rm peri,min}$, DMO'], binsize=50, limits=((0,400), (0,200)), file_path_and_name=directory+'/median/d_peri_both_vs_dz0_compare_iso_zoom.pdf')







# t_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_sim_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,6)), file_path_and_name=directory+'/median/t_sim_vs_dz0_compare_iso_zoom.pdf')

# t_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_min_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,10)), file_path_and_name=directory+'/median/t_min_vs_dz0_compare_iso_zoom.pdf')

# Infall time vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_infall_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/t_infall_vs_dz0_compare_iso_zoom.pdf')

# Infall time (any) vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_infall_any_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/t_infall_any_vs_dz0_compare_iso_zoom.pdf')

# vtan vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/vtan_z0_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,300)), file_path_and_name=directory+'/median/vtan_z0_vs_dz0_compare_iso_zoom.pdf')

# vrad vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/vrad_z0_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/vrad_z0_vs_dz0_compare_iso_zoom.pdf')

# vtot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/vtot_z0_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,350)), file_path_and_name=directory+'/median/vtot_z0_vs_dz0_compare_iso_zoom.pdf')

# Ltot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/Ltot_vs_dz0_compare_iso.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,4)), file_path_and_name=directory+'/median/Ltot_vs_dz0_compare_iso_zoom.pdf')

# N_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,5)), file_path_and_name=directory+'/median/N_sim_vs_Mhalo_peak_compare_iso_zoom.pdf')

# d_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/d_sim_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,300)), file_path_and_name=directory+'/median/d_sim_vs_Mhalo_peak_compare_iso_zoom.pdf')

# d_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/d_min_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,200)), file_path_and_name=directory+'/median/d_min_vs_Mhalo_peak_compare_iso_zoom.pdf')

# d(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[dz0_tot, dz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/dz0_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[dz0_tot, dz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,400)), file_path_and_name=directory+'/median/dz0_vs_Mhalo_peak_compare_iso_zoom.pdf')

# t_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_sim_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,6)), file_path_and_name=directory+'/median/t_sim_vs_Mhalo_peak_compare_iso_zoom.pdf')

# t_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_min_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,10)), file_path_and_name=directory+'/median/t_min_vs_Mhalo_peak_compare_iso_zoom.pdf')

# t_infall vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_infall_vs_Mhalo_peak_compare_iso.pdf')

# t_infall (any) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_infall_any_vs_Mhalo_peak_compare_iso.pdf')

# v_tan(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/vtan_z0_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,250)), file_path_and_name=directory+'/median/vtan_z0_vs_Mhalo_peak_compare_iso_zoom.pdf')

# v_rad(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/vrad_z0_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (-150,150)), file_path_and_name=directory+'/median/vrad_z0_vs_Mhalo_peak_compare_iso_zoom.pdf')

# v_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/vtot_z0_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,250)), file_path_and_name=directory+'/median/vtot_z0_vs_Mhalo_peak_compare_iso_zoom.pdf')

# L_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/Ltot_vs_Mhalo_peak_compare_iso.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=0.5, limits=(None, (0,4)), file_path_and_name=directory+'/median/Ltot_vs_Mhalo_peak_compare_iso_zoom.pdf')




"""
    Same plots as above except no m12r
"""

# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
data_total = summary.data_read(directory=sim_data.home_dir, hosts='iso_dmo', sim_type='all_baryon')
data_total_dmo = summary.data_read(directory=sim_data.home_dir, hosts='iso_dmo', sim_type='dmo')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='iso_dmo')
masks_infall_dmo = summary.data_mask(data_total_dmo, peri_sim=False, peri_model=False, hosts='iso_dmo')
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1/baryon_vs_dmo_no_m12r'


### Generate all of the data for the plots below
# Hydro
mask_selection = masks_infall
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='iso_dmo', sim_type='baryon_all')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso_dmo', sim_type='baryon_all')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='iso_dmo', sim_type='baryon_all')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='iso_dmo', sim_type='baryon_all')
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso_dmo', sim_type='baryon_all')
t_min_tot = summary.tperi_min(data_total, mask_selection, oversample=True, hosts='iso_dmo', sim_type='baryon_all')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='iso_dmo', sim_type='baryon_all')
t_in_any_tot = summary.first_infall_any(data_total, mask_selection, oversample=True, hosts='iso_dmo', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='iso_dmo', sim_type='baryon_all')
vtan_tot = summary.velocities(data_total, mask_selection, selection='tan', oversample=True, hosts='iso_dmo', sim_type='baryon_all')
vrad_tot = summary.velocities(data_total, mask_selection, selection='rad', oversample=True, hosts='iso_dmo', sim_type='baryon_all')
vz0_tot = summary.v_z0(data_total, mask_selection, oversample=True, hosts='iso_dmo', sim_type='baryon_all')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='iso_dmo', sim_type='baryon_all')
#
# DMO
mask_selection = masks_infall_dmo
N_sim_tot_dmo = summary.nperi(data_total_dmo, mask_selection, oversample=True, selection='sim', hosts='iso_dmo', sim_type='dmo')
d_sim_tot_dmo = summary.dperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso_dmo', sim_type='dmo')
d_min_tot_dmo = summary.dperi_min(data_total_dmo, mask_selection, oversample=True, hosts='iso_dmo', sim_type='dmo')
dz0_tot_dmo = summary.d_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso_dmo', sim_type='dmo')
t_sim_tot_dmo = summary.tperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso_dmo', sim_type='dmo')
t_min_tot_dmo = summary.tperi_min(data_total_dmo, mask_selection, oversample=True, hosts='iso_dmo', sim_type='dmo')
t_in_tot_dmo = summary.first_infall(data_total_dmo, mask_selection, oversample=True, hosts='iso_dmo', sim_type='dmo')
t_in_any_tot_dmo = summary.first_infall_any(data_total_dmo, mask_selection, oversample=True, hosts='iso_dmo', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, mask_selection, selection='peak', oversample=True, hosts='iso_dmo', sim_type='dmo')
vtan_tot_dmo = summary.velocities(data_total_dmo, mask_selection, selection='tan', oversample=True, hosts='iso_dmo', sim_type='dmo')
vrad_tot_dmo = summary.velocities(data_total_dmo, mask_selection, selection='rad', oversample=True, hosts='iso_dmo', sim_type='dmo')
vz0_tot_dmo = summary.v_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso_dmo', sim_type='dmo')
L_tot_dmo = summary.L_z0(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso_dmo', sim_type='dmo')


### Histograms
# Recent pericenter distances
summary_plot.plot_hist_mult(x=[d_sim_tot, d_sim_tot_dmo], xtype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, file_path_and_name=directory+'/histogram/d_sim_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[d_sim_tot, d_sim_tot_dmo], xtype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, xlimits=[-5,400], file_path_and_name=directory+'/histogram/d_sim_comare_iso_zoom_no_r.pdf')

# Minimum pericenter distances
summary_plot.plot_hist_mult(x=[d_min_tot, d_min_tot_dmo], xtype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, file_path_and_name=directory+'/histogram/d_min_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[d_min_tot, d_min_tot_dmo], xtype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, xlimits=[-5,400], file_path_and_name=directory+'/histogram/d_min_comare_iso_zoom_no_r.pdf')

# Recent pericenter times
summary_plot.plot_hist_mult(x=[t_sim_tot, t_sim_tot_dmo], xtype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_sim_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[t_sim_tot, t_sim_tot_dmo], xtype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,11], file_path_and_name=directory+'/histogram/t_sim_comare_iso_zoom_no_r.pdf')

# Time of minimum pericenter
summary_plot.plot_hist_mult(x=[t_min_tot, t_min_tot_dmo], xtype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_min_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[t_min_tot, t_min_tot_dmo], xtype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,11], file_path_and_name=directory+'/histogram/t_min_comare_iso_zoom_no_r.pdf')

# Pericenter number
summary_plot.plot_hist_mult(x=[N_sim_tot, N_sim_tot_dmo], xtype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=1, pdf=True, file_path_and_name=directory+'/histogram/N_sim_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[N_sim_tot, N_sim_tot_dmo], xtype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=1, pdf=True, xlimits=[0,11], file_path_and_name=directory+'/histogram/N_sim_comare_iso_zoom_no_r.pdf')

# Infall times
summary_plot.plot_hist_mult(x=[t_in_tot, t_in_tot_dmo], xtype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_infall_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[t_in_tot, t_in_tot_dmo], xtype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,13.8], file_path_and_name=directory+'/histogram/t_infall_comare_iso_zoom_no_r.pdf')

# Infall times (any)
summary_plot.plot_hist_mult(x=[t_in_any_tot, t_in_any_tot_dmo], xtype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_infall_any_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[t_in_any_tot, t_in_any_tot_dmo], xtype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, xlimits=[-0.1,13.8], file_path_and_name=directory+'/histogram/t_infall_any_comare_iso_zoom_no_r.pdf')

# d(z = 0)
summary_plot.plot_hist_mult(x=[dz0_tot, dz0_tot_dmo], xtype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, file_path_and_name=directory+'/histogram/dz0_comare_iso_no_r.pdf')
summary_plot.plot_hist_mult(x=[dz0_tot, dz0_tot_dmo], xtype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=20, pdf=True, xlimits=[0,400], file_path_and_name=directory+'/histogram/dz0_comare_iso_zoom_no_r.pdf')

# Mhalo (peak)
summary_plot.plot_hist_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], labels=['Baryon', 'DMO'], binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/Mhalo_peak_comare_iso_no_r.pdf')

# Tangential velocity
summary_plot.plot_hist_mult(x=[vtan_tot, vtan_tot_dmo], xtype=['v.tan','v.tan'], labels=['Baryon', 'DMO'], binsize=10, pdf=True, file_path_and_name=directory+'/histogram/vtan_z0_comare_iso_no_r.pdf')

# Radial velocity
summary_plot.plot_hist_mult(x=[vrad_tot, vrad_tot_dmo], xtype=['v.rad','v.rad'], labels=['Baryon', 'DMO'], binsize=10, pdf=True, file_path_and_name=directory+'/histogram/vrad_z0_comare_iso_no_r.pdf')

# Total velocity
summary_plot.plot_hist_mult(x=[vz0_tot, vz0_tot_dmo], xtype=['v.tot','v.tot'], labels=['Baryon', 'DMO'], binsize=10, pdf=True, file_path_and_name=directory+'/histogram/vtot_z0_comare_iso_no_r.pdf')

# Total angular momentum
summary_plot.plot_hist_mult(x=[L_tot/1e4, L_tot_dmo/1e4], xtype=['L.tot','L.tot'], labels=['Baryon', 'DMO'], binsize=0.1, pdf=True, file_path_and_name=directory+'/histogram/Ltot_z0_comare_iso_no_r.pdf')


### Median plots
# N_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/N_sim_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,9)), file_path_and_name=directory+'/median/N_sim_vs_dz0_compare_iso_zoom_no_r.pdf')

# d_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/d_sim_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,200)), file_path_and_name=directory+'/median/d_sim_vs_dz0_compare_iso_zoom_no_r.pdf')

# d_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/d_min_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,200)), file_path_and_name=directory+'/median/d_min_vs_dz0_compare_iso_zoom_no_r.pdf')

# t_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_sim_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,6)), file_path_and_name=directory+'/median/t_sim_vs_dz0_compare_iso_zoom_no_r.pdf')

# t_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_min_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,10)), file_path_and_name=directory+'/median/t_min_vs_dz0_compare_iso_zoom_no_r.pdf')

# Infall time vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_infall_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/t_infall_vs_dz0_compare_iso_zoom_no_r.pdf')

# Infall time (any) vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/t_infall_any_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/t_infall_any_vs_dz0_compare_iso_zoom_no_r.pdf')

# vtan vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/vtan_z0_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,300)), file_path_and_name=directory+'/median/vtan_z0_vs_dz0_compare_iso_zoom_no_r.pdf')

# vrad vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/vrad_z0_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/vrad_z0_vs_dz0_compare_iso_zoom_no_r.pdf')

# vtot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/vtot_z0_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,350)), file_path_and_name=directory+'/median/vtot_z0_vs_dz0_compare_iso_zoom_no_r.pdf')

# Ltot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/median/Ltot_vs_dz0_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,4)), file_path_and_name=directory+'/median/Ltot_vs_dz0_compare_iso_zoom_no_r.pdf')

# N_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.sim', 'N.sim'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,5)), file_path_and_name=directory+'/median/N_sim_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# d_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/d_sim_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim', 'd.sim'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,300)), file_path_and_name=directory+'/median/d_sim_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# d_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/d_min_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.sim.min', 'd.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,200)), file_path_and_name=directory+'/median/d_min_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# d(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[dz0_tot, dz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/dz0_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[dz0_tot, dz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.z0', 'd.z0'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,400)), file_path_and_name=directory+'/median/dz0_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# t_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_sim_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim', 't.sim'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,6)), file_path_and_name=directory+'/median/t_sim_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# t_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_min_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.sim.min', 't.sim.min'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,10)), file_path_and_name=directory+'/median/t_min_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# t_infall vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall', 't.infall'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_infall_vs_Mhalo_peak_compare_iso_no_r.pdf')

# t_infall (any) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/t_infall_any_vs_Mhalo_peak_compare_iso_no_r.pdf')

# v_tan(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/vtan_z0_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,250)), file_path_and_name=directory+'/median/vtan_z0_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# v_rad(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/vrad_z0_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vrad_tot, vrad_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.rad', 'v.rad'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (-150,150)), file_path_and_name=directory+'/median/vrad_z0_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# v_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/vtot_z0_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,250)), file_path_and_name=directory+'/median/vtot_z0_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')

# L_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/median/Ltot_vs_Mhalo_peak_compare_iso_no_r.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5), (0,4)), file_path_and_name=directory+'/median/Ltot_vs_Mhalo_peak_compare_iso_zoom_no_r.pdf')














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
