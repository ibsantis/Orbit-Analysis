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
data_total = summary.data_read(directory=sim_data.home_dir, hosts='iso')
data_total_dmo = summary.data_read(directory=sim_data.home_dir, hosts='iso', dmo=True)
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='iso')
masks_infall_dmo = summary.data_mask(data_total_dmo, peri_sim=False, peri_model=False, hosts='iso')
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1'


### Generate all of the data for the plots below
# Hydro
mask_selection = masks_infall
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='iso')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='iso')
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='iso')
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='iso')
#
# DMO
mask_selection = masks_infall_dmo
N_sim_tot_dmo = summary.nperi(data_total_dmo, mask_selection, oversample=True, selection='sim', hosts='iso', dmo=True)
d_sim_tot_dmo = summary.dperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso', dmo=True)
dz0_tot_dmo = summary.d_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso', dmo=True)
t_sim_tot_dmo = summary.tperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso', dmo=True)
t_in_tot_dmo = summary.first_infall(data_total_dmo, mask_selection, oversample=True, hosts='iso', dmo=True)
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, mask_selection, selection='peak', oversample=True, hosts='iso', dmo=True)


### Plots
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
