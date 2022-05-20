#!/usr/bin/python3

"""
    ==================
    = Paper II Plots =
    ==================

    Create figures to be featured in Paper II


    Mostly just fucking around right now.

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
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2'



"""
    Properties vs Mstar
"""
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
#
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
t_min_sim = summary.tperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
t_min_mod = summary.tperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
#
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
#
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all', sim_type='baryon')


# Pericenter distances vs Mstar
summary_plot.plot_hist(x=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='delta.d.frac', pdf=True, xlimits=None, binsize=0.05, title='Recent Pericenters', file_path_and_name=directory+'/dfrac_recent_hist.pdf')
summary_plot.plot_hist(x=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='delta.d.frac', pdf=True, xlimits=(-1,3), binsize=0.05, title='Recent Pericenters', file_path_and_name=directory+'/dfrac_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(d_min_mod-d_min_sim)/d_min_sim, xtype='delta.d.frac', pdf=True, xlimits=None, binsize=0.05, title='Minimum Pericenters', file_path_and_name=directory+'/dfrac_min_hist.pdf')
summary_plot.plot_hist(x=(d_min_mod-d_min_sim)/d_min_sim, xtype='delta.d.frac', pdf=True, xlimits=(-1,5), binsize=0.05, title='Minimum Pericenters', file_path_and_name=directory+'/dfrac_min_hist_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),(-1,1)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dfrac_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),(-1,2.5)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dfrac_min_vs_mstar_zoom.pdf')
#
# Pericenter times vs Mstar
summary_plot.plot_hist(x=(t_rec_mod-t_rec_sim), xtype='delta.t', pdf=True, binsize=0.25, title='Recent Pericenters', file_path_and_name=directory+'/dt_recent_hist.pdf')
summary_plot.plot_hist(x=(t_rec_mod-t_rec_sim), xtype='delta.t', pdf=True, xlimits=(-4,4), binsize=0.25, title='Recent Pericenters', file_path_and_name=directory+'/dt_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(t_min_mod-t_min_sim), xtype='delta.t', pdf=True, binsize=0.25, title='Minimum Pericenters', file_path_and_name=directory+'/dt_min_hist.pdf')
summary_plot.plot_hist(x=(t_min_mod-t_min_sim), xtype='delta.t', pdf=True, xlimits=(-11,12), binsize=0.25, title='Minimum Pericenters', file_path_and_name=directory+'/dt_min_hist_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0, y=(t_rec_mod-t_rec_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-13.8,13.8)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dt_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_rec_mod-t_rec_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-2,3)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dt_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_min_mod-t_min_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-15,15)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dt_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_min_mod-t_min_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-5,13)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dt_min_vs_mstar_zoom.pdf')
#
# Pericenter velocities vs Mstar
summary_plot.plot_hist(x=(v_rec_mod-v_rec_sim), xtype='delta.v', pdf=True, binsize=10, title='Recent Pericenters', file_path_and_name=directory+'/dv_recent_hist.pdf')
summary_plot.plot_hist(x=(v_rec_mod-v_rec_sim), xtype='delta.v', pdf=True, xlimits=(-300,200), binsize=10, title='Recent Pericenters', file_path_and_name=directory+'/dv_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(v_min_mod-v_min_sim), xtype='delta.v', pdf=True, binsize=10, title='Minimum Pericenters', file_path_and_name=directory+'/dv_min_hist.pdf')
summary_plot.plot_hist(x=(v_min_mod-v_min_sim), xtype='delta.v', pdf=True, xlimits=(-300,200), binsize=10, title='Minimum Pericenters', file_path_and_name=directory+'/dv_min_hist_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0, y=(v_rec_mod-v_rec_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dv_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(v_rec_mod-v_rec_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),(-150,50)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dv_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(v_min_mod-v_min_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dv_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(v_min_mod-v_min_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),(-150,200)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dv_min_vs_mstar_zoom.pdf')


# Apocenter distances vs Mstar
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all', sim_type='baryon')
tapo_rec_sim = summary.tapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all', sim_type='baryon')
tapo_rec_mod = summary.tapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_apo, selection='z0', oversample=True, hosts='all', sim_type='baryon')
#
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim), xtype='delta.dapo', pdf=True, binsize=10, title='Recent apocenter', file_path_and_name=directory+'/d_dapo_hist.pdf')
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim), xtype='delta.dapo', pdf=True, xlimits=(-100,200), binsize=10, title='Recent apocenter', file_path_and_name=directory+'/d_dapo_hist_zoom.pdf')
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='delta.dapo.frac', pdf=True, binsize=0.1, title='Recent apocenter', file_path_and_name=directory+'/dfrac_dapo_hist.pdf')
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='delta.dapo.frac', pdf=True, xlimits=(-1,2), binsize=0.1, title='Recent apocenter', file_path_and_name=directory+'/dfrac_dapo_hist_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim), xtype='M.star.z0', ytype='delta.dapo', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_dapo_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim), xtype='M.star.z0', ytype='delta.dapo', limits=((4.5,9.5),(-50,150)), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_dapo_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='M.star.z0', ytype='delta.dapo.frac', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='M.star.z0', ytype='delta.dapo.frac', limits=((4.5,9.5),(-0.5,1)), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_mstar_zoom.pdf')
#
# Apocenter times vs Mstar
summary_plot.plot_hist(x=(tapo_rec_mod-tapo_rec_sim), xtype='delta.tapo', title='Recent apocenter', pdf=True, binsize=0.5, file_path_and_name=directory+'/d_tapo_hist.pdf')
summary_plot.plot_hist(x=(tapo_rec_mod-tapo_rec_sim), xtype='delta.tapo', title='Recent apocenter', pdf=True, xlimits=(-5,5), binsize=0.5, file_path_and_name=directory+'/d_tapo_hist_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0, y=(tapo_rec_mod-tapo_rec_sim), xtype='M.star.z0', title='Recent apocenter', ytype='delta.tapo', limits=((4.5,9.5),(-13.8,13.8)), binsize=0.5, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/d_tapo_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(tapo_rec_mod-tapo_rec_sim), xtype='M.star.z0', title='Recent apocenter', ytype='delta.tapo', limits=((4.5,9.5),(-1,3)), binsize=0.5, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/d_tapo_vs_mstar_zoom.pdf')


# Pericenter number vs Mstar
N_sim = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all', sim_type='baryon')
N_mod = summary.nperi(data_total, masks_infall, oversample=True, selection='model', hosts='all', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all', sim_type='baryon')
#
summary_plot.plot_hist(x=(N_mod-N_sim), xtype='N.delta', pdf=True, binsize=1, file_path_and_name=directory+'/dN_hist.pdf')
summary_plot.plot_hist(x=(N_mod-N_sim), xtype='N.delta', pdf=True, xlimits=(-5,5), binsize=1, file_path_and_name=directory+'/dN_hist_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0, y=(N_mod-N_sim), xtype='M.star.z0', ytype='N.delta', limits=((4.5,9.5),None), binsize=1, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/dN_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(N_mod-N_sim), xtype='M.star.z0', ytype='N.delta', limits=((4.5,9.5),(-1,6)), binsize=1, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/dN_vs_mstar_zoom.pdf')




"""
    Properties vs d(z = 0)
"""
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
#
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
t_min_sim = summary.tperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
t_min_mod = summary.tperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
#
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')
#
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all', sim_type='baryon')


# Pericenter distance vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='d.z0', ytype='delta.d.frac', limits=((0,400),(-1,1.5)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='d.z0', ytype='delta.d.frac', limits=((0,400),(-1,2.5)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dfrac_min_vs_dz0.pdf')
#
# Pericenter time vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=(t_rec_mod-t_rec_sim), xtype='d.z0', ytype='delta.t', limits=((0,400),(-13,13)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dt_recent_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(t_min_mod-t_min_sim), xtype='d.z0', ytype='delta.t', limits=((0,400),(-13,13)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dt_min_vs_dz0.pdf')










# Eccentricity + period histograms

ecc = summary.eccentricity(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all', sim_type='baryon')
ecc_model = summary.eccentricity(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all', sim_type='baryon')

per = summary.period(data_total, masks_infall_peri, selection='sim', oversample=True)
per_model = summary.period(data_total, masks_infall_peri, selection='model', oversample=True)

summary_plot.plot_hist_mult(x=[ecc, ecc_model], xtype=['ecc', 'ecc.model'], labels=['Simulation', 'Model'], binsize=0.05, file_path_and_name=directory+'/ecc_hist.pdf', pdf=True)
summary_plot.plot_hist_mult(x=[per, per_model], xtype=['period', 'period.model'], labels=['Simulation', 'Model'], binsize=0.5, file_path_and_name=directory+'/period_hist.pdf', pdf=True)
