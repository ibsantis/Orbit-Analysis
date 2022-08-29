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
    Infall time comparisons
"""
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_diagnostics(data_total, masks_infall, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
mask_finite = np.isfinite(t_in_mod)*(t_in_mod != -1)
mask_finite_R200m = np.isfinite(t_in_mod_R200m)
#
summary_plot.plot_hist(x=(t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m]), xtype='t.infall.text', binsize=0.5, pdf=True, xlimits=(-10,13), title='$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z=0)$', file_path_and_name=directory+'/infall_comp_Rz0.pdf')
summary_plot.plot_hist(x=(t_in_mod[mask_finite]-t_in_sim[mask_finite]), xtype='t.infall.text', binsize=0.5, pdf=True, xlimits=(-10,10), title='$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z)$', file_path_and_name=directory+'/infall_comp_Rz.pdf')
summary_plot.plot_hist_mult(x=[(t_in_mod[mask_finite]-t_in_sim[mask_finite]),(t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m])], xtype=['t.infall.text','t.infall.text'], labels=['$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z)$','$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z=0)$'], binsize=0.5, pdf=True, xlimits=(-10,10), leg_loc='center left', med_location=[0.38,0.35], file_path_and_name=directory+'/infall_comp_both.pdf')
#
summary_plot.median_plot_mult(x=(t_in_sim[mask_finite_R200m], t_in_sim[mask_finite]), y=(t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m], t_in_mod[mask_finite]-t_in_sim[mask_finite]), xtype=['t.infall.text','t.infall.text'], ytype=['delta_t_infall','delta_t_infall'], binsize=1, limits=((0,13.8),None), labels=['$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z=0)$','$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z)$'], hl=True, file_path_and_name=directory+'/dt_infall_vs_t_infall.pdf')

t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_diagnostics(data_total, masks_infall, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
mask_finite = np.isfinite(t_in_mod)
mask_finite_R200m = np.isfinite(t_in_mod_R200m)
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
summary_plot.median_plot_mult(x=(Mstar_z0[mask_finite_R200m], Mstar_z0[mask_finite]), y=(t_in_mod_R200m[mask_finite_R200m]-t_in_sim[mask_finite_R200m], t_in_mod[mask_finite]-t_in_sim[mask_finite]), xtype=['M.star.z0','M.star.z0'], ytype=['delta_t_infall','delta_t_infall'], binsize=0.5, limits=((4,9.5),None), labels=['$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z=0)$','$t_{\\rm infall,model}$ w/ $R_{\\rm 200m}(z)$'], hl=True, file_path_and_name=directory+'/dt_infall_vs_Mstar.pdf')



"""
    Pericenter distance histograms
"""
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_d
summary_plot.plot_hist(x=(d_rec_mod-d_rec_sim), xtype='delta.d', pdf=True, xlimits=None, binsize=5, title='Recent Pericenters', file_path_and_name=directory+'/d_peri_recent_hist.pdf')
summary_plot.plot_hist(x=(d_rec_mod-d_rec_sim), xtype='delta.d', pdf=True, xlimits=(-110,150), binsize=5, title='Recent Pericenters', file_path_and_name=directory+'/d_peri_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(d_min_mod-d_min_sim), xtype='delta.d', pdf=True, xlimits=None, binsize=5, title='Minimum Pericenters', file_path_and_name=directory+'/d_peri_min_hist.pdf')
summary_plot.plot_hist(x=(d_min_mod-d_min_sim), xtype='delta.d', pdf=True, xlimits=(-110,150), binsize=5, title='Minimum Pericenters', file_path_and_name=directory+'/d_peri_min_hist_zoom.pdf')
#
# d_frac
summary_plot.plot_hist(x=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='delta.d.frac', pdf=True, xlimits=None, binsize=0.05, title='Recent Pericenters', file_path_and_name=directory+'/dfrac_recent_hist.pdf')
summary_plot.plot_hist(x=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='delta.d.frac', pdf=True, xlimits=(-1,3), binsize=0.05, title='Recent Pericenters', file_path_and_name=directory+'/dfrac_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(d_min_mod-d_min_sim)/d_min_sim, xtype='delta.d.frac', pdf=True, xlimits=None, binsize=0.05, title='Minimum Pericenters', file_path_and_name=directory+'/dfrac_min_hist.pdf')
summary_plot.plot_hist(x=(d_min_mod-d_min_sim)/d_min_sim, xtype='delta.d.frac', pdf=True, xlimits=(-1,5), binsize=0.05, title='Minimum Pericenters', file_path_and_name=directory+'/dfrac_min_hist_zoom.pdf')


"""
    Pericenter distance vs Mstar
"""
# delta_d
summary_plot.median_plot(x=Mstar_z0, y=(d_rec_mod-d_rec_sim), xtype='M.star.z0', ytype='delta.d', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_rec_mod-d_rec_sim), xtype='M.star.z0', ytype='delta.d', limits=((4.5,9.5),(-30,60)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_min_mod-d_min_sim), xtype='M.star.z0', ytype='delta.d', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_d_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_min_mod-d_min_sim), xtype='M.star.z0', ytype='delta.d', limits=((4.5,9.5),(-30,80)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_d_min_vs_mstar_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=Mstar_z0, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),(-1,1)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dfrac_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='M.star.z0', ytype='delta.d.frac', limits=((4.5,9.5),(-1,2.5)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dfrac_min_vs_mstar_zoom.pdf')
#
# Both d_frac on same plot
summary_plot.median_plot_mult(x=[Mstar_z0, Mstar_z0], y=[(d_rec_mod-d_rec_sim)/d_rec_sim, (d_min_mod-d_min_sim)/d_min_sim], xtype=['M.star.z0','M.star.z0'], ytype=['delta.d.frac','delta.d.frac'], limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), labels=['Recent','Minimum'], hl=True, file_path_and_name=directory+'/dfrac_both_vs_mstar.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0, Mstar_z0], y=[(d_rec_mod-d_rec_sim)/d_rec_sim, (d_min_mod-d_min_sim)/d_min_sim], xtype=['M.star.z0','M.star.z0'], ytype=['delta.d.frac','delta.d.frac'], limits=((4.5,9.5),(-1,2)), binsize=0.5, binedges=(4.5,10), labels=['Recent','Minimum'], hl=True, file_path_and_name=directory+'/dfrac_both_vs_mstar_zoom.pdf')



"""
    Pericenter time histograms
"""
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_sim = summary.tperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_mod = summary.tperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_t
summary_plot.plot_hist(x=(t_rec_mod-t_rec_sim), xtype='delta.t', pdf=True, binsize=0.25, title='Recent Pericenters', file_path_and_name=directory+'/dt_recent_hist.pdf')
summary_plot.plot_hist(x=(t_rec_mod-t_rec_sim), xtype='delta.t', pdf=True, xlimits=(-4,4), binsize=0.25, title='Recent Pericenters', file_path_and_name=directory+'/dt_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(t_min_mod-t_min_sim), xtype='delta.t', pdf=True, binsize=0.25, title='Minimum Pericenters', file_path_and_name=directory+'/dt_min_hist.pdf')
summary_plot.plot_hist(x=(t_min_mod-t_min_sim), xtype='delta.t', pdf=True, xlimits=(-11,12), binsize=0.25, title='Minimum Pericenters', file_path_and_name=directory+'/dt_min_hist_zoom.pdf')
#
# t_frac
summary_plot.plot_hist(x=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='delta.t.frac', pdf=True, binsize=0.05, title='Recent Pericenters', file_path_and_name=directory+'/tfrac_recent_hist.pdf')
summary_plot.plot_hist(x=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='delta.t.frac', pdf=True, xlimits=(-1.05,2), binsize=0.05, title='Recent Pericenters', file_path_and_name=directory+'/tfrac_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(t_min_mod-t_min_sim)/t_min_sim, xtype='delta.t.frac', pdf=True, binsize=0.05, title='Minimum Pericenters', file_path_and_name=directory+'/tfrac_min_hist.pdf')
summary_plot.plot_hist(x=(t_min_mod-t_min_sim)/t_min_sim, xtype='delta.t.frac', pdf=True, xlimits=(-1.05,6), binsize=0.05, title='Minimum Pericenters', file_path_and_name=directory+'/tfrac_min_hist_zoom.pdf')



"""
    Pericenter times vs Mstar
"""
# delta_t
summary_plot.median_plot(x=Mstar_z0, y=(t_rec_mod-t_rec_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-13.8,13.8)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dt_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_rec_mod-t_rec_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-2,3)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dt_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_min_mod-t_min_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-15,15)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dt_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_min_mod-t_min_sim), xtype='M.star.z0', ytype='delta.t', limits=((4.5,9.5),(-5,13)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dt_min_vs_mstar_zoom.pdf')
#
# t_frac
summary_plot.median_plot(x=Mstar_z0, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='M.star.z0', ytype='delta.t.frac', limits=((4.5,9.5),(-2,160)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='M.star.z0', ytype='delta.t.frac', limits=((4.5,9.5),(-1,2)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='M.star.z0', ytype='delta.t.frac', limits=((4.5,9.5),(-2,500)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/tfrac_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='M.star.z0', ytype='delta.t.frac', limits=((4.5,9.5),(-1,10)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/tfrac_min_vs_mstar_zoom.pdf')
#
# Both delta_t on same plot
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(t_rec_mod-t_rec_sim),(t_min_mod-t_min_sim)], xtype=['M.star.z0','M.star.z0'], ytype=['delta.t','delta.t'], limits=((4.5,9.5),(-13.8,13.8)), binsize=0.5, binedges=(4.5,10), labels=['Recent','Minimum'], hl=True, file_path_and_name=directory+'/dt_both_vs_mstar.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(t_rec_mod-t_rec_sim),(t_min_mod-t_min_sim)], xtype=['M.star.z0','M.star.z0'], ytype=['delta.t','delta.t'], limits=((4.5,9.5),(-5,10)), binsize=0.5, binedges=(4.5,10), labels=['Recent','Minimum'], hl=True, file_path_and_name=directory+'/dt_both_vs_mstar_zoom.pdf')



"""
    Pericenter velocity histograms
"""
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_v
summary_plot.plot_hist(x=(v_rec_mod-v_rec_sim), xtype='delta.v', pdf=True, binsize=10, title='Recent Pericenters', file_path_and_name=directory+'/dv_recent_hist.pdf')
summary_plot.plot_hist(x=(v_rec_mod-v_rec_sim), xtype='delta.v', pdf=True, xlimits=(-300,200), binsize=10, title='Recent Pericenters', file_path_and_name=directory+'/dv_recent_hist_zoom.pdf')
summary_plot.plot_hist(x=(v_min_mod-v_min_sim), xtype='delta.v', pdf=True, binsize=10, title='Minimum Pericenters', file_path_and_name=directory+'/dv_min_hist.pdf')
summary_plot.plot_hist(x=(v_min_mod-v_min_sim), xtype='delta.v', pdf=True, xlimits=(-300,200), binsize=10, title='Minimum Pericenters', file_path_and_name=directory+'/dv_min_hist_zoom.pdf')


"""
    Pericenter velocity vs Mstar
"""
# delta_v
summary_plot.median_plot(x=Mstar_z0, y=(v_rec_mod-v_rec_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dv_recent_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(v_rec_mod-v_rec_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),(-150,50)), binsize=0.5, binedges=(4.5,10), title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dv_recent_vs_mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(v_min_mod-v_min_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dv_min_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(v_min_mod-v_min_sim), xtype='M.star.z0', ytype='delta.v', limits=((4.5,9.5),(-150,200)), binsize=0.5, binedges=(4.5,10), title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dv_min_vs_mstar_zoom.pdf')
#
# Both delta_v on same plot
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(v_rec_mod-v_rec_sim),(v_min_mod-v_min_sim)], xtype=['M.star.z0','M.star.z0'], ytype=['delta.v','delta.v'], limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), labels=['Recent','Minimum'], hl=True, file_path_and_name=directory+'/dv_both_vs_mstar.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(v_rec_mod-v_rec_sim),(v_min_mod-v_min_sim)], xtype=['M.star.z0','M.star.z0'], ytype=['delta.v','delta.v'], limits=((4.5,9.5),(-150,180)), binsize=0.5, binedges=(4.5,10), labels=['Recent','Minimum'], hl=True, file_path_and_name=directory+'/dv_both_vs_mstar_zoom.pdf')



"""
    Apocenter distance histograms
"""
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_sim = summary.dmax(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_mod = summary.dmax(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
tapo_rec_sim = summary.tapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
tapo_rec_mod = summary.tapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_apo, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_d
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim), xtype='delta.dapo', pdf=True, binsize=10, title='Recent apocenter', file_path_and_name=directory+'/d_dapo_hist.pdf')
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim), xtype='delta.dapo', pdf=True, xlimits=(-100,200), binsize=10, title='Recent apocenter', file_path_and_name=directory+'/d_dapo_hist_zoom.pdf')
#
# d_frac
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='delta.dapo.frac', pdf=True, binsize=0.05, title='Recent apocenter', file_path_and_name=directory+'/dfrac_dapo_hist.pdf')
summary_plot.plot_hist(x=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='delta.dapo.frac', pdf=True, xlimits=(-1,2), binsize=0.05, title='Recent apocenter', file_path_and_name=directory+'/dfrac_dapo_hist_zoom.pdf')

"""
    Apocenter distance vs Mstar
"""
# delta_d
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim), xtype='M.star.z0', ytype='delta.dapo', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_dapo_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim), xtype='M.star.z0', ytype='delta.dapo', limits=((4.5,9.5),(-50,150)), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_dapo_vs_mstar_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='M.star.z0', ytype='delta.dapo.frac', limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='M.star.z0', ytype='delta.dapo.frac', limits=((4.5,9.5),(-0.5,1)), binsize=0.5, binedges=(4.5,10), title='Recent apocenter', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_mstar_zoom.pdf')
#
# Both d_frac on same page
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['M.star.z0','M.star.z0'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((4.5,9.5),None), binsize=0.5, binedges=(4.5,10), labels=['Maximim','Recent'], hl=True, file_path_and_name=directory+'/dapo_both_vs_mstar.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['M.star.z0','M.star.z0'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((4.5,9.5),(-1,1)), binsize=0.5, binedges=(4.5,10), labels=['Maximim','Recent'], hl=True, file_path_and_name=directory+'/dapo_both_vs_mstar_zoom.pdf')


"""
    Apocenter time histograms
"""
# delta_t
summary_plot.plot_hist(x=(tapo_rec_mod-tapo_rec_sim), xtype='delta.tapo', title='Recent apocenter', pdf=True, binsize=0.5, file_path_and_name=directory+'/d_tapo_hist.pdf')
summary_plot.plot_hist(x=(tapo_rec_mod-tapo_rec_sim), xtype='delta.tapo', title='Recent apocenter', pdf=True, xlimits=(-5,5), binsize=0.5, file_path_and_name=directory+'/d_tapo_hist_zoom.pdf')

"""
    Apocenter time vs Mstar
"""
# delta_t
summary_plot.median_plot(x=Mstar_z0, y=(tapo_rec_mod-tapo_rec_sim), xtype='M.star.z0', title='Recent apocenter', ytype='delta.tapo', limits=((4.5,9.5),(-13.8,13.8)), binsize=0.5, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/d_tapo_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(tapo_rec_mod-tapo_rec_sim), xtype='M.star.z0', title='Recent apocenter', ytype='delta.tapo', limits=((4.5,9.5),(-1,3)), binsize=0.5, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/d_tapo_vs_mstar_zoom.pdf')


"""
    Pericenter number histograms
"""
N_sim = summary.nperi(data_total, masks_infall_peri, oversample=True, selection='sim', hosts='all_no_r', sim_type='baryon')
N_mod = summary.nperi(data_total, masks_infall_peri, oversample=True, selection='model', hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist(x=(N_mod-N_sim), xtype='N.delta', pdf=True, binsize=1, file_path_and_name=directory+'/dN_hist.pdf')
summary_plot.plot_hist(x=(N_mod-N_sim), xtype='N.delta', pdf=True, xlimits=(-5,5), binsize=1, file_path_and_name=directory+'/dN_hist_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0, y=(N_mod-N_sim), xtype='M.star.z0', ytype='N.delta', limits=((4.5,9.5),None), binsize=1, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/dN_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(N_mod-N_sim), xtype='M.star.z0', ytype='N.delta', limits=((4.5,9.5),(-1,6)), binsize=1, binedges=(4.5,10), hl=True, file_path_and_name=directory+'/dN_vs_mstar_zoom.pdf')



"""
    Periceneter distances vs Infall time (simulation)
"""
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_d
summary_plot.median_plot(x=t_in_sim, y=(d_rec_mod-d_rec_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),None), binsize=0.5, title='Recent pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_rec_mod-d_rec_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),(-25,75)), binsize=0.5, title='Recent pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_t_in_sim_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_min_mod-d_min_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),None), binsize=0.5, title='Minimum pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/delta_d_min_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_min_mod-d_min_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),(-25,75)), binsize=0.5, title='Minimum pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/delta_d_min_vs_t_in_sim_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=t_in_sim, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),None), binsize=0.5, title='Recent pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),(-0.5,1.2)), binsize=0.5, title='Recent pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_t_in_sim_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),None), binsize=0.5, title='Minimum pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_min_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),(-1,10)), binsize=0.5, title='Minimum pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_min_vs_t_in_sim_zoom.pdf')
#
# both d_frac on same plot
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(d_min_mod-d_min_sim)/d_min_sim, (d_rec_mod-d_rec_sim)/d_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,13.8),None), binsize=0.5, labels=['Minimum','Recent'], title='Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_both_vs_t_in_sim.pdf')
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(d_min_mod-d_min_sim)/d_min_sim, (d_rec_mod-d_rec_sim)/d_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,13.8),(-1,10)), binsize=0.5, labels=['Minimum','Recent'], title='Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_both_vs_t_in_sim_zoom.pdf')

"""
    Periceneter distances vs Infall time (model)
"""
# delta_d
summary_plot.median_plot(x=t_in_mod, y=(d_rec_mod-d_rec_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),None), binsize=0.5, title='Recent pericenter, Model infall time', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(d_rec_mod-d_rec_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),(-25,120)), binsize=0.5, title='Recent pericenter, Model infall time', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_t_in_mod_zoom.pdf')
summary_plot.median_plot(x=t_in_mod, y=(d_min_mod-d_min_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),None), binsize=0.5, title='Minimum pericenter, Model infall time', hl=True, file_path_and_name=directory+'/delta_d_min_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(d_min_mod-d_min_sim), xtype='t.infall.text', ytype='delta.d', limits=((0,13.8),(-25,120)), binsize=0.5, title='Minimum pericenter, Model infall time', hl=True, file_path_and_name=directory+'/delta_d_min_vs_t_in_mod_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=t_in_mod, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),None), binsize=0.5, title='Recent pericenter, Model infall time', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),(-0.5,2.5)), binsize=0.5, title='Recent pericenter, Model infall time', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_t_in_mod_zoom.pdf')
summary_plot.median_plot(x=t_in_mod, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),None), binsize=0.5, title='Minimum pericenter, Model infall time', hl=True, file_path_and_name=directory+'/dfrac_min_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),(-1,4)), binsize=0.5, title='Minimum pericenter, Model infall time', hl=True, file_path_and_name=directory+'/dfrac_min_vs_t_in_mod_zoom.pdf')
#
# both d_frac on same plot
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(d_min_mod-d_min_sim)/d_min_sim, (d_rec_mod-d_rec_sim)/d_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,13.8),None), binsize=0.5, labels=['Minimum','Recent'], title='Model infall time', hl=True, file_path_and_name=directory+'/dfrac_both_vs_t_in_mod.pdf')
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(d_min_mod-d_min_sim)/d_min_sim, (d_rec_mod-d_rec_sim)/d_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,13.8),(-1,4)), binsize=0.5, labels=['Minimum','Recent'], title='Model infall time', hl=True, file_path_and_name=directory+'/dfrac_both_vs_t_in_mod_zoom.pdf')


"""
    Pericenter times vs Infall time (simulation)
"""
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_sim = summary.tperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_mod = summary.tperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_t
summary_plot.median_plot(x=t_in_sim, y=(t_rec_mod-t_rec_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-13.8,13.8)), binsize=0.5, title='Recent Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/dt_recent_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(t_rec_mod-t_rec_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-2,4)), binsize=0.5, title='Recent Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/dt_recent_vs_t_in_sim_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(t_min_mod-t_min_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-13.8,13.8)), binsize=0.5, title='Minimum Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/dt_min_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(t_min_mod-t_min_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-11,10)), binsize=0.5, title='Minimum Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/dt_min_vs_t_in_sim_zoom.pdf')
#
# t_frac
summary_plot.median_plot(x=t_in_sim, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-2,160)), binsize=0.5, title='Recent Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-1.1,1)), binsize=0.5, title='Recent Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_t_in_sim_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-2,500)), binsize=0.5, title='Minimum Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/tfrac_min_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-2,10)), binsize=0.5, title='Minimum Pericenters, Simulation infall time', hl=True, file_path_and_name=directory+'/tfrac_min_vs_t_in_sim_zoom.pdf')
#
# both delta_t on same plot
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(t_min_mod-t_min_sim),(t_rec_mod-t_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.t','delta.t'], limits=((0,13.8),(-13.8,13.8)), binsize=0.5, labels=['Minimum','Recent'], title='Simulation infall time', hl=True, file_path_and_name=directory+'/dt_both_vs_t_in_sim.pdf')
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(t_min_mod-t_min_sim),(t_rec_mod-t_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.t','delta.t'], limits=((0,13.8),(-10,10)), binsize=0.5, labels=['Minimum','Recent'], title='Simulation infall time', hl=True, file_path_and_name=directory+'/dt_both_vs_t_in_sim_zoom.pdf')

"""
    Pericenter times vs Infall time (model)
"""
# delta_t
summary_plot.median_plot(x=t_in_mod, y=(t_rec_mod-t_rec_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-13.8,13.8)), binsize=0.5, title='Recent Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/dt_recent_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(t_rec_mod-t_rec_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-6,6)), binsize=0.5, title='Recent Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/dt_recent_vs_t_in_mod_zoom.pdf')
summary_plot.median_plot(x=t_in_mod, y=(t_min_mod-t_min_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-13.8,13.8)), binsize=0.5, title='Minimum Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/dt_min_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(t_min_mod-t_min_sim), xtype='t.infall.text', ytype='delta.t', limits=((0,13.8),(-8,12)), binsize=0.5, title='Minimum Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/dt_min_vs_t_in_mod_zoom.pdf')
#
# t_frac
summary_plot.median_plot(x=t_in_mod, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-2,160)), binsize=0.5, title='Recent Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-1.1,4)), binsize=0.5, title='Recent Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_t_in_mod_zoom.pdf')
summary_plot.median_plot(x=t_in_mod, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-2,500)), binsize=0.5, title='Minimum Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/tfrac_min_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='t.infall.text', ytype='delta.t.frac', limits=((0,13.8),(-1.1,15)), binsize=0.5, title='Minimum Pericenters, Model infall time', hl=True, file_path_and_name=directory+'/tfrac_min_vs_t_in_mod_zoom.pdf')
#
# both delta_t on same plot
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(t_min_mod-t_min_sim),(t_rec_mod-t_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.t','delta.t'], limits=((0,13.8),(-13.8,13.8)), binsize=0.5, labels=['Minimum','Recent'], title='Model infall time', hl=True, file_path_and_name=directory+'/dt_both_vs_t_in_mod.pdf')
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(t_min_mod-t_min_sim),(t_rec_mod-t_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.t','delta.t'], limits=((0,13.8),(-6,12)), binsize=0.5, labels=['Minimum','Recent'], title='Model infall time', hl=True, file_path_and_name=directory+'/dt_both_vs_t_in_mod_zoom.pdf')


"""
    Pericenter velocities vs Infall time (Simulation)
"""
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_v
summary_plot.median_plot(x=t_in_sim, y=(v_rec_mod-v_rec_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),None), binsize=0.5, title='Recent Pericenters, Simulation Infall time', hl=True, file_path_and_name=directory+'/dv_recent_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(v_rec_mod-v_rec_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),(-200,50)), binsize=0.5, title='Recent Pericenters, Simulation Infall time', hl=True, file_path_and_name=directory+'/dv_recent_vs_t_in_sim_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(v_min_mod-v_min_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),None), binsize=0.5, title='Minimum Pericenters, Simulation Infall time', hl=True, file_path_and_name=directory+'/dv_min_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(v_min_mod-v_min_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),(-180,100)), binsize=0.5, title='Minimum Pericenters, Simulation Infall time', hl=True, file_path_and_name=directory+'/dv_min_vs_t_in_sim_zoom.pdf')
#
# Both delta_v on same plot
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(v_min_mod-v_min_sim),(v_rec_mod-v_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.v','delta.v'], limits=((0,13.8),None), binsize=0.5, labels=['Minimum','Recent'], title='Simulation Infall time', hl=True, file_path_and_name=directory+'/dv_both_vs_t_in_sim.pdf')
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(v_min_mod-v_min_sim),(v_rec_mod-v_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.v','delta.v'], limits=((0,13.8),(-200,100)), binsize=0.5, labels=['Minimum','Recent'], title='Simulation Infall time', hl=True, file_path_and_name=directory+'/dv_both_vs_t_in_sim_zoom.pdf')

"""
    Pericenter velocities vs Infall time (Model)
"""
# delta_v
summary_plot.median_plot(x=t_in_mod, y=(v_rec_mod-v_rec_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),None), binsize=0.5, title='Recent Pericenters, Model Infall time', hl=True, file_path_and_name=directory+'/dv_recent_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(v_rec_mod-v_rec_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),(-180,100)), binsize=0.5, title='Recent Pericenters, Model Infall time', hl=True, file_path_and_name=directory+'/dv_recent_vs_t_in_mod_zoom.pdf')
summary_plot.median_plot(x=t_in_mod, y=(v_min_mod-v_min_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),None), binsize=0.5, title='Minimum Pericenters, Model Infall time', hl=True, file_path_and_name=directory+'/dv_min_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(v_min_mod-v_min_sim), xtype='t.infall.text', ytype='delta.v', limits=((0,13.8),(-180,180)), binsize=0.5, title='Minimum Pericenters, Model Infall time', hl=True, file_path_and_name=directory+'/dv_min_vs_t_in_mod_zoom.pdf')
#
# Both delta_v on same plot
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(v_min_mod-v_min_sim),(v_rec_mod-v_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.v','delta.v'], limits=((0,13.8),None), binsize=0.5, labels=['Minimum','Recent'], title='Model Infall time', hl=True, file_path_and_name=directory+'/dv_both_vs_t_in_mod.pdf')
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(v_min_mod-v_min_sim),(v_rec_mod-v_rec_sim)], xtype=['t.infall.text','t.infall.text'], ytype=['delta.v','delta.v'], limits=((0,13.8),(-200,200)), binsize=0.5, labels=['Minimum','Recent'], title='Model Infall time', hl=True, file_path_and_name=directory+'/dv_both_vs_t_in_mod_zoom.pdf')


"""
    Apocenter distances vs Infall time (Simulation)
"""
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_sim = summary.dmax(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_mod = summary.dmax(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_d
summary_plot.median_plot(x=t_in_sim, y=(dapo_rec_mod-dapo_rec_sim), xtype='t.infall.text', ytype='delta.dapo', limits=((0,13.8),None), binsize=0.5, title='Recent apocenter, Simulation Infall time', hl=True, file_path_and_name=directory+'/d_dapo_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(dapo_rec_mod-dapo_rec_sim), xtype='t.infall.text', ytype='delta.dapo', limits=((0,13.8),(-50,300)), binsize=0.5, title='Recent apocenter, Simulation Infall time', hl=True, file_path_and_name=directory+'/d_dapo_vs_t_in_sim_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=t_in_sim, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='t.infall.text', ytype='delta.dapo.frac', limits=((0,13.8),None), binsize=0.5, title='Recent apocenter, Simulation Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='t.infall.text', ytype='delta.dapo.frac', limits=((0,13.8),(-0.5,2)), binsize=0.5, title='Recent apocenter, Simulation Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_t_in_sim_zoom.pdf')
#
# Both d_frac on same plot
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((0,13.8),None), binsize=0.5, labels=['Maximum','Recent'], title='Simulation Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_both_vs_t_in_sim.pdf')
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((0,13.8),(-1,2)), binsize=0.5, labels=['Maximum','Recent'], title='Simulation Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_both_vs_t_in_sim_zoom.pdf')

"""
    Apocenter distances vs Infall time (Model)
"""
# delta_d
summary_plot.median_plot(x=t_in_mod, y=(dapo_rec_mod-dapo_rec_sim), xtype='t.infall.text', ytype='delta.dapo', limits=((0,13.8),None), binsize=0.5, title='Recent apocenter, Model Infall time', hl=True, file_path_and_name=directory+'/d_dapo_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(dapo_rec_mod-dapo_rec_sim), xtype='t.infall.text', ytype='delta.dapo', limits=((0,13.8),(-50,500)), binsize=0.5, title='Recent apocenter, Model Infall time', hl=True, file_path_and_name=directory+'/d_dapo_vs_t_in_mod_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=t_in_mod, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='t.infall.text', ytype='delta.dapo.frac', limits=((0,13.8),None), binsize=0.5, title='Recent apocenter, Model Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='t.infall.text', ytype='delta.dapo.frac', limits=((0,13.8),(-0.5,3)), binsize=0.5, title='Recent apocenter, Model Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_t_in_mod_zoom.pdf')
#
# Both d_frac on same plot
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((0,13.8),None), binsize=0.5, labels=['Maximum','Recent'], title='Model Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_both_vs_t_in_mod.pdf')
summary_plot.median_plot_mult(x=[t_in_mod,t_in_mod], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((0,13.8),(-1,3)), binsize=0.5, labels=['Maximum','Recent'], title='Model Infall time', hl=True, file_path_and_name=directory+'/dfrac_dapo_both_vs_t_in_mod_zoom.pdf')


"""
    Apocenter time vs Infall time (simulation)
"""
tapo_rec_sim = summary.tapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
tapo_rec_mod = summary.tapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_t
summary_plot.median_plot(x=t_in_sim, y=(tapo_rec_mod-tapo_rec_sim), xtype='t.infall.text', title='Recent apocenter, Simulation infall time', ytype='delta.tapo', limits=((0,13.8),(-13.8,13.8)), binsize=0.5, hl=True, file_path_and_name=directory+'/d_tapo_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(tapo_rec_mod-tapo_rec_sim), xtype='t.infall.text', title='Recent apocenter, Simulation infall time', ytype='delta.tapo', limits=((0,13.8),(-2,7)), binsize=0.5, hl=True, file_path_and_name=directory+'/d_tapo_vs_t_in_sim_zoom.pdf')
"""
    Apocenter time vs Infall time (model)
"""
# delta_t
summary_plot.median_plot(x=t_in_mod, y=(tapo_rec_mod-tapo_rec_sim), xtype='t.infall.text', title='Recent apocenter, Model infall time', ytype='delta.tapo', limits=((0,13.8),(-13.8,13.8)), binsize=0.5, hl=True, file_path_and_name=directory+'/d_tapo_vs_t_in_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(tapo_rec_mod-tapo_rec_sim), xtype='t.infall.text', title='Recent apocenter, Model infall time', ytype='delta.tapo', limits=((0,13.8),(-4,8)), binsize=0.5, hl=True, file_path_and_name=directory+'/d_tapo_vs_t_in_mod_zoom.pdf')



"""
    Pericenter distance vs d(z = 0)
"""
d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_d
summary_plot.median_plot(x=dz0_tot, y=(d_rec_mod-d_rec_sim), xtype='d.z0', ytype='delta.d', limits=((0,400),(None)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(d_rec_mod-d_rec_sim), xtype='d.z0', ytype='delta.d', limits=((0,400),(-25,75)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_d_recent_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_tot, y=(d_min_mod-d_min_sim), xtype='d.z0', ytype='delta.d', limits=((0,400),(None)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_d_min_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(d_min_mod-d_min_sim), xtype='d.z0', ytype='delta.d', limits=((0,400),(-25,75)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_d_min_vs_dz0_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=dz0_tot, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='d.z0', ytype='delta.d.frac', limits=((0,400),(None)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='d.z0', ytype='delta.d.frac', limits=((0,400),(-1,1.5)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/dfrac_recent_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_tot, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='d.z0', ytype='delta.d.frac', limits=((0,400),(None)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dfrac_min_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='d.z0', ytype='delta.d.frac', limits=((0,400),(-1,2.2)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/dfrac_min_vs_dz0_zoom.pdf')
#
# both d_frac on same plot
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(d_min_mod-d_min_sim)/d_min_sim,(d_rec_mod-d_rec_sim)/d_rec_sim], xtype=['d.z0','d.z0'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,400),(None)), binsize=50, labels=['Minimum','Recent'], hl=True, file_path_and_name=directory+'/dfrac_both_vs_dz0.pdf')
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(d_min_mod-d_min_sim)/d_min_sim,(d_rec_mod-d_rec_sim)/d_rec_sim], xtype=['d.z0','d.z0'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,400),(-1,2)), binsize=50, labels=['Minimum','Recent'], hl=True, file_path_and_name=directory+'/dfrac_both_vs_dz0_zoom.pdf')



"""
    Pericenter time vs d(z = 0)
"""
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_sim = summary.tperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_mod = summary.tperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_t
summary_plot.median_plot(x=dz0_tot, y=(t_rec_mod-t_rec_sim), xtype='d.z0', ytype='delta.t', limits=((0,400),(None)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_t_recent_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(t_rec_mod-t_rec_sim), xtype='d.z0', ytype='delta.t', limits=((0,400),(-1,2)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_t_recent_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_tot, y=(t_min_mod-t_min_sim), xtype='d.z0', ytype='delta.t', limits=((0,400),(None)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_t_min_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(t_min_mod-t_min_sim), xtype='d.z0', ytype='delta.t', limits=((0,400),(-6,10)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_t_min_vs_dz0_zoom.pdf')
#
# t_frac
summary_plot.median_plot(x=dz0_tot, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='d.z0', ytype='delta.t.frac', limits=((0,400),(None)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(t_rec_mod-t_rec_sim)/t_rec_sim, xtype='d.z0', ytype='delta.t.frac', limits=((0,400),(-1,1)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/tfrac_recent_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_tot, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='d.z0', ytype='delta.t.frac', limits=((0,400),(None)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/tfrac_min_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(t_min_mod-t_min_sim)/t_min_sim, xtype='d.z0', ytype='delta.t.frac', limits=((0,400),(-1.5,7)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/tfrac_min_vs_dz0_zoom.pdf')
#
# both delta_t on same plot
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(t_min_mod-t_min_sim),(t_rec_mod-t_rec_sim)], xtype=['d.z0','d.z0'], ytype=['delta.t','delta.t'], limits=((0,400),(None)), binsize=50, labels=['Minimum','Recent'], hl=True, file_path_and_name=directory+'/delta_t_both_vs_dz0.pdf')
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(t_min_mod-t_min_sim),(t_rec_mod-t_rec_sim)], xtype=['d.z0','d.z0'], ytype=['delta.t','delta.t'], limits=((0,400),(-6,10)), binsize=50, labels=['Minimum','Recent'], hl=True, file_path_and_name=directory+'/delta_t_both_vs_dz0_zoom.pdf')


"""
    Pericenter velocity vs d(z = 0)
"""
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_v
summary_plot.median_plot(x=dz0_tot, y=(v_rec_mod-v_rec_sim), xtype='d.z0', ytype='delta.v', limits=((0,400),(None)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_v_recent_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(v_rec_mod-v_rec_sim), xtype='d.z0', ytype='delta.v', limits=((0,400),(-150,50)), binsize=50, title='Recent Pericenters', hl=True, file_path_and_name=directory+'/delta_v_recent_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_tot, y=(v_min_mod-v_min_sim), xtype='d.z0', ytype='delta.v', limits=((0,400),(None)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_v_min_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(v_min_mod-v_min_sim), xtype='d.z0', ytype='delta.v', limits=((0,400),(-150,75)), binsize=50, title='Minimum Pericenters', hl=True, file_path_and_name=directory+'/delta_v_min_vs_dz0_zoom.pdf')
#
# both delta_v on same plot
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(v_min_mod-v_min_sim),(v_rec_mod-v_rec_sim)], xtype=['d.z0','d.z0'], ytype=['delta.v','delta.v'], limits=((0,400),(None)), binsize=50, labels=['Minimum', 'Recent'], hl=True, file_path_and_name=directory+'/delta_v_both_vs_dz0.pdf')
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(v_min_mod-v_min_sim),(v_rec_mod-v_rec_sim)], xtype=['d.z0','d.z0'], ytype=['delta.v','delta.v'], limits=((0,400),(-140,80)), binsize=50, labels=['Minimum', 'Recent'], hl=True, file_path_and_name=directory+'/delta_v_both_vs_dz0_zoom.pdf')


"""
    Apocenter distances vs d(z = 0)
"""
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_sim = summary.dmax(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_mod = summary.dmax(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_apo, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_d
summary_plot.median_plot(x=dz0_tot, y=(dapo_rec_mod-dapo_rec_sim), xtype='d.z0', ytype='delta.dapo', limits=((0,400),None), binsize=50, title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_dapo_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(dapo_rec_mod-dapo_rec_sim), xtype='d.z0', ytype='delta.dapo', limits=((0,400),(-50,300)), binsize=50, title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_dapo_vs_dz0_zoom.pdf')
#
# d_frac
summary_plot.median_plot(x=dz0_tot, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='d.z0', ytype='delta.dapo.frac', limits=((0,400),None), binsize=50, title='Recent apocenter', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim, xtype='d.z0', ytype='delta.dapo.frac', limits=((0,400),(-0.25,2.5)), binsize=50, title='Recent apocenter', hl=True, file_path_and_name=directory+'/dfrac_dapo_vs_dz0_zoom.pdf')
#
# both d_frac on same plot
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['d.z0','d.z0'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((0,400),None), binsize=50, labels=['Maximum','Recent'], hl=True, file_path_and_name=directory+'/dfrac_dapo_both_vs_dz0.pdf')
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(dmax_mod-dmax_sim)/dmax_sim,(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim], xtype=['d.z0','d.z0'], ytype=['delta.dapo.frac','delta.dapo.frac'], limits=((0,400),(-1,2.5)), binsize=50, labels=['Maximum','Recent'], hl=True, file_path_and_name=directory+'/dfrac_dapo_both_vs_dz0_zoom.pdf')


"""
    Apocenter times vs d(z = 0)
"""
tapo_rec_sim = summary.tapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
tapo_rec_mod = summary.tapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_apo, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_t
summary_plot.median_plot(x=dz0_tot, y=(tapo_rec_mod-tapo_rec_sim), xtype='d.z0', ytype='delta.tapo', limits=((0,400),None), binsize=50, title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_tapo_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(tapo_rec_mod-tapo_rec_sim), xtype='d.z0', ytype='delta.tapo', limits=((0,400),(-2,6)), binsize=50, title='Recent apocenter', hl=True, file_path_and_name=directory+'/d_tapo_vs_dz0_zoom.pdf')



"""
    Compare the different model pericenter number metrics all on same histogram
"""
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod = summary.nperi(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_sim_infall = summary.nperi_model(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_300 = summary.nperi_model(data_total, masks_infall, selection='model.300', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[n_mod, n_mod_sim_infall, n_mod_mod_infall, n_mod_300, n_mod_r200], xtype=['N.model','N.model','N.model','N.model','N.model'], binsize=1, labels=['No infall','Sim infall time','Model infall time', '300 kpc threshold', 'R200m threshold'], pdf=True, file_path_and_name=directory+'/nperi_model_comparisons.pdf')



"""
    Pericenter number vs Mstar
"""
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_N
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(n_mod_mod_infall-n_sim), (n_mod_r200-n_sim)], xtype=['M.star.z0','M.star.z0'], ytype=['N.delta','N.delta'], binsize=1, binedges=(4.5,10), limits=((4.5,10),None), labels=['Model w/$R_{\\rm 200m}(z)$','Model w/$R_{\\rm 200m}(z=0)$'], file_path_and_name=directory+'/nperi_vs_mstar_both.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0,Mstar_z0], y=[(n_mod_mod_infall-n_sim), (n_mod_r200-n_sim)], xtype=['M.star.z0','M.star.z0'], ytype=['N.delta','N.delta'], binsize=1, binedges=(4.5,10), limits=((4.5,10),(-2,6)), labels=['Model w/$R_{\\rm 200m}(z)$','Model w/$R_{\\rm 200m}(z=0)$'], file_path_and_name=directory+'/nperi_vs_mstar_both_zoom.pdf')



"""
    Pericenter number vs t_infall
"""
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_diagnostics(data_total, masks_infall, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.median_plot_mult(x=[t_in_sim, t_in_mod], y=[n_sim, n_mod_mod_infall], xtype=['t.infall.text','t.infall.text'], ytype=['N.peri.text','N.peri.text'], w_scatter=True, binsize=0.5, limits=((0,14),None), labels=['Simulation','Model w/$R_{\\rm 200m}(z)$'], file_path_and_name=directory+'/nperi_vs_t_infall_both_mod_infall.pdf')
summary_plot.median_plot_mult(x=[t_in_sim, t_in_mod_R200m], y=[n_sim, n_mod_r200], xtype=['t.infall.text','t.infall.text'], ytype=['N.peri.text','N.peri.text'], w_scatter=True, binsize=0.5, limits=((0,14),None), labels=['Simulation','Model w/$R_{\\rm 200m}(z=0)$'], file_path_and_name=directory+'/nperi_vs_t_infall_both_mod_infall_r200m.pdf')



"""
    Pericenter number vs d(z = 0)
"""
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_N
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(n_mod_mod_infall-n_sim), (n_mod_r200-n_sim)], xtype=['d.z0','d.z0'], ytype=['N.delta','N.delta'], binsize=50, limits=((0,400),None), labels=['Model w/$R_{\\rm 200m}(z)$','Model w/$R_{\\rm 200m}(z=0)$'], file_path_and_name=directory+'/nperi_vs_dz0_both.pdf')
summary_plot.median_plot_mult(x=[dz0_tot,dz0_tot], y=[(n_mod_mod_infall-n_sim), (n_mod_r200-n_sim)], xtype=['d.z0','d.z0'], ytype=['N.delta','N.delta'], binsize=50, limits=((0,400),(-3,4)), labels=['Model w/$R_{\\rm 200m}(z)$','Model w/$R_{\\rm 200m}(z=0)$'], file_path_and_name=directory+'/nperi_vs_dz0_both_zoom.pdf')



"""
    Eccentricity vs Mstar
"""
ecc = summary.eccentricity(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
ecc_model = summary.eccentricity(data_total, masks_infall_peri, selection='model.apsis', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_ecc
summary_plot.median_plot(x=Mstar_z0, y=(ecc_model-ecc), xtype='M.star.z0', ytype='ecc.delta', w_scatter=True, hl=True, binsize=0.5, binedges=(4.5,10), limits=((4.5,9.5),None), file_path_and_name=directory+'/delta_ecc_vs_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(ecc_model-ecc), xtype='M.star.z0', ytype='ecc.delta', w_scatter=True, hl=True, binsize=0.5, binedges=(4.5,10), limits=((4.5,9.5),(-0.3,0.3)), file_path_and_name=directory+'/delta_ecc_vs_mstar_zoom.pdf')


"""
    Eccentricity vs t_infalls
"""
ecc = summary.eccentricity(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
ecc_model = summary.eccentricity(data_total, masks_infall_peri, selection='model.apsis', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# t_infall (simulation)
summary_plot.median_plot(x=t_in_sim, y=(ecc_model-ecc), xtype='t.infall.text', ytype='ecc.delta', w_scatter=True, hl=True, binsize=0.5, limits=((0,14),None), file_path_and_name=directory+'/delta_ecc_vs_t_infall_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(ecc_model-ecc), xtype='t.infall.text', ytype='ecc.delta', w_scatter=True, hl=True, binsize=0.5, limits=((0,14),(-0.4,0.25)), file_path_and_name=directory+'/delta_ecc_vs_t_infall_sim_zoom.pdf')
#
# t_infall (model)
summary_plot.median_plot(x=t_in_mod, y=(ecc_model-ecc), xtype='t.infall.text', ytype='ecc.delta', w_scatter=True, hl=True, binsize=0.5, limits=((0,14),None), file_path_and_name=directory+'/delta_ecc_vs_t_infall_mod.pdf')
summary_plot.median_plot(x=t_in_mod, y=(ecc_model-ecc), xtype='t.infall.text', ytype='ecc.delta', w_scatter=True, hl=True, binsize=0.5, limits=((0,14),(-0.3,0.4)), file_path_and_name=directory+'/delta_ecc_vs_t_infall_mod_zoom.pdf')


"""
    Eccentricity vs d(z = 0)
"""
ecc = summary.eccentricity(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
ecc_model = summary.eccentricity(data_total, masks_infall_peri, selection='model.apsis', oversample=True, hosts='all_no_r', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_r', sim_type='baryon')
#
# delta_ecc
summary_plot.median_plot(x=dz0_tot, y=(ecc_model-ecc), xtype='d.z0', ytype='ecc.delta', hl=True, binsize=50, limits=((0,400),None), file_path_and_name=directory+'/delta_ecc_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(ecc_model-ecc), xtype='d.z0', ytype='ecc.delta', hl=True, binsize=50, limits=((0,400),(-0.25,0.4)), file_path_and_name=directory+'/delta_ecc_vs_dz0_zoom.pdf')



# Eccentricity + period histograms

ecc_rec_sim = summary.eccentricity_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
ecc_rec_mod = summary.eccentricity_recent(data_total, masks_infall_peri, selection='model.apsis', oversample=True, hosts='all_no_r', sim_type='baryon')
summary_plot.plot_hist_mult(x=[ecc_rec_sim, ecc_rec_mod], xtype=['ecc', 'ecc.model'], labels=['Simulation', 'Model'], title='Recent eccentricities', binsize=0.05, file_path_and_name=directory+'/ecc_comp_recent.pdf', pdf=True)
#
ecc_avg_sim = summary.eccentricity(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
ecc_avg_mod = summary.eccentricity(data_total, masks_infall_peri, selection='model.apsis', oversample=True, hosts='all_no_r', sim_type='baryon')
summary_plot.plot_hist_mult(x=[ecc_avg_sim, ecc_avg_mod], xtype=['ecc', 'ecc.model'], labels=['Simulation', 'Model'], title='Average eccentricities', binsize=0.05, file_path_and_name=directory+'/ecc_comp_average.pdf', pdf=True)
#
summary_plot.plot_hist_mult(x=[ecc_rec_sim, ecc_avg_sim], xtype=['ecc', 'ecc'], labels=['Recent', 'Average'], title='Simulation eccentricities', binsize=0.05, file_path_and_name=directory+'/ecc_rec_vs_avg_sim.pdf', pdf=True)
summary_plot.plot_hist_mult(x=[ecc_rec_mod, ecc_avg_mod], xtype=['ecc', 'ecc'], labels=['Recent', 'Average'], title='Model eccentricities', binsize=0.05, file_path_and_name=directory+'/ecc_rec_vs_avg_mod.pdf', pdf=True)














x = []
y = []
# Loop over hosts
for name in summary.host_names['all_no_r']:
    # Loop over satellites
    for i in range(0, len(data_total[name]['eccentricity.sim'][masks_infall_apo[name]])):
        # Loop over the phase
        for j in range(0, np.min((len(data_total[name]['eccentricity.sim'][masks_infall_apo[name]][i]), len(data_total[name]['eccentricity.model.apsis'][masks_infall_apo[name]][i])))):
            # Make sure there is an event in both the sim and model
            if (data_total[name]['eccentricity.sim'][masks_infall_apo[name]][i][j] != -1) & (data_total[name]['eccentricity.model.apsis'][masks_infall_apo[name]][i][j] != -1):
                # Save the difference
                x.append(data_total[name]['eccentricity.model.apsis'][masks_infall_apo[name]][i][j]-data_total[name]['eccentricity.sim'][masks_infall_apo[name]][i][j])
                # Save the phase
                y.append(j+1)
x = np.asarray(x)
y = np.asarray(y)
#
# THINK MORE ABOUT WHAT TO REALLY PLOT AGAINST
onesigp = 84.13
onesigm = 15.87
twosigp = 100
twosigm = 0
meds = np.zeros(np.max(y))
upper = np.zeros(np.max(y))
lower = np.zeros(np.max(y))
highest = np.zeros(np.max(y))
lowest = np.zeros(np.max(y))
for i in range(0, np.max(y)):
    mask = (y == i+1)
    meds[i] = np.nanmedian(x[mask])
    upper[i] = np.nanpercentile(x[mask], onesigp)
    lower[i] = np.nanpercentile(x[mask], onesigm)
    highest[i] = np.nanpercentile(x[mask], twosigp)
    lowest[i] = np.nanpercentile(x[mask], twosigm)
#
f, ax = plt.subplots(1, 1, figsize=(10,10))
plt.scatter(np.arange(np.max(y))+1, meds, s=50., marker='s', color=summary_plot.colors[1])
for j in range(0, np.max(y)):
    plt.errorbar(np.arange(np.max(y))[j]+1, meds[j], yerr=np.array([[meds[j]-lowest[j]],[highest[j]-meds[j]]]), alpha=0.3, color=summary_plot.colors[1])
    plt.errorbar(np.arange(np.max(y))[j]+1, meds[j], yerr=np.array([[meds[j]-lower[j]],[upper[j]-meds[j]]]), alpha=0.7, color=summary_plot.colors[1])
plt.hlines(0, 0, np.max(y)+1, linestyle='dotted', color='k', alpha=0.5)
plt.xlim(0.5, np.max(y)+0.5)
plt.xlabel('Subsequent peri/apo calculations')
plt.ylabel('$e_{\\rm model} - e_{\\rm sim}$')
#plt.show()
plt.tight_layout()
plt.savefig(directory+'/ecc_comp_vs_phase.pdf')
plt.close()











# Plotting the orbital periods defined by recent peris and apos
per_peri = summary.period_recent(data_total, masks_infall_peri, selection='sim', choice='peri', hosts='all_no_r', oversample=True)
per_apo = summary.period_recent(data_total, masks_infall_peri, selection='sim', choice='apo', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_peri, per_apo], xtype=['period', 'period'], labels=['Recent Pericenters', 'Recent Apocenters'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_recent_peri_vs_apo_sim.pdf', pdf=True)
#
per_peri_mod = summary.period_recent(data_total, masks_infall_peri, selection='model', choice='peri', hosts='all_no_r', oversample=True)
per_apo_mod = summary.period_recent(data_total, masks_infall_peri, selection='model', choice='apo', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_peri_mod, per_apo_mod], xtype=['period.model', 'period.model'], labels=['Recent Pericenters', 'Recent Apocenters'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_recent_peri_vs_apo_model.pdf', pdf=True)





# Plotting the average orbital periods
per_peri = summary.period_average(data_total, masks_infall_peri, selection='sim', choice='peri', hosts='all_no_r', oversample=True)
per_apo = summary.period_average(data_total, masks_infall_peri, selection='sim', choice='apo', hosts='all_no_r', oversample=True)
per_both = summary.period_average(data_total, masks_infall_peri, selection='sim', choice='both', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_peri, per_apo, per_both], xtype=['period', 'period', 'period'], labels=['Pericenter average', 'Apocenter average', 'All average'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_average_comp_sim.pdf', pdf=True)
#
per_peri_mod = summary.period_average(data_total, masks_infall_peri, selection='model', choice='peri', hosts='all_no_r', oversample=True)
per_apo_mod = summary.period_average(data_total, masks_infall_peri, selection='model', choice='apo', hosts='all_no_r', oversample=True)
per_both_mod = summary.period_average(data_total, masks_infall_peri, selection='model', choice='both', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_peri_mod, per_apo_mod, per_both_mod], xtype=['period.model', 'period.model', 'period.model'], labels=['Pericenter average', 'Apocenter average', 'All average'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_average_comp_model.pdf', pdf=True)
#
summary_plot.plot_hist_mult(x=[per_peri, per_peri_mod], xtype=['period', 'period.model'], labels=['Pericenter average (sim)', 'Pericenter average (model)'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_average_peri_comp.pdf', pdf=True)
summary_plot.plot_hist_mult(x=[per_apo, per_apo_mod], xtype=['period', 'period.model'], labels=['Apocenter average (sim)', 'Apocenter average (model)'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_average_apo_comp.pdf', pdf=True)
summary_plot.plot_hist_mult(x=[per_both, per_both_mod], xtype=['period', 'period.model'], labels=['All average (sim)', 'All average (model)'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_average_all_comp.pdf', pdf=True)




# Plot the recent vs average in simulation
per_peri_rec = summary.period_recent(data_total, masks_infall_peri, selection='sim', choice='peri', hosts='all_no_r', oversample=True)
per_peri_avg = summary.period_average(data_total, masks_infall_peri, selection='sim', choice='peri', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_peri_rec, per_peri_avg], xtype=['period', 'period'], labels=['Recent Pericenters', 'Average Pericenters'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_peri_recent_vs_average_sim.pdf', pdf=True)
#
per_apo_rec = summary.period_recent(data_total, masks_infall_peri, selection='sim', choice='apo', hosts='all_no_r', oversample=True)
per_apo_avg = summary.period_average(data_total, masks_infall_peri, selection='sim', choice='apo', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_apo_rec, per_apo_avg], xtype=['period', 'period'], labels=['Recent Apocenters', 'Average Apocenters'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_apo_recent_vs_average_sim.pdf', pdf=True)

# Plot the recent vs average in model
per_peri_rec = summary.period_recent(data_total, masks_infall_peri, selection='model', choice='peri', hosts='all_no_r', oversample=True)
per_peri_avg = summary.period_average(data_total, masks_infall_peri, selection='model', choice='peri', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_peri_rec, per_peri_avg], xtype=['period.model', 'period.model'], labels=['Recent Pericenters', 'Average Pericenters'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_peri_recent_vs_average_model.pdf', pdf=True)
#
per_apo_rec = summary.period_recent(data_total, masks_infall_peri, selection='model', choice='apo', hosts='all_no_r', oversample=True)
per_apo_avg = summary.period_average(data_total, masks_infall_peri, selection='model', choice='apo', hosts='all_no_r', oversample=True)
summary_plot.plot_hist_mult(x=[per_apo_rec, per_apo_avg], xtype=['period.model', 'period.model'], labels=['Recent Apocenters', 'Average Apocenters'], binsize=0.5, xlimits=(0, 10), file_path_and_name=directory+'/period_apo_recent_vs_average_model.pdf', pdf=True)



# OLD, this was only plotting the recent periods I think
#per_model = summary.period(data_total, masks_infall_peri, hosts='all_no_r', selection='model', oversample=True)
#summary_plot.plot_hist_mult(x=[per, per_model], xtype=['period', 'period.model'], labels=['Simulation', 'Model'], binsize=0.5, file_path_and_name=directory+'/period_hist.pdf', pdf=True)

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
x_peri = np.asarray(x)
y_peri = np.asarray(y)
#
x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.apo.sim'][masks_infall_apo[name]])):
        for j in range(0, np.min((len(data_total[name]['orbit.period.apo.sim'][masks_infall_apo[name]][i]), len(data_total[name]['orbit.period.apo.model'][masks_infall_apo[name]][i])))):
            if (data_total[name]['orbit.period.apo.sim'][masks_infall_apo[name]][i][j] != -1) & (data_total[name]['orbit.period.apo.model'][masks_infall_apo[name]][i][j] != -1):
                x.append(j+1)
                y.append(data_total[name]['orbit.period.apo.model'][masks_infall_apo[name]][i][j]-data_total[name]['orbit.period.apo.sim'][masks_infall_apo[name]][i][j])
x_apo = np.asarray(x)
y_apo = np.asarray(y)
#
onesigp = 84.13
onesigm = 15.87
twosigp = 100
twosigm = 0
#
meds_peri = np.zeros(np.max(x_peri))
upper_peri = np.zeros(np.max(x_peri))
lower_peri = np.zeros(np.max(x_peri))
highest_peri = np.zeros(np.max(x_peri))
lowest_peri = np.zeros(np.max(x_peri))
for i in range(0, np.max(x_peri)):
    mask = (x_peri == i+1)
    meds_peri[i] = np.nanmedian(y_peri[mask])
    upper_peri[i] = np.nanpercentile(y_peri[mask], onesigp)
    lower_peri[i] = np.nanpercentile(y_peri[mask], onesigm)
    highest_peri[i] = np.nanpercentile(y_peri[mask], twosigp)
    lowest_peri[i] = np.nanpercentile(y_peri[mask], twosigm)
#
meds_apo = np.zeros(np.max(x_apo))
upper_apo = np.zeros(np.max(x_apo))
lower_apo = np.zeros(np.max(x_apo))
highest_apo = np.zeros(np.max(x_apo))
lowest_apo = np.zeros(np.max(x_apo))
for i in range(0, np.max(x_apo)):
    mask = (x_apo == i+1)
    meds_apo[i] = np.nanmedian(y_apo[mask])
    upper_apo[i] = np.nanpercentile(y_apo[mask], onesigp)
    lower_apo[i] = np.nanpercentile(y_apo[mask], onesigm)
    highest_apo[i] = np.nanpercentile(y_apo[mask], twosigp)
    lowest_apo[i] = np.nanpercentile(y_apo[mask], twosigm)
#
f, ax = plt.subplots(1, 1, figsize=(10,10))
plt.scatter(np.arange(np.max(x))+1-0.1, meds_peri, s=50., marker='s', color=summary_plot.colors[1], label='Pericenter period')
for j in range(0, np.max(x_peri)):
    plt.errorbar(np.arange(np.max(x_peri))[j]+1-0.1, meds_peri[j], yerr=np.array([[meds_peri[j]-lowest_peri[j]],[highest_peri[j]-meds_peri[j]]]), alpha=0.3, color=summary_plot.colors[1])
    plt.errorbar(np.arange(np.max(x_peri))[j]+1-0.1, meds_peri[j], yerr=np.array([[meds_peri[j]-lower_peri[j]],[upper_peri[j]-meds_peri[j]]]), alpha=0.7, color=summary_plot.colors[1])
#
plt.scatter(np.arange(np.max(x))+1+0.1, meds_apo, s=50., marker='s', color=summary_plot.colors[3], label='Apocenter period')
for j in range(0, np.max(x_apo)):
    plt.errorbar(np.arange(np.max(x_apo))[j]+1+0.1, meds_apo[j], yerr=np.array([[meds_apo[j]-lowest_apo[j]],[highest_apo[j]-meds_apo[j]]]), alpha=0.3, color=summary_plot.colors[3])
    plt.errorbar(np.arange(np.max(x_apo))[j]+1+0.1, meds_apo[j], yerr=np.array([[meds_apo[j]-lower_apo[j]],[upper_apo[j]-meds_apo[j]]]), alpha=0.7, color=summary_plot.colors[3])
plt.hlines(0, 0, np.max(x_peri)+1, linestyle='dotted', color='k', alpha=0.5)
plt.xlim(0.5, np.max(x_peri)+0.5)
plt.xlabel('Orbit Number')
plt.ylabel('$T_{\\rm model} - T_{\\rm sim}$')
plt.legend(prop={'size': 24}, loc='best')
#plt.show()
plt.tight_layout()
plt.savefig(directory+'/period_vs_phase_both.pdf')
plt.close()





"""
    Period vs Mstar (both)
"""

x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['M.star.z0'][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
M_peri = np.hstack(x)
delta_ecc_peri = np.hstack(y)
#
x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['M.star.z0'][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
M_apo = np.hstack(x)
delta_ecc_apo = np.hstack(y)
#
f, ax = plt.subplots(1, 1, figsize=(10,8))
#
binss, half_binss = summary_plot.binning_scheme(x=M_peri, xtype='M.star.z0', binsize=0.5, binedges=(4.5,10))
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=M_peri, y=delta_ecc_peri, xtype='M.star.z0', ytype='ecc.delta', bins=binss)
plt.plot(10**(binss[:-1]+half_binss), med, color=summary_plot.colors[1], markersize=10, alpha=0.5, label='Pericenter periods')
plt.fill_between(10**(binss[:-1]+half_binss), upper, lower, color=summary_plot.colors[1], alpha=0.3)
plt.fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=summary_plot.colors[1], alpha=0.15)
#
binss, half_binss = summary_plot.binning_scheme(x=M_apo, xtype='M.star.z0', binsize=0.5, binedges=(4.5,10))
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=M_apo, y=delta_ecc_apo, xtype='M.star.z0', ytype='ecc.delta', bins=binss)
plt.plot(10**(binss[:-1]+half_binss), med, color=summary_plot.colors[3], markersize=10, alpha=0.5, label='Apocenter periods')
plt.fill_between(10**(binss[:-1]+half_binss), upper, lower, color=summary_plot.colors[3], alpha=0.3)
plt.fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=summary_plot.colors[3], alpha=0.15)

#
ax.set_xscale('log')
ax.set_yscale('linear')
plt.hlines(0, 10**(4.5), 10**(9.5), linestyle='dotted', color='k', alpha=0.5)
plt.xlim(10**(4.5), 10**(9.5))
plt.ylim(-2,2.5)
plt.xlabel('$M_{\\rm star}$ [$M_{\\odot}$]')
plt.ylabel('$T_{\\rm model} - T_{\\rm sim}$')
plt.legend(prop={'size': 24}, loc='best')
#plt.show()
plt.tight_layout()
plt.savefig(directory+'/delta_period_vs_Mstar_both_zoom.pdf')
plt.close()



"""
    Period vs t_infall,sim (both)
"""

x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['first.infall.time.lb'][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
t_in_sim_peri = np.hstack(x)
delta_ecc_peri = np.hstack(y)
#
x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['first.infall.time.lb'][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
t_in_sim_apo = np.hstack(x)
delta_ecc_apo = np.hstack(y)
#
f, ax = plt.subplots(1, 1, figsize=(10,8))
#
binss, half_binss = summary_plot.binning_scheme(x=t_in_sim_peri, xtype='t.infall.text', binsize=1)
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=t_in_sim_peri, y=delta_ecc_peri, xtype='t.infall.text', ytype='ecc.delta', bins=binss)
plt.plot(binss[:-1]+half_binss, med, '-s', color=summary_plot.colors[1], markersize=10, alpha=0.5, label='Pericenter periods')
plt.fill_between(binss[:-1]+half_binss, upper, lower, color=summary_plot.colors[1], alpha=0.3)
plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=summary_plot.colors[1], alpha=0.15)
#
binss, half_binss = summary_plot.binning_scheme(x=t_in_sim_apo, xtype='t.infall.text', binsize=1)
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=t_in_sim_apo, y=delta_ecc_apo, xtype='t.infall.text', ytype='ecc.delta', bins=binss)
plt.plot(binss[:-1]+half_binss, med, '-s', color=summary_plot.colors[3], markersize=10, alpha=0.5, label='Apocenter periods')
plt.fill_between(binss[:-1]+half_binss, upper, lower, color=summary_plot.colors[3], alpha=0.3)
plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=summary_plot.colors[3], alpha=0.15)

#
ax.set_xscale('linear')
ax.set_yscale('linear')
plt.hlines(0, 0, 14, linestyle='dotted', color='k', alpha=0.5)
plt.xlim(3, 13)
plt.ylim(-2,4)
plt.xlabel('Lookback Infall Time [Gyr]')
plt.ylabel('$T_{\\rm model} - T_{\\rm sim}$')
plt.legend(prop={'size': 24}, loc='best')
#plt.show()
plt.tight_layout()
plt.savefig(directory+'/delta_period_vs_t_in_sim_both_zoom.pdf')
plt.close()



"""
    Period vs t_infall,model (both)
"""

x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['first.infall.time.lb.model'][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
t_in_mod_peri = np.hstack(x)
delta_ecc_peri = np.hstack(y)
#
x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['first.infall.time.lb.model'][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
t_in_mod_apo = np.hstack(x)
delta_ecc_apo = np.hstack(y)
#
f, ax = plt.subplots(1, 1, figsize=(10,8))
#
binss, half_binss = summary_plot.binning_scheme(x=t_in_mod_peri, xtype='t.infall.text', binsize=1)
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=t_in_mod_peri, y=delta_ecc_peri, xtype='t.infall.text', ytype='ecc.delta', bins=binss)
plt.plot(binss[:-1]+half_binss, med, '-s', color=summary_plot.colors[1], markersize=10, alpha=0.5, label='Pericenter periods')
plt.fill_between(binss[:-1]+half_binss, upper, lower, color=summary_plot.colors[1], alpha=0.3)
plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=summary_plot.colors[1], alpha=0.15)
#
binss, half_binss = summary_plot.binning_scheme(x=t_in_mod_apo, xtype='t.infall.text', binsize=1)
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=t_in_mod_apo, y=delta_ecc_apo, xtype='t.infall.text', ytype='ecc.delta', bins=binss)
plt.plot(binss[:-1]+half_binss, med, '-s', color=summary_plot.colors[3], markersize=10, alpha=0.5, label='Apocenter periods')
plt.fill_between(binss[:-1]+half_binss, upper, lower, color=summary_plot.colors[3], alpha=0.3)
plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=summary_plot.colors[3], alpha=0.15)

#
ax.set_xscale('linear')
ax.set_yscale('linear')
plt.hlines(0, 0, 14, linestyle='dotted', color='k', alpha=0.5)
plt.xlim(0,14)
plt.xlim(5, 13)
plt.ylim(-2,2)
plt.xlabel('Lookback Infall Time [Gyr]')
plt.ylabel('$T_{\\rm model} - T_{\\rm sim}$')
plt.legend(prop={'size': 24}, loc='best')
#plt.show()
plt.tight_layout()
plt.savefig(directory+'/delta_period_vs_t_in_mod_both_zoom.pdf')
plt.close()


"""
    Period vs d(z = 0) (both)
"""

x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['d.tot.sim'][:,0][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.peri.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.peri.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
dz0_peri = np.hstack(x)
delta_ecc_peri = np.hstack(y)
#
x = []
y = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]])):
        m_sim = (data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i] != -1)
        m_mod = (data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i] != -1)
        if (np.sum(m_sim) != 0) & (np.sum(m_mod) != 0):
            x.append(np.repeat(data_total[name]['d.tot.sim'][:,0][masks_infall_peri[name]][i], summary.oversample['baryon'][name]))
            y.append( np.repeat( np.average(data_total[name]['orbit.period.apo.model'][masks_infall_peri[name]][i][m_mod]) - np.average(data_total[name]['orbit.period.apo.sim'][masks_infall_peri[name]][i][m_sim]), summary.oversample['baryon'][name] ) )
dz0_apo = np.hstack(x)
delta_ecc_apo = np.hstack(y)
#
f, ax = plt.subplots(1, 1, figsize=(10,8))
#
binss, half_binss = summary_plot.binning_scheme(x=dz0_peri, xtype='d.z0', binsize=50)
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=dz0_peri, y=delta_ecc_peri, xtype='d.z0', ytype='ecc.delta', bins=binss)
plt.plot(binss[:-1]+half_binss, med, '-s', color=summary_plot.colors[1], markersize=10, alpha=0.5, label='Pericenter periods')
plt.fill_between(binss[:-1]+half_binss, upper, lower, color=summary_plot.colors[1], alpha=0.3)
plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=summary_plot.colors[1], alpha=0.15)
#
binss, half_binss = summary_plot.binning_scheme(x=dz0_apo, xtype='d.z0', binsize=50)
med, upper, lower, highest, lowest = summary_plot.median_and_scatter(x=dz0_apo, y=delta_ecc_apo, xtype='d.z0', ytype='ecc.delta', bins=binss)
plt.plot(binss[:-1]+half_binss, med, '-s', color=summary_plot.colors[3], markersize=10, alpha=0.5, label='Apocenter periods')
plt.fill_between(binss[:-1]+half_binss, upper, lower, color=summary_plot.colors[3], alpha=0.3)
plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=summary_plot.colors[3], alpha=0.15)

#
ax.set_xscale('linear')
ax.set_yscale('linear')
plt.hlines(0, 0, 400, linestyle='dotted', color='k', alpha=0.5)
plt.xlim(0,400)
plt.ylim(-2,3)
plt.xlabel('Host Distance, $d$ [kpc]')
plt.ylabel('$T_{\\rm model} - T_{\\rm sim}$')
plt.legend(prop={'size': 24}, loc='best')
#plt.show()
plt.tight_layout()
plt.savefig(directory+'/delta_period_vs_dz0_both_zoom.pdf')
plt.close()
