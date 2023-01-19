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
#
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_no_r')
#
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
summary_plot.median_plot(x=t_in_sim, y=(d_rec_mod-d_rec_sim)/d_rec_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13),(-1,2)), binsize=1, hl=True, file_path_and_name=directory+'/dfrac_recent_vs_t_in_sim_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),None), binsize=0.5, title='Minimum pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_min_vs_t_in_sim.pdf')
summary_plot.median_plot(x=t_in_sim, y=(d_min_mod-d_min_sim)/d_min_sim, xtype='t.infall.text', ytype='delta.d.frac', limits=((0,13.8),(-1,10)), binsize=0.5, title='Minimum pericenter, Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_min_vs_t_in_sim_zoom.pdf')
#
# both d_frac on same plot
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(d_min_mod-d_min_sim)/d_min_sim, (d_rec_mod-d_rec_sim)/d_rec_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,13.8),None), binsize=0.5, labels=['Minimum','Recent'], title='Simulation infall time', hl=True, file_path_and_name=directory+'/dfrac_both_vs_t_in_sim.pdf')
summary_plot.median_plot_mult(x=[t_in_sim,t_in_sim], y=[(d_rec_mod-d_rec_sim)/d_rec_sim, (d_min_mod-d_min_sim)/d_min_sim], xtype=['t.infall.text','t.infall.text'], ytype=['delta.d.frac','delta.d.frac'], limits=((0,13),(-1,2)), binsize=1, labels=['Recent','Minimum'], hl=True, file_path_and_name=directory+'/dfrac_both_vs_t_in_sim_zoom.pdf')

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






dperi_1 = summary.dperi_select(data_total, lb_number=1, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
dperi_2 = summary.dperi_select(data_total, lb_number=2, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
dperi_3 = summary.dperi_select(data_total, lb_number=3, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
#
nperi_mask_1 = summary.data_mask_nperi(data_total, nperi=1, select='sim', hosts='all_no_r')
nperi_mask_2 = summary.data_mask_nperi(data_total, nperi=2, select='sim', hosts='all_no_r')
nperi_mask_3 = summary.data_mask_nperi(data_total, nperi=3, select='sim', hosts='all_no_r')
#
t_in_sim_1 = summary.first_infall(data_total, nperi_mask_1, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_2 = summary.first_infall(data_total, nperi_mask_2, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_3 = summary.first_infall(data_total, nperi_mask_3, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.median_plot_mult(x=[t_in_sim_1, t_in_sim_2, t_in_sim_3], y=[dperi_1, dperi_2, dperi_3], xtype=['t.infall.text','t.infall.text','t.infall.text'], ytype=['d.peri.text','d.peri.text','d.peri.text'], labels=['N=1','N=2','N=3'], binsize=1, limits=((0,13.5),None), file_path_and_name=directory+'/diagnostics/dperi_comp_sim.pdf')



dperi_1 = summary.dperi_select(data_total, lb_number=1, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
dperi_2 = summary.dperi_select(data_total, lb_number=2, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
dperi_3 = summary.dperi_select(data_total, lb_number=3, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
dperi_4 = summary.dperi_select(data_total, lb_number=4, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
dperi_5 = summary.dperi_select(data_total, lb_number=5, mask_selection='sim', selection='sim', oversample=True, hosts='all_no_r')
#
nperi_mask_1 = summary.data_mask_nperi(data_total, nperi=1, select='sim', hosts='all_no_r')
nperi_mask_2 = summary.data_mask_nperi(data_total, nperi=2, select='sim', hosts='all_no_r')
nperi_mask_3 = summary.data_mask_nperi(data_total, nperi=3, select='sim', hosts='all_no_r')
nperi_mask_4 = summary.data_mask_nperi(data_total, nperi=4, select='sim', hosts='all_no_r')
nperi_mask_5 = summary.data_mask_nperi(data_total, nperi=5, select='sim', hosts='all_no_r')
#
t_in_sim_1 = summary.first_infall(data_total, nperi_mask_1, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_2 = summary.first_infall(data_total, nperi_mask_2, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_3 = summary.first_infall(data_total, nperi_mask_3, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_4 = summary.first_infall(data_total, nperi_mask_4, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_5 = summary.first_infall(data_total, nperi_mask_5, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')

xs = [t_in_sim_1, t_in_sim_2, t_in_sim_3, t_in_sim_4, t_in_sim_5]
ys = [dperi_1, dperi_2, dperi_3, dperi_4, dperi_5]
#
xtypes = ['t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text']
ytypes = ['d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text']
labels = ['$N_{\\rm peri} = 1$','$N_{\\rm peri} = 2$','$N_{\\rm peri} = 3$','$N_{\\rm peri} = 4$','$N_{\\rm peri} = 5$']
#
plt.rcParams["font.family"] = "serif"
f, ax = plt.subplots(1, 1, figsize=(11,8))
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=1)
    meds, up1, low1, up2, low2 = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    ax.plot(binss[:-1]+half_binss, meds, label=labels[i])
    if i == 0:
        ax.fill_between(binss[:-1]+half_binss, up1, low1, color='k', alpha=0.3)
        ax.fill_between(binss[:-1]+half_binss, up2, low2, color='k', alpha=0.15)
ax.set_xlim(0, 13.5)
ax.legend(prop={'size': 24})
ax.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
ax.set_xlabel('Lookback Time [Gyr]', fontsize=26)
ax.set_ylabel('Simulation Pericenter Distance [kpc]', fontsize=24)
#
plt.tight_layout()
plt.savefig(directory+'/diagnostics/dperi_mult_one_scatter.pdf')
plt.close()








dperi_1 = summary.dperi_select(data_total, lb_number=1, mask_selection='both', selection='sim', oversample=True, hosts='all_no_r')
dperi_2 = summary.dperi_select(data_total, lb_number=2, mask_selection='both', selection='sim', oversample=True, hosts='all_no_r')
dperi_3 = summary.dperi_select(data_total, lb_number=3, mask_selection='both', selection='sim', oversample=True, hosts='all_no_r')
dperi_4 = summary.dperi_select(data_total, lb_number=4, mask_selection='both', selection='sim', oversample=True, hosts='all_no_r')
dperi_5 = summary.dperi_select(data_total, lb_number=5, mask_selection='both', selection='sim', oversample=True, hosts='all_no_r')
#
dperi_mod_1 = summary.dperi_select(data_total, lb_number=1, mask_selection='both', selection='model', oversample=True, hosts='all_no_r')
dperi_mod_2 = summary.dperi_select(data_total, lb_number=2, mask_selection='both', selection='model', oversample=True, hosts='all_no_r')
dperi_mod_3 = summary.dperi_select(data_total, lb_number=3, mask_selection='both', selection='model', oversample=True, hosts='all_no_r')
dperi_mod_4 = summary.dperi_select(data_total, lb_number=4, mask_selection='both', selection='model', oversample=True, hosts='all_no_r')
dperi_mod_5 = summary.dperi_select(data_total, lb_number=5, mask_selection='both', selection='model', oversample=True, hosts='all_no_r')
#
nperi_mask_1 = summary.data_mask_nperi(data_total, nperi=1, select='both', hosts='all_no_r')
nperi_mask_2 = summary.data_mask_nperi(data_total, nperi=2, select='both', hosts='all_no_r')
nperi_mask_3 = summary.data_mask_nperi(data_total, nperi=3, select='both', hosts='all_no_r')
nperi_mask_4 = summary.data_mask_nperi(data_total, nperi=4, select='both', hosts='all_no_r')
nperi_mask_5 = summary.data_mask_nperi(data_total, nperi=5, select='both', hosts='all_no_r')
nperi_mask_1['m12f'][59] = False
nperi_mask_2['m12f'][59] = False
nperi_mask_3['m12f'][59] = False
nperi_mask_4['m12f'][59] = False
nperi_mask_5['m12f'][59] = False
#
t_in_sim_1 = summary.first_infall(data_total, nperi_mask_1, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_2 = summary.first_infall(data_total, nperi_mask_2, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_3 = summary.first_infall(data_total, nperi_mask_3, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_4 = summary.first_infall(data_total, nperi_mask_4, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim_5 = summary.first_infall(data_total, nperi_mask_5, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')

xs = [t_in_sim_1, t_in_sim_2, t_in_sim_3, t_in_sim_4, t_in_sim_5]
ys = [(dperi_mod_1-dperi_1)/dperi_1, (dperi_mod_2-dperi_2)/dperi_2, (dperi_mod_3-dperi_3)/dperi_3, (dperi_mod_4-dperi_4)/dperi_4, (dperi_mod_5-dperi_5)/dperi_5]
#
xtypes = ['t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text']
ytypes = ['d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text']
labels = ['$N_{\\rm peri} = 1$','$N_{\\rm peri} = 2$','$N_{\\rm peri} = 3$','$N_{\\rm peri} = 4$','$N_{\\rm peri} = 5$']
#
plt.rcParams["font.family"] = "serif"
f, ax = plt.subplots(1, 1, figsize=(11,8))
for i in range(0, len(xs)):
    binss, half_binss = summary_plot.binning_scheme(x=xs[i], xtype=xtypes[i], binsize=1)
    meds, up1, low1, up2, low2 = summary_plot.median_and_scatter(x=xs[i], y=ys[i], xtype=xtypes[i], ytype=ytypes[i], bins=binss)
    ax.plot(binss[:-1]+half_binss, meds, label=labels[i])
    if i == 0:
        print(binss)
        print(meds)
        print(ys[i])
        print(len(ys[i]))
        ax.fill_between(binss[:-1]+half_binss, up1, low1, color='k', alpha=0.3)
        ax.fill_between(binss[:-1]+half_binss, up2, low2, color='k', alpha=0.15)
ax.set_xlim(0,13.5)
ax.set_ylim(-1, 1)
ax.legend(prop={'size': 24})
ax.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
ax.set_xlabel('Lookback Time [Gyr]', fontsize=26)
ax.set_ylabel('$(d_{\\rm peri, model} - d_{\\rm peri,sim})/d_{\\rm peri,sim}$', fontsize=24)
#
plt.tight_layout()
plt.savefig(directory+'/diagnostics/dperi_frac_mult.pdf')
plt.close()




in_mask_1 = summary.data_mask_ninfall(data_total, 1, select='both', hosts='all_no_r')
d_rec_sim_1 = summary.dperi_recent(data_total, in_mask_1, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_1 = summary.dperi_recent(data_total, in_mask_1, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
in_mask_2 = summary.data_mask_ninfall(data_total, 2, select='both', hosts='all_no_r')
d_rec_sim_2 = summary.dperi_recent(data_total, in_mask_2, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_2 = summary.dperi_recent(data_total, in_mask_2, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[d_rec_sim_1, d_rec_sim_2], xtype=['d.peri.text','d.peri.text'], labels=['$N_{\\rm infall}=1$','$N_{\\rm infall}=2$'], binsize=20, xlimits=(0, 250), pdf=True, file_path_and_name=directory+'/diagnostics/dperi_infall_hist.pdf')
summary_plot.plot_hist_mult(x=[(d_rec_mod_1-d_rec_sim_1)/d_rec_sim_1, (d_rec_mod_2-d_rec_sim_2)/d_rec_sim_2], xtype=['delta.d.frac','delta.d.frac'], labels=['$N_{\\rm infall}=1$','$N_{\\rm infall}=2$'], xlimits=(-1,1), binsize=0.05, pdf=True, file_path_and_name=directory+'/diagnostics/dperi_comp_infall_hist.pdf')



"""
    Plotting a 2x2 figure with the host mass and radius vs time
"""

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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
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
        if j == 1:
            print(np.hstack(temp_med))
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
f, axs = plt.subplots(2, 2, figsize=(18,10))
# Plot the scatter for the recent and minimum pericenters
axs[0,0].fill_between(binss[0][:-1]+half_bins[0], uppers[0]/1e12, lowers[0]/1e12, color='g', alpha=0.3)
axs[0,0].fill_between(binss[0][:-1]+half_bins[0], highests[0]/1e12, lowests[0]/1e12, color='g', alpha=0.15)
axs[0,0].plot(binss[0][:-1]+half_bins[0], medians[0]/1e12, 'k', alpha=0.5, lw=4)
#
axs[1,0].fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color='g', alpha=0.3)
axs[1,0].fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color='g', alpha=0.15)
axs[1,0].plot(binss[1][:-1]+half_bins[1], medians[1], 'k', alpha=0.5, lw=4)
#
axs[0,1].fill_between(binss[2][:-1]+half_bins[2], uppers[2], lowers[2], color='g', alpha=0.3)
axs[0,1].fill_between(binss[2][:-1]+half_bins[2], highests[2], lowests[2], color='g', alpha=0.15)
axs[0,1].plot(binss[2][:-1]+half_bins[2], medians[2], 'k', alpha=0.5, lw=4)
#
axs[1,1].fill_between(binss[3][:-1]+half_bins[3], uppers[3], lowers[3], color='g', alpha=0.3)
axs[1,1].fill_between(binss[3][:-1]+half_bins[3], highests[3], lowests[3], color='g', alpha=0.15)
axs[1,1].plot(binss[3][:-1]+half_bins[3], medians[3], 'k', alpha=0.5, lw=4)
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0,0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=22)
axz.set_xlim(13.5,0)
axz.set_xlabel(axis_z_label, fontsize=26, labelpad=9)
axz.tick_params(pad=3)
#
axis_z2_label = 'redshift'
axis_z2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_z2_tick_values = [float(v) for v in axis_z2_tick_labels]
axis_z2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z2_tick_values)
axz2 = axs[0,1].twiny()
axz2.set_xscale('linear')
axz2.set_yscale('linear')
axz2.set_xticks(axis_z2_tick_locations)
axz2.set_xticklabels(axis_z2_tick_labels, fontsize=22)
axz2.set_xlim(13.5,0)
axz2.set_xlabel(axis_z_label, fontsize=26, labelpad=9)
axz2.tick_params(pad=3)
#
axs[0,0].set_xlim(13.5,0)
axs[1,0].set_xlim(13.5,0)
axs[1,1].set_xlim(13.5,0)
axs[0,1].set_xlim(13.5,0)
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=22, labelbottom=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=22, labelbottom=False)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
#
axs[0,0].set_ylabel('$M_{\\rm 200m}(t_{\\rm lb})\ [10^{12}\ M_{\\odot}]$', fontsize=24)
axs[0,0].get_yaxis().set_label_coords(-0.13,0.5)
axs[1,0].set_ylabel('$M_{\\rm 200m}(t_{\\rm lb})\ /\ M_{\\rm 200m}(t_{\\rm lb}=0)$', fontsize=24)
axs[1,0].get_yaxis().set_label_coords(-0.13,0.5)
axs[0,1].set_ylabel('$R_{\\rm 200m}(t_{\\rm lb})\ [\\rm kpc]$', fontsize=24)
axs[0,1].get_yaxis().set_label_coords(-0.13,0.5)
axs[1,1].set_ylabel('$R_{\\rm 200m}(t_{\\rm lb})\ /\ R_{\\rm 200m}(t_{\\rm lb}=0)$', fontsize=24)
axs[1,1].get_yaxis().set_label_coords(-0.13,0.5)
#
axs[1,0].set_xlabel('Lookback Time [Gyr]', fontsize=26)
axs[1,1].set_xlabel('Lookback Time [Gyr]', fontsize=26)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.savefig(directory+'/host_mass_and_radius_scatter.pdf')
plt.close()


"""
    Plotting the da/dr stuff
"""
dadr = summary.da_dr(mass_profile=data_mp, hosts='all_no_r')
dadr_dperi_min = summary.da_dr_dperi_min(data_total, masks_infall_peri, data_mp, dadr, oversample=True, hosts='all_no_r')
dadr_max = summary.da_dr_max(data_total, masks_infall_peri, data_mp, dadr, oversample=True, hosts='all_no_r')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[np.log10(dadr_dperi_min['dadr']), np.log10(dadr_max['dadr'])], xtype=['dadr.log', 'dadr'], labels=['log($|da/dr|_{\\rm dperi,min}$)', 'log($|da/dr|_{\\rm max}$)'], binedges=(-34,-29), binsize=0.1, pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_comp.pdf')
summary_plot.plot_hist_mult(x=[dadr_dperi_min['dadr.time.lb.interp'], dadr_max['dadr.time.lb.interp']], xtype=['t.lb', 't.lb'], labels=['$t_{\\rm da/dr, dperi, min}$', '$t_{\\rm da/dr, max}$'], binsize=0.5, pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_comp_time.pdf')
summary_plot.plot_hist(x=np.log10(dadr_dperi_min['dadr'])-np.log10(dadr_max['dadr']), xtype='dadr.diff.log', binsize=0.01, xlimits=(-0.5,0.1), pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_diff.pdf')
#summary_plot.plot_hist(x=(np.log10(dadr_dperi_min['dadr'])-np.log10(dadr_max['dadr']))/np.log10(dadr_dperi_min['dadr']), xtype='dadr.frac', binsize=0.01, pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_frac.pdf')
summary_plot.plot_hist(x=(dadr_dperi_min['dadr']-dadr_max['dadr'])/dadr_dperi_min['dadr'], xtype='dadr.frac', binsize=0.01, pdf=True, xlimits=(-0.5,0.5), file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_frac_nolog.pdf')
summary_plot.plot_hist(x=dadr_dperi_min['dadr.time.lb.interp']-dadr_max['dadr.time.lb.interp'], xtype='dadr.diff.t', binsize=0.05, xlimits=(-1,1), pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_diff_time.pdf')
#
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#35cddc']
binedges = (-34,-30)
binsize = 0.25
#limits=((0,13.8),None)
#
x = [np.log10(dadr_max['dadr'])]
y = [(dadr_dperi_min['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['dadr']
ytype = ['dadr.frac']
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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
#plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel('$log |da/dr|_{\\rm max}$', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm peri,min} - |da/dr|_{\\rm max})/ |da/dr|_{\\rm max}$', fontsize=20)
#ax1.legend(prop={'size': 21}, loc='best')
#ax1.set_ylim(-2.5,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/diagnostics/dadr/dadr_frac_vs_dadr_max.pdf')



f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#35cddc']
binedges = None
binsize = 1
limits=((0,13.8),None)
#
x = [t_in_sim]
y = [(dadr_dperi_min['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['t.infall.text']
ytype = ['dadr.frac']
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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
#plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel('Lookback infall time [Gyr]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm peri,min} - |da/dr|_{\\rm max})/ |da/dr|_{\\rm max}$', fontsize=20)
#ax1.legend(prop={'size': 21}, loc='best')
ax1.set_ylim(-1,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/diagnostics/dadr/dadr_frac_vs_t_infall.pdf')



f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#35cddc']
binedges = None
binsize = 0.5
limits=(None,None)
#
x = [np.log10(Mstar_z0)]
y = [(dadr_dperi_min['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['t.infall.text']
ytype = ['dadr.frac']
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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
#plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel('$M_{\\rm star}$ [$M_{\\odot}$]]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm peri,min} - |da/dr|_{\\rm max})/ |da/dr|_{\\rm max}$', fontsize=20)
#ax1.legend(prop={'size': 21}, loc='best')
ax1.set_ylim(-0.5,0.2)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/diagnostics/dadr/dadr_frac_vs_mstar.pdf')



"""
    Comparing da/dr stuff between the model and simulation
"""
dadr = summary.da_dr(mass_profile=data_mp, hosts='all_no_r')
#
dadr_dperi_min = summary.da_dr_dperi_min(data_total, masks_infall_peri, data_mp, dadr, selection='sim', oversample=True, hosts='all_no_r')
dadr_dperi_min_mod = summary.da_dr_dperi_min(data_total, masks_infall_peri, data_mp, dadr, selection='model', oversample=True, hosts='all_no_r')
dadr_max = summary.da_dr_max(data_total, masks_infall_peri, data_mp, dadr, oversample=True, hosts='all_no_r')
dadr_max_mod = summary.da_dr_max(data_total, masks_infall_peri, data_mp, dadr, selection='model', oversample=True, hosts='all_no_r')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[np.log10(dadr_dperi_min['dadr']), np.log10(dadr_dperi_min_mod['dadr'])], xtype=['dadr.log', 'dadr'], labels=['log($|da/dr|_{\\rm dperi,min,sim}$)', 'log($|da/dr|_{\\rm peri,min,model}$)'], binedges=(-34,-29), binsize=0.1, pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_dpmin_sim_v_model_comp.pdf')
summary_plot.plot_hist_mult(x=[np.log10(dadr_max['dadr']), np.log10(dadr_max_mod['dadr'])], xtype=['dadr.log', 'dadr'], labels=['log($|da/dr|_{\\rm max,sim}$)', 'log($|da/dr|_{\\rm max,model}$)'], binedges=(-34,-29), binsize=0.1, pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_max_sim_v_model_comp.pdf')
summary_plot.plot_hist_mult(x=[(dadr_dperi_min['dadr']-dadr_max['dadr'])/dadr_max['dadr'], (dadr_dperi_min_mod['dadr']-dadr_max_mod['dadr'])/dadr_max_mod['dadr']], xtype=['dadr.frac', 'dadr'], labels=['Simulation', 'Model'], binsize=0.01, pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_frac_sim_v_model.pdf')
#summary_plot.plot_hist_mult(x=[dadr_dperi_min['dadr.time.lb.interp'], dadr_dperi_min_mod['dadr.time.lb.interp']], xtype=['t.lb', 't.lb'], labels=['$t_{\\rm dperi,min,sim}$)', '$t_{\\rm peri,min,model}$)'], binsize=0.1, pdf=True, file_path_and_name=directory+'/diagnostics/dadr/dadr_hist_tpmin_sim_v_model_comp.pdf')
#
# Plotting vs dadr max
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#006400', '#000080']
binedges = (-34,-30)
binsize = 0.5
#limits=((0,13.8),None)
#
x = [np.log10(dadr_max['dadr']), np.log10(dadr_max_mod['dadr'])]
y = [(dadr_dperi_min['dadr']-dadr_max['dadr'])/dadr_max['dadr'], (dadr_dperi_min_mod['dadr']-dadr_max_mod['dadr'])/dadr_max_mod['dadr']]
#
xtype = ['dadr', 'dadr']
ytype = ['dadr.frac', 'dadr.frac']
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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
ax1.fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], linewidth=3.5, alpha=0.9, label='Simulation')
ax1.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], linewidth=3.5, alpha=0.9, label='Model')
#
#plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel('$log |da/dr|_{\\rm max}$', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm peri,min} - |da/dr|_{\\rm max})/ |da/dr|_{\\rm max}$', fontsize=20)
ax1.legend(prop={'size': 21}, loc='best')
#ax1.set_ylim(-2.5,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/diagnostics/dadr/dadr_frac_vs_dadr_max_sim_v_mod.pdf')





dadr_max = summary.da_dr_max(data_total, masks_infall_peri, data_mp, dadr, selection='sim', oversample=True, hosts='all_no_r')
dadr_max_mod = summary.da_dr_max(data_total, masks_infall_peri, data_mp, dadr, selection='model', oversample=True, hosts='all_no_r')
t_in_sim = summary.first_infall(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_r', sim_type='baryon')
#
# Plotting vs t_infall
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#006400', '#000080']
binedges = None
binsize = 0.5
limits=((0,13.8),(-1,3))
#
x = [t_in_sim]
y = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['t.infall.text']
ytype = ['dadr.frac']
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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
plt.hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], colors='k', linestyles='dotted', alpha=0.5)
#
ax1.set_xlabel('Lookback Infall Time [Gyr]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm model} - |da/dr|_{\\rm sim})/ |da/dr|_{\\rm sim}$', fontsize=20)
#ax1.legend(prop={'size': 21}, loc='best')
ax1.set_xlim(limits[0])
ax1.set_ylim(limits[1])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/dadr_frac_vs_t_infall.pdf')
plt.close()


# Plotting vs t_infall
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#006400', '#000080']
binedges = None
binsize = 50
limits=((0,400),(-1,2.5))
#
x = [dz0_tot]
y = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['d.z0']
ytype = ['dadr.frac']
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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
plt.hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], colors='k', linestyles='dotted', alpha=0.5)
#
ax1.set_xlabel('Host Distance, r [kpc]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm model} - |da/dr|_{\\rm sim})/ |da/dr|_{\\rm sim}$', fontsize=20)
#ax1.legend(prop={'size': 21}, loc='best')
ax1.set_xlim(limits[0])
ax1.set_ylim(limits[1])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/dadr_frac_vs_dz0.pdf')
plt.close()



# Plotting vs t_infall
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#006400', '#000080']
binedges = None
binsize = 0.5
limits=((4,9.5),(-1,1.5))
#
x = [np.log10(Mstar_z0)]
y = [(dadr_max_mod['dadr']-dadr_max['dadr'])/dadr_max['dadr']]
#
xtype = ['M.star.z0']
ytype = ['dadr.frac']
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
    onesigp = 84.13
    onesigm = 15.87
    twosigp = 100
    twosigm = 0
    #
    med = np.zeros(len(bins)-1)
    lower = np.zeros(len(bins)-1)
    upper = np.zeros(len(bins)-1)
    lowest = np.zeros(len(bins)-1)
    highest = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        upper[i] = np.nanpercentile(y[j][mask], onesigp)
        lower[i] = np.nanpercentile(y[j][mask], onesigm)
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
#
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], linewidth=3.5, alpha=0.9)
#
plt.hlines(y=0, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
ax1.set_xlabel('$M_{\\rm star}$ [$M_{\\odot}$]', fontsize=30)
ax1.set_ylabel('($|da/dr|_{\\rm model} - |da/dr|_{\\rm sim})/ |da/dr|_{\\rm sim}$', fontsize=20)
ax1.set_xscale('log')
ax1.set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
ax1.set_ylim(limits[1])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/dadr_frac_vs_Mstar.pdf')
plt.close()



data = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new')
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_res7100')
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_energy_new', sim_type='baryon')
sub_energy = summary.energies(data_total, masks_infall, data, data_mp, snaps, oversample=True, hosts='all_energy_new')
#
# Plot the energy differences vs infall time
summary_plot.median_plot(x=t_in_sim, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.infall']), xtype='t.infall.text', ytype='E.tot', limits=((0,13.5),None), binsize=1, hl=True, axis_labels=['Lookback Infall time [Gyr]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm infall}$|'], file_path_and_name=directory+'/energy/E_norm_infall_vs_t_infall.pdf')
summary_plot.median_plot(x=t_in_sim, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.infall']), xtype='t.infall.text', ytype='E.tot', limits=((0,13.5),(-10,2)), binsize=1, hl=True, axis_labels=['Lookback Infall time [Gyr]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm infall}$|'], file_path_and_name=directory+'/energy/E_norm_infall_vs_t_infall_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.z0']), xtype='t.infall.text', ytype='E.tot', limits=((0,13.5),None), binsize=1, hl=True, axis_labels=['Lookback Infall time [Gyr]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E(z=0)$|'], file_path_and_name=directory+'/energy/E_norm_z0_vs_t_infall.pdf')
summary_plot.median_plot(x=t_in_sim, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.z0']), xtype='t.infall.text', ytype='E.tot', limits=((0,13.5),(-5,2)), binsize=1, hl=True, axis_labels=['Lookback Infall time [Gyr]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E(z=0)$|'], file_path_and_name=directory+'/energy/E_norm_z0_vs_t_infall_zoom.pdf')
summary_plot.median_plot(x=t_in_sim, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir']), xtype='t.infall.text', ytype='E.tot', limits=((0,13.5),None), binsize=1, hl=True, axis_labels=['Lookback Infall time [Gyr]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm vir}$|'], file_path_and_name=directory+'/energy/E_norm_vir_vs_t_infall.pdf')
summary_plot.median_plot(x=t_in_sim, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir']), xtype='t.infall.text', ytype='E.tot', limits=((0,13.5),(-4,0.5)), binsize=1, hl=True, axis_labels=['Lookback Infall time [Gyr]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm vir}$|'], file_path_and_name=directory+'/energy/E_norm_vir_vs_t_infall_zoom.pdf')
#
# Plot the energy differences vs Mstar
sub_energy = summary.energies(data_total, masks_infall_peri, data, data_mp, snaps, oversample=True, hosts='all_energy_new')
Mstar_z0 = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_energy_new', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_energy_new', sim_type='baryon')
#
summary_plot.median_plot(x=Mstar_z0, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.infall']), xtype='M.star.z0', ytype='E.tot', limits=((4,9.5),None), binsize=0.5, hl=True, axis_labels=['$M_{\\rm star} \ [M_{\\odot}]$', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm infall}$|'], file_path_and_name=directory+'/energy/E_norm_infall_vs_Mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.infall']), xtype='M.star.z0', ytype='E.tot', limits=((4,9.5),(-10,0.5)), binsize=0.5, hl=True, axis_labels=['$M_{\\rm star} \ [M_{\\odot}]$', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm infall}$|'], file_path_and_name=directory+'/energy/E_norm_infall_vs_Mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.z0']), xtype='M.star.z0', ytype='E.tot', limits=((4,9.5),None), binsize=0.5, hl=True, axis_labels=['$M_{\\rm star} \ [M_{\\odot}]$', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E(z=0)$|'], file_path_and_name=directory+'/energy/E_norm_z0_vs_Mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.z0']), xtype='M.star.z0', ytype='E.tot', limits=((4,9.5),(-8,0.5)), binsize=0.5, hl=True, axis_labels=['$M_{\\rm star} \ [M_{\\odot}]$', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E(z=0)$|'], file_path_and_name=directory+'/energy/E_norm_z0_vs_Mstar_zoom.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir']), xtype='M.star.z0', ytype='E.tot', limits=((4,9.5),None), binsize=0.5, hl=True, axis_labels=['$M_{\\rm star} \ [M_{\\odot}]$', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm vir}$|'], file_path_and_name=directory+'/energy/E_norm_vir_vs_Mstar.pdf')
summary_plot.median_plot(x=Mstar_z0, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir']), xtype='M.star.z0', ytype='E.tot', limits=((4,9.5),(-3,0.5)), binsize=0.5, hl=True, axis_labels=['$M_{\\rm star} \ [M_{\\odot}]$', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm vir}$|'], file_path_and_name=directory+'/energy/E_norm_vir_vs_Mstar_zoom.pdf')
#
# Plot the energy differences vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.infall']), xtype='d.z0', ytype='E.tot', limits=((0,400),None), binsize=50, hl=True, axis_labels=['Host distance, d [kpc]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm infall}$|'], file_path_and_name=directory+'/energy/E_norm_infall_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.infall']), xtype='d.z0', ytype='E.tot', limits=((0,400),(-10,1)), binsize=50, hl=True, axis_labels=['Host distance, d [kpc]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm infall}$|'], file_path_and_name=directory+'/energy/E_norm_infall_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_tot, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.z0']), xtype='d.z0', ytype='E.tot', limits=((0,400),None), binsize=50, hl=True, axis_labels=['Host distance, d [kpc]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E(z=0)$|'], file_path_and_name=directory+'/energy/E_norm_z0_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['energy.z0']), xtype='d.z0', ytype='E.tot', limits=((0,400),(-10,0.5)), binsize=50, hl=True, axis_labels=['Host distance, d [kpc]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E(z=0)$|'], file_path_and_name=directory+'/energy/E_norm_z0_vs_dz0_zoom.pdf')
summary_plot.median_plot(x=dz0_tot, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir']), xtype='d.z0', ytype='E.tot', limits=((0,400),None), binsize=50, hl=True, axis_labels=['Host distance, d [kpc]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm vir}$|'], file_path_and_name=directory+'/energy/E_norm_vir_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(sub_energy['energy.z0']-sub_energy['energy.infall'])/np.abs(sub_energy['E.vir']), xtype='d.z0', ytype='E.tot', limits=((0,400),(-4, 0.5)), binsize=50, hl=True, axis_labels=['Host distance, d [kpc]', '($E(z=0)$ -  $E_{\\rm infall}$)/|$E_{\\rm vir}$|'], file_path_and_name=directory+'/energy/E_norm_vir_vs_dz0_zoom.pdf')








summary_plot.plot_hist(x=(d_rec_mod - d_rec_mod_aligned), xtype='d.model', x_labels='$d_{\\rm peri,unaligned} - d_{\\rm peri,aligned}$ [kpc]', title='Recent Model Pericenters', pdf=True, binsize=0.05, xlimits=(-0.6,0.6), file_path_and_name=directory+'/d_mod_recent_hist.pdf')
summary_plot.plot_hist(x=(d_rec_mod - d_rec_mod_aligned), xtype='d.model', x_labels='$d_{\\rm peri,unaligned} - d_{\\rm peri,aligned}$ [kpc]', title='Recent Model Pericenters', pdf=False, binsize=0.05, xlimits=(-0.6,0.6), file_path_and_name=directory+'/d_mod_recent_raw.pdf')





summary_plot.plot_hist(x=(sub_energy['energy.z0'] - sub_energy['energy.infall'])/sub_energy['energy.infall'], xtype='E.tot', binsize=1, pdf=True, xlimits=(-15,5), x_labels='($E(z=0)-E_{\\rm infall})/E_{\\rm infall}$', file_path_and_name=directory+'/energy_hist_zoom.pdf')
