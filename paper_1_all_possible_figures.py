#!/usr/bin/python3

"""
    =========================
    = Paper I Summary Plots =
    =========================

    Create figures for Paper I

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
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_z', sim_type='baryon')
data_potentials = summary.data_read_potential(directory=sim_data.home_dir, hosts='all_energy', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_z')
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs'


### Generate all of the data for the plots below
mask_selection = masks_infall
# Fix for the outlier in the Mstar-Mhalo relation
mask_selection['m12f'][57] = False
#
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
d_1st_tot = summary.dperi_first(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
d_apo_tot = summary.dapo_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
t_min_tot = summary.tperi_min(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_any_tot = summary.first_infall_any(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_peak_tot = summary.mstar(data_total, mask_selection, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon')
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon')
vtan_tot = summary.velocities(data_total, mask_selection, selection='tan', oversample=True, hosts='all_no_z', sim_type='baryon')
vrad_tot = summary.velocities(data_total, mask_selection, selection='rad', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')

"""
    Plotting luminous halos with Mstar > 3e4

    - Excluding m12z from this selection
"""

### Median plots
# Mstar vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot], y=[Mstar_z0_tot, Mstar_peak_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['M.star.z0', 'M.star.z0'], labels=['$M_{\\rm star,z=0}$', '$M_{\\rm star,peak}$'], binsize=0.5, binedges=(8,12), file_path_and_name=directory+'/mstar_mhalo_both.pdf')
summary_plot.median_plot(x=Mhalo_peak_tot, y=Mstar_peak_tot, xtype='M.halo.peak', ytype='M.star.peak', binsize=0.5, binedges=(8,12), file_path_and_name=directory+'/mstar_peak_vs_mhalo_peak.pdf')
summary_plot.median_plot(x=Mhalo_peak_tot, y=Mstar_z0_tot, xtype='M.halo.peak', ytype='M.star.z0', binsize=0.5, binedges=(8,12), limits=((7.9,11.5),(4,10)), file_path_and_name=directory+'/mstar_z0_vs_mhalo_peak.pdf')
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=Mstar_z0_tot, xtype='M.halo.peak', ytype='M.star.z0', file_path_and_name=directory+'/mstar_z0_vs_mhalo_z0_scatter.pdf')
summary_plot.scatter_plot(x=Mhalo_peak_tot, y=Mstar_peak_tot, xtype='M.halo.peak', ytype='M.star.peak', file_path_and_name=directory+'/mstar_peak_vs_mhalo_peak_scatter.pdf')

# d_peri vs d_min
summary_plot.median_plot(x=d_sim_tot, y=d_min_tot, xtype='d.peri.recent', ytype='d.peri.min', binsize=50, limits=((0,350),(0,350)), file_path_and_name=directory+'/d_sim_vs_d_min.pdf')

# t_peri vs t_min
summary_plot.median_plot(x=t_sim_tot, y=t_min_tot, xtype='t.peri.recent', ytype='t.peri.min', binsize=1, limits=((0,11),(0,13.8)), file_path_and_name=directory+'/t_sim_vs_t_min.pdf')

# N_sim vs Infall Time
summary_plot.median_plot(x=t_in_tot, y=N_sim_tot, xtype='t.infall.text', ytype='N.peri.text', binsize=1, limits=((0,13.8),(None)), file_path_and_name=directory+'/N_sim_vs_t_infall.pdf')

# vtot vs Infall Time
summary_plot.median_plot(x=t_in_tot, y=vz0_tot, xtype='t.infall.text', ytype='v.tot', binsize=1, limits=((0,13.8),(None)), file_path_and_name=directory+'/final/vtot_vs_t_infall.pdf')

# Ltot vs Infall Time
summary_plot.median_plot(x=t_in_tot, y=L_tot/1e4, xtype='t.infall.text', ytype='L.tot', binsize=1, limits=((0,13.8),(0,4)), file_path_and_name=directory+'/final/Ltot_vs_t_infall.pdf')


# N_sim vs Mstar(z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=N_sim_tot, xtype='M.star.z0', ytype='N.peri.text', binsize=0.5, file_path_and_name=directory+'/N_sim_vs_Mstar_z0.pdf')
summary_plot.median_plot(x=Mstar_z0_tot, y=N_sim_tot, xtype='M.star.z0', ytype='N.peri.text', binedges=(4.5,9.5), binsize=0.5, limits=((4,9.5),(0,6)), file_path_and_name=directory+'/N_sim_vs_Mstar_z0_zoom.pdf')

# d_min & d_recent vs Mstar(z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot], y=[d_sim_tot, d_min_tot], xtype=['M.star.z0', 'M.star.z0'], ytype=['d.peri.text', 'd.peri.text'], labels=['Recent', 'Minimum'], binsize=0.5, file_path_and_name=directory+'/d_peri_both_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot], y=[d_sim_tot, d_min_tot], xtype=['M.star.z0', 'M.star.z0'], ytype=['d.peri.text', 'd.peri.text'], labels=['Recent', 'Minimum'], binedges=(4.5,9.5), binsize=0.5, limits=((4,9.5),(0,225)), file_path_and_name=directory+'/d_peri_both_vs_Mstar_z0_zoom.pdf')

# t_min & t_recent vs Mstar(z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot], y=[t_sim_tot, t_min_tot], xtype=['M.star.z0', 'M.star.z0'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binsize=0.5, binedges=(4.5,9.5), limits=((4,9.5),(0,11)), file_path_and_name=directory+'/t_peri_both_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot], y=[t_sim_tot, t_min_tot], xtype=['M.star.z0', 'M.star.z0'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binsize=0.5, binedges=(4.5,9.5), limits=((4,9.5),(0,9)), file_path_and_name=directory+'/t_peri_both_vs_Mstar_z0_zoom.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot], y=[t_sim_tot, t_min_tot], xtype=['M.star.z0', 'M.star.z0'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binsize=0.5, binedges=(4.5,9.5), limits=((4,9.5),(0,6)), file_path_and_name=directory+'/t_peri_both_vs_Mstar_z0_zoom2.pdf')

# t_infall (both) vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot], y=[t_in_tot, t_in_any_tot], xtype=['M.star.z0', 'M.star.z0'], ytype=['t.infall.text', 't.infall.text'], labels=['MW/M31 halo', 'Any halo'], binedges=(4.5,9.5), binsize=0.5, limits=((4,9.5),(0,14)), file_path_and_name=directory+'/t_infall_both_vs_Mstar_z0.pdf')

# v_tot vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=vz0_tot, xtype='M.star.z0', ytype='v.tot', binsize=0.5, file_path_and_name=directory+'/vtot_vs_Mstar_z0.pdf')
summary_plot.median_plot(x=Mstar_z0_tot, y=vz0_tot, xtype='M.star.z0', ytype='v.tot', binsize=0.5, binedges=(4.5,9.5), limits=((4,9.5),(0,280)), file_path_and_name=directory+'/vtot_vs_Mstar_z0_zoom.pdf')

# L_tot vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=L_tot/1e4, xtype='M.star.z0', ytype='L.tot', binsize=0.5, file_path_and_name=directory+'/Ltot_vs_Mstar_z0.pdf')
summary_plot.median_plot(x=Mstar_z0_tot, y=L_tot/1e4, xtype='M.star.z0', ytype='L.tot', binsize=0.5, binedges=(4.5,9.5), limits=((4,9.5),(0,4)), file_path_and_name=directory+'/Ltot_vs_Mstar_z0_zoom.pdf')

# N_sim vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=N_sim_tot, xtype='d.z0', ytype='N.peri.text', binsize=50, file_path_and_name=directory+'/N_sim_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=N_sim_tot, xtype='d.z0', ytype='N.peri.text', binsize=50, limits=((0,400),(0,8)), file_path_and_name=directory+'/N_sim_vs_dz0_zoom.pdf')

# d_min & d_recent vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot], y=[d_sim_tot, d_min_tot], xtype=['d.z0', 'd.z0'], ytype=['d.peri.text', 'd.peri.text'], labels=['Recent', 'Minimum'], binsize=50, file_path_and_name=directory+'/d_peri_both_vs_dz0.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot], y=[d_sim_tot, d_min_tot], xtype=['d.z0', 'd.z0'], ytype=['d.peri.text', 'd.peri.text'], labels=['Recent', 'Minimum'], binsize=50, limits=((0,400),(0,200)), file_path_and_name=directory+'/d_peri_both_vs_dz0_zoom.pdf')

# t_min & t_recent vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot], y=[t_sim_tot, t_min_tot], xtype=['d.z0', 'd.z0'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_peri_both_vs_dz0.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot], y=[t_sim_tot, t_min_tot], xtype=['d.z0', 'd.z0'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binsize=50, limits=((0,400),(0,10)), file_path_and_name=directory+'/t_peri_both_vs_dz0_zoom.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot], y=[t_sim_tot, t_min_tot], xtype=['d.z0', 'd.z0'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binsize=50, limits=((0,400),(0,6)), file_path_and_name=directory+'/t_peri_both_vs_dz0_zoom2.pdf')

# Infall time (both) vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot], y=[t_in_tot, t_in_any_tot], xtype=['d.z0', 'd.z0'], ytype=['t.infall.text', 't.infall.text'], labels=['MW/M31 halo', 'Any halo'], binsize=50, limits=((0,400),(0,14)), file_path_and_name=directory+'/t_infall_both_vs_dz0.pdf')

# vtan vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=vtan_tot, xtype='d.z0', ytype='v.tan', binsize=50, file_path_and_name=directory+'/vtan_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=vtan_tot, xtype='d.z0', ytype='v.tan', binsize=50, limits=((0,400),None), file_path_and_name=directory+'/vtan_vs_dz0_zoom.pdf')

# vtot vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=vz0_tot, xtype='d.z0', ytype='v.tot', binsize=50, file_path_and_name=directory+'/vtot_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=vz0_tot, xtype='d.z0', ytype='v.tot', binsize=50, limits=((0,400),(0,340)), file_path_and_name=directory+'/vtot_vs_dz0_zoom.pdf')

# Ltot vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=L_tot/1e4, xtype='d.z0', ytype='L.tot', binsize=50, file_path_and_name=directory+'/Ltot_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=L_tot/1e4, xtype='d.z0', ytype='L.tot', binsize=50, limits=((0,400), (0,5)), file_path_and_name=directory+'/Ltot_vs_dz0_zoom.pdf')


"""
    Energy plots
"""
potential_tot = summary.potential(data_potentials, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon')


# E_tot vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=(ke_z0_tot+potential_tot)/1e4, xtype='M.star.z0', ytype='E.tot', binsize=0.5, file_path_and_name=directory+'/Etot_vs_Mstar_z0.pdf')
summary_plot.median_plot(x=Mstar_z0_tot, y=(ke_z0_tot+potential_tot)/1e4, xtype='M.star.z0', ytype='E.tot', binedges=(4.5,9.5), binsize=0.5, limits=((4,9.5),(-4,1)), file_path_and_name=directory+'/Etot_vs_Mstar_z0_zoom.pdf')

# E_tot vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=(ke_z0_tot+potential_tot)/1e4, xtype='d.z0', ytype='E.tot', binsize=50, file_path_and_name=directory+'/Etot_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=(ke_z0_tot+potential_tot)/1e4, xtype='d.z0', ytype='E.tot', binsize=50, limits=((0,400),(-5,2)), file_path_and_name=directory+'/Etot_vs_dz0_zoom.pdf')

# Etot vs Infall time
summary_plot.median_plot(x=t_in_tot, y=(ke_z0_tot+potential_tot)/1e4, xtype='t.infall.text', ytype='E.tot', binsize=1, limits=((0,13.8),(-4,1)), file_path_and_name=directory+'/final/Etot_vs_t_infall.pdf')


"""
    Plotting properties vs peak halo mass for all subhalos (including dark ones)
    from the baryonic simulations
        - Mhalo,peak > 1e8 Msun

    NOTE: This is for ALL hosts
"""

### Generate all of the data for the plots below
data_total_all = summary.data_read(directory=sim_data.home_dir, hosts='all_no_z', sim_type='all_baryon')
data_potentials_all = summary.data_read_potential(directory=sim_data.home_dir, hosts='all_energy', sim_type='all_baryon')
masks_infall = summary.data_mask(data_total_all, peri_sim=False, peri_model=False, hosts='all_no_z')
mask_selection = masks_infall
#
N_sim_tot_all = summary.nperi(data_total_all, mask_selection, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon_all')
dz0_tot_all = summary.d_z0(data_total_all, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon_all')
d_sim_tot_all = summary.dperi_recent(data_total_all, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon_all')
d_min_tot_all = summary.dperi_min(data_total_all, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon_all')
t_sim_tot_all = summary.tperi_recent(data_total_all, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon_all')
t_min_tot_all = summary.tperi_min(data_total_all, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon_all')
t_in_tot_all = summary.first_infall(data_total_all, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon_all')
t_in_any_tot_all = summary.first_infall_any(data_total_all, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon_all')
Mhalo_peak_tot_all = summary.mhalo(data_total_all, mask_selection, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon_all')
vtan_tot_all = summary.velocities(data_total_all, mask_selection, selection='tan', oversample=True, hosts='all_no_z', sim_type='baryon_all')
vrad_tot_all = summary.velocities(data_total_all, mask_selection, selection='rad', oversample=True, hosts='all_no_z', sim_type='baryon_all')
vz0_tot_all = summary.v_z0(data_total_all, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon_all')
L_tot_all = summary.L_z0(data_total_all, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon_all')

### Median plots
# N_sim vs Mhalo (peak)
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=N_sim_tot_all, xtype='M.halo.peak', ytype='N.peri.text', binsize=0.5, file_path_and_name=directory+'/N_sim_vs_Mhalo_peak_baryon_all.pdf')
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=N_sim_tot_all, xtype='M.halo.peak', ytype='N.peri.text', binedges=(8,11.5), binsize=0.5, limits=((8,11.5),(0,5)), file_path_and_name=directory+'/N_sim_vs_Mhalo_peak_baryon_all_zoom.pdf')

# d_recent & d_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_all, Mhalo_peak_tot_all], y=[d_sim_tot_all, d_min_tot_all], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.text', 'd.peri.text'], labels=['Recent', 'Minimum'], binsize=0.5, file_path_and_name=directory+'/d_peri_both_vs_Mhalo_peak_baryon_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_all, Mhalo_peak_tot_all], y=[d_sim_tot_all, d_min_tot_all], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.text', 'd.peri.text'], labels=['Recent', 'Minimum'], binedges=(8,11.5), binsize=0.5, limits=((8,11.5),(0,250)), file_path_and_name=directory+'/d_peri_both_vs_Mhalo_peak_baryon_all_zoom.pdf')

# t_recent & t_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_all, Mhalo_peak_tot_all], y=[t_sim_tot_all, t_min_tot_all], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/t_peri_both_vs_Mhalo_peak_baryon_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_all, Mhalo_peak_tot_all], y=[t_sim_tot_all, t_min_tot_all], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binedges=(8,11.5), binsize=0.5, limits=((8,11.5),(0,8.5)), file_path_and_name=directory+'/t_peri_both_vs_Mhalo_peak_baryon_all_zoom.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_all, Mhalo_peak_tot_all], y=[t_sim_tot_all, t_min_tot_all], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.text', 't.peri.text'], labels=['Recent', 'Minimum'], binedges=(8,11.5), binsize=0.5, limits=((8,11.5),(0,5)), file_path_and_name=directory+'/t_peri_both_vs_Mhalo_peak_baryon_all_zoom2.pdf')

# t_infall,host & t_infall,any vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_all, Mhalo_peak_tot_all], y=[t_in_tot_all, t_in_any_tot_all], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.text', 't.infall.text'], labels=['MW/M31 halo', 'Any halo'], binedges=(8,11.5), limits=((8,11.5), None), binsize=0.5, file_path_and_name=directory+'/t_infall_both_vs_Mhalo_peak_baryon_all.pdf')

# v_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=vz0_tot_all, xtype='M.halo.peak', ytype='v.tot', binsize=0.5, file_path_and_name=directory+'/vtot_z0_vs_Mhalo_peak_baryon_all.pdf')
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=vz0_tot_all, xtype='M.halo.peak', ytype='v.tot', binsize=0.5, limits=((8,11.5),(0,250)), file_path_and_name=directory+'/vtot_z0_vs_Mhalo_peak_baryon_all_zoom.pdf')

# L_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=L_tot_all/1e4, xtype='M.halo.peak', ytype='L.tot', binsize=0.5, file_path_and_name=directory+'/Ltot_vs_Mhalo_peak_baryon_all.pdf')
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=L_tot_all/1e4, xtype='M.halo.peak', ytype='L.tot', binsize=0.5, limits=((8,11.5),(0,3.5)), file_path_and_name=directory+'/Ltot_vs_Mhalo_peak_baryon_all_zoom.pdf')

#### Energy plots
#masks_infall = summary.data_mask(data_total_all, peri_sim=False, peri_model=False, hosts='all_energy')
#mask_selection = masks_infall
#
potential_tot_all = summary.potential(data_potentials_all, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon_all', norm='kinetic')
ke_z0_tot_all = summary.kinetic_energy(data_total_all, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon_all')
#
Mhalo_peak_tot_all = summary.mhalo(data_total_all, mask_selection, selection='peak', oversample=True, hosts='all_energy', sim_type='baryon_all')
dz0_tot_all = summary.d_z0(data_total_all, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon_all')

# E_tot vs Mhalo (peak)
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=(ke_z0_tot_all+potential_tot_all)/1e4, xtype='M.halo.peak', ytype='E.tot', binsize=0.5, file_path_and_name=directory+'/Etot_vs_Mhalo_peak_baryon_all.pdf')
summary_plot.median_plot(x=Mhalo_peak_tot_all, y=(ke_z0_tot_all+potential_tot_all)/1e4, xtype='M.halo.peak', ytype='E.tot', binsize=0.5, limits=((8,11.5),(-5,1)), file_path_and_name=directory+'/Etot_vs_Mhalo_peak_baryon_all_zoom.pdf')

# E_tot vs d(z = 0)
summary_plot.median_plot(x=dz0_tot_all, y=(ke_z0_tot_all+potential_tot_all)/1e4, xtype='d.z0', ytype='E.tot', binsize=50, file_path_and_name=directory+'/Etot_vs_dz0_baryon_all.pdf')
summary_plot.median_plot(x=dz0_tot_all, y=(ke_z0_tot_all+potential_tot_all)/1e4, xtype='d.z0', ytype='E.tot', binsize=50, limits=((0,400),(-4,2)), file_path_and_name=directory+'/Etot_vs_dz0_baryon_all_zoom.pdf')



"""
    Comparing the halos with the two different selections:
        - Isolated
        - Paired
"""

### Generate all of the data for the plots below
data_total_iso = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='baryon')
data_potentials_iso = summary.data_read_potential(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='baryon')
masks_infall = summary.data_mask(data_total_iso, peri_sim=False, peri_model=False, hosts='iso_no_z')
mask_selection_iso = masks_infall
mask_selection_iso['m12f'][57] = False
#
N_sim_iso = summary.nperi(data_total_iso, mask_selection_iso, oversample=True, selection='sim', hosts='iso_no_z', sim_type='baryon')
d_sim_iso = summary.dperi_recent(data_total_iso, mask_selection_iso, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon')
d_min_iso = summary.dperi_min(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
dz0_iso = summary.d_z0(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
t_sim_iso = summary.tperi_recent(data_total_iso, mask_selection_iso, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon')
t_min_iso = summary.tperi_min(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
t_in_iso = summary.first_infall(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
t_in_any_iso = summary.first_infall_any(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
Mstar_z0_iso = summary.mstar(data_total_iso, mask_selection_iso, selection='z0', oversample=True, hosts='iso_no_z', sim_type='baryon')
Mhalo_peak_iso = summary.mhalo(data_total_iso, mask_selection_iso, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon')
vtan_iso = summary.velocities(data_total_iso, mask_selection_iso, selection='tan', oversample=True, hosts='iso_no_z', sim_type='baryon')
vz0_iso = summary.v_z0(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
L_iso = summary.L_z0(data_total_iso, mask_selection_iso, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon')
#
data_total_lg = summary.data_read(directory=sim_data.home_dir, hosts='lg', sim_type='baryon')
data_potentials_lg = summary.data_read_potential(directory=sim_data.home_dir, hosts='lg_no_RR', sim_type='baryon')
masks_infall = summary.data_mask(data_total_lg, peri_sim=False, peri_model=False, hosts='lg')
mask_selection_lg = masks_infall
#
N_sim_lg = summary.nperi(data_total_lg, mask_selection_lg, oversample=True, selection='sim', hosts='lg', sim_type='baryon')
d_sim_lg = summary.dperi_recent(data_total_lg, mask_selection_lg, selection='sim', oversample=True, hosts='lg', sim_type='baryon')
d_min_lg = summary.dperi_min(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon')
dz0_lg = summary.d_z0(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon')
t_sim_lg = summary.tperi_recent(data_total_lg, mask_selection_lg, selection='sim', oversample=True, hosts='lg', sim_type='baryon')
t_min_lg = summary.tperi_min(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon')
t_in_lg = summary.first_infall(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon')
t_in_any_lg = summary.first_infall_any(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon')
Mstar_z0_lg = summary.mstar(data_total_lg, mask_selection_lg, selection='z0', oversample=True, hosts='lg', sim_type='baryon')
Mhalo_peak_lg = summary.mhalo(data_total_lg, mask_selection_lg, selection='peak', oversample=True, hosts='lg', sim_type='baryon')
vtan_lg = summary.velocities(data_total_lg, mask_selection_lg, selection='tan', oversample=True, hosts='lg', sim_type='baryon')
vrad_lg = summary.velocities(data_total_lg, mask_selection_lg, selection='rad', oversample=True, hosts='lg', sim_type='baryon')
vz0_lg = summary.v_z0(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon')
L_lg = summary.L_z0(data_total_lg, mask_selection_lg, selection='sim', oversample=True, hosts='lg', sim_type='baryon')


### Median plots
# N_sim vs Mstar(z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[N_sim_iso, N_sim_lg], xtype=['M.star.z0','M.star.z0'], ytype=['N.peri.text','N.peri.text'], labels=['Isolated', 'Paired'], binsize=1, file_path_and_name=directory+'/N_sim_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[N_sim_iso, N_sim_lg], xtype=['M.star.z0','M.star.z0'], ytype=['N.peri.text','N.peri.text'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,5.5)), file_path_and_name=directory+'/N_sim_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# d_sim vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[d_sim_iso, d_sim_lg], xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.recent','d.peri.recent'], labels=['Isolated', 'Paired'], binsize=1, file_path_and_name=directory+'/d_sim_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[d_sim_iso, d_sim_lg], xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.recent','d.peri.recent'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,200)), file_path_and_name=directory+'/d_sim_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# d_min vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[d_min_iso, d_min_lg], xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.min','d.peri.min'], labels=['Isolated', 'Paired'], binsize=1, file_path_and_name=directory+'/d_min_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[d_min_iso, d_min_lg], xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.min','d.peri.min'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,175)), file_path_and_name=directory+'/d_min_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# t_sim vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_sim_iso, t_sim_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.peri.recent','t.peri.recent'], labels=['Isolated', 'Paired'], binsize=1, limits=((4,9.5),(0,13.8)), file_path_and_name=directory+'/t_sim_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_sim_iso, t_sim_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.peri.recent','t.peri.recent'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,6)), file_path_and_name=directory+'/t_sim_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# t_min vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_min_iso, t_min_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.peri.min','t.peri.min'], labels=['Isolated', 'Paired'], binsize=1, limits=((4,9.5),(0,13.8)), file_path_and_name=directory+'/t_min_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_min_iso, t_min_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.peri.min','t.peri.min'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,9)), file_path_and_name=directory+'/t_min_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# t_infall vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_in_iso, t_in_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.infall.text','t.infall.text'], labels=['Isolated', 'Paired'], binsize=1, limits=((4,9.5),(0,13.8)), file_path_and_name=directory+'/t_infall_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_in_iso, t_in_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.infall.text','t.infall.text'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,11)), file_path_and_name=directory+'/t_infall_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# t_infall (any) vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_in_any_iso, t_in_any_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.infall.any','t.infall.any'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,13.8)), file_path_and_name=directory+'/t_infall_any_vs_Mstar_z0_iso_vs_lg.pdf')

# v_tot vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[vz0_iso, vz0_lg], xtype=['M.star.z0','M.star.z0'], ytype=['v.tot','v.tot'], labels=['Isolated', 'Paired'], binsize=1, file_path_and_name=directory+'/vtot_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[vz0_iso, vz0_lg], xtype=['M.star.z0','M.star.z0'], ytype=['v.tot','v.tot'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,250)), file_path_and_name=directory+'/vtot_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# L_tot vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[L_iso/1e4, L_lg/1e4], xtype=['M.star.z0','M.star.z0'], ytype=['L.tot','L.tot'], labels=['Isolated', 'Paired'], binsize=1, file_path_and_name=directory+'/Ltot_vs_Mstar_z0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[L_iso/1e4, L_lg/1e4], xtype=['M.star.z0','M.star.z0'], ytype=['L.tot','L.tot'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,3.5)), file_path_and_name=directory+'/Ltot_vs_Mstar_z0_iso_vs_lg_zoom.pdf')

# N_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[N_sim_iso, N_sim_lg], xtype=['d.z0', 'd.z0'], ytype=['N.peri.text', 'N.peri.text'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/N_sim_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[N_sim_iso, N_sim_lg], xtype=['d.z0', 'd.z0'], ytype=['N.peri.text', 'N.peri.text'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,8)), file_path_and_name=directory+'/N_sim_vs_dz0_iso_vs_lg_zoom.pdf')

# d_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[d_sim_iso, d_sim_lg], xtype=['d.z0', 'd.z0'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/d_sim_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[d_sim_iso, d_sim_lg], xtype=['d.z0', 'd.z0'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,250)), file_path_and_name=directory+'/d_sim_vs_dz0_iso_vs_lg_zoom.pdf')

# d_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[d_min_iso, d_min_lg], xtype=['d.z0', 'd.z0'], ytype=['d.peri.min', 'd.peri.min'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/d_min_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[d_min_iso, d_min_lg], xtype=['d.z0', 'd.z0'], ytype=['d.peri.min', 'd.peri.min'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,250)), file_path_and_name=directory+'/d_min_vs_dz0_iso_vs_lg_zoom.pdf')

# t_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_sim_iso, t_sim_lg], xtype=['d.z0', 'd.z0'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_sim_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_sim_iso, t_sim_lg], xtype=['d.z0', 'd.z0'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,5.5)), file_path_and_name=directory+'/t_sim_vs_dz0_iso_vs_lg_zoom.pdf')

# t_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_min_iso, t_min_lg], xtype=['d.z0', 'd.z0'], ytype=['t.peri.min', 't.peri.min'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_min_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_min_iso, t_min_lg], xtype=['d.z0', 'd.z0'], ytype=['t.peri.min', 't.peri.min'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,10)), file_path_and_name=directory+'/t_min_vs_dz0_iso_vs_lg_zoom.pdf')

# Infall time vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_in_iso, t_in_lg], xtype=['d.z0', 'd.z0'], ytype=['t.infall.text', 't.infall.text'], labels=['Isolated', 'Paired'], binsize=50, limits=(None,(0,13.8)), file_path_and_name=directory+'/t_infall_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_in_iso, t_in_lg], xtype=['d.z0', 'd.z0'], ytype=['t.infall.text', 't.infall.text'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_infall_vs_dz0_iso_vs_lg_zoom.pdf')

# Infall time (any) vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_in_any_iso, t_in_any_lg], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Isolated', 'Paired'], binsize=50, limits=(None,(0,13.8)), file_path_and_name=directory+'/t_infall_any_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[t_in_any_iso, t_in_any_lg], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_infall_any_vs_dz0_iso_vs_lg_zoom.pdf')

# vtan vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[vtan_iso, vtan_lg], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/vtan_z0_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[vtan_iso, vtan_lg], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,300)), file_path_and_name=directory+'/vtan_z0_vs_dz0_iso_vs_lg_zoom.pdf')

# vtot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[vz0_iso, vz0_lg], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/vtot_z0_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[vz0_iso, vz0_lg], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,350)), file_path_and_name=directory+'/vtot_z0_vs_dz0_iso_vs_lg_zoom.pdf')

# Ltot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[L_iso/1e4, L_lg/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/Ltot_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_iso, dz0_lg], y=[L_iso/1e4, L_lg/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(0,4)), file_path_and_name=directory+'/Ltot_vs_dz0_iso_vs_lg_zoom.pdf')


"""
    Energy plots
"""
potential_tot_iso = summary.potential(data_potentials_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon', norm='kinetic')
potential_tot_lg = summary.potential(data_potentials_lg, mask_selection_lg, oversample=True, hosts='lg_no_RR', sim_type='baryon', norm='kinetic')
ke_z0_tot_iso = summary.kinetic_energy(data_total_iso, mask_selection_iso, ke_type='z0', oversample=True, hosts='iso_no_z', sim_type='baryon')
ke_z0_tot_lg = summary.kinetic_energy(data_total_lg, mask_selection_lg, ke_type='z0', oversample=True, hosts='lg_no_RR', sim_type='baryon')
#
Mstar_z0_tot_iso = summary.mstar(data_total_iso, mask_selection_iso, selection='z0', oversample=True, hosts='iso_no_z', sim_type='baryon')
Mstar_z0_tot_lg = summary.mstar(data_total_lg, mask_selection_lg, selection='z0', oversample=True, hosts='lg_no_RR', sim_type='baryon')
dz0_tot_iso = summary.d_z0(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
dz0_tot_lg = summary.d_z0(data_total_lg, mask_selection_lg, oversample=True, hosts='lg_no_RR', sim_type='baryon')

# E_tot vs Mstar (z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_tot_iso, Mstar_z0_tot_lg], y=[(ke_z0_tot_iso+potential_tot_iso)/1e4, (ke_z0_tot_lg+potential_tot_lg)/1e4], xtype=['M.star.z0', 'M.star.z0'], ytype=['E.tot', 'E.tot'], labels=['Isolated', 'Paired'], binsize=1, file_path_and_name=directory+'/Etot_vs_Mstar_z0_iso_vs_lg.pdf')

# E_tot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot_iso, dz0_tot_lg], y=[(ke_z0_tot_iso+potential_tot_iso)/1e4, (ke_z0_tot_lg+potential_tot_lg)/1e4], xtype=['d.z0', 'd.z0'], ytype=['E.tot', 'E.tot'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/Etot_vs_dz0_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_tot_iso, dz0_tot_lg], y=[(ke_z0_tot_iso+potential_tot_iso)/1e4, (ke_z0_tot_lg+potential_tot_lg)/1e4], xtype=['d.z0', 'd.z0'], ytype=['E.tot', 'E.tot'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(-5,2)), file_path_and_name=directory+'/Etot_vs_dz0_iso_vs_lg_zoom.pdf')



"""
    Comparing properties vs Mpeak with the two different selections:
        - Isolated
        - Paired

    NOTE: Doing this for all subhalos, luminous and dark
"""

### Generate all of the data for the plots below
data_total_iso = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='all_baryon')
data_potentials_iso = summary.data_read_potential(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='all_baryon')
masks_infall = summary.data_mask(data_total_iso, peri_sim=False, peri_model=False, hosts='iso_no_z')
mask_selection_iso = masks_infall
#
N_sim_iso = summary.nperi(data_total_iso, mask_selection_iso, oversample=True, selection='sim', hosts='iso_no_z', sim_type='baryon_all')
d_sim_iso = summary.dperi_recent(data_total_iso, mask_selection_iso, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
d_min_iso = summary.dperi_min(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
dz0_iso = summary.d_z0(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_sim_iso = summary.tperi_recent(data_total_iso, mask_selection_iso, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_min_iso = summary.tperi_min(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_in_iso = summary.first_infall(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_in_any_iso = summary.first_infall_any(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_iso = summary.mhalo(data_total_iso, mask_selection_iso, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
vtan_iso = summary.velocities(data_total_iso, mask_selection_iso, selection='tan', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
vrad_iso = summary.velocities(data_total_iso, mask_selection_iso, selection='rad', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
vz0_iso = summary.v_z0(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
L_iso = summary.L_z0(data_total_iso, mask_selection_iso, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
#
data_total_lg = summary.data_read(directory=sim_data.home_dir, hosts='lg', sim_type='all_baryon')
data_potentials_lg = summary.data_read_potential(directory=sim_data.home_dir, hosts='lg_no_RR', sim_type='all_baryon')
masks_infall = summary.data_mask(data_total_lg, peri_sim=False, peri_model=False, hosts='lg')
mask_selection_lg = masks_infall
#
N_sim_lg = summary.nperi(data_total_lg, mask_selection_lg, oversample=True, selection='sim', hosts='lg', sim_type='baryon_all')
d_sim_lg = summary.dperi_recent(data_total_lg, mask_selection_lg, selection='sim', oversample=True, hosts='lg', sim_type='baryon_all')
d_min_lg = summary.dperi_min(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon_all')
dz0_lg = summary.d_z0(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon_all')
t_sim_lg = summary.tperi_recent(data_total_lg, mask_selection_lg, selection='sim', oversample=True, hosts='lg', sim_type='baryon_all')
t_min_lg = summary.tperi_min(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon_all')
t_in_lg = summary.first_infall(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon_all')
t_in_any_lg = summary.first_infall_any(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon_all')
Mhalo_peak_lg = summary.mhalo(data_total_lg, mask_selection_lg, selection='peak', oversample=True, hosts='lg', sim_type='baryon_all')
vtan_lg = summary.velocities(data_total_lg, mask_selection_lg, selection='tan', oversample=True, hosts='lg', sim_type='baryon_all')
vrad_lg = summary.velocities(data_total_lg, mask_selection_lg, selection='rad', oversample=True, hosts='lg', sim_type='baryon_all')
vz0_lg = summary.v_z0(data_total_lg, mask_selection_lg, oversample=True, hosts='lg', sim_type='baryon_all')
L_lg = summary.L_z0(data_total_lg, mask_selection_lg, selection='sim', oversample=True, hosts='lg', sim_type='baryon_all')


## Median plots
# N_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[N_sim_iso, N_sim_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.peri.text', 'N.peri.text'], labels=['Isolated', 'Paired'], binsize=0.5, file_path_and_name=directory+'/N_sim_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[N_sim_iso, N_sim_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.peri.text', 'N.peri.text'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,5)), file_path_and_name=directory+'/N_sim_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# d_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[d_sim_iso, d_sim_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Isolated', 'Paired'], binsize=0.5, file_path_and_name=directory+'/d_sim_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[d_sim_iso, d_sim_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,300)), file_path_and_name=directory+'/d_sim_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# d_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[d_min_iso, d_min_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.min', 'd.peri.min'], labels=['Isolated', 'Paired'], binsize=0.5, file_path_and_name=directory+'/d_min_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[d_min_iso, d_min_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.min', 'd.peri.min'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,175)), file_path_and_name=directory+'/d_min_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# t_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_sim_iso, t_sim_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/t_sim_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_sim_iso, t_sim_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,6)), file_path_and_name=directory+'/t_sim_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# t_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_min_iso, t_min_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.min', 't.peri.min'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/t_min_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_min_iso, t_min_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.min', 't.peri.min'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,9)), file_path_and_name=directory+'/t_min_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# t_infall vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_in_iso, t_in_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.text', 't.infall.text'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/t_infall_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_in_iso, t_in_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.text', 't.infall.text'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,10)), file_path_and_name=directory+'/t_infall_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# t_infall (any) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_in_any_iso, t_in_any_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.any', 't.infall.any'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/t_infall_any_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[t_in_any_iso, t_in_any_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.any', 't.infall.any'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,11.5)), file_path_and_name=directory+'/t_infall_any_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# v_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[vz0_iso, vz0_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Isolated', 'Paired'], binsize=0.5, file_path_and_name=directory+'/vtot_z0_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[vz0_iso, vz0_lg], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5),(0,250)), file_path_and_name=directory+'/vtot_z0_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

# L_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[L_iso/1e4, L_lg/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['Isolated', 'Paired'], binsize=0.5, file_path_and_name=directory+'/Ltot_vs_Mhalo_peak_iso_vs_lg_all.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_iso, Mhalo_peak_lg], y=[L_iso/1e4, L_lg/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['Isolated', 'Paired'], binsize=0.5, limits=((8,11.5), (0,3.5)), file_path_and_name=directory+'/Ltot_vs_Mhalo_peak_iso_vs_lg_all_zoom.pdf')

"""
    Energy plots
"""
potential_tot_iso = summary.potential(data_potentials_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all', norm='kinetic')
potential_tot_lg = summary.potential(data_potentials_lg, mask_selection_lg, oversample=True, hosts='lg_no_RR', sim_type='baryon_all', norm='kinetic')
ke_z0_tot_iso = summary.kinetic_energy(data_total_iso, mask_selection_iso, ke_type='z0', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
ke_z0_tot_lg = summary.kinetic_energy(data_total_lg, mask_selection_lg, ke_type='z0', oversample=True, hosts='lg_no_RR', sim_type='baryon_all')
#
Mhalo_peak_tot_iso = summary.mhalo(data_total_iso, mask_selection_iso, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot_lg = summary.mhalo(data_total_lg, mask_selection_lg, selection='peak', oversample=True, hosts='lg_no_RR', sim_type='baryon_all')
dz0_tot_iso = summary.d_z0(data_total_iso, mask_selection_iso, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
dz0_tot_lg = summary.d_z0(data_total_lg, mask_selection_lg, oversample=True, hosts='lg_no_RR', sim_type='baryon_all')

# E_tot vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_iso, Mhalo_peak_tot_lg], y=[(ke_z0_tot_iso+potential_tot_iso)/1e4, (ke_z0_tot_lg+potential_tot_lg)/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['E.tot', 'E.tot'], labels=['Isolated', 'Paired'], binsize=0.5, file_path_and_name=directory+'/Etot_vs_Mhalo_peak_baryon_all_iso_vs_lg.pdf')

# E_tot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot_iso, dz0_tot_lg], y=[(ke_z0_tot_iso+potential_tot_iso)/1e4, (ke_z0_tot_lg+potential_tot_lg)/1e4], xtype=['d.z0', 'd.z0'], ytype=['E.tot', 'E.tot'], labels=['Isolated', 'Paired'], binsize=50, file_path_and_name=directory+'/Etot_vs_dz0_baryon_all_iso_vs_lg.pdf')
summary_plot.median_plot_mult(x=[dz0_tot_iso, dz0_tot_lg], y=[(ke_z0_tot_iso+potential_tot_iso)/1e4, (ke_z0_tot_lg+potential_tot_lg)/1e4], xtype=['d.z0', 'd.z0'], ytype=['E.tot', 'E.tot'], labels=['Isolated', 'Paired'], binsize=50, limits=((0,400),(-5,2)), file_path_and_name=directory+'/Etot_vs_dz0_baryon_all_iso_vs_lg_zoom.pdf')



"""
    DMO comparison
"""

data_total = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='all_baryon')
data_total_dmo = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='dmo')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='iso_no_z')
masks_infall_dmo = summary.data_mask(data_total_dmo, peri_sim=False, peri_model=False, hosts='iso_no_z')

### Generate all of the data for the plots below
# Hydro
mask_selection = masks_infall
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='iso_no_z', sim_type='baryon_all')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_min_tot = summary.tperi_min(data_total, mask_selection, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_in_any_tot = summary.first_infall_any(data_total, mask_selection, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
vtan_tot = summary.velocities(data_total, mask_selection, selection='tan', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
vrad_tot = summary.velocities(data_total, mask_selection, selection='rad', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
vz0_tot = summary.v_z0(data_total, mask_selection, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
#
# DMO
mask_selection = masks_infall_dmo
N_sim_tot_dmo = summary.nperi(data_total_dmo, mask_selection, oversample=True, selection='sim', hosts='iso_no_z', sim_type='dmo')
d_sim_tot_dmo = summary.dperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso_no_z', sim_type='dmo')
d_min_tot_dmo = summary.dperi_min(data_total_dmo, mask_selection, oversample=True, hosts='iso_no_z', sim_type='dmo')
dz0_tot_dmo = summary.d_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso_no_z', sim_type='dmo')
t_sim_tot_dmo = summary.tperi_recent(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso_no_z', sim_type='dmo')
t_min_tot_dmo = summary.tperi_min(data_total_dmo, mask_selection, oversample=True, hosts='iso_no_z', sim_type='dmo')
t_in_tot_dmo = summary.first_infall(data_total_dmo, mask_selection, oversample=True, hosts='iso_no_z', sim_type='dmo')
t_in_any_tot_dmo = summary.first_infall_any(data_total_dmo, mask_selection, oversample=True, hosts='iso_no_z', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, mask_selection, selection='peak', oversample=True, hosts='iso_no_z', sim_type='dmo')
vtan_tot_dmo = summary.velocities(data_total_dmo, mask_selection, selection='tan', oversample=True, hosts='iso_no_z', sim_type='dmo')
vrad_tot_dmo = summary.velocities(data_total_dmo, mask_selection, selection='rad', oversample=True, hosts='iso_no_z', sim_type='dmo')
vz0_tot_dmo = summary.v_z0(data_total_dmo, mask_selection, oversample=True, hosts='iso_no_z', sim_type='dmo')
L_tot_dmo = summary.L_z0(data_total_dmo, mask_selection, selection='sim', oversample=True, hosts='iso_no_z', sim_type='dmo')


### Median plots
# N_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['N.peri.text', 'N.peri.text'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/N_sim_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[N_sim_tot, N_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['N.peri.text', 'N.peri.text'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,9)), file_path_and_name=directory+'/N_sim_vs_dz0_compare_dmo_zoom.pdf')

# d_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/d_sim_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,200)), file_path_and_name=directory+'/d_sim_vs_dz0_compare_dmo_zoom.pdf')

# d_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.peri.min', 'd.peri.min'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/d_min_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[d_min_tot, d_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['d.peri.min', 'd.peri.min'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,200)), file_path_and_name=directory+'/d_min_vs_dz0_compare_dmo_zoom.pdf')

# t_sim vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_sim_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,6)), file_path_and_name=directory+'/t_sim_vs_dz0_compare_dmo_zoom.pdf')

# t_min vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.peri.min', 't.peri.min'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_min_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_min_tot, t_min_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.peri.min', 't.peri.min'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,10)), file_path_and_name=directory+'/t_min_vs_dz0_compare_dmo_zoom.pdf')

# Infall time vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.text', 't.infall.text'], labels=['Baryon', 'DMO'], binsize=50, limits=(None,(0,13.8)), file_path_and_name=directory+'/t_infall_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_tot, t_in_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.text', 't.infall.text'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_infall_vs_dz0_compare_dmo_zoom.pdf')

# Infall time (any) vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=50, limits=(None,(0,13.8)), file_path_and_name=directory+'/t_infall_any_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[t_in_any_tot, t_in_any_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['t.infall.any', 't.infall.any'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,13.8)), file_path_and_name=directory+'/t_infall_any_vs_dz0_compare_dmo_zoom.pdf')

# vtan vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/vtan_z0_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vtan_tot, vtan_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tan', 'v.tan'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,300)), file_path_and_name=directory+'/vtan_z0_vs_dz0_compare_dmo_zoom.pdf')

# vtot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/vtot_z0_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['d.z0', 'd.z0'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,350)), file_path_and_name=directory+'/vtot_z0_vs_dz0_compare_dmo_zoom.pdf')

# Ltot vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=50, file_path_and_name=directory+'/Ltot_vs_dz0_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_tot_dmo], y=[L_tot/1e4, L_tot_dmo/1e4], xtype=['d.z0', 'd.z0'], ytype=['L.tot', 'L.tot'], labels=['Baryon', 'DMO'], binsize=50, limits=((0,400),(0,4)), file_path_and_name=directory+'/Ltot_vs_dz0_compare_dmo_zoom.pdf')

# N_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[N_sim_tot_dmo, N_sim_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.peri.text', 'N.peri.text'], labels=['DMO', 'Baryon'], binsize=0.5, legend_on=False, file_path_and_name=directory+'/N_sim_vs_Mhalo_peak_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[N_sim_tot_dmo, N_sim_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['N.peri.text', 'N.peri.text'], labels=['DMO', 'Baryon'], binsize=0.5, legend_on=False, limits=((8,11.5),(0,8)), file_path_and_name=directory+'/N_sim_vs_Mhalo_peak_compare_dmo_zoom.pdf')

# d_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/d_sim_vs_Mhalo_peak_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[d_sim_tot, d_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.recent', 'd.peri.recent'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5),(0,300)), file_path_and_name=directory+'/d_sim_vs_Mhalo_peak_compare_dmo_zoom.pdf')

# d_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[d_min_tot_dmo, d_min_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.min', 'd.peri.min'], labels=['DMO', 'Baryon'], binsize=0.5, file_path_and_name=directory+'/d_min_vs_Mhalo_peak_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[d_min_tot_dmo, d_min_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['d.peri.min', 'd.peri.min'], labels=['DMO', 'Baryon'], binsize=0.5, limits=((8,11.5),(0,175)), file_path_and_name=directory+'/d_min_vs_Mhalo_peak_compare_dmo_zoom.pdf')

# t_sim vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/t_sim_vs_Mhalo_peak_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[t_sim_tot, t_sim_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.recent', 't.peri.recent'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5),(0,5)), file_path_and_name=directory+'/t_sim_vs_Mhalo_peak_compare_dmo_zoom.pdf')

# t_min vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[t_min_tot_dmo, t_min_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.min', 't.peri.min'], labels=['DMO', 'Baryon'], legend_on=False, binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/t_min_vs_Mhalo_peak_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[t_min_tot_dmo, t_min_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.peri.min', 't.peri.min'], labels=['DMO', 'Baryon'], legend_on=False, binsize=0.5, limits=((8,11.5),(0,9)), file_path_and_name=directory+'/t_min_vs_Mhalo_peak_compare_dmo_zoom.pdf')

# t_infall vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[t_in_tot_dmo, t_in_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.text', 't.infall.text'], labels=['DMO', 'Baryon'], binsize=0.5, limits=((8,11.5),(0,13.8)), legend_on=False, file_path_and_name=directory+'/t_infall_vs_Mhalo_peak_compare_dmo.pdf')

# t_infall (any) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[t_in_any_tot_dmo, t_in_any_tot], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['t.infall.any', 't.infall.any'], labels=['DMO', 'Baryon'], binsize=0.5, limits=((8,11.5),(0,13.8)), legend_on=False, file_path_and_name=directory+'/t_infall_any_vs_Mhalo_peak_compare_dmo.pdf')

# v_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=0.5, file_path_and_name=directory+'/vtot_z0_vs_Mhalo_peak_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot, Mhalo_peak_tot_dmo], y=[vz0_tot, vz0_tot_dmo], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['v.tot', 'v.tot'], labels=['Baryon', 'DMO'], binsize=0.5, limits=((8,11.5),(0,250)), file_path_and_name=directory+'/vtot_z0_vs_Mhalo_peak_compare_dmo_zoom.pdf')

# L_tot(z = 0) vs Mhalo (peak)
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[L_tot_dmo/1e4, L_tot/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['DMO', 'Baryon'], binsize=0.5, legend_on=False, file_path_and_name=directory+'/Ltot_vs_Mhalo_peak_compare_dmo.pdf')
summary_plot.median_plot_mult(x=[Mhalo_peak_tot_dmo, Mhalo_peak_tot], y=[L_tot_dmo/1e4, L_tot/1e4], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['L.tot', 'L.tot'], labels=['DMO', 'Baryon'], binsize=0.5, legend_on=False, limits=((8,11.5),(0,3)), file_path_and_name=directory+'/Ltot_vs_Mhalo_peak_compare_dmo_zoom.pdf')






"""
Everything below here is only temporary, delete if we dont include in the paper
"""
# dperi fraction plots
mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0.05)
summary_plot.plot_hist(x=((d_min_tot-d_sim_tot)/d_sim_tot)[mask_delta_d], xtype='delta_d_frac', binsize=0.05, pdf=True, xlimits=(-1,0), file_path_and_name=directory+'/delta_d_frac_histogram.pdf')
summary_plot.median_plot(x=Mstar_z0_tot[mask_delta_d], y=((d_min_tot-d_sim_tot)/d_sim_tot)[mask_delta_d], xtype='M.star.z0', ytype='delta_d_frac', binedges=(4.5,9.5), binsize=0.5, limits=((4,9.5),(-1,0)), file_path_and_name=directory+'/delta_d_frac_vs_Mstar_z0.pdf')
summary_plot.median_plot(x=dz0_tot[mask_delta_d], y=((d_min_tot-d_sim_tot)/d_sim_tot)[mask_delta_d], xtype='d.z0', ytype='delta_d_frac', binsize=50, limits=(None,(-1,0)), file_path_and_name=directory+'/delta_d_frac_vs_dz0.pdf')
summary_plot.median_plot(x=d_sim_tot[mask_delta_d], y=((d_min_tot-d_sim_tot)/d_sim_tot)[mask_delta_d], xtype='d.peri.recent', ytype='delta_d_frac', binsize=50, limits=(None,(-1,0)), file_path_and_name=directory+'/delta_d_frac_vs_d_sim.pdf')
#
mask_delta_d_all = (np.abs((d_min_tot_all - d_sim_tot_all)/d_sim_tot_all) > 0.05)
summary_plot.median_plot(x=Mhalo_peak_tot_all[mask_delta_d_all], y=((d_min_tot_all-d_sim_tot_all)/d_sim_tot_all)[mask_delta_d_all], xtype='M.halo.peak', ytype='delta_d_frac', binedges=(8,11.5), binsize=0.5, limits=((8,11.5),(-1,0)), file_path_and_name=directory+'/delta_d_frac_vs_Mhalo_peak.pdf')




# delta tperi plots
mask_delta_t = (np.abs((t_min_tot - t_sim_tot)/t_sim_tot) > 0.05)
summary_plot.plot_hist(x=(t_min_tot-t_sim_tot)[mask_delta_t], xtype='delta_t', binsize=0.5, pdf=True, file_path_and_name=directory+'/delta_t_histogram.pdf')
summary_plot.median_plot(x=Mstar_z0_tot[mask_delta_t], y=(t_min_tot-t_sim_tot)[mask_delta_t], xtype='M.star.z0', ytype='delta_t', binedges=(4.5,9.5), binsize=0.5, limits=((4,9.5),None), file_path_and_name=directory+'/delta_t_vs_Mstar_z0.pdf')
summary_plot.median_plot(x=dz0_tot[mask_delta_t], y=(t_min_tot-t_sim_tot)[mask_delta_t], xtype='d.z0', ytype='delta_t', binsize=50, file_path_and_name=directory+'/delta_t_vs_dz0.pdf')
summary_plot.median_plot(x=t_sim_tot[mask_delta_t], y=(t_min_tot-t_sim_tot)[mask_delta_t], xtype='t.peri.recent', ytype='delta_t', binsize=0.5, limits=((0,6),(0,13.8)), file_path_and_name=directory+'/delta_t_vs_t_sim.pdf')
#
mask_delta_t_all = (np.abs((t_min_tot_all - t_sim_tot_all)/t_sim_tot_all) > 0.05)
summary_plot.median_plot(x=Mhalo_peak_tot_all[mask_delta_t_all], y=(t_min_tot_all-t_sim_tot_all)[mask_delta_t_all], xtype='M.halo.peak', ytype='delta_t', binedges=(8,11.5), binsize=0.5, limits=((8,11.5),(0,13.8)), file_path_and_name=directory+'/delta_t_vs_Mhalo_peak.pdf')




### Plot the energy histogram for the outlier population
potential_tot = summary.potential(data_potentials, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon')
#
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_energy', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon')
mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0.05)
#
summary_plot.plot_hist(x=((ke_z0_tot+potential_tot)/1e4)[mask_delta_d], xtype='E.tot', binsize=0.1, pdf=True, file_path_and_name=directory+'/Etot_outlier_histogram.pdf')
summary_plot.plot_hist_mult(x=[((ke_z0_tot+potential_tot)/1e4)[mask_delta_d], ((ke_z0_tot+potential_tot)/1e4)], xtype=['E.tot','E.tot'], labels=['Outliers','Total'], binsize=0.1, pdf=True, file_path_and_name=directory+'/Etot_outlier_comparison_histogram.pdf')
#
# Plot the energy vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot[mask_delta_d], y=((ke_z0_tot+potential_tot)/1e4)[mask_delta_d], xtype='M.star.z0', ytype='E.tot', binedges=(4.5,9.5), binsize=0.5, file_path_and_name=directory+'/Etot_outliers_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot[mask_delta_d]], y=[((ke_z0_tot+potential_tot)/1e4), ((ke_z0_tot+potential_tot)/1e4)[mask_delta_d]], xtype=['M.star.z0', 'M.star.z0'], ytype=['E.tot', 'E.tot'], labels=['Total', 'Outliers'], binsize=0.5, file_path_and_name=directory+'/Etot_outliers_vs_Mstar_z0_comparison.pdf')




#### Plot the angular momentum histogram for the outlier population
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0.05)
#
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
summary_plot.plot_hist(x=L_tot[mask_delta_d]/1e4, xtype='L.tot', binsize=0.1, pdf=True, file_path_and_name=directory+'/Ltot_outlier_histogram.pdf')
summary_plot.plot_hist_mult(x=[L_tot[mask_delta_d]/1e4, L_tot/1e4], xtype=['L.tot','L.tot'], labels=['Outliers','Total'], binsize=0.1, pdf=True, xlimits=(0,5), file_path_and_name=directory+'/Ltot_outlier_comparison_histogram.pdf')
#
# Plot the angular momentum vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot[mask_delta_d], y=L_tot[mask_delta_d]/1e4, xtype='M.star.z0', ytype='L.tot', binedges=(4.5,9.5), binsize=0.5, file_path_and_name=directory+'/Ltot_outliers_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_tot[mask_delta_d]], y=[L_tot/1e4, L_tot[mask_delta_d]/1e4], xtype=['M.star.z0', 'M.star.z0'], ytype=['L.tot', 'L.tot'], labels=['Total', 'Outliers'], binedges=(4.5,9.5), binsize=0.5, legend_on=False, limits=((4,9.5),(0,4)), file_path_and_name=directory+'/Ltot_outliers_vs_Mstar_z0_comparison.pdf')




#### Plot the total velocity histogram for the outlier population
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0.05)
#
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
#
summary_plot.plot_hist(x=vz0_tot[mask_delta_d], xtype='v.tot', binsize=10, pdf=True, file_path_and_name=directory+'/vtot_outlier_histogram.pdf')
summary_plot.plot_hist_mult(x=[vz0_tot[mask_delta_d], vz0_tot], xtype=['v.tot','v.tot'], labels=['Outliers','Total'], binsize=10, pdf=True, file_path_and_name=directory+'/vtot_outlier_comparison_histogram.pdf')
#
# Plot the total velocity vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot[mask_delta_d], y=vz0_tot[mask_delta_d], xtype='M.star.z0', ytype='v.tot', binedges=(4.5,9.5), binsize=0.5, file_path_and_name=directory+'/vtot_outliers_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot[mask_delta_d], Mstar_z0_tot], y=[vz0_tot[mask_delta_d], vz0_tot], xtype=['M.star.z0', 'M.star.z0'], ytype=['v.tot', 'v.tot'], labels=['Outliers', 'Total'], binedges=(4.5,9.5), binsize=0.5, limits=((4,9.5),(0,400)), file_path_and_name=directory+'/vtot_outliers_vs_Mstar_z0_comparison.pdf')




# Look at histograms between satellites on their first infall, sats with one peri, and sats with more than one peri
nperi_0_mask = summary.data_mask_nperi(data_total, nperi=0, hosts='all_no_z')
nperi_1_mask = summary.data_mask_nperi(data_total, nperi=1, hosts='all_no_z')
nperi_2_mask = summary.data_mask_nperi(data_total, nperi=2, hosts='all_no_z')

# Energy plot
potential_tot_0 = summary.potential(data_potentials, nperi_0_mask, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot_0 = summary.kinetic_energy(data_total, nperi_0_mask, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
Mstar_z0_tot_0 = summary.mstar(data_total, nperi_0_mask, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
potential_tot_1 = summary.potential(data_potentials, nperi_1_mask, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot_1 = summary.kinetic_energy(data_total, nperi_1_mask, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
Mstar_z0_tot_1 = summary.mstar(data_total, nperi_1_mask, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
potential_tot_2 = summary.potential(data_potentials, nperi_2_mask, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot_2 = summary.kinetic_energy(data_total, nperi_2_mask, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
Mstar_z0_tot_2 = summary.mstar(data_total, nperi_2_mask, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[(potential_tot_0+ke_z0_tot_0)/1e4, (potential_tot_1+ke_z0_tot_1)/1e4, (potential_tot_2+ke_z0_tot_2)/1e4], xtype=['E.tot','E.tot','E.tot'], labels=['$N_{\\rm peri}$ = 0','$N_{\\rm peri}$ = 1','$N_{\\rm peri}$ > 1'], med_location=[1.3, 1.25, 1.225], binsize=0.2, legend_on=False, pdf=True, file_path_and_name=directory+'/Etot_comparison_nperis_histogram.pdf')
#
summary_plot.median_plot_mult(x=[Mstar_z0_tot_0, Mstar_z0_tot_1, Mstar_z0_tot_2], y=[(potential_tot_0+ke_z0_tot_0)/1e4, (potential_tot_1+ke_z0_tot_1)/1e4, (potential_tot_2+ke_z0_tot_2)/1e4], xtype=['M.star.z0', 'M.star.z0', 'M.star.z0'], ytype=['E.tot', 'E.tot', 'E.tot'], labels=['N$_{\\rm peri}$ = 0','N$_{\\rm peri}$ = 1','N$_{\\rm peri}$ > 1'], binsize=0.5, file_path_and_name=directory+'/Etot_vs_Mstar_z0_nperi_pops.pdf')



# Angular momentum plot
L_tot_0 = summary.L_z0(data_total, nperi_0_mask, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot_1 = summary.L_z0(data_total, nperi_1_mask, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot_2 = summary.L_z0(data_total, nperi_2_mask, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot_0 = summary.mstar(data_total, nperi_0_mask, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot_1 = summary.mstar(data_total, nperi_1_mask, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot_2 = summary.mstar(data_total, nperi_2_mask, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[L_tot_0/1e4, L_tot_1/1e4, L_tot_2/1e4], xtype=['L.tot','L.tot','L.tot'], labels=['$N_{\\rm peri}$ = 0','$N_{\\rm peri}$ = 1','$N_{\\rm peri}$ > 1'], binsize=0.2, pdf=True, xlimits=(0,7), med_location=[1.025, 0.975, 1.05,], legend_on=False, file_path_and_name=directory+'/Ltot_comparison_nperis_histogram.pdf')
#
summary_plot.median_plot_mult(x=[Mstar_z0_tot_0, Mstar_z0_tot_1, Mstar_z0_tot_2], y=[L_tot_0/1e4, L_tot_1/1e4, L_tot_2/1e4], xtype=['M.star.z0', 'M.star.z0', 'M.star.z0'], ytype=['L.tot', 'L.tot', 'L.tot'], labels=['N$_{\\rm peri}$ = 0','N$_{\\rm peri}$ = 1','N$_{\\rm peri}$ > 1'], binsize=0.5, file_path_and_name=directory+'/Ltot_vs_Mstar_z0_nperi_pops.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot_0, Mstar_z0_tot_1, Mstar_z0_tot_2], y=[L_tot_0/1e4, L_tot_1/1e4, L_tot_2/1e4], xtype=['M.star.z0', 'M.star.z0', 'M.star.z0'], ytype=['L.tot', 'L.tot', 'L.tot'], labels=['N$_{\\rm peri}$ = 0','N$_{\\rm peri}$ = 1','N$_{\\rm peri}$ > 1'], binsize=0.5, limits=((4,9.5),(0,5)), file_path_and_name=directory+'/Ltot_vs_Mstar_z0_nperi_pops_zoom.pdf')

# vtot plot
vz0_tot_0 = summary.v_z0(data_total, nperi_0_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot_1 = summary.v_z0(data_total, nperi_1_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot_2 = summary.v_z0(data_total, nperi_2_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[vz0_tot_0, vz0_tot_1, vz0_tot_2], xtype=['v.tot','v.tot','v.tot'], labels=['N$_{\\rm peri}$ = 0','N$_{\\rm peri}$ = 1','N$_{\\rm peri}$ > 1'], binsize=10, pdf=True, file_path_and_name=directory+'/vtot_comparison_nperis_histogram.pdf')

# d(z = 0) plot
dz0_tot_0 = summary.d_z0(data_total, nperi_0_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_tot_1 = summary.d_z0(data_total, nperi_1_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_tot_2 = summary.d_z0(data_total, nperi_2_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot_0 = summary.mstar(data_total, nperi_0_mask, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot_1 = summary.mstar(data_total, nperi_1_mask, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot_2 = summary.mstar(data_total, nperi_2_mask, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[dz0_tot_0, dz0_tot_1, dz0_tot_2], xtype=['d.z0','d.z0','d.z0'], labels=['$N_{\\rm peri}$ = 0','$N_{\\rm peri}$ = 1','$N_{\\rm peri}$ > 1'], binsize=20, pdf=True, xlimits=(0,500), med_location=[0.00925, 0.00875, 0.009], legend_on=True, file_path_and_name=directory+'/dz0_comparison_nperis_histogram.pdf')
#
summary_plot.median_plot_mult(x=[Mstar_z0_tot_0, Mstar_z0_tot_1, Mstar_z0_tot_2], y=[dz0_tot_0, dz0_tot_1, dz0_tot_2], xtype=['M.star.z0', 'M.star.z0', 'M.star.z0'], ytype=['d.z0','d.z0','d.z0'], labels=['N$_{\\rm peri}$ = 0','N$_{\\rm peri}$ = 1','N$_{\\rm peri}$ > 1'], binsize=0.5, file_path_and_name=directory+'/dz0_vs_Mstar_z0_nperi_pops.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot_0, Mstar_z0_tot_1, Mstar_z0_tot_2], y=[dz0_tot_0, dz0_tot_1, dz0_tot_2], xtype=['M.star.z0', 'M.star.z0', 'M.star.z0'], ytype=['d.z0','d.z0','d.z0'], labels=['N$_{\\rm peri}$ = 0','N$_{\\rm peri}$ = 1','N$_{\\rm peri}$ > 1'], binsize=0.5, limits=((4,9.5),(0,400)), file_path_and_name=directory+'/dz0_vs_Mstar_z0_nperi_pops_zoom.pdf')


"""
    Apocenter Plots
"""
summary = summary_io.SummaryDataSort()
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_z', sim_type='baryon')
data_potentials = summary.data_read_potential(directory=sim_data.home_dir, hosts='all_energy', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_z')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_z')
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs'


### Generate all of the data for the plots below
# Fix for the outlier in the Mstar-Mhalo relation
masks_infall['m12f'][57] = False
masks_infall_apo['m12f'][57] = False
#
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_apo_only = summary.d_z0(data_total, masks_infall_apo, oversample=True, hosts='all_no_z', sim_type='baryon')
#
d_apo_tot = summary.dapo_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_apo_only = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_apo_only = summary.first_infall(data_total, masks_infall_apo, oversample=True, hosts='all_no_z', sim_type='baryon')
#
t_in_any_tot = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_any_apo_only = summary.first_infall_any(data_total, masks_infall_apo, oversample=True, hosts='all_no_z', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_apo_only = summary.mstar(data_total, masks_infall_apo, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')


summary_plot.median_plot(x=dz0_tot, y=d_apo_tot, xtype='d.z0', ytype='d.apo.text', binsize=50, limits=((0,400),(0,800)), file_path_and_name=directory+'/d_apo_vs_dz0.pdf')
summary_plot.median_plot(x=Mstar_z0_tot, y=d_apo_tot, xtype='M.star.z0', ytype='d.apo.text', binsize=0.5, binedges=(4.5,9.5), limits=((4,9.5),(0,600)), file_path_and_name=directory+'/d_apo_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[t_in_tot, t_in_any_tot], y=[d_apo_tot, d_apo_tot], xtype=['t.infall.text', 't.infall.text'], ytype=['d.apo.text', 'd.apo.text'], labels=['MW-mass halo', 'Any halo'], binsize=1, limits=((0,14),(0,800)), file_path_and_name=directory+'/d_apo_vs_t_infall.pdf')


# Apocenter vs d(z = 0)
summary_plot.median_plot_mult(x=[dz0_tot, dz0_apo_only], y=[d_apo_tot, d_apo_only], xtype=['d.z0','d.z0'], ytype=['d.apo.text', 'd.apo.text'], labels=['Including no apocenters', 'Real Apocenters'], binsize=50, limits=((0,400),(0,800)), file_path_and_name=directory+'/d_apo_vs_dz0_comp.pdf')
summary_plot.median_plot_mult(x=[dz0_tot, dz0_apo_only], y=[d_apo_tot, d_apo_only], xtype=['d.z0','d.z0'], ytype=['d.apo.text', 'd.apo.text'], labels=['Including no apocenters', 'Real Apocenters'], binsize=50, limits=((0,400),(0,500)), file_path_and_name=directory+'/d_apo_vs_dz0_comp_zoom.pdf')

# Apocenter vs Mstar(z = 0)
summary_plot.median_plot_mult(x=[Mstar_z0_tot, Mstar_z0_apo_only], y=[d_apo_tot, d_apo_only], xtype=['M.star.z0', 'M.star.z0'], ytype=['d.apo.text', 'd.apo.text'], labels=['Including no apocenters', 'Real Apocenters'], binsize=0.5, binedges=(4.5,9.5), limits=((4,9.5),(0,600)), file_path_and_name=directory+'/d_apo_vs_Mstar_z0_comp.pdf')

# Apocenter vs t_infall (MW-mass host)
summary_plot.median_plot_mult(x=[t_in_tot, t_in_apo_only], y=[d_apo_tot, d_apo_only], xtype=['t.infall.text', 't.infall.text'], ytype=['d.apo.text', 'd.apo.text'], labels=['Including no apocenters', 'Real Apocenters'], binsize=1, limits=((0,14),(0,800)), file_path_and_name=directory+'/d_apo_vs_t_infall_MW_comp.pdf')

# Apocenter vs t_infall (any host)
summary_plot.median_plot_mult(x=[t_in_any_tot, t_in_any_apo_only], y=[d_apo_tot, d_apo_only], xtype=['t.infall.text', 't.infall.text'], ytype=['d.apo.text', 'd.apo.text'], labels=['Including no apocenters', 'Real Apocenters'], binsize=1, limits=((0,14),(0,800)), file_path_and_name=directory+'/d_apo_vs_t_infall_any_comp.pdf')



# Checking correlations between pericenter properties with infall time and merger times
t90star = np.array([1.27, 0.98, 1.19, 1.45, 1.49, 0.83, 0.92, 2.13, 1.57, 1.08, 1.52, 1.38, 1.89])
t10halo = np.array([11.57, 10.57, 11.65, 11.79, 11.29, 0, 11.65, 12.54, 12.49, 10.85, 12.23, 11.57, 12.14])
t15in = np.array([10.99, 10.24, 12.04, 11.65, 9.4, 0, 11.29, 12.82, 12.66, 9.86, 12.54, 12.66, 12.04])

rec = np.zeros(len(summary.host_names['all_no_z']))
minn = np.zeros(len(summary.host_names['all_no_z']))
for i, name in enumerate(summary.host_names['all_no_z']):
    rec[i] = np.median(t_sim_tot[np.where(name == names['host'])[0]])
    minn[i] = np.median(t_min_tot[np.where(name == names['host'])[0]])

####

mergers_b = 13.78-np.array([10.96, 3.94, 3.33, 3.21, 2.77])
mergers_c = 13.78-np.array([5.47, 5.34, 5.09, 4.81, 3.68])
mergers_f = 13.78-np.array([12.38, 7.51, 7.27, 4.94, 3.14, 2.73, 2.27, 2.23])
mergers_i = 13.78-np.array([8.19, 8.11, 5.60, 3.80, 3.30, 2.49, 2.27])
mergers_m = 13.78-np.array([4.96, 4.94, 4.24, 3.19, 3.16, 2.93, 2.82])
mergers_r = 13.78-np.array([13.19, 8.03, 3.80])
mergers_w = 13.78-np.array([8.55, 8.34, 5.63, 3.51, 2.66])
mergers_juliet = 13.78-np.array([13.78, 13.42])
mergers_thelma = 13.78-np.array([9.10, 8.06, 6.07, 5.76, 5.19, 5.16, 4.99, 3.87, 3.25, 2.69])
mergers_louise = 13.78-np.array([4.73, 3.19, 3.05])
mergers_romulus = 13.78-np.array([7.93, 6.20, 5.27, 2.49, 2.47, 2.38])
mergers_remus = 13.78-np.array([5.45, 5.01, 3.25, 2.21])
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=False, hosts='all_no_z', sim_type='baryon')
t_in_any_tot = summary.first_infall_any(data_total, masks_infall, oversample=False, hosts='all_no_z', sim_type='baryon')


#
name = 'Remus'
mergers = mergers_remus
#
rec = t_sim_tot[np.where(name == names['host'])[0]]
minn = t_min_tot[np.where(name == names['host'])[0]]

f, ax1 = plt.subplots(1, 1, figsize=(10,8))
binss, asdf = summary_plot.binning_scheme(rec, binsize=0.5, xtype='t.infall.text')
ax1.hist(x=rec, bins=binss, color=summary_plot.colors[3], alpha=0.5, edgecolor='k', label='MW-mass halo')
binss, asdf = summary_plot.binning_scheme(minn, binsize=0.5, xtype='t.infall.text')
ax1.hist(x=minn, bins=binss, color=summary_plot.colors[2], alpha=0.5, edgecolor='k', label='Any halo')
#ax1.set_xlim(0,11)
#ax1.set_ylim(-0.005,1.01)
ax1.set_xlabel('Infall lookback time [Gyr]', fontsize=32)
ax1.set_ylabel('N', fontsize=24)
ax1.set_title(name, fontsize=24)
ax1.legend(prop={'size': 20}, loc='best')
for i in range(0, len(mergers)):
    plt.vlines(x=mergers[i], ymin=0, ymax=13, alpha=0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/'+name+'_mergers_and_tinfall.pdf')
plt.close()




# Checking the ell difference at infall vs ell at z = 0
# plotting vs infall time and Mstar
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
L_in = summary.L_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
summary_plot.median_plot(x=t_in_tot, y=(L_tot-L_in)/1e4, xtype='t.infall.text', ytype='L.diff', binsize=1, limits=((0,13),None), file_path_and_name=directory+'/delta_ell_infall.pdf')
summary_plot.median_plot(x=t_in_tot, y=(L_tot-L_in)/1e4, xtype='t.infall.text', ytype='L.diff', binsize=1, limits=((0,13),(-2,2)), file_path_and_name=directory+'/delta_ell_infall_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0_tot, y=(L_tot-L_in)/1e4, xtype='M.star.z0', ytype='L.diff', binsize=0.5, limits=((4.5,9.5),None), file_path_and_name=directory+'/delta_ell_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0_tot, y=(L_tot-L_in)/1e4, xtype='M.star.z0', ytype='L.diff', binsize=0.5, limits=((4.5,9.5),(-1.5,1.5)), file_path_and_name=directory+'/delta_ell_mstar_zoom.pdf')
#
summary_plot.median_plot(x=t_in_tot, y=(L_tot-L_in)/L_in, xtype='t.infall.text', ytype='L.frac', binsize=1, limits=((0,13),None), file_path_and_name=directory+'/delta_ell_frac_infall.pdf')
summary_plot.median_plot(x=t_in_tot, y=(L_tot-L_in)/L_in, xtype='t.infall.text', ytype='L.frac', binsize=1, limits=((0,13),(-1,4)), file_path_and_name=directory+'/delta_ell_frac_infall_zoom.pdf')
#
summary_plot.median_plot(x=Mstar_z0_tot, y=(L_tot-L_in)/L_in, xtype='M.star.z0', ytype='L.frac', binsize=0.5, limits=((4.5,9.5),None), file_path_and_name=directory+'/delta_ell_frac_mstar.pdf')
summary_plot.median_plot(x=Mstar_z0_tot, y=(L_tot-L_in)/L_in, xtype='M.star.z0', ytype='L.frac', binsize=0.5, limits=((4.5,9.5),(-1.5,1.5)), file_path_and_name=directory+'/delta_ell_frac_mstar_zoom.pdf')



# Checking pericenter properties of high-mass vs low-mass hosts
d_sim_hi = summary.dperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='high-mass', sim_type='baryon')
d_min_hi = summary.dperi_min(data_total, masks_infall, oversample=True, hosts='high-mass', sim_type='baryon')
dz0_hi = summary.d_z0(data_total, masks_infall, oversample=True, hosts='high-mass', sim_type='baryon')
t_sim_hi = summary.tperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='high-mass', sim_type='baryon')
t_min_hi = summary.tperi_min(data_total, masks_infall, oversample=True, hosts='high-mass', sim_type='baryon')
t_in_hi = summary.first_infall(data_total, masks_infall, oversample=True, hosts='high-mass', sim_type='baryon')
t_in_any_hi = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='high-mass', sim_type='baryon')
Mstar_z0_hi = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='high-mass', sim_type='baryon')
#
d_sim_lo = summary.dperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='low-mass', sim_type='baryon')
d_min_lo = summary.dperi_min(data_total, masks_infall, oversample=True, hosts='low-mass', sim_type='baryon')
dz0_lo = summary.d_z0(data_total, masks_infall, oversample=True, hosts='low-mass', sim_type='baryon')
t_sim_lo = summary.tperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='low-mass', sim_type='baryon')
t_min_lo = summary.tperi_min(data_total, masks_infall, oversample=True, hosts='low-mass', sim_type='baryon')
t_in_lo = summary.first_infall(data_total, masks_infall, oversample=True, hosts='low-mass', sim_type='baryon')
t_in_any_lo = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='low-mass', sim_type='baryon')
Mstar_z0_lo = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='low-mass', sim_type='baryon')

# Recent pericenter
summary_plot.median_plot_mult(x=[Mstar_z0_hi,Mstar_z0_lo], y=[d_sim_hi,d_sim_lo], binsize=0.5, limits=((4.5,9.5),None), xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Recent Pericenter', file_path_and_name=directory+'/drec_vs_mstar_hinlo.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_hi,Mstar_z0_lo], y=[d_sim_hi,d_sim_lo], binsize=0.5, limits=((4.5,9.5),(0,250)), xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Recent Pericenter', file_path_and_name=directory+'/drec_vs_mstar_hinlo_zoom.pdf')
summary_plot.median_plot_mult(x=[t_in_hi,t_in_lo], y=[d_sim_hi,d_sim_lo], binsize=1, limits=((0,14),None), xtype=['t.infall.text','t.infall.text'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Recent Pericenter', file_path_and_name=directory+'/drec_vs_tinmw_hinlo.pdf')
summary_plot.median_plot_mult(x=[t_in_hi,t_in_lo], y=[d_sim_hi,d_sim_lo], binsize=1, limits=((0,14),(0,300)), xtype=['t.infall.text','t.infall.text'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Recent Pericenter', file_path_and_name=directory+'/drec_vs_tinmw_hinlo_zoom.pdf')
#
# Minimum pericenter
summary_plot.median_plot_mult(x=[Mstar_z0_hi,Mstar_z0_lo], y=[d_min_hi,d_min_lo], binsize=0.5, limits=((4.5,9.5),None), xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Minimum Pericenter', file_path_and_name=directory+'/dmin_vs_mstar_hinlo.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_hi,Mstar_z0_lo], y=[d_min_hi,d_min_lo], binsize=0.5, limits=((4.5,9.5),(0,250)), xtype=['M.star.z0','M.star.z0'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Minimum Pericenter', file_path_and_name=directory+'/dmin_vs_mstar_hinlo_zoom.pdf')
summary_plot.median_plot_mult(x=[t_in_hi,t_in_lo], y=[d_min_hi,d_min_lo], binsize=1, limits=((0,14),None), xtype=['t.infall.text','t.infall.text'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Minimum Pericenter', file_path_and_name=directory+'/dmin_vs_tinmw_hinlo.pdf')
summary_plot.median_plot_mult(x=[t_in_hi,t_in_lo], y=[d_min_hi,d_min_lo], binsize=1, limits=((0,14),(0,300)), xtype=['t.infall.text','t.infall.text'], ytype=['d.peri.text','d.peri.text'], labels=['High-mass', 'Low-mass'], title='Minimum Pericenter', file_path_and_name=directory+'/dmin_vs_tinmw_hinlo_zoom.pdf')
#
mask_hi = (np.abs((d_min_hi - d_sim_hi)/d_sim_hi) > 0)
mask_lo = (np.abs((d_min_lo - d_sim_lo)/d_sim_lo) > 0)
summary_plot.plot_hist_mult(x=[((d_min_hi-d_sim_hi)/d_sim_hi)[mask_hi], ((d_min_lo-d_sim_lo)/d_sim_lo)[mask_lo]], xtype=['delta_d_frac','delta_d_frac'], labels=['High-mass', 'Low-mass'], binsize=0.05, binedges=(-1,0), pdf=True, xlimits=(-1,0), file_path_and_name=directory+'/delta_d_frac_hinlo.pdf')
