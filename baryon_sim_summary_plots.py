#!/usr/bin/python3

"""
    ==========================================
    = Baryonic Simulation Halo Summary Plots =
    ==========================================

    Create histograms of orbit properties for just the halos in the
    baryonic simulations. Also, plot different properties vs:
        - Stellar mass
        - Peak halo mass
        - Present-day distance

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
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all')
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1/baryon'


### Generate all of the data for the plots below
mask_selection = masks_infall
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='all', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all', sim_type='baryon')
t_sim_tot = summary.tperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='all', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all', sim_type='baryon')
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='all', sim_type='baryon')
#
mask_low = np.where(Mhalo_peak_tot < 3.16e9)[0]
mask_high = np.where(Mhalo_peak_tot > 3.16e9)[0]

### Histograms
# Number of pericenters
summary_plot.plot_hist(x=N_sim_tot, xtype='N.sim', binsize=1, pdf=True, file_path_and_name=directory+'/histogram/N_sim_pdf.pdf')
summary_plot.plot_hist(x=N_sim_tot, xtype='N.sim', binsize=1, pdf=True, xlimits=(-0.5, 13), file_path_and_name=directory+'/histogram/N_sim_pdf_zoom.pdf')

# Recent pericenter distance
summary_plot.plot_hist(x=d_sim_tot, xtype='d.sim', binsize=20, pdf=True, file_path_and_name=directory+'/histogram/d_sim_pdf.pdf')
summary_plot.plot_hist(x=d_sim_tot, xtype='d.sim', binsize=20, pdf=True, xlimits=(0,400), file_path_and_name=directory+'/histogram/d_sim_pdf_zoom.pdf')

# Present-day distance
summary_plot.plot_hist(x=dz0_tot, xtype='d.z0', binsize=20, pdf=True, file_path_and_name=directory+'/histogram/dz0_pdf.pdf')
summary_plot.plot_hist(x=dz0_tot, xtype='d.z0', binsize=20, pdf=True, xlimits=(0,400), file_path_and_name=directory+'/histogram/dz0_pdf_zoom.pdf')

# Recent pericenter times
summary_plot.plot_hist(x=t_sim_tot, xtype='t.sim', binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_sim_pdf.pdf')
summary_plot.plot_hist(x=t_sim_tot, xtype='t.sim', binsize=0.5, pdf=True, xlimits=(0,10), file_path_and_name=directory+'/histogram/t_sim_pdf_zoom.pdf')

# Infall times
summary_plot.plot_hist(x=t_in_tot, xtype='t.infall', binsize=0.5, pdf=True, file_path_and_name=directory+'/histogram/t_inall_pdf.pdf')
summary_plot.plot_hist(x=t_in_tot, xtype='t.infall', binsize=0.5, pdf=True, xlimits=(0,13.8), file_path_and_name=directory+'/histogram/t_inall_pdf_zoom.pdf')

# Mstar(z = 0)
summary_plot.plot_hist(x=Mstar_z0_tot, xtype='M.star.z0', binsize=0.1, pdf=True, file_path_and_name=directory+'/histogram/Mstar_z0_pdf.pdf')
summary_plot.plot_hist(x=Mstar_z0_tot, xtype='M.star.z0', binsize=0.1, pdf=True, xlimits=(4,10), file_path_and_name=directory+'/histogram/Mstar_z0_pdf_zoom.pdf')

# Mhalo (peak)
summary_plot.plot_hist(x=Mhalo_peak_tot, xtype='M.halo.peak', binsize=0.1, pdf=True, file_path_and_name=directory+'/histogram/Mhalo_peak_pdf.pdf')
summary_plot.plot_hist(x=Mhalo_peak_tot, xtype='M.halo.peak', binsize=0.1, pdf=True, xlimits=(7,12), file_path_and_name=directory+'/histogram/Mhalo_peak_pdf_zoom.pdf')


### Median plots
# Mstar (z = 0) vs Mhalo (peak)
summary_plot.median_plot(x=Mhalo_peak_tot, y=Mstar_z0_tot, xtype='M.halo.peak', ytype='M.star.z0', binsize=0.5, file_path_and_name=directory+'/median/mstar_z0_vs_mhalo_peak.pdf')

# N_sim vs Mstar(z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=N_sim_tot, xtype='M.star.z0', ytype='N.sim', binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot[mask_low], Mstar_z0_tot[mask_high]], y=[N_sim_tot[mask_low], N_sim_tot[mask_high]], xtype=['M.star.z0', 'M.star.z0'], ytype=['N.sim', 'N.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=0.5, file_path_and_name=directory+'/median/N_sim_vs_Mstar_z0_massbins.pdf')

# d_sim vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=d_sim_tot, xtype='M.star.z0', ytype='d.sim', binsize=0.5, file_path_and_name=directory+'/median/d_sim_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot[mask_low], Mstar_z0_tot[mask_high]], y=[d_sim_tot[mask_low], d_sim_tot[mask_high]], xtype=['M.star.z0', 'M.star.z0'], ytype=['d.sim', 'd.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=0.5, file_path_and_name=directory+'/median/d_sim_vs_Mstar_z0_massbins.pdf')

# d(z = 0) vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=dz0_tot, xtype='M.star.z0', ytype='d.z0', binsize=0.5, file_path_and_name=directory+'/median/dz0_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot[mask_low], Mstar_z0_tot[mask_high]], y=[dz0_tot[mask_low], dz0_tot[mask_high]], xtype=['M.star.z0', 'M.star.z0'], ytype=['d.z0', 'd.z0'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=0.5, file_path_and_name=directory+'/median/dz0_vs_Mstar_z0_massbins.pdf')

# t_sim vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=t_sim_tot, xtype='M.star.z0', ytype='t.sim', binsize=0.5, file_path_and_name=directory+'/median/t_sim_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot[mask_low], Mstar_z0_tot[mask_high]], y=[t_sim_tot[mask_low], t_sim_tot[mask_high]], xtype=['M.star.z0', 'M.star.z0'], ytype=['t.sim', 't.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=0.5, file_path_and_name=directory+'/median/t_sim_vs_Mstar_z0_massbins.pdf')

# t_infall vs Mstar (z = 0)
summary_plot.median_plot(x=Mstar_z0_tot, y=t_in_tot, xtype='M.star.z0', ytype='t.infall', binsize=0.5, file_path_and_name=directory+'/median/t_infall_vs_Mstar_z0.pdf')
summary_plot.median_plot_mult(x=[Mstar_z0_tot[mask_low], Mstar_z0_tot[mask_high]], y=[t_in_tot[mask_low], t_in_tot[mask_high]], xtype=['M.star.z0', 'M.star.z0'], ytype=['t.infall', 't.infall'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=0.5, file_path_and_name=directory+'/median/t_infall_vs_Mstar_z0_massbins.pdf')

# N_sim vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=N_sim_tot, xtype='d.z0', ytype='N.sim', binsize=50, file_path_and_name=directory+'/median/N_sim_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=N_sim_tot, xtype='d.z0', ytype='N.sim', binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/N_sim_vs_dz0_zoom.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[N_sim_tot[mask_low], N_sim_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['N.sim', 'N.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, file_path_and_name=directory+'/median/N_sim_vs_dz0_massbins.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[N_sim_tot[mask_low], N_sim_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['N.sim', 'N.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, limits=((0,400), None), file_path_and_name=directory+'/median/N_sim_vs_dz0_massbins_zoom.pdf')

# d_sim vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=d_sim_tot, xtype='d.z0', ytype='d.sim', binsize=50, file_path_and_name=directory+'/median/d_sim_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=d_sim_tot, xtype='d.z0', ytype='d.sim', binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/d_sim_vs_dz0_zoom.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[d_sim_tot[mask_low], d_sim_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['d.sim', 'd.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, file_path_and_name=directory+'/median/d_sim_vs_dz0_massbins.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[d_sim_tot[mask_low], d_sim_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['d.sim', 'd.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, limits=((0,400), None), file_path_and_name=directory+'/median/d_sim_vs_dz0_massbins_zoom.pdf')

# t_sim vs d(z = 0)
summary_plot.median_plot(x=dz0_tot, y=t_sim_tot, xtype='d.z0', ytype='t.sim', binsize=50, file_path_and_name=directory+'/median/t_sim_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=t_sim_tot, xtype='d.z0', ytype='t.sim', binsize=50, limits=((0,400),None), file_path_and_name=directory+'/median/t_sim_vs_dz0_zoom.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[t_sim_tot[mask_low], t_sim_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['t.sim', 't.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, file_path_and_name=directory+'/median/t_sim_vs_dz0_massbins.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[t_sim_tot[mask_low], t_sim_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['t.sim', 't.sim'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, limits=((0,400), None), file_path_and_name=directory+'/median/t_sim_vs_dz0_massbins_zoom.pdf')



# Infall time vs d(z = 0) [ALL]
summary_plot.median_plot(x=dz0_tot, y=t_in_tot, xtype='d.z0', ytype='t.infall', binsize=50, file_path_and_name=directory+'/median/t_infall_vs_dz0.pdf')
summary_plot.median_plot(x=dz0_tot, y=t_in_tot, xtype='d.z0', ytype='t.infall', binsize=50, limits=((0,400), None), file_path_and_name=directory+'/median/t_infall_vs_dz0_zoom.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[t_in_tot[mask_low], t_in_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, file_path_and_name=directory+'/median/t_infall_vs_dz0_massbins.pdf')
summary_plot.median_plot_mult(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[t_in_tot[mask_low], t_in_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, limits=((0,400), None), file_path_and_name=directory+'/median/t_infall_vs_dz0_massbins_zoom.pdf')




######### TESTING THE NEW PLOTTING FUNCTION...
summary_plot.median_plot_mult_one_scatter(x=[dz0_tot[mask_low], dz0_tot[mask_high]], y=[t_in_tot[mask_low], t_in_tot[mask_high]], xtype=['d.z0', 'd.z0'], ytype=['t.infall', 't.infall'], labels=['log M$_{\\rm halo,peak}$ < 9.5', 'log M$_{\\rm halo,peak}$ > 9.5'], binsize=50, limits=((0,400), None), file_path_and_name=directory+'/test.pdf')





# Splitting into mass bins
t_in_mask = summary.mass_masking_property(data_total, masks_infall, prop='t.infall', mass_array=[1e8,3.16e9], mass_type='Mhalo.peak', oversample=True, hosts='all', sim_type='baryon')
#
dz0_mask = summary.mass_masking_property(data_total, masks_infall, prop='dz0', mass_array=[1e8,3.16e9], mass_type='Mhalo.peak', oversample=True, hosts='all', sim_type='baryon')
#
# Baryon plots
y = t_in_tot
x = dz0_tot
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
#
for i in range(0, len(bins)-1):
    mask = (x >= bins[i]) & (x <= bins[i+1])
    med[i] = np.nanmedian(y[mask])
    upper[i] = np.nanpercentile(y[mask], onesigp)
    lower[i] = np.nanpercentile(y[mask], onesigm)
#
for i in range(0, len(bins)-1):
    mask_low = (dz0_mask['low'] >= bins[i]) & (dz0_mask['low'] <= bins[i+1])
    med_low[i] = np.nanmedian(t_in_mask['low'][mask_low])
    mask_mid = (dz0_mask['mid'] >= bins[i]) & (dz0_mask['mid'] <= bins[i+1])
    med_mid[i] = np.nanmedian(t_in_mask['mid'][mask_mid])
#
f, ax = plt.subplots(figsize=(10, 8))
plt.plot(bins[:-1]+half_bin, med, color='k', alpha=0.5)
plt.fill_between(bins[:-1]+half_bin, upper, lower, color='k', alpha=0.3)
#
plt.plot(bins[:-1]+half_bin, med_low, color=summary_plot.colors[1], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm halo}$ = [8, 9.5]')
plt.plot(bins[:-1]+half_bin, med_mid, color=summary_plot.colors[2], marker='s', markersize=5, alpha=0.3, label='log M$_{\\rm halo}$ > 9.5')
#
plt.xlim(0, 400)
plt.xlabel('d(z = 0) [kpc]', fontsize=28)
plt.ylabel('t$_{\\rm infall,lb}$ [Gyr]', fontsize=28)
plt.legend(prop={'size': 16})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(directory+'/median/infall_vs_d_z0_mass_bins_zoom.pdf')
plt.close()




#### This was all done to check whether or not we should use 1e10 as the cutoff for the most massive subhalos or 3.16e9; we choose the latter
# 38 halos with Mhalo,peak > 1e10
# 154 halos with Mhalo,peak > 3.16e9
### Plot the mass function of galaxies with Mpeak > 1e10
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=False, hosts='all', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=False, hosts='all', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=False, hosts='all', sim_type='baryon')
summary_plot.plot_hist(x=Mhalo_peak_tot[np.where(Mhalo_peak_tot > 3.16e9)[0]], xtype='M.halo.peak', binsize=0.1, pdf=False, file_path_and_name=directory+'/histogram/Mhalo_peak_3.16e9.pdf')
summary_plot.plot_hist(x=Mstar_z0_tot[np.where(Mhalo_peak_tot > 3.16e9)[0]], xtype='M.star.z0', binsize=0.1, pdf=False, file_path_and_name=directory+'/histogram/Mstar_z0_3.16e9.pdf')
summary_plot.plot_hist(x=dz0_tot[np.where(Mhalo_peak_tot > 1e10)[0]], xtype='d.z0', binsize=20, pdf=False, file_path_and_name=directory+'/histogram/Mhalo_peak_3.16e9_dz0.pdf')
