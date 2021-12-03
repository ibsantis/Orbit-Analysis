#!/usr/bin/python3

"""
    =========================
    = Paper I Outlier Plots =
    =========================
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
#mask_selection['m12f'][57] = False
#
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')

# Create the masks
# Total sample, green with scatter
# Doesn't need a mask for anything

# Outlier sample, purple with scatter
mask_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) > 0.05)

# N > 1 sample, red with scatter
mask_mult = (N_sim_tot > 1)

# Total - Outlier sample, dashed green no scatter
mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0.05)

# N > 1 - Outlier sample, dashed red no scatter
mask_mult = (N_sim_tot > 1)
mask_mult_no_out = (np.abs((d_min_tot[mask_mult]-d_sim_tot[mask_mult])/d_sim_tot[mask_mult]) < 0.05)

"""
# For the energy plots
N_sim_tot = summary.nperi(data_total, mask_selection, oversample=True, selection='sim', hosts='all_energy', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, mask_selection, selection='sim', oversample=True, hosts='all_energy', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon')
potential_tot = summary.potential(data_potentials, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
# Outlier sample, purple with scatter
mask_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) > 0.05)
# N > 1 sample, red with scatter
mask_mult = (N_sim_tot > 1)
# Total - Outlier sample, dashed green no scatter
mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0.05)
# N > 1 - Outlier sample, dashed red no scatter
mask_mult = (N_sim_tot > 1)
mask_mult_no_out = (np.abs((d_min_tot[mask_mult]-d_sim_tot[mask_mult])/d_sim_tot[mask_mult]) < 0.05)
"""

f, ax = plt.subplots(figsize=(11, 8))
colorss = ['#000080', '#006400', '#8b0000']
binedges = None
binsize = 0.5
binedges = (4.5, 9.5)
limits = ((4,9.5),(-4,0.5))
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult], Mstar_z0_tot[mask_no_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
#y = [d_min_tot, d_min_tot[mask_out], d_min_tot[mask_mult], d_min_tot[mask_no_out], d_min_tot[mask_mult][mask_mult_no_out]]
#y = [L_tot/1e4, (L_tot[mask_out])/1e4, (L_tot[mask_mult])/1e4, (L_tot[mask_no_out])/1e4, (L_tot[mask_mult][mask_mult_no_out])/1e4]
#y = [t_in_tot, t_in_tot[mask_out], t_in_tot[mask_mult], t_in_tot[mask_no_out], t_in_tot[mask_mult][mask_mult_no_out]]
y = [(potential_tot+ke_z0_tot)/1e4, ((potential_tot+ke_z0_tot)/1e4)[mask_out], ((potential_tot+ke_z0_tot)/1e4)[mask_mult], ((potential_tot+ke_z0_tot)/1e4)[mask_no_out], ((potential_tot+ke_z0_tot)/1e4)[mask_mult][mask_mult_no_out]]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0',]
#ytype = ['d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text']
#ytype = ['L.tot', 'L.tot', 'L.tot', 'L.tot', 'L.tot']
#ytype = ['t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text']
ytype = ['E.tot','E.tot','E.tot','E.tot','E.tot']
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
    if 'M.' in xtype[j]:
        x[j] = np.log10(x[j])
    if 'M.' in ytype[j]:
        y[j] = np.log10(y[j])
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
    scatter = np.zeros(len(bins)-1)
    #
    for i in range(0, len(bins)-1):
        mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
        med[i] = np.nanmedian(y[j][mask])
        scatter[i] = np.nanstd(y[j][mask])
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
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
plt.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3) # Total sample
#plt.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#plt.fill_between(10**(binss[2][:-1]+half_bins[2]), uppers[2], lowers[2], color=colorss[2], alpha=0.3) # N > 1 sample
#plt.fill_between(binss[2][:-1]+half_bins[2], highests[2], lowests[2], color=colorss[2], alpha=0.15)
plt.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3) # Outlier sample
#plt.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
plt.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Total Sample')
#plt.plot(10**(binss[3][:-1]+half_bins[3]), medians[3], color=colorss[1], markersize=10, linestyle='--', alpha=0.5, label='Total Sample, no outliers')
#
plt.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Outliers')
#
#plt.plot(10**(binss[2][:-1]+half_bins[2]), medians[2], color=colorss[2], markersize=10, alpha=0.5, label='N > 1')
plt.plot(10**(binss[4][:-1]+half_bins[4]), medians[4], color=colorss[2], markersize=10, linestyle='--', alpha=0.5, label='N > 1, no outliers')
#
plt.xlim(10**(limits[0][0]), 10**(limits[0][1]))
#plt.xlim(limits[0])
plt.ylim(limits[1])
plt.xscale('log')
if 't.' in ytype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_2_label = 'redshift'
    axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
    axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
    ax2 = ax.twinx()
    ax2.set_xscale('log')
    ax2.set_yscale('linear')
    ax2.set_yticks(axis_2_tick_locations)
    ax2.set_yticklabels(axis_2_tick_labels, fontsize=28)
    ax2.set_ylim(limits[1])
    ax2.set_ylabel(axis_2_label, labelpad=9)
    ax2.tick_params(pad=3)
ax.set_xlabel('$M_{\\rm star} \ [M_{\\odot}]$', fontsize=28)
#ax.set_ylabel('Min pericenter distance [kpc]', fontsize=28)
#ax.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
#ax.set_ylabel('Infall lookback time [Gyr]', fontsize=28)
ax.set_ylabel('$E$ [$10^4$ km$^2$ s$^{-1}$]', fontsize=28)
#ax.legend(prop={'size': 20}, loc='best')
ax.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.show()
