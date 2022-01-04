
## Import all of the tools for analysis
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
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
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
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
    Pericenter distance and number.
"""
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = (4.5, 9.5)
binsize = 0.5
limits = ((4,9.5),(0,200))
#
x = [Mstar_z0_tot, Mstar_z0_tot, Mstar_z0_tot]
y = [d_sim_tot, d_min_tot, N_sim_tot]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0']
ytype = ['d.peri.text','d.peri.text','N.peri.text']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent') # Recent, M < 1e7
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum') #, label='MW/M31-mass host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Recent M > 1e7
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
ax1.set_ylim(limits[1])
#
x = [Mstar_z0_tot]
y = [N_sim_tot]
#
xtype = ['M.star.z0']
ytype = ['N.peri.text']
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
        med[i] = np.nanmean(y[j][mask])
        scatter[i] = np.nanstd(y[j][mask])
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
        upper[i] = med[i]+scatter[i]
        lower[i] = med[i]-scatter[i]
        if (upper[i] > highest[i]):
            upper[i] = highest[i]
        if (lower[i] < lowest[i]):
            lower[i] = lowest[i]
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
ax2.set_ylim(0, 5.5)
#
ax2.set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Pericenter distance [kpc]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('Pericenter Number', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
ax1.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_2.pdf')


################################################################################


"""
    Pericenter time and Infall time
"""
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = (4.5, 9.5)
binsize = 0.5
limits_1 = ((4,9.5),(0,13.8))
limits_2 = ((4,9.5),(0,9))
#
x = [Mstar_z0_tot, Mstar_z0_tot]
y = [t_in_tot, t_in_any_tot]
#
xtype = ['M.star.z0', 'M.star.z0']
ytype = ['t.infall.text','t.infall.text']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='MW/M31-mass halo') # Recent, M < 1e7
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Any halo') # Recent, M < 1e7
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
if 't.' in ytype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_3_label = 'redshift'
    axis_3_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_3_tick_values = [float(v) for v in axis_3_tick_labels]
    axis_3_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_3_tick_values)
    ax3 = ax1.twinx()
    ax3.set_xscale('log')
    ax3.set_yscale('linear')
    ax3.set_yticks(axis_3_tick_locations)
    ax3.set_yticklabels(axis_3_tick_labels, fontsize=28)
    ax3.set_ylim(limits_1[1])
    ax3.set_ylabel(axis_3_label, labelpad=9)
    ax3.tick_params(pad=3)
#
x = [Mstar_z0_tot, Mstar_z0_tot]
y = [t_sim_tot, t_min_tot]
#
xtype = ['M.star.z0', 'M.star.z0']
ytype = ['t.peri.text','t.peri.text']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent') # Recent, M < 1e7
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum') #, label='MW/M31-mass host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Recent M > 1e7
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
if 't.' in ytype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_4_label = 'redshift'
    axis_4_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_4_tick_values = [float(v) for v in axis_4_tick_labels]
    axis_4_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_4_tick_values)
    ax4 = ax2.twinx()
    ax4.set_xscale('log')
    ax4.set_yscale('linear')
    ax4.set_yticks(axis_4_tick_locations)
    ax4.set_yticklabels(axis_4_tick_labels, fontsize=24)
    ax4.set_ylim(limits_2[1])
    ax4.set_ylabel(axis_4_label, labelpad=9)
    ax4.tick_params(pad=3)
#
ax2.set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Infall Lookback Time [Gyr]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.075,0.5)
ax2.set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.075,0.5)
ax2.legend(prop={'size': 20}, loc='best')
ax1.legend(prop={'size': 20}, loc='best')
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax1.xaxis.set_major_locator(LogLocator(base=10))
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax1.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_3.pdf')


################################################################################


"""
    Plotting Figure 5 in Paper I: Dynamics
"""
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5,16))
colorss = ['#000080', '#006400']
binedges = (4.5, 9.5)
binsize = 0.5
limits_1 = ((4,9.5),(0,300))
limits_2 = ((4,9.5),(0,4.5))
limits_3 = ((4,9.5),(-4,0.5))
#
x = [Mstar_z0_tot]
y = [vz0_tot]
#
xtype = ['M.star.z0']
ytype = ['v.tot']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
x = [Mstar_z0_tot]
y = [L_tot/1e4]
#
xtype = ['M.star.z0']
ytype = ['L.tot']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
potential_tot = summary.potential(data_potentials, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
x = [Mstar_z0_tot]
y = [(potential_tot+ke_z0_tot)/1e4]
#
xtype = ['M.star.z0']
ytype = ['E.tot']
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
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax3.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax3.set_xscale('log')
ax3.set_xlim(10**(limits_3[0][0]), 10**(limits_3[0][1]))
ax3.set_ylim(limits_3[1])
#
ax3.set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
ax3.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=24)
ax3.get_yaxis().set_label_coords(-0.09,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax3.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax3.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
ax3.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_5.pdf')



"""
    Plotting Figure 6 in Paper I: Pericenter distance and number.
"""
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges=None
binsize = 50
limits = ((0,400),(0,250))
#
x = [dz0_tot, dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high], dz0_tot[mass_low], dz0_tot[mass_high]]
y = [d_sim_tot, d_min_tot, d_sim_tot[mass_low], d_sim_tot[mass_high], d_min_tot[mass_low], d_min_tot[mass_high]]
#
xtype = ['d.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0']
ytype = ['d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text']
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax1.fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, alpha=0.5, label='Recent')
ax1.plot(binss[3][:-1]+half_bins[3], medians[3], color=colorss[1], linestyle='--', markersize=10, alpha=0.5)
ax1.plot(binss[4][:-1]+half_bins[4], medians[4], color=colorss[0], markersize=10, alpha=0.5, label='Minimum')
ax1.plot(binss[5][:-1]+half_bins[5], medians[5], color=colorss[0], linestyle='--', markersize=10, alpha=0.5)
#
ax1.set_xscale('linear')
ax1.set_xlim(limits[0][0], limits[0][1])
ax1.set_ylim(limits[1])
#
x = [dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high]]
y = [N_sim_tot, N_sim_tot[mass_low], N_sim_tot[mass_high]]
#
xtype = ['d.z0','d.z0','d.z0']
ytype = ['N.peri.text','N.peri.text','N.peri.text']
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
        med[i] = np.nanmean(y[j][mask])
        scatter[i] = np.nanstd(y[j][mask])
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
        upper[i] = med[i]+scatter[i]
        lower[i] = med[i]-scatter[i]
        if (upper[i] > highest[i]):
            upper[i] = highest[i]
        if (lower[i] < lowest[i]):
            lower[i] = lowest[i]
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
ax2.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5, label='$M_{\\rm star} < 10^7 M_{\\odot}$') # Recent, M < 1e7
ax2.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], linestyle='--', markersize=10, alpha=0.5, label='$M_{\\rm star} > 10^7 M_{\\odot}$') # Recent, M < 1e7
#
ax2.set_xscale('linear')
ax2.set_xlim(limits[0][0], limits[0][1])
ax2.set_ylim(0, 6.5)
#
ax2.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax1.set_ylabel('Pericenter distance [kpc]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('Pericenter Number', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
ax1.legend(prop={'size': 20}, loc='best')
ax2.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_6.pdf')



################################################################################

"""
    Plotting Figure 3 in Paper I: Pericenter time and Infall time
"""
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges=None
binsize = 50
limits_1 = ((0, 400),(0,13.5))
limits_2 = ((0, 400),(0,10))
#
x = [dz0_tot, dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high], dz0_tot[mass_low], dz0_tot[mass_high]]
y = [t_in_tot, t_in_any_tot, t_in_tot[mass_low], t_in_tot[mass_high], t_in_any_tot[mass_low], t_in_any_tot[mass_high]]
#
xtype = ['d.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0']
ytype = ['t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text']
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax1.fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, alpha=0.5, label='MW/M31-mass halo') # Recent, M < 1e7
ax1.plot(binss[3][:-1]+half_bins[3], medians[3], color=colorss[1], linestyle='--', markersize=10, alpha=0.5) # Recent, M < 1e7
ax1.plot(binss[4][:-1]+half_bins[4], medians[4], color=colorss[0], markersize=10, alpha=0.5, label='Any halo') # Recent, M < 1e7
ax1.plot(binss[5][:-1]+half_bins[5], medians[5], color=colorss[0], linestyle='--', markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax1.set_xscale('linear')
ax1.set_xlim(limits_1[0][0], limits_1[0][1])
ax1.set_ylim(limits_1[1])
#
if 't.' in ytype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_3_label = 'redshift'
    axis_3_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_3_tick_values = [float(v) for v in axis_3_tick_labels]
    axis_3_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_3_tick_values)
    ax3 = ax1.twinx()
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.set_yticks(axis_3_tick_locations)
    ax3.set_yticklabels(axis_3_tick_labels, fontsize=24)
    ax3.set_ylim(limits_1[1])
    ax3.set_ylabel(axis_3_label, labelpad=9)
    ax3.tick_params(pad=3)
#
x = [dz0_tot, dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high], dz0_tot[mass_low], dz0_tot[mass_high]]
y = [t_sim_tot, t_min_tot, t_sim_tot[mass_low], t_sim_tot[mass_high], t_min_tot[mass_low], t_min_tot[mass_high]]
#
xtype = ['d.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0']
ytype = ['t.peri.text','t.peri.text','t.peri.text','t.peri.text','t.peri.text','t.peri.text']
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
ax2.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax2.fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax2.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, alpha=0.5, label='Recent')
ax2.plot(binss[3][:-1]+half_bins[3], medians[3], color=colorss[1], linestyle='--', markersize=10, alpha=0.5)
ax2.plot(binss[4][:-1]+half_bins[4], medians[4], color=colorss[0], markersize=10, alpha=0.5, label='Minimum')
ax2.plot(binss[5][:-1]+half_bins[5], medians[5], color=colorss[0], linestyle='--', markersize=10, alpha=0.5)
#
ax2.set_xscale('linear')
ax2.set_xlim(limits_2[0][0], limits_2[0][1])
ax2.set_ylim(limits_2[1])
#
if 't.' in ytype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_4_label = 'redshift'
    axis_4_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_4_tick_values = [float(v) for v in axis_4_tick_labels]
    axis_4_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_4_tick_values)
    ax4 = ax2.twinx()
    ax4.set_xscale('linear')
    ax4.set_yscale('linear')
    ax4.set_yticks(axis_4_tick_locations)
    ax4.set_yticklabels(axis_4_tick_labels, fontsize=24)
    ax4.set_ylim(limits_2[1])
    ax4.set_ylabel(axis_4_label, labelpad=9)
    ax4.tick_params(pad=3)
#
ax2.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax1.set_ylabel('Infall Lookback Time [Gyr]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.075,0.5)
ax2.set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.075,0.5)
ax1.legend(prop={'size': 20}, loc='best')
ax2.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_7.pdf')



################################################################################

"""
    Plotting Figure 8 in Paper I: Dynamics
"""
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vtan_tot = summary.velocities(data_total, mask_selection, selection='tan', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
#
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
limits_1 = ((0,400),(0,350))
limits_2 = ((0,400),(0,325))
#
x = [dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high]]
y = [vz0_tot, vz0_tot[mass_low], vz0_tot[mass_high]]
#
xtype = ['d.z0', 'd.z0', 'd.z0']
ytype = ['v.tot', 'v.tot', 'v.tot']
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5, label='$M_{\\rm star} < 10^7 M_{\\odot}$')
ax1.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], linestyle='--', markersize=10, alpha=0.5, label='$M_{\\rm star} > 10^7 M_{\\odot}$')
#
ax1.set_xscale('linear')
ax1.set_xlim(limits_1[0])
ax1.set_ylim(limits_1[1])
#
############
#
x = [dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high]]
y = [vtan_tot, vtan_tot[mass_low], vtan_tot[mass_high]]
#
xtype = ['d.z0', 'd.z0', 'd.z0']
ytype = ['v.tan', 'v.tan', 'v.tan']
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
ax2.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5)
ax2.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], linestyle='--', markersize=10, alpha=0.5)
#
ax2.set_xscale('linear')
ax2.set_xlim(limits_2[0])
ax2.set_ylim(limits_2[1])
#
ax2.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax1.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
ax1.legend(prop={'size': 24}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_8a.pdf')
#
############
#
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
limits_1 = ((0,400),(0,4))
limits_2 = ((0,400),(-6,0.5))
#
x = [dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high]]
y = [L_tot/1e4, (L_tot[mass_low])/1e4, (L_tot[mass_high])/1e4]
#
xtype = ['d.z0', 'd.z0', 'd.z0']
ytype = ['L.tot', 'L.tot', 'L.tot']
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5)
ax1.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], linestyle='--', markersize=10, alpha=0.5)
#
ax1.set_xscale('linear')
ax1.set_xlim(limits_1[0])
ax1.set_ylim(limits_1[1])
#
############
#
potential_tot = summary.potential(data_potentials, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon')
#
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
x = [dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high]]
y = [(potential_tot+ke_z0_tot)/1e4, (potential_tot[mass_low]+ke_z0_tot[mass_low])/1e4, (potential_tot[mass_high]+ke_z0_tot[mass_high])/1e4]
#
xtype = ['d.z0', 'd.z0', 'd.z0']
ytype = ['E.tot', 'E.tot', 'E.tot']
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
ax2.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5)
ax2.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], linestyle='--', markersize=10, alpha=0.5)
#
ax2.set_xscale('linear')
ax2.set_xlim(limits_2[0])
ax2.set_ylim(limits_2[1])
#
ax2.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax1.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.075,0.5)
ax2.set_ylabel('$E$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.075,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_8b.pdf')



################################################################################

"""
    Plotting Figure 12 in Paper I: Outliers
"""

# Outlier sample, purple with scatter
mask_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) > 0.05)

# N > 1 sample, red with scatter
mask_mult = (N_sim_tot > 1)

# Total - Outlier sample, dashed green no scatter
mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0.05)

# N > 1 - Outlier sample, dashed red no scatter
mask_mult = (N_sim_tot > 1)
mask_mult_no_out = (np.abs((d_min_tot[mask_mult]-d_sim_tot[mask_mult])/d_sim_tot[mask_mult]) < 0.05)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400', '#8b0000']
binedges = None
binsize = 0.5
binedges = (4.5, 9.5)
limits_1 = ((4,9.5),(0,12))
limits_2 = ((4,9.5),(0,190))
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult], Mstar_z0_tot[mask_no_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [t_in_tot, t_in_tot[mask_out], t_in_tot[mask_mult], t_in_tot[mask_no_out], t_in_tot[mask_mult][mask_mult_no_out]]
#y = [d_min_tot, d_min_tot[mask_out], d_min_tot[mask_mult], d_min_tot[mask_no_out], d_min_tot[mask_mult][mask_mult_no_out]]
#y = [L_tot/1e4, (L_tot[mask_out])/1e4, (L_tot[mask_mult])/1e4, (L_tot[mask_no_out])/1e4, (L_tot[mask_mult][mask_mult_no_out])/1e4]
#y = [(potential_tot+ke_z0_tot)/1e4, ((potential_tot+ke_z0_tot)/1e4)[mask_out], ((potential_tot+ke_z0_tot)/1e4)[mask_mult], ((potential_tot+ke_z0_tot)/1e4)[mask_no_out], ((potential_tot+ke_z0_tot)/1e4)[mask_mult][mask_mult_no_out]]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0',]
ytype = ['t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text']
#ytype = ['d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text']
#ytype = ['L.tot', 'L.tot', 'L.tot', 'L.tot', 'L.tot']
#ytype = ['E.tot','E.tot','E.tot','E.tot','E.tot']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3) # Total sample
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3) # Outlier sample
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Total Sample')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Outliers')
ax1.plot(10**(binss[4][:-1]+half_bins[4]), medians[4], color=colorss[2], markersize=10, alpha=0.5, label='N > 1, no outliers')
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
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
    ax3 = ax1.twinx()
    ax3.set_xscale('log')
    ax3.set_yscale('linear')
    ax3.set_yticks(axis_2_tick_locations)
    ax3.set_yticklabels(axis_2_tick_labels, fontsize=24)
    ax3.set_ylim(limits_1[1])
    ax3.set_ylabel(axis_2_label, labelpad=9)
    ax3.tick_params(pad=3)
#ax1.set_xlabel('$M_{\\rm star} \ [M_{\\odot}]$', fontsize=28)
#ax1.set_ylabel('Infall lookback time [Gyr]', fontsize=28)
#ax.set_ylabel('Min pericenter distance [kpc]', fontsize=28)
#ax.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
#ax.set_ylabel('$E$ [$10^4$ km$^2$ s$^{-1}$]', fontsize=28)
#ax1.legend(prop={'size': 20}, loc='best')
#ax1.tick_params(axis='both', which='major', labelsize=28)
#
############
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult], Mstar_z0_tot[mask_no_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [d_min_tot, d_min_tot[mask_out], d_min_tot[mask_mult], d_min_tot[mask_no_out], d_min_tot[mask_mult][mask_mult_no_out]]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0',]
ytype = ['d.peri.text','d.peri.text','d.peri.text','d.peri.text','d.peri.text']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3) # Total sample
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3) # Outlier sample
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Total Sample')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Outliers')
ax2.plot(10**(binss[4][:-1]+half_bins[4]), medians[4], color=colorss[2], markersize=10, alpha=0.5, label='N > 1, no outliers')
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
ax1.set_ylabel('Infall lookback time [Gyr]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.1,0.5)
ax2.set_ylabel('Min pericenter distance [kpc]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
ax2.set_xlabel('$M_{\\rm star} \ [M_{\\odot}]$', fontsize=28)
ax1.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_12a.pdf')
#
#
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400', '#8b0000']
binedges = None
binsize = 0.5
binedges = (4.5, 9.5)
limits_1 = ((4,9.5),(0,3.5))
limits_2 = ((4,9.5),(-4,0.5))
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult], Mstar_z0_tot[mask_no_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [L_tot/1e4, (L_tot[mask_out])/1e4, (L_tot[mask_mult])/1e4, (L_tot[mask_no_out])/1e4, (L_tot[mask_mult][mask_mult_no_out])/1e4]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0',]
ytype = ['L.tot', 'L.tot', 'L.tot', 'L.tot', 'L.tot']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3) # Total sample
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3) # Outlier sample
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Total Sample')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Outliers')
ax1.plot(10**(binss[4][:-1]+half_bins[4]), medians[4], color=colorss[2], markersize=10, alpha=0.5, label='N > 1, no outliers')
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
############
#
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
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult], Mstar_z0_tot[mask_no_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [(potential_tot+ke_z0_tot)/1e4, ((potential_tot+ke_z0_tot)/1e4)[mask_out], ((potential_tot+ke_z0_tot)/1e4)[mask_mult], ((potential_tot+ke_z0_tot)/1e4)[mask_no_out], ((potential_tot+ke_z0_tot)/1e4)[mask_mult][mask_mult_no_out]]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0', 'M.star.z0',]
ytype = ['E.tot','E.tot','E.tot','E.tot','E.tot']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3) # Total sample
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3) # Outlier sample
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Total Sample')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Outliers')
ax2.plot(10**(binss[4][:-1]+half_bins[4]), medians[4], color=colorss[2], markersize=10, alpha=0.5, label='N > 1, no outliers')
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
ax1.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.1,0.5)
ax2.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
ax2.set_xlabel('$M_{\\rm star} \ [M_{\\odot}]$', fontsize=28)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_12b.pdf')



################################################################################

"""
    Plotting Figure A1 in Paper I: Pericenter distance and number versus Mhalo
"""
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8,11.5),(0,225))
limits_2 = ((8,11.5),(0,4.5))
#
x = [Mhalo_peak_tot_all, Mhalo_peak_tot_all]
y = [d_sim_tot_all, d_min_tot_all]
#
xtype = ['M.halo.peak', 'M.halo.peak']
ytype = ['d.peri.text','d.peri.text']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent') # Recent, M < 1e7
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum') #, label='MW/M31-mass host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Recent M > 1e7
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
x = [Mhalo_peak_tot_all]
y = [N_sim_tot_all]
#
xtype = ['M.halo.peak']
ytype = ['N.peri.text']
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
        med[i] = np.nanmean(y[j][mask])
        scatter[i] = np.nanstd(y[j][mask])
        highest[i] = np.nanpercentile(y[j][mask], twosigp)
        lowest[i] = np.nanpercentile(y[j][mask], twosigm)
        upper[i] = med[i]+scatter[i]
        lower[i] = med[i]-scatter[i]
        if (upper[i] > highest[i]):
            upper[i] = highest[i]
        if (lower[i] < lowest[i]):
            lower[i] = lowest[i]
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
ax2.set_xlabel('$M_{\\rm halo,peak} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Pericenter distance [kpc]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('Pericenter Number', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
ax1.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e8, 1e9, 1e10, 1e11])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_a1.pdf')



"""
    Plotting Figure A2 in Paper I: Pericenter time and Infall time
"""
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8,11.5),(0,11.5))
limits_2 = ((8,11.5),(0,9))
#
x = [Mhalo_peak_tot_all, Mhalo_peak_tot_all]
y = [t_in_tot_all, t_in_any_tot_all]
#
xtype = ['M.halo.peak', 'M.halo.peak']
ytype = ['t.infall.text','t.infall.text']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='MW/M31-mass halo') # Recent, M < 1e7
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Any halo') # Recent, M < 1e7
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
if 't.' in ytype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_3_label = 'redshift'
    axis_3_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_3_tick_values = [float(v) for v in axis_3_tick_labels]
    axis_3_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_3_tick_values)
    ax3 = ax1.twinx()
    ax3.set_xscale('log')
    ax3.set_yscale('linear')
    ax3.set_yticks(axis_3_tick_locations)
    ax3.set_yticklabels(axis_3_tick_labels, fontsize=24)
    ax3.set_ylim(limits_1[1])
    ax3.set_ylabel(axis_3_label, labelpad=9)
    ax3.tick_params(pad=3)
#
x = [Mhalo_peak_tot_all, Mhalo_peak_tot_all]
y = [t_sim_tot_all, t_min_tot_all]
#
xtype = ['M.halo.peak', 'M.halo.peak']
ytype = ['t.peri.text','t.peri.text']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent') # Recent, M < 1e7
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum') #, label='MW/M31-mass host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Recent M > 1e7
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
if 't.' in ytype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_4_label = 'redshift'
    axis_4_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_4_tick_values = [float(v) for v in axis_4_tick_labels]
    axis_4_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_4_tick_values)
    ax4 = ax2.twinx()
    ax4.set_xscale('log')
    ax4.set_yscale('linear')
    ax4.set_yticks(axis_4_tick_locations)
    ax4.set_yticklabels(axis_4_tick_labels, fontsize=24)
    ax4.set_ylim(limits_2[1])
    ax4.set_ylabel(axis_4_label, labelpad=9)
    ax4.tick_params(pad=3)
#
ax2.set_xlabel('$M_{\\rm halo,peak} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Infall Lookback Time [Gyr]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.075,0.5)
ax2.set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.075,0.5)
ax1.legend(prop={'size': 20}, loc='best')
ax2.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e8, 1e9, 1e10, 1e11])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_a2.pdf')




"""
    Plotting Figure A3 in Paper I: Dynamics
"""
Mhalo_peak_tot_all = summary.mhalo(data_total_all, mask_selection, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon_all')
vz0_tot_all = summary.v_z0(data_total_all, mask_selection, oversample=True, hosts='all_no_z', sim_type='baryon_all')
L_tot_all = summary.L_z0(data_total_all, mask_selection, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon_all')
#
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5,16))
colorss = ['#000080', '#006400']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8,11.5),(0,300))
limits_2 = ((8,11.5),(0,4.5))
limits_3 = ((8,11.5),(-5,0.5))
#
x = [Mhalo_peak_tot_all]
y = [vz0_tot_all]
#
xtype = ['M.halo.peak']
ytype = ['v.tot']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
x = [Mhalo_peak_tot_all]
y = [L_tot_all/1e4]
#
xtype = ['M.halo.peak']
ytype = ['L.tot']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
potential_tot_all = summary.potential(data_potentials_all, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon_all', norm='kinetic')
ke_z0_tot_all = summary.kinetic_energy(data_total_all, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon_all')
#
Mhalo_peak_tot_all = summary.mhalo(data_total_all, mask_selection, selection='peak', oversample=True, hosts='all_energy', sim_type='baryon_all')
#
x = [Mhalo_peak_tot_all]
y = [(potential_tot_all+ke_z0_tot_all)/1e4]
#
xtype = ['M.halo.peak']
ytype = ['E.tot']
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
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax3.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5) # Recent, M < 1e7
#
ax3.set_xscale('log')
ax3.set_xlim(10**(limits_3[0][0]), 10**(limits_3[0][1]))
ax3.set_ylim(limits_3[1])
#
ax3.set_xlabel('$M_{\\rm halo,peak} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
ax3.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=24)
ax3.get_yaxis().set_label_coords(-0.09,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax3.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax3.set_xticks([1e8, 1e9, 1e10, 1e11])
ax3.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final/figure_a3.pdf')














"""
    Making Nperi population plots
"""

nperi_0_mask = summary.data_mask_nperi(data_total, nperi=0, hosts='all_no_z')
nperi_1_mask = summary.data_mask_nperi(data_total, nperi=1, hosts='all_no_z')
nperi_2_mask = summary.data_mask_nperi(data_total, nperi=2, hosts='all_no_z')

# d(z = 0) plot
dz0_tot_0 = summary.d_z0(data_total, nperi_0_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_tot_1 = summary.d_z0(data_total, nperi_1_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_tot_2 = summary.d_z0(data_total, nperi_2_mask, oversample=True, hosts='all_no_z', sim_type='baryon')
#
colorss = ['#2f4f4f', '#006400', '#8b0000', '#000080', '#00ced1',\
           '#ff8c00', '#c71585', '#7fff00', '#00fa9a', '#0000ff',\
           '#ff00ff', '#1e90ff', '#f0e68c', '#ffc0cb']
x=[dz0_tot_0, dz0_tot_1, dz0_tot_2]
xtype=['d.z0','d.z0','d.z0']
labels=['$N_{\\rm peri}$ = 0','$N_{\\rm peri}$ = 1','$N_{\\rm peri}$ > 1']
binsize=20
xlimits=(0,500)
med_location=[0.00925, 0.00875, 0.009]
#
# Plot the data
plt.figure(figsize=(10, 8))
#
for i in range(0, len(x)):
    minn = binsize*np.floor(np.min(x[i])/binsize)
    maxx = binsize*np.ceil(np.max(x[i])/binsize)
    if minn < 0:
        bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
    else:
        bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
    bin_array = np.linspace(minn, maxx, bin_num)
    #
    # Calculate the scatter
    onesigp = 84.13
    onesigm = 15.87
    sigma_one_op = np.nanpercentile(x[i], onesigp)
    sigma_one_om = np.nanpercentile(x[i], onesigm)
    #
    if med_location:
        y_med = med_location[i]
    else:
        y_med = np.max(np.histogram(x[i], bin_array, density=pdf)[0])*1.1
    #
    plt.hist(x[i], bin_array, density=True, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
    plt.errorbar(np.median(x[i]), y_med, xerr=np.array([[np.median(x[i])-sigma_one_om],[sigma_one_op-np.median(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.8)
    plt.scatter(np.median(x[i]), y_med, s=250, marker='s', c=colorss[i], alpha=0.8)
#
plt.xlim(xlimits)
plt.xlabel('Host distance $d$ [kpc]', fontsize=36)
plt.ylabel('PDF', fontsize=36)
plt.legend(prop={'size': 24}, loc='center right')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/final/figure_9a.pdf')
plt.close()



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
x=[(potential_tot_0+ke_z0_tot_0)/1e4, (potential_tot_1+ke_z0_tot_1)/1e4, (potential_tot_2+ke_z0_tot_2)/1e4]
xtype=['E.tot','E.tot','E.tot']
labels=['$N_{\\rm peri}$ = 0','$N_{\\rm peri}$ = 1','$N_{\\rm peri}$ > 1']
med_location=[1.3, 1.25, 1.225]
binsize=0.2
xlimits=(-6,2)
#
# Plot the data
plt.figure(figsize=(10, 8))
#
for i in range(0, len(x)):
    minn = binsize*np.floor(np.min(x[i])/binsize)
    maxx = binsize*np.ceil(np.max(x[i])/binsize)
    if minn < 0:
        bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
    else:
        bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
    bin_array = np.linspace(minn, maxx, bin_num)
    #
    # Calculate the scatter
    onesigp = 84.13
    onesigm = 15.87
    sigma_one_op = np.nanpercentile(x[i], onesigp)
    sigma_one_om = np.nanpercentile(x[i], onesigm)
    #
    y_med = med_location[i]
    plt.hist(x[i], bin_array, density=True, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
    plt.errorbar(np.median(x[i]), y_med, xerr=np.array([[np.median(x[i])-sigma_one_om],[sigma_one_op-np.median(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.8)
    plt.scatter(np.median(x[i]), y_med, s=250, marker='s', c=colorss[i], alpha=0.8)
#
plt.xlim(xlimits)
plt.xlabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=36)
plt.ylabel('PDF', fontsize=36)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/final/figure_9b.pdf')
plt.close()



# Angular momentum plot
L_tot_0 = summary.L_z0(data_total, nperi_0_mask, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot_1 = summary.L_z0(data_total, nperi_1_mask, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot_2 = summary.L_z0(data_total, nperi_2_mask, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[L_tot_0/1e4, L_tot_1/1e4, L_tot_2/1e4], xtype=['L.tot','L.tot','L.tot'], labels=['$N_{\\rm peri}$ = 0','$N_{\\rm peri}$ = 1','$N_{\\rm peri}$ > 1'], binsize=0.2, pdf=True, xlimits=(0,7), med_location=[1.025, 0.975, 1.05,], legend_on=False, file_path_and_name=directory+'/Ltot_comparison_nperis_histogram.pdf')
#
x=[L_tot_0/1e4, L_tot_1/1e4, L_tot_2/1e4]
xtype=['L.tot','L.tot','L.tot']
labels=['$N_{\\rm peri}$ = 0','$N_{\\rm peri}$ = 1','$N_{\\rm peri}$ > 1']
binsize=0.2
xlimits=(0,6)
med_location=[1.025, 0.975, 1.05]
#
# Plot the data
plt.figure(figsize=(10, 8))
#
for i in range(0, len(x)):
    minn = binsize*np.floor(np.min(x[i])/binsize)
    maxx = binsize*np.ceil(np.max(x[i])/binsize)
    if minn < 0:
        bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
    else:
        bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
    bin_array = np.linspace(minn, maxx, bin_num)
    #
    # Calculate the scatter
    onesigp = 84.13
    onesigm = 15.87
    sigma_one_op = np.nanpercentile(x[i], onesigp)
    sigma_one_om = np.nanpercentile(x[i], onesigm)
    #
    y_med = med_location[i]
    plt.hist(x[i], bin_array, density=True, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
    plt.errorbar(np.median(x[i]), y_med, xerr=np.array([[np.median(x[i])-sigma_one_om],[sigma_one_op-np.median(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.8)
    plt.scatter(np.median(x[i]), y_med, s=250, marker='s', c=colorss[i], alpha=0.8)
#
plt.xlim(xlimits)
plt.xlabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=36)
plt.ylabel('PDF', fontsize=36)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/final/figure_9c.pdf')
plt.close()
