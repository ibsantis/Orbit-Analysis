#!/usr/bin/python3

"""
    =======================
    = Paper I Final Plots =
    =======================

    Create figures to be featured in Paper I

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
from matplotlib.ticker import MultipleLocator
from matplotlib import gridspec
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
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_z', sim_type='baryon')
data_potentials = summary.data_read_potential(directory=sim_data.home_dir, hosts='all_energy', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_z')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_z')
summary_plot = summary_io.SummaryDataPlot()


# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_1/paper_figs/final'

### Generate all of the data for the plots below
#
# Fix for the outlier in the Mstar-Mhalo relation
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
#
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
d_1st_tot = summary.dperi_first(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_sim_tot = summary.tperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
t_min_tot = summary.tperi_min(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_any_tot = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mhalo_peak_tot = summary.mhalo(data_total, masks_infall, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
L_in = summary.L_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
################################################################################



################################################################################


"""
    Figure 1:
        SMHM Relation
"""
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
Mhalo_peak_tot = summary.mhalo(data_total, masks_infall, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon')
#
# Create the data of satellites that never fell in
mask_1Mpc = dict()
for name in summary.host_names['all_no_z']:
    mask_1Mpc[name] = ~data_total[name]['infall.check']
#
Mstar_z0_1Mpc = []
Mhalo_peak_1Mpc = []
for name in summary.host_names['all_no_z']:
    for i in range(0, len(mask_1Mpc[name])):
        if (mask_1Mpc[name][i]) and (data_total[name]['d.tot.sim'][:,0][i] > 1000):
            Mstar_z0_1Mpc.append(np.repeat(data_total[name]['M.star.z0'][i], summary.oversample['baryon'][name]))
            Mhalo_peak_1Mpc.append(np.repeat(data_total[name]['M.halo.peak'][i], summary.oversample['baryon'][name]))
Mstar_z0_1Mpc = np.hstack(Mstar_z0_1Mpc)
Mhalo_peak_1Mpc = np.hstack(Mhalo_peak_1Mpc)
#
#summary_plot.median_plot(x=Mhalo_peak_tot, y=Mstar_z0_tot, xtype='M.halo.peak', ytype='M.star.z0', binsize=0.5, binedges=(8,12), limits=((7.9,11.5),(4,10)), file_path_and_name=directory+'/smhm.pdf')

# Generate the Behroozi median extrapolation
df = pd.read_csv('~/simulation/orbit_data/smhm_behroozi_values.txt', sep=' ', header=0, skiprows=[1,2])
x_behroozi = df['Log10(Mpeak/Msun)'][:15]
y_behroozi = df['Log10(Median_SM/Msun)'][:15]
#
smhm_behroozi = interpolate.interp1d(x=x_behroozi, y=y_behroozi, bounds_error=False, fill_value='extrapolate')
x_behroozi_new = np.linspace(8.25,10.5,300)

# Generate the Moster (2013) function
def moster_smhm(xs):
    m1 = 10**11.59# + 1.195*(0.1/(0.1+1))
    N = 0.0351# + (-0.0247)*(0.1/(0.1+1))
    beta = 1.376# + (-0.826)*(0.1/(0.1+1))
    gamma = 0.608# + 0.329*(0.1/(0.1+1))
    return xs*2*N*( (xs/m1)**((-1)*beta) + (xs/m1)**(gamma) )**(-1)
x_moster = np.logspace(8.25,11.25,300)
smhm_moster = moster_smhm(x_moster)

# Generate SGK (2017) function
def sgk_smhm(xs, alpha, b):
    return xs*(alpha)+(-1)*b
x_sgk = np.linspace(8.25,11.25,300)
smhm_sgk_1 = sgk_smhm(x_sgk, 1.8, 11.3)
smhm_sgk_2 = sgk_smhm(x_sgk, 2.6, 20.4)

f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#35cddc', '#3579DC', '#4535dc']
binedges = (8,12)
binsize = 0.5
limits=((7.99,11.3),(4,10))
#
x = [np.log10(Mhalo_peak_tot), np.log10(Mhalo_peak_1Mpc)]
y = [np.log10(Mstar_z0_tot), np.log10(Mstar_z0_1Mpc)]
#
xtype = ['M.halo.peak', 'M.halo.peak']
ytype = ['M.star.z0', 'M.star.z0']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), 10**(uppers[0]), 10**(lowers[0]), color=colorss[1], alpha=0.3)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), 10**(highests[0]), 10**(lowests[0]), color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), 10**(medians[0]), color=colorss[1], linewidth=3.5, alpha=0.9, label='This work')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), 10**(medians[1]), color=colorss[1], linewidth=3.5, linestyle='-.', alpha=0.9, label='This work, non-satellites')
#
ax1.set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
ax1.set_ylim(10**(limits[1][0]), 10**(limits[1][1]))
#
# Plot the Behroozi extrapolation
ax1.plot(x_moster, smhm_moster, color=summary_plot.colors[2], linestyle='--', alpha=0.4, label='Moster+ (2013)')
ax1.plot(10**(x_sgk), 10**(smhm_sgk_1), color='#006400', linestyle='--', alpha=0.4, label='Garrison-Kimmel+ (2017a)') # alpha = 1.8
#ax1.plot(10**(x_sgk), 10**(smhm_sgk_2), color=summary_plot.colors[9], linestyle='--', alpha=0.75, label='Garrison-Kimmel+ (2017a, $\\alpha=2.6$)')
ax1.plot(10**(x_behroozi), 10**(y_behroozi), color='k', linestyle='--', alpha=0.4)
ax1.plot(10**(x_behroozi_new), 10**(smhm_behroozi(x_behroozi_new)), linestyle='--', color='k', alpha=0.4, label='Behroozi+ (2020)')
#
# Plot Fitts points as +s
x_fitts = np.array([8.16e9, 9.29e9, 8.92e9, 8.43e9, 1.02e10, 7.82e9, 8.56e9, 7.92e9, 1.28e10, 1.06e10, 1.10e10, 1.15e10, 1.06e10, 1.15e10])
y_fitts = np.array([1e5, 4.56e5, 5.75e5, 1.53e6, 1.98e6, 2.08e6, 4.11e6, 5.70e6, 7.80e6, 8.01e6, 9.74e6, 1.04e7, 1.30e7, 1.44e7])
y_fitts_upper = np.percentile(y_fitts, onesigp)
y_fitts_lower = np.percentile(y_fitts, onesigm)
y_fitts_highest = np.percentile(y_fitts, twosigp)
y_fitts_lowest = np.percentile(y_fitts, twosigm)
#
ax1.plot((np.min(x_fitts),np.max(x_fitts)), (np.median(y_fitts), np.median(y_fitts)), color='#660066', alpha=0.5, label='Fitts+ (2017)')
ax1.fill_between(x=(np.min(x_fitts), np.max(x_fitts)), y1=y_fitts_lower, y2=y_fitts_upper, color='#660066', alpha=0.25)
ax1.fill_between(x=(np.min(x_fitts), np.max(x_fitts)), y1=y_fitts_lowest, y2=y_fitts_highest, color='#660066', alpha=0.15)
#ax1.scatter(x=x_fitts, y=y_fitts, marker='+', s=150, c='k', alpha=0.7, label='Fitts+ (2017)')
#
# Plot Coral's points as stars
x_wheeler = np.array([2.5e9, 3.2e9, 9e9, 7.7e9])
y_wheeler = np.array([12e3, 41e3, 330e3, 5200e3])
ax1.scatter(x=x_wheeler, y=y_wheeler, marker='*', s=150, c='m', alpha=0.7, label='Wheeler+ (2019)')
#
plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
#
ax1.text(5*10**9,4*10**4, '$M_{\\rm star}$-limited selection', fontsize=24)
#
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('$M_{\\rm halo,peak}$ [$M_{\\odot}$]', fontsize=30)
ax1.set_ylabel('$M_{\\rm star}$ [$M_{\\odot}$]', fontsize=30)
ax1.legend(prop={'size': 21}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/smhm_w_comparisons.pdf')


################################################################################



################################################################################

"""
    Figure 2:
        Dynamics vs Mstar, d(z = 0), and t_infall
"""
#f, axs = plt.subplots(3, 3, figsize=(24,16))
f = plt.figure(figsize=(28,16))
colorss = ['#6096e4', '#2a1da7', '#1d55a7']
gs = gridspec.GridSpec(4, 3, height_ratios=[1,4.5,4.5,4.5])
ax00 = plt.subplot(gs[0,0])
ax01 = plt.subplot(gs[0,1])
ax02 = plt.subplot(gs[0,2])
ax10 = plt.subplot(gs[1,0])
ax11 = plt.subplot(gs[1,1])
ax12 = plt.subplot(gs[1,2])
ax20 = plt.subplot(gs[2,0])
ax21 = plt.subplot(gs[2,1])
ax22 = plt.subplot(gs[2,2])
ax30 = plt.subplot(gs[3,0])
ax31 = plt.subplot(gs[3,1])
ax32 = plt.subplot(gs[3,2])
#
"""
    Plotting versus t_infall
"""
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
ts = summary.first_infall(data_total, masks_infall, oversample=False, hosts='all_no_z', sim_type='baryon')
tbins = summary_plot.binning_scheme(ts, 't.infall', binsize=0.5)[0]
ax00.hist(ts, bins=tbins, density=False, color='k', alpha=0.35)
ax00.set_xlim([0,13])
#
binedges = None
binsize = 1
limits_1 = ((0,13),(0,350))
limits_2 = ((0,13),(-5.5,0.5))
limits_3 = ((0,13),(0,3.8))
#
x = [t_in_tot]
y = [vz0_tot]
#
xtype = ['t.infall.text']
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
ax10.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.5)
ax10.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax10.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], markersize=10, alpha=0.7)
ax10.axvspan(2.85, 3.01, alpha=0.1, color='k')
ax10.text(3.1,160,'Free-fall time',fontsize=24,rotation=90)
ax10.axhspan(121, 150, alpha=0.1, color='k')
#
ax10.set_xlim(limits_1[0])
ax10.set_ylim(limits_1[1])
#
potential_tot = summary.potential(data_potentials, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, masks_infall, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon')
#
x = [t_in_tot]
y = [(potential_tot+ke_z0_tot)/1e4]
#
xtype = ['t.infall.text']
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
ax20.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[2], alpha=0.5)
ax20.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[2], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax20.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[2], markersize=10, alpha=0.7)
ax20.axvspan(2.85, 3.01, alpha=0.1, color='k')
#
ax20.set_xlim(limits_2[0])
ax20.set_ylim(limits_2[1])
#
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
x = [t_in_tot]
y = [L_tot/1e4]
#
xtype = ['t.infall.text']
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
ax30.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax30.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax30.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.7)
ax30.axvspan(2.85, 3.01, alpha=0.1, color='k')
ax30.axhspan(27951/1e4, 60494/1e4, alpha=0.1, color='k')
#
ax30.set_xlim(limits_3[0])
ax30.set_ylim(limits_3[1])


"""
    Plotting versus d(z = 0)
"""
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
dzs = summary.d_z0(data_total, masks_infall, oversample=False, hosts='all_no_z', sim_type='baryon')
dbins = summary_plot.binning_scheme(dzs, 't.infall', binsize=20)[0]
ax01.hist(dzs, bins=dbins, density=False, color='k', alpha=0.35)
ax01.set_xlim([0,415])
#
binedges = None
binsize = 50
limits_1 = ((0,415),(0,350))
limits_2 = ((0,415),(-5.5,0.5))
limits_3 = ((0,415),(0,3.8))
#
x = [dz0_tot]
y = [vz0_tot]
#
xtype = ['d.z0']
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
# Make circular velocity line
def vc(m,r):
    return np.sqrt((6.67*10**(-11)*10**(12)*2*10**(30)*m)/(1000*3.086*10**(16)*r))/1000
ds = np.linspace(25, 425, 1000)
vcirc = np.zeros(len(ds))
for i in range(0, len(ds)):
    vcirc[i] = vc(1, ds[i])
#
def vc_nfw(amp, rs, r):
    return np.sqrt((amp*6.67*10**(-11)*2*10**(30)*(np.log((rs+r)/rs)+rs/(rs+r)-1))/(r*1000*3.086*10**(16)))/1000
#
def vvir(m,r):
    return np.sqrt((m/r)*(6.67*10**(-11)*10**(12)*2*10**(30))/(1000*3.086*10**(16)))/1000
#
vcirc_nfw_mean = np.zeros(len(ds))
vcirc_nfw_median = np.zeros(len(ds))
for i in range(0, len(ds)):
    vcirc_nfw_mean[i] = vc_nfw(554760646622.4431, 20.52222313571508, ds[i])
    vcirc_nfw_median[i] = vc_nfw(416798929010.74097, 20.0940230187877, ds[i])
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax11.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.5)
ax11.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax11.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], markersize=10, alpha=0.7)
#ax11.plot(ds, vcirc, color='k', linestyle='dotted', alpha=0.8, label='$\\upsilon_{\\rm c}$: point mass')
#ax11.plot(ds, vcirc_nfw_mean, color='b', linestyle='dotted', alpha=0.8, label='$\\upsilon_{\\rm c}$: NFW mean')
#ax11.plot(ds, vcirc_nfw_median, color='r', linestyle='dotted', alpha=0.8, label='$\\upsilon_{\\rm c}$: NFW median')
#
ax11.axvspan(319, 406, alpha=0.1, color='k')
ax11.text(330,300,'$R_{\\rm 200m}$', fontsize=30)
#
ax11.axhspan(121, 150, alpha=0.1, color='k')
ax11.text(30, 125, '$V_{\\rm 200m}$', fontsize=30)
#
ax11.set_xscale('linear')
ax11.set_xlim(limits_1[0])
ax11.set_ylim(limits_1[1])
#ax11.legend(prop={'size': 20}, loc='best')
#
############
#
potential_tot = summary.potential(data_potentials, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, masks_infall, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon')
#
x = [dz0_tot, dz0_tot]
y = [(potential_tot+ke_z0_tot)/1e4, potential_tot/1e4]
#
xtype = ['d.z0', 'd.z0']
ytype = ['E.tot', 'E.tot']
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
# Calculate simple potentials
def phi_pm(m,r):
    return (-1)*((6.67*10**(-11)*10**(12)*2*10**(30))/(r*1000*3.086*10**(16)))*(1/1000)**2
phi_point = np.zeros(len(ds))
for i in range(0, len(ds)):
    phi_point[i] = phi_pm(1, ds[i])
#
def phi_nfw(amp, rs, r):
    return (-1)*((6.67*10**(-11)*amp*2*10**(30))/(r*1000*3.086*10**(16)))*np.log(1+r/rs)*(1/1000)**2
phi_mean = np.zeros(len(ds))
phi_median = np.zeros(len(ds))
for i in range(0, len(ds)):
    phi_mean[i] = phi_nfw(554760646622.4431, 20.52222313571508, ds[i])
    phi_median[i] = phi_nfw(416798929010.74097, 20.0940230187877, ds[i])
#
def phi_vir(m,r):
    return (-m/r)*((6.67*10**(-11)*10**(12)*2*10**(30))/(1000*3.086*10**(16)))*(1/1000)**2
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax21.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[2], alpha=0.5)
ax21.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[2], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax21.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[2], markersize=10, alpha=0.7)
#ax31.plot(ds, phi_point/1e4, color='k', linestyle='dotted', alpha=0.7, label='$\\Phi$: point mass')
#ax31.plot(ds, phi_mean/1e4, color='b', linestyle='dotted', alpha=0.7, label='$\\Phi$: NFW mean')
#ax31.plot(ds, phi_median/1e4, color='r', linestyle='dotted', alpha=0.7, label='$\\Phi$: NFW median')
#ax31.plot(binss[1][:-1]+half_bins[1], medians[1], color='g', linestyle='dotted', markersize=10, alpha=0.7, label='$\\Phi$: Simulations')
#
ax21.axvspan(319, 406, alpha=0.1, color='k')
#ax31.axhspan(-14813/1e4, -22359/1e4, alpha=0.1, color='k')
#ax31.text(270, -2, '$\\Phi_{\\rm vir}$', fontsize=30)
#
ax21.set_xscale('linear')
ax21.set_xlim(limits_2[0])
ax21.set_ylim(limits_2[1])
#ax31.legend(prop={'size': 20}, loc='best')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
x = [dz0_tot]
y = [L_tot/1e4]
#
xtype = ['d.z0']
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
# Define the circular angular momentum
def lc(m,r):
    return np.sqrt((6.67*10**(-11)*2*10**(30)*m*r)/(1000*3.086*10**(16)))/1000
lcirc = np.zeros(len(ds))
for i in range(0, len(ds)):
    lcirc[i] = lc(10**(12), ds[i])
#
def lc_nfw(amp,rs,r):
    return np.sqrt(((6.67*10**(-11)*2*10**(30)*amp*r)/(1000*3.086*10**(16))*(np.log((rs+r)/rs)+(rs/(rs+r))-1)))/1000
lcirc_mean_nfw = np.zeros(len(ds))
lcirc_median_nfw = np.zeros(len(ds))
for i in range(0, len(ds)):
    lcirc_mean_nfw[i] = lc_nfw(554760646622.4431, 20.52222313571508, ds[i])
    lcirc_median_nfw[i] = lc_nfw(416798929010.74097, 20.0940230187877, ds[i])
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
ax31.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax31.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax31.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.7)
#ax21.plot(ds, lcirc/1e4, color='k', linestyle='dotted', label='$\\ell_{\\rm c}$: point mass')
#ax21.plot(ds, lcirc_mean_nfw/1e4, color='b', linestyle='dotted', label='$\\ell_{\\rm c}$: NFW mean')
#ax21.plot(ds, lcirc_median_nfw/1e4, color='r', linestyle='dotted', label='$\\ell_{\\rm c}$: NFW median')
#
ax31.axvspan(319, 406, alpha=0.1, color='k')
ax31.axhspan(27951/1e4, 60494/1e4, alpha=0.1, color='k')
ax31.text(30, 3.2, '$L_{\\rm 200m}$', fontsize=30)
#
ax31.set_xscale('linear')
ax31.set_xlim(limits_3[0])
ax31.set_ylim(limits_3[1])
#ax21.legend(prop={'size': 20}, loc='best')

"""
    Plotting versus Mstar
"""
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
mms = summary.mstar(data_total, masks_infall, selection='z0', oversample=False, hosts='all_no_z', sim_type='baryon')
mbins = summary_plot.binning_scheme(mms, 'M.star.z0', binsize=0.5, binedges=(4.5,9.5))[0]
ax02.hist(mms, bins=10**mbins, density=False, color='k', alpha=0.35)
ax02.set_xlim([10**(4.5),10**(9.5)])
ax02.set_xscale('log')
#
binedges = (4.5, 9.5)
binsize = 0.5
limits_1 = ((4.5,9.5),(0,350))
limits_2 = ((4.5,9.5),(-5.5,0.5))
limits_3 = ((4.5,9.5),(0,3.8))
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
ax12.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.5)
ax12.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax12.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], markersize=10, alpha=0.7)
ax12.axhspan(121, 150, alpha=0.1, color='k')
#
ax12.set_xscale('log')
ax12.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax12.set_ylim(limits_1[1])
#
potential_tot = summary.potential(data_potentials, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, masks_infall, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
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
ax22.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[2], alpha=0.5)
ax22.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[2], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax22.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[2], markersize=10, alpha=0.7)
#
ax22.set_xscale('log')
ax22.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax22.set_ylim(limits_2[1])
#
#
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
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
ax32.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax32.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax32.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7)
ax32.axhspan(27951/1e4, 60494/1e4, alpha=0.1, color='k')
#
ax32.set_xscale('log')
ax32.set_xlim(10**(limits_3[0][0]), 10**(limits_3[0][1]))
ax32.set_ylim(limits_3[1])
#
ax11.get_yaxis().set_label_coords(-0.14,0.5)
ax21.get_yaxis().set_label_coords(-0.14,0.5)
ax11.set_ylabel(' ', fontsize=28)
ax21.set_ylabel(' ', fontsize=28)
ax31.set_ylabel(' ', fontsize=28)
ax31.get_yaxis().set_label_coords(-0.14,0.5)
#
ax12.get_yaxis().set_label_coords(-0.14,0.5)
ax22.get_yaxis().set_label_coords(-0.14,0.5)
ax32.get_yaxis().set_label_coords(-0.14,0.5)
ax12.set_ylabel(' ', fontsize=24)
ax22.set_ylabel(' ', fontsize=24)
ax32.set_ylabel(' ', fontsize=24)
#
ax12.xaxis.set_major_locator(LogLocator(base=10))
ax12.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax22.xaxis.set_major_locator(LogLocator(base=10))
ax22.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax32.xaxis.set_major_locator(LogLocator(base=10))
ax32.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax12.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
ax22.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
ax32.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
#
ax00.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax01.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax02.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax10.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False)
ax20.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False)
ax30.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28)
ax11.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=False, labelbottom=False)
ax21.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=False, labelbottom=False)
ax31.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=False)
ax12.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=False, labelbottom=False)
ax22.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=False, labelbottom=False)
ax32.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelleft=False)
#
ax30.set_xlabel('Infall lookback time [Gyr]', fontsize=30)
ax31.set_xlabel('Host distance $d$ [kpc]', fontsize=30)
ax32.set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=30)
#
ax00.set_ylabel('$N_{\\rm sat}$', fontsize=28)
ax00.get_yaxis().set_label_coords(-0.08,0.5)
ax10.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=28)
ax10.get_yaxis().set_label_coords(-0.14,0.5)
ax20.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=30)
ax20.get_yaxis().set_label_coords(-0.14,0.5)
ax30.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=30)
ax30.get_yaxis().set_label_coords(-0.14,0.5)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.show()
plt.savefig(directory+'/dynamics.pdf')


################################################################################


################################################################################


"""
    Figure 3: Ell vs Infall time and Mstar(z = 0)
"""
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
L_in = summary.L_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,8))
colorss = ['#1d55a7']
limits_1 = ((0,13),(-1,2))
limits_2 = ((4.5,9.5),(-1,2))
binedges=None
binsize = 1
#
x = [t_in_tot]
y = [(L_tot-L_in)/L_in]
#
xtype = ['t.infall.text']
ytype = ['L.frac']
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.5)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], markersize=10, alpha=0.7)
#
ax1.set_xscale('linear')
ax1.set_xlim(limits_1[0][0], limits_1[0][1])
ax1.set_ylim(limits_1[1])
#
ax1.axvspan(2.85, 3.01, alpha=0.1, color='k')
ax1.text(3.1,0.7,'Free-fall time',fontsize=24,rotation=90)
ax1.hlines(y=0, xmin=0, xmax=13, color='k', linestyle='dotted', alpha=0.3)
#
if 't.' in xtype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_3_label = 'redshift'
    axis_3_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_3_tick_values = [float(v) for v in axis_3_tick_labels]
    axis_3_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_3_tick_values)
    ax3 = ax1.twiny()
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.set_xticks(axis_3_tick_locations)
    ax3.set_xticklabels(axis_3_tick_labels, fontsize=26)
    ax3.set_xlim(limits_1[0])
    ax3.set_xlabel(axis_3_label, fontsize=28, labelpad=9)
    ax3.tick_params(pad=3)
#
binedges = (4.5, 9.5)
binsize = 0.5
limits_1 = ((4.5,9.5),(-1,1.5))
#
x = [Mstar_z0_tot]
y = [(L_tot-L_in)/L_in]
#
xtype = ['M.star.z0']
ytype = ['L.frac']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.5)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], markersize=10, alpha=0.7)
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
ax2.hlines(y=0, xmin=10**(4.5), xmax=10**(9.5), color='k', linestyle='dotted', alpha=0.3)
#
ax1.set_ylabel('($\\ell(z=0) - \\ell_{\\rm infall,MW}$)/$\\ell_{\\rm infall,MW}$', fontsize=28)
ax1.set_xlabel('Infall lookback time [Gyr]', fontsize=28)
ax2.set_xlabel('$M_{\\rm star}$ [$M_{\\odot}$]', fontsize=28)
ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelsize=26)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelleft=False, labelsize=26)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(directory+'/ell_evo.pdf')


################################################################################


################################################################################


"""
    Figure 4:
        Pericenter time and Infall time vs Mstar
"""
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_any_tot = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
#colorss = ['#aa2222', '#42193d']
colorss = ['#000000', '#ff4900']
binedges = (4.5, 9.5)
binsize = 0.5
limits_1 = ((4.5,9.5), (0,13.8))
limits_2 = ((4.5,9.5), (0,9))
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
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.6, label='MW-mass halo')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.6, label='Any halo')
ax1.hlines(y=12.84, xmin=10**(limits_1[0][0]), xmax=10**(limits_1[0][1]), colors='k', linestyles='dotted', alpha=0.5)
ax1.text(6e5, 12.1, 'End of Reionization', fontsize=20)

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
t_sim_tot = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
t_min_tot = summary.tperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
x = [Mstar_z0_tot, Mstar_z0_tot]
y = [t_sim_tot, t_min_tot]
#
xtype = ['M.star.z0', 'M.star.z0']
ytype = ['t.peri.text','t.peri.text']
colorss = ['#000080', '#006400']
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
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum')
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
    ax4.set_yticklabels(axis_4_tick_labels, fontsize=28)
    ax4.set_ylim(limits_2[1])
    ax4.set_ylabel(axis_4_label, labelpad=9)
    ax4.tick_params(pad=3)
#
ax1.legend(prop={'size': 20}, loc='best')
ax2.legend(prop={'size': 20}, loc='best')
#
ax1.xaxis.set_major_locator(LogLocator(base=10))
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax1.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
ax1.yaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
ax1.set_yticks([0,2,4,6,8,10,12])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False)
ax2.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28)
#
ax1.set_ylabel('Infall Lookback Time [Gyr]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.075,0.5)
ax2.set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.075,0.5)
ax2.set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=28)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/times_vs_mstar.pdf')


################################################################################


################################################################################


"""
    Figure 5:
        Infall time versus d(z = 0)
"""
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_any_tot = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
#
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#000000', '#ff4900']
binedges=None
binsize = 50
limits_1 = ((0, 400),(0,13.5))
#
x = [dz0_tot, dz0_tot]
y = [t_in_tot, t_in_any_tot]
#
xtype = ['d.z0', 'd.z0']
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax1.fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax1.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.5, label='MW-mass halo')
ax1.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Any halo')
plt.hlines(y=12.84, xmin=limits_1[0][0], xmax=limits_1[0][1], colors='k', linestyles='dotted', alpha=0.5)
ax1.text(85, 12.1, 'End of Reionization', fontsize=20)

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
    ax3.set_yticklabels(axis_3_tick_labels, fontsize=28)
    ax3.set_ylim(limits_1[1])
    ax3.set_ylabel(axis_3_label, fontsize=28, labelpad=9)
    ax3.tick_params(pad=3)
#
ax1.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax1.set_ylabel('Infall Lookback Time [Gyr]', fontsize=28)
ax1.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/t_infall_vs_dz0.pdf')


###############################################################################


###############################################################################


"""
    Figure 6 :
        Pericenter number vs infall time
"""
f, axs = plt.subplots(2, 3, figsize=(26,16))
#colorss = ['#000080', '#006400']
colorss = ['#A02422']
#colorss = ['#3579DC']
#
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
#
x = [t_in_tot]
y = [N_sim_tot]
#
binsize = 1
binedges = None
limits = ((0,13),(0,10))
#
xtype = ['t.infall.text']
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
axs[0,0].fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.5)
axs[0,0].fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
axs[0,0].axvspan(2.85, 3.01, alpha=0.15, color='k')
axs[0,0].text(3.2, 5.8, 'Free-fall time', fontsize=28, rotation=90)
#
# Plot the medians for the two mass bins (low-mass)
axs[0,0].plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[0], markersize=10, alpha=0.7)
#
axs[0,0].set_xlim(limits[0])
axs[0,0].set_ylim(limits[1])
#
if 't.' in xtype[0]:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_3_label = 'redshift'
    axis_3_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.1', '0']
    axis_3_tick_values = [float(v) for v in axis_3_tick_labels]
    axis_3_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_3_tick_values)
    ax3 = axs[0,0].twiny()
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.set_xticks(axis_3_tick_locations)
    ax3.set_xticklabels(axis_3_tick_labels, fontsize=28)
    ax3.set_xlim(limits[0])
    ax3.set_xlabel(axis_3_label, labelpad=9, fontsize=28)
    ax3.tick_params(pad=3)
#
"""
        Pericenter number vs d(z = 0), binned by Mstar
"""
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
binedges = None
binsize = 50
limits = ((0,400),(0,8))
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
axs[0,1].fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.5)
axs[0,1].fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
axs[0,1].axvspan(319, 406, alpha=0.15, color='k')
axs[0,1].text(325, 6, '$R_{\\rm 200m}$', fontsize=30)
#
# Plot the medians for the two mass bins (low-mass)
axs[0,1].plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[0], markersize=10, alpha=0.7, label='$M_{\\rm star} < 10^7 M_{\\odot}$')
axs[0,1].plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[0], linestyle='--', markersize=10, alpha=0.7, label='$M_{\\rm star} > 10^7 M_{\\odot}$')
#
axs[0,1].set_xscale('linear')
axs[0,1].set_xlim(limits[0])
axs[0,1].set_ylim(limits[1])
#
"""
    Pericenter number vs Mstar
"""
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
x = [Mstar_z0_tot]
y = [N_sim_tot]
binedges = (4.5, 9.5)
binsize = 0.5
limits = ((4.5,9.5),(0,5))
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
axs[0,2].fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.5)
axs[0,2].fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
axs[0,2].plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], markersize=10, alpha=0.7) # Recent, M < 1e7
#
axs[0,2].set_xscale('log')
axs[0,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
axs[0,2].set_ylim(limits[1])
#
"""
        Pericenter distance versus t_infall
"""
d_min_tot = summary.dperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
#d_1st_tot = summary.dperi_first(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
binsize = 1
binedges = None
limits = ((0,13),(0,220))
colorss = ['#000080', '#006400']
#
x = [t_in_tot, t_in_tot]
y = [d_sim_tot, d_min_tot]
#
xtype = ['t.infall.text','t.infall.text']
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
axs[1,0].fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
axs[1,0].fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
axs[1,0].fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
axs[1,0].fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
axs[1,0].axvspan(2.85, 3.01, alpha=0.15, color='k')
#
# Plot the medians for the two mass bins (low-mass)
axs[1,0].plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.8, label='Recent')
axs[1,0].plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum')
#
axs[1,0].set_xlim(limits[0])
axs[1,0].set_ylim(limits[1])
#
"""
        Pericenter distance versus d(z = 0)
"""
d_sim_tot = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
#
binedges = None
binsize = 50
limits = ((0,400),(0,165))
#
x = [dz0_tot, dz0_tot]
y = [d_sim_tot, d_min_tot]
#
xtype = ['d.z0','d.z0']
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
axs[1,1].fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
axs[1,1].fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
axs[1,1].fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
axs[1,1].fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
axs[1,1].axvspan(319, 406, alpha=0.15, color='k')
#
# Plot the medians for the two mass bins (low-mass)
axs[1,1].plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent') # Recent, M < 1e7
axs[1,1].plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum') #, label='MW/M31-mass host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Recent M > 1e7
#
axs[1,1].set_xlim(limits[0])
axs[1,1].set_ylim(limits[1])
#
"""
    Pericenter distance vs Mstar
"""
d_sim_tot = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
binedges = (4.5, 9.5)
binsize = 0.5
limits = ((4.5,9.5),(0,165))
#
x = [Mstar_z0_tot, Mstar_z0_tot]
y = [d_sim_tot, d_min_tot]
#
xtype = ['M.star.z0', 'M.star.z0']
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
axs[1,2].fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
axs[1,2].fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
axs[1,2].fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
axs[1,2].fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
axs[1,2].plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent') # Recent, M < 1e7
axs[1,2].plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum') #, label='MW/M31-mass host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Recent M > 1e7
#
axs[1,2].set_xscale('log')
axs[1,2].set_xlim(10**(limits[0][0]), 10**(limits[0][1]))
axs[1,2].set_ylim(limits[1])
#
## Labels and other stuff
#
axs[1,0].set_xlabel('Infall lookback time [Gyr]', fontsize=30)
axs[1,1].set_xlabel('Host distance $d$ [kpc]', fontsize=32)
axs[1,2].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=32)
axs[0,0].set_ylabel('Pericenter Number', fontsize=32)
axs[1,0].set_ylabel('Pericenter distance [kpc]', fontsize=30)
axs[0,0].get_yaxis().set_label_coords(-0.12,0.5)
axs[1,0].get_yaxis().set_label_coords(-0.12,0.5)
#
axs[0,1].legend(prop={'size': 26}, loc='best')
axs[1,0].legend(prop={'size': 26}, loc='best')
#axs[1,1].legend(prop={'size': 26}, loc='best')
#axs[1,2].legend(prop={'size': 26}, loc='best')
#
axs[0,2].xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
axs[1,2].xaxis.set_major_locator(LogLocator(base=10))
axs[1,2].xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
axs[1,2].set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False)
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28)
axs[0,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=False)
axs[1,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.16, hspace=0)
#plt.show()
plt.savefig(directory+'/peri_dn.pdf')


################################################################################


################################################################################


"""
    Figure 8:
        Outlier Histograms
            - Used to only select outliers with fractions > 5%
              i.e., used to select
"""
N_sim_tot = summary.nperi(data_total, masks_infall_peri, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
t_sim_tot = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
t_min_tot = summary.tperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
L_rec_tot = summary.L_rec(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
L_min_tot = summary.L_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
#
# dperi fraction plots
mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0)
#mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0.05)
#mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0.1)
summary_plot.plot_hist(x=((d_min_tot-d_sim_tot)/d_sim_tot)[mask_delta_d], xtype='delta_d_frac', binsize=0.05, pdf=True, xlimits=(-1,0), file_path_and_name=directory+'/delta_d_frac_histogram.pdf')
#summary_plot.plot_hist(x=((d_min_tot[m]-d_sim_tot[m])/d_sim_tot[m]), xtype='delta_d_frac', binsize=0.05, pdf=True, xlimits=(-1,0), file_path_and_name=directory+'/delta_d_frac_histogram_all_multnperi.pdf')
#
# delta tperi plots
mask_delta_t = (np.abs((t_min_tot - t_sim_tot)/t_sim_tot) > 0)
summary_plot.plot_hist(x=(t_min_tot-t_sim_tot)[mask_delta_t], xtype='delta_t', binsize=0.5, pdf=True, file_path_and_name=directory+'/delta_t_histogram.pdf')
#
# delta Lperi plots
mask_delta_d = (np.abs((d_min_tot - d_sim_tot)/d_sim_tot) > 0)
summary_plot.plot_hist(x=((L_min_tot-L_rec_tot)/L_rec_tot)[mask_delta_d], xtype='delta_ell_frac', binsize=0.05, pdf=True, xlimits=(-1, 0), file_path_and_name=directory+'/delta_ell_frac_histogram.pdf')


################################################################################


################################################################################


"""
    Figure 7:
        Fraction of satellites with growing pericenters vs N_peri
"""
N_sim_all = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
binss, binwidth = summary_plot.binning_scheme(N_sim_all, 'N.peri.text', binsize=1)
#
hist_all, edges_all = np.histogram(N_sim_all, bins=binss)
hist_out, edges_out = np.histogram(N_sim_tot[mask_delta_d], bins=binss)
#
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
ax1.bar(x=binss[1:]-0.5, height=hist_out/hist_all, color=summary_plot.colors[3], alpha=0.5, linewidth=3.5, edgecolor='k', width=1)
ax1.set_xlim(0,11)
ax1.set_ylim(-0.005,1.01)
ax1.set_xlabel('Pericenter number', fontsize=32)
ax1.set_ylabel('Fraction with growing pericenters', fontsize=24)
ax1.xaxis.set_major_locator(AutoLocator())
ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax1.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax1.yaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
ax1.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/fraction_vs_Nperi.pdf')


################################################################################


################################################################################


"""
    Figure 9:
        Orbits and L evolution subplots
"""
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
times = data_total['m12b']['time.sim'][-1] - np.flip(data_total['m12b']['time.sim'])
#
orbit_1 = data_total['m12b']['d.tot.sim'][33][data_total['m12b']['d.tot.sim'][33]>0]
orbit_2 = data_total['m12f']['d.tot.sim'][44][data_total['m12f']['d.tot.sim'][44]>0] # ORIGINALLY WAS SUBHALO 42 IN THE OLDER DATA, BUT 45 IN THE APRIL 18 DATA
orbit_3 = data_total['Thelma']['d.tot.sim'][100][data_total['Thelma']['d.tot.sim'][100]>0]
orbit_4 = data_total['m12w']['d.tot.sim'][38][data_total['m12w']['d.tot.sim'][38]>0]
#
ang_1 = data_total['m12b']['L.tot.sim'][33][data_total['m12b']['L.tot.sim'][33]>0]/1e4
ang_2 = data_total['m12f']['L.tot.sim'][44][data_total['m12f']['L.tot.sim'][44]>0]/1e4
ang_3 = data_total['Thelma']['L.tot.sim'][100][data_total['Thelma']['L.tot.sim'][100]>0]/1e4
ang_4 = data_total['m12w']['L.tot.sim'][38][data_total['m12w']['L.tot.sim'][38]>0]/1e4
#
f, axs = plt.subplots(2, 4, figsize=(22,10))
colorss = ['#000080', '#006400']
xlimit = (13.8,0)
ylimit_1 = (0,400)
ylimit_2 = (0, 3.4)
#
# PLOTTING
axs[0,0].plot(times[:len(data_total['m12b']['host.radius'])], data_total['m12b']['host.radius'], color='k', markersize=10, alpha=0.3)#, label='$R_{\\rm 200m,host}$')
axs[0,0].plot(times[:len(orbit_1)], orbit_1, color='b', markersize=10, alpha=0.5)
arrow0m = mpatches.FancyArrowPatch((data_total['m12b']['pericenter.time.lb.sim'][33][1], 50), (data_total['m12b']['pericenter.time.lb.sim'][33][1], 5), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow0r = mpatches.FancyArrowPatch((data_total['m12b']['pericenter.time.lb.sim'][33][0], 50), (data_total['m12b']['pericenter.time.lb.sim'][33][0], 5), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[0,0].add_patch(arrow0m)
axs[0,0].add_patch(arrow0r)
axs[0,0].scatter([],[],marker='$\\rightarrow$', label='Min = 70 kpc', color=colorss[0], s=300)
axs[0,0].scatter([],[],marker='$\\rightarrow$', label='Recent = 79 kpc', color=colorss[1], s=300)
axs[0,0].plot([],[], ' ', label='frac diff = 12%')
axs[0,0].legend(prop={'size': 20}, loc='upper left')
axs[0,0].text(x=7, y=180,s='$R_{\\rm 200m,host}$', fontsize=25, rotation=43)
#
axs[0,1].plot(times[:len(data_total['m12f']['host.radius'])], data_total['m12f']['host.radius'], color='k', markersize=10, alpha=0.3)
axs[0,1].plot(times[:len(orbit_2)], orbit_2, color='b', markersize=10, alpha=0.5)
arrow1m = mpatches.FancyArrowPatch((data_total['m12f']['pericenter.time.lb.sim'][44][2], 45), (data_total['m12f']['pericenter.time.lb.sim'][44][2], 5), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow1r = mpatches.FancyArrowPatch((data_total['m12f']['pericenter.time.lb.sim'][44][0], 50), (data_total['m12f']['pericenter.time.lb.sim'][44][0], 5), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[0,1].add_patch(arrow1m)
axs[0,1].add_patch(arrow1r)
axs[0,1].scatter([],[],marker='$\\rightarrow$', label='Min = 45 kpc', color=colorss[0], s=300)
axs[0,1].scatter([],[],marker='$\\rightarrow$', label='Recent = 76 kpc', color=colorss[1], s=300)
axs[0,1].plot([],[], ' ', label='frac diff = 41%')
axs[0,1].legend(prop={'size': 20}, loc='upper center')
#
axs[0,2].plot(times[:len(data_total['Thelma']['host.radius'])], data_total['Thelma']['host.radius'], color='k', markersize=10, alpha=0.3)
axs[0,2].plot(times[:len(orbit_3)], orbit_3, color='b', markersize=10, alpha=0.5)
arrow2m = mpatches.FancyArrowPatch((data_total['Thelma']['pericenter.time.lb.sim'][100][5], 125), (data_total['Thelma']['pericenter.time.lb.sim'][100][5], 75), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow2r = mpatches.FancyArrowPatch((data_total['Thelma']['pericenter.time.lb.sim'][100][0], 50), (data_total['Thelma']['pericenter.time.lb.sim'][100][0], 5), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[0,2].add_patch(arrow2m)
axs[0,2].add_patch(arrow2r)
axs[0,2].scatter([],[],marker='$\\rightarrow$', label='Min = 25 kpc', color=colorss[0], s=300)
axs[0,2].scatter([],[],marker='$\\rightarrow$', label='Recent = 68 kpc', color=colorss[1], s=300)
axs[0,2].plot([],[], ' ', label='frac diff = 63%')
axs[0,2].legend(prop={'size': 20}, loc='upper center')
#
axs[0,3].plot(times[:len(data_total['m12w']['host.radius'])], data_total['m12w']['host.radius'], color='k', markersize=10, alpha=0.3)
axs[0,3].plot(times[:len(orbit_4)], orbit_4, color='b', markersize=10, alpha=0.5)
arrow3m = mpatches.FancyArrowPatch((data_total['m12w']['pericenter.time.lb.sim'][38][1], 175), (data_total['m12w']['pericenter.time.lb.sim'][38][1], 125), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow3r = mpatches.FancyArrowPatch((data_total['m12w']['pericenter.time.lb.sim'][38][0], 50), (data_total['m12w']['pericenter.time.lb.sim'][38][0], 5), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[0,3].add_patch(arrow3m)
axs[0,3].add_patch(arrow3r)
axs[0,3].scatter([],[],marker='$\\rightarrow$', label='Min = 6 kpc', color=colorss[0], s=300)
axs[0,3].scatter([],[],marker='$\\rightarrow$', label='Recent = 80 kpc', color=colorss[1], s=300)
axs[0,3].plot([],[], ' ', label='frac diff = 93%')
axs[0,3].legend(prop={'size': 20}, loc='upper center')
#
# ANGULAR MOMENTUM
axs[1,0].plot(times[:len(ang_1)], ang_1, color='b', markersize=10, alpha=0.5)
arrow0m = mpatches.FancyArrowPatch((data_total['m12b']['pericenter.time.lb.sim'][33][1], 2.5), (data_total['m12b']['pericenter.time.lb.sim'][33][1], 2), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow0r = mpatches.FancyArrowPatch((data_total['m12b']['pericenter.time.lb.sim'][33][0], 2.5), (data_total['m12b']['pericenter.time.lb.sim'][33][0], 2), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[1,0].add_patch(arrow0m)
axs[1,0].add_patch(arrow0r)
axs[1,0].text(5,1.5,'Rapid',fontsize=28)
#
axs[1,1].plot(times[:len(ang_2)], ang_2, color='b', markersize=10, alpha=0.5)
arrow1m = mpatches.FancyArrowPatch((data_total['m12f']['pericenter.time.lb.sim'][44][2], 2.1), (data_total['m12f']['pericenter.time.lb.sim'][44][2], 1.6), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow1r = mpatches.FancyArrowPatch((data_total['m12f']['pericenter.time.lb.sim'][44][0], 2.5), (data_total['m12f']['pericenter.time.lb.sim'][44][0], 2), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[1,1].add_patch(arrow1m)
axs[1,1].add_patch(arrow1r)
axs[1,1].text(5,1.5,'Rapid',fontsize=28)
#
axs[1,2].plot(times[:len(ang_3)], ang_3, color='b', markersize=10, alpha=0.5)
arrow2m = mpatches.FancyArrowPatch((data_total['Thelma']['pericenter.time.lb.sim'][100][5], 1.1), (data_total['Thelma']['pericenter.time.lb.sim'][100][5], 0.6), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow2r = mpatches.FancyArrowPatch((data_total['Thelma']['pericenter.time.lb.sim'][100][0], 2.1), (data_total['Thelma']['pericenter.time.lb.sim'][100][0], 1.6), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[1,2].add_patch(arrow2m)
axs[1,2].add_patch(arrow2r)
axs[1,2].text(5,1,'Gradual',fontsize=28)
#
axs[1,3].plot(times[:len(ang_4)], ang_4, color='b', markersize=10, alpha=0.5)
arrow3m = mpatches.FancyArrowPatch((data_total['m12w']['pericenter.time.lb.sim'][38][1], 0.8), (data_total['m12w']['pericenter.time.lb.sim'][38][1], 0.3), color=colorss[0], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # minimum
arrow3r = mpatches.FancyArrowPatch((data_total['m12w']['pericenter.time.lb.sim'][38][0], 2.4), (data_total['m12w']['pericenter.time.lb.sim'][38][0], 1.9), color=colorss[1], arrowstyle=mpatches.ArrowStyle('Fancy',head_length=9, head_width=9, tail_width=3)) # recent
axs[1,3].add_patch(arrow3m)
axs[1,3].add_patch(arrow3r)
axs[1,3].text(5,1.1,'Rapid',fontsize=28)
#
axs[0,0].set_xlim(xlimit)
axs[0,1].set_xlim(xlimit)
axs[0,2].set_xlim(xlimit)
axs[0,3].set_xlim(xlimit)
axs[0,0].set_ylim(ylimit_1)
axs[0,1].set_ylim(ylimit_1)
axs[0,2].set_ylim(ylimit_1)
axs[0,3].set_ylim(ylimit_1)
#
axs[1,0].set_xlim(xlimit)
axs[1,1].set_xlim(xlimit)
axs[1,2].set_xlim(xlimit)
axs[1,3].set_xlim(xlimit)
axs[1,0].set_ylim(ylimit_2)
axs[1,1].set_ylim(ylimit_2)
axs[1,2].set_ylim(ylimit_2)
axs[1,3].set_ylim(ylimit_2)
#
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False)
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False, labelleft=False)
axs[0,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False, labelleft=False)
axs[0,3].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=False, labelleft=False)
#
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True)
axs[1,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
axs[1,2].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
axs[1,3].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=22, labelbottom=True, labelleft=False)
#
majors = [0,2,4,6,8,10,12]
axs[0,0].xaxis.set_major_locator(ticker.FixedLocator(majors))
axs[0,1].xaxis.set_major_locator(ticker.FixedLocator(majors))
axs[0,2].xaxis.set_major_locator(ticker.FixedLocator(majors))
axs[0,3].xaxis.set_major_locator(ticker.FixedLocator(majors))
#
axs[1,0].xaxis.set_major_locator(ticker.FixedLocator(majors))
axs[1,1].xaxis.set_major_locator(ticker.FixedLocator(majors))
axs[1,2].xaxis.set_major_locator(ticker.FixedLocator(majors))
axs[1,3].xaxis.set_major_locator(ticker.FixedLocator(majors))
#
r1 = mpatches.Rectangle(xy=(1.75,0.2),width=-1.35,height=0.35, color='k', alpha=0.2)
r2 = mpatches.Rectangle(xy=(1.75,0.2),width=-1.35,height=0.35, color='k', alpha=0.2)
r3 = mpatches.Rectangle(xy=(1.75,0.2),width=-1.35,height=0.35, color='k', alpha=0.2)
r4 = mpatches.Rectangle(xy=(1.75,0.2),width=-1.35,height=0.35, color='k', alpha=0.2)
axs[1,0].text(1.5,0.25,'A',fontsize=28)
axs[1,1].text(1.5,0.25,'B',fontsize=28)
axs[1,2].text(1.5,0.25,'C',fontsize=28)
axs[1,3].text(1.5,0.25,'D',fontsize=28)
axs[1,0].add_patch(r1)
axs[1,1].add_patch(r2)
axs[1,2].add_patch(r3)
axs[1,3].add_patch(r4)
#
axs[0,0].set_ylabel('Host Distance $d$ [kpc]', fontsize=24)
axs[0,0].set_xlabel('Lookback time [Gyr]', fontsize=24)
axs[0,1].set_xlabel('Lookback time [Gyr]', fontsize=24)
axs[0,2].set_xlabel('Lookback time [Gyr]', fontsize=24)
axs[0,3].set_xlabel('Lookback time [Gyr]', fontsize=24)
#
axs[1,0].set_ylabel('$\\ell$ [10$^{4}$ kpc km s$^{-1}$]', fontsize=24)
axs[1,0].set_xlabel('Lookback time [Gyr]', fontsize=24)
axs[1,1].set_xlabel('Lookback time [Gyr]', fontsize=24)
axs[1,2].set_xlabel('Lookback time [Gyr]', fontsize=24)
axs[1,3].set_xlabel('Lookback time [Gyr]', fontsize=24)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0)
plt.savefig(directory+'/outlier_orbits_and_angs.pdf')


################################################################################


################################################################################


"""
    Figure 10:
        Outlier properties versus Mstar
"""
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
# Outlier sample, purple with scatter
mask_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) > 0)
# N > 1 sample, red with scatter
#mask_mult = (N_sim_tot > 1)
# Total - Outlier sample, dashed green no scatter
#mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0.05)
#mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0)
# N > 1 - Outlier sample, dashed red no scatter
mask_mult = (N_sim_tot > 1)
mask_mult_no_out = (np.abs((d_min_tot[mask_mult]-d_sim_tot[mask_mult])/d_sim_tot[mask_mult]) == 0)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#8b0000', '#3579DC', '#000000']
binedges = None
binsize = 0.5
binedges = (4.5, 9.5)
limits_1 = ((4.5,9.5),(0,12))
limits_2 = ((4.5,9.5),(0,165))
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [t_in_tot, t_in_tot[mask_out], t_in_tot[mask_mult][mask_mult_no_out]]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0']
ytype = ['t.infall.text','t.infall.text','t.infall.text']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.5)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Total Sample')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='$N_{\\rm peri}$ > 1, growing pericenters')
ax1.plot(10**(binss[2][:-1]+half_bins[2]), medians[2], color=colorss[2], markersize=10, alpha=0.7, label='$N_{\\rm peri}$ > 1, shrinking pericenters')
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
    ax3.set_yticklabels(axis_2_tick_labels, fontsize=28)
    ax3.set_ylim(limits_1[1])
    ax3.set_ylabel(axis_2_label, fontsize=28, labelpad=9)
    ax3.tick_params(pad=3)
#
############
#
N_sim_tot = summary.nperi(data_total, masks_infall_peri, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
# Outlier sample, purple with scatter
mask_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) > 0)
# N > 1 sample, red with scatter
#mask_mult = (N_sim_tot > 1)
# Total - Outlier sample, dashed green no scatter
#mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0.05)
# N > 1 - Outlier sample, dashed red no scatter
mask_mult = (N_sim_tot > 1)
mask_mult_no_out = (np.abs((d_min_tot[mask_mult]-d_sim_tot[mask_mult])/d_sim_tot[mask_mult]) == 0)
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [d_min_tot, d_min_tot[mask_out], d_min_tot[mask_mult][mask_mult_no_out]]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0']
ytype = ['d.peri.text','d.peri.text','d.peri.text']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.5)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Total Sample')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='Outliers')
ax2.plot(10**(binss[2][:-1]+half_bins[2]), medians[2], color=colorss[2], markersize=10, alpha=0.7, label='N > 1, no outliers')
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
ax1.set_ylabel('Infall lookback time [Gyr]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Min pericenter distance [kpc]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_xlabel('$M_{\\rm star} \ [M_{\\odot}]$', fontsize=28)
ax1.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
ax2.yaxis.set_major_locator(AutoLocator())
ax2.yaxis.set_minor_locator(MultipleLocator(5))
ax2.set_yticks([0,25,50,75,100,125,150])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#
plt.savefig(directory+'/outlier_props_vs_mstar_1.pdf')
#
############
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#8b0000', '#3579DC', '#000000']
binedges = None
binsize = 0.5
binedges = (4.5, 9.5)
limits_1 = ((4.5,9.5),(0,3.5))
limits_2 = ((4.5,9.5),(-4,0.5))
#
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
d_1st_tot = summary.dperi_first(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
# Outlier sample, purple with scatter
mask_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) > 0)
# N > 1 sample, red with scatter
#mask_mult = (N_sim_tot > 1)
# Total - Outlier sample, dashed green no scatter
#mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0.05)
# N > 1 - Outlier sample, dashed red no scatter
mask_mult = (N_sim_tot > 1)
mask_mult_no_out = (np.abs((d_min_tot[mask_mult]-d_sim_tot[mask_mult])/d_sim_tot[mask_mult]) == 0)
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [L_tot/1e4, (L_tot[mask_out])/1e4, (L_tot[mask_mult][mask_mult_no_out])/1e4]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0']
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax1.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.5)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Total Sample')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='Outliers')
ax1.plot(10**(binss[2][:-1]+half_bins[2]), medians[2], color=colorss[2], markersize=10, alpha=0.7, label='N > 1, no outliers')
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
############
#
# For the energy plots
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_energy', sim_type='baryon')
d_sim_tot = summary.dperi_recent(data_total, masks_infall, selection='sim', oversample=True, hosts='all_energy', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon')
potential_tot = summary.potential(data_potentials, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, masks_infall, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
# Outlier sample, purple with scatter
mask_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) > 0)
# N > 1 sample, red with scatter
#mask_mult = (N_sim_tot > 1)
# Total - Outlier sample, dashed green no scatter
#mask_no_out = (np.abs((d_min_tot-d_sim_tot)/d_sim_tot) < 0.05)
# N > 1 - Outlier sample, dashed red no scatter
mask_mult = (N_sim_tot > 1)
mask_mult_no_out = (np.abs((d_min_tot[mask_mult]-d_sim_tot[mask_mult])/d_sim_tot[mask_mult]) == 0)
#
x = [Mstar_z0_tot, Mstar_z0_tot[mask_out], Mstar_z0_tot[mask_mult][mask_mult_no_out]]
y = [(potential_tot+ke_z0_tot)/1e4, ((potential_tot+ke_z0_tot)/1e4)[mask_out], ((potential_tot+ke_z0_tot)/1e4)[mask_mult][mask_mult_no_out]]
#
xtype = ['M.star.z0', 'M.star.z0', 'M.star.z0']
ytype = ['E.tot','E.tot','E.tot']
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.5)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Total Sample')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='Outliers')
ax2.plot(10**(binss[2][:-1]+half_bins[2]), medians[2], color=colorss[2], markersize=10, alpha=0.7, label='N > 1, no outliers')
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
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#
plt.savefig(directory+'/outlier_props_vs_mstar_2.pdf')


################################################################################


################################################################################


"""
    Figure A1:
        Dynamics versus Mhalo
"""
data_total_all = summary.data_read(directory=sim_data.home_dir, hosts='all_no_z', sim_type='all_baryon')
data_potentials_all = summary.data_read_potential(directory=sim_data.home_dir, hosts='all_energy', sim_type='all_baryon')
masks_infall_all = summary.data_mask(data_total_all, peri_sim=False, peri_model=False, hosts='all_no_z')
masks_infall_all_peri = summary.data_mask(data_total_all, peri_sim=True, peri_model=False, hosts='all_no_z')
#
Mhalo_peak_tot_all = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon_all')
vz0_tot_all = summary.v_z0(data_total_all, masks_infall_all, oversample=True, hosts='all_no_z', sim_type='baryon_all')
L_tot_all = summary.L_z0(data_total_all, masks_infall_all, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon_all')
#
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5,16))
colorss = ['#6096e4', '#2a1da7', '#1d55a7']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8,11.5),(0,300))
limits_2 = ((8,11.5),(-5,0.5))
limits_3 = ((8,11.5),(0,4.5))
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.5)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], markersize=10, alpha=0.7)
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
potential_tot_all = summary.potential(data_potentials_all, masks_infall_all, oversample=True, hosts='all_energy', sim_type='baryon_all', norm='kinetic')
ke_z0_tot_all = summary.kinetic_energy(data_total_all, masks_infall_all, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon_all')
Mhalo_peak_tot_all = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='all_energy', sim_type='baryon_all')
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[2], alpha=0.5)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[2], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[2], markersize=10, alpha=0.7)
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
#
Mhalo_peak_tot_all = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon_all')
vz0_tot_all = summary.v_z0(data_total_all, masks_infall_all, oversample=True, hosts='all_no_z', sim_type='baryon_all')
L_tot_all = summary.L_z0(data_total_all, masks_infall_all, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon_all')
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
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.5)
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax3.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7)
#
ax3.set_xscale('log')
ax3.set_xlim(10**(limits_3[0][0]), 10**(limits_3[0][1]))
ax3.set_ylim(limits_3[1])
#
ax3.set_xlabel('$M_{\\rm halo,peak} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=26)
ax1.get_yaxis().set_label_coords(-0.1,0.5)
ax2.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
ax3.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
ax3.get_yaxis().set_label_coords(-0.1,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax3.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax3.set_xticks([1e8, 1e9, 1e10, 1e11])
ax3.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/dynamics_vs_mhalo.pdf')


################################################################################


################################################################################


"""
    Figure A2:
        Pericenter time and Infall time versus Mhalo
"""
t_in_tot_all = summary.first_infall(data_total_all, masks_infall_all, oversample=True, hosts='all_no_z', sim_type='baryon_all')
t_in_any_tot_all = summary.first_infall_any(data_total_all, masks_infall_all, oversample=True, hosts='all_no_z', sim_type='baryon_all')
Mhalo_peak_tot_all = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon_all')
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000000', '#ff4900']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8,11.5),(0,13.5))
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
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.6, label='MW-mass halo')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.6, label='Any halo')
#
ax1.hlines(y=12.84, xmin=10**(limits_1[0][0]), xmax=10**(limits_1[0][1]), colors='k', linestyles='dotted', alpha=0.5)
ax1.text(3*10**8, 12.1, 'End of Reionization', fontsize=20)
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
t_sim_tot_all = summary.tperi_recent(data_total_all, masks_infall_all_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon_all')
t_min_tot_all = summary.tperi_min(data_total_all, masks_infall_all_peri, oversample=True, hosts='all_no_z', sim_type='baryon_all')
Mhalo_peak_tot_all = summary.mhalo(data_total_all, masks_infall_all_peri, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon_all')
#
x = [Mhalo_peak_tot_all, Mhalo_peak_tot_all]
y = [t_sim_tot_all, t_min_tot_all]
#
xtype = ['M.halo.peak', 'M.halo.peak']
ytype = ['t.peri.text','t.peri.text']
colorss = ['#000080', '#006400']

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
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum')
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
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
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
plt.savefig(directory+'/times_vs_mhalo.pdf')


################################################################################


################################################################################


"""
    Figure A3:
        Pericenter number versus Mhalo
"""
N_sim_tot_all = summary.nperi(data_total_all, masks_infall_all, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon_all')
Mhalo_peak_tot_all = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='all_no_z', sim_type='baryon_all')
#
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#A02422']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8,11.5),(0,4))
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
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[0], alpha=0.5)
ax1.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[0], alpha=0.3)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[0], markersize=10, alpha=0.7)
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
ax1.set_xlabel('$M_{\\rm halo,peak} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Pericenter Number', fontsize=30)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax1.set_xticks([1e8, 1e9, 1e10, 1e11])
ax1.set_yticks([0,1,2,3,4])
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/nperi_vs_mhalo.pdf')


################################################################################


################################################################################


"""
    Figure B1:
        DMO versus Baryonic comparison
"""
data_total_all = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='all_baryon')
masks_infall_all = summary.data_mask(data_total_all, peri_sim=False, peri_model=False, hosts='iso_no_z')
masks_infall_all_peri = summary.data_mask(data_total_all, peri_sim=True, peri_model=False, hosts='iso_no_z')
#
data_total_dmo = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='dmo')
masks_infall_dmo = summary.data_mask(data_total_dmo, peri_sim=False, peri_model=False, hosts='iso_no_z')
masks_infall_dmo_peri = summary.data_mask(data_total_dmo, peri_sim=True, peri_model=False, hosts='iso_no_z')
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#2b2031', '#AB0B23']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8, 11.5),(0,13))
limits_2 = ((8, 11.5),(0,145))
#
t_in_any_tot = summary.first_infall_any(data_total_all, masks_infall_all, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
t_in_any_tot_dmo = summary.first_infall_any(data_total_dmo, masks_infall_dmo, oversample=True, hosts='iso_no_z', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, masks_infall_dmo, selection='peak', oversample=True, hosts='iso_no_z', sim_type='dmo')
x = [Mhalo_peak_tot, Mhalo_peak_tot_dmo]
y = [t_in_any_tot, t_in_any_tot_dmo]
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
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Baryonic')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='DMO')
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
    ax3.set_ylabel(axis_3_label, fontsize=28, labelpad=9)
    ax3.tick_params(pad=3)
#
d_min_tot = summary.dperi_min(data_total_all, masks_infall_all_peri, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total_all, masks_infall_all_peri, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
d_min_tot_dmo = summary.dperi_min(data_total_dmo, masks_infall_dmo_peri, oversample=True, hosts='iso_no_z', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, masks_infall_dmo_peri, selection='peak', oversample=True, hosts='iso_no_z', sim_type='dmo')
x = [Mhalo_peak_tot, Mhalo_peak_tot_dmo]
y = [d_min_tot, d_min_tot_dmo]
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
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Baryonic')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='DMO')
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
ax2.set_xlabel('$M_{\\rm halo,peak} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('First infall lookback time [Gyr]', fontsize=22)
ax1.get_yaxis().set_label_coords(-0.11,0.5)
ax2.set_ylabel('Min pericenter distance [kpc]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.11,0.5)
ax1.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax1.yaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
ax1.set_yticks([0,2,4,6,8,10,12])
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e8, 1e9, 1e10, 1e11])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/dmo_1.pdf')
#
###
#
Mhalo_peak_tot = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
L_tot = summary.L_z0(data_total_all, masks_infall_all, selection='sim', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, masks_infall_dmo, selection='peak', oversample=True, hosts='iso_no_z', sim_type='dmo')
L_tot_dmo = summary.L_z0(data_total_dmo, masks_infall_dmo, selection='sim', oversample=True, hosts='iso_no_z', sim_type='dmo')
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8, 11.5),(0,3))
limits_2 = ((8, 11.5),(0,7.5))
#
x = [Mhalo_peak_tot, Mhalo_peak_tot_dmo]
y = [L_tot/1e4, L_tot_dmo/1e4]
#
xtype = ['M.halo.peak', 'M.halo.peak']
ytype = ['L.tot','L.tot']
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
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Baryonic')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='DMO')
#
ax1.set_xscale('log')
ax1.set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
ax1.set_ylim(limits_1[1])
#
N_sim_tot = summary.nperi(data_total_all, masks_infall_all, oversample=True, selection='sim', hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
N_sim_tot_dmo = summary.nperi(data_total_dmo, masks_infall_dmo, oversample=True, selection='sim', hosts='iso_no_z', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, masks_infall_dmo, selection='peak', oversample=True, hosts='iso_no_z', sim_type='dmo')
x = [Mhalo_peak_tot, Mhalo_peak_tot_dmo]
y = [N_sim_tot, N_sim_tot_dmo]
#
xtype = ['M.halo.peak', 'M.halo.peak']
ytype = ['N.peri.text','N.peri.text']
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
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
ax2.fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.7, label='Baryonic')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.7, label='DMO')
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
#
ax2.set_xlabel('$M_{\\rm halo,peak} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('$\\ell$ [kpc km s$^{-1}$]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.1,0.5)
ax2.set_ylabel('Pericenter number', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27, labelbottom=False)
ax1.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
ax2.set_xticks([1e8, 1e9, 1e10, 1e11])
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=27)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/dmo_2.pdf')
