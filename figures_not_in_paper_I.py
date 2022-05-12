
"""
    These are all figures that were either removed from the paper, or figures
    that I made as checks for my analysis.
"""

ang_before_min = np.array([6942.83, 15696.42, 16544.51, 16155.70, 13106.67, 14827.38, 2732.80, 5296.33, 3420.01, 1490.72, 2810.88])
ang_after_min = np.array([6863.16, 14647.43, 15272.48, 13108.21, 11702.31, 13221.56, 3862.71, 4741.37, 3529.65, 2248.67, 1986.19])
mass_before_min = np.array([1.06e08, 6.60e08, 1.00e09, 1.52e09, 7.44e09, 1.79e09, 1.44e08, 1.61e09, 2.07e08, 5.62e07, 2.65e09])
mass_after_min = np.array([9.70e07, 6.51e08, 8.46e08, 1.40e09, 7.24e09, 2.20e09, 2.30e08, 1.42e09, 2.03e08, 5.29e07, 2.42e09])
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
colorss = ['#000080', '#006400']
#
# PLOTTING
x1 = np.ones(len(ang_before_min))
x2 = 2*x1
x3 = np.array([x1,x2])
data = np.array([ang_before_min/1e4, ang_after_min/1e4])
data2 = np.array([mass_before_min, mass_after_min])
ax1.scatter(x1, ang_before_min/1e4, s=30, color='b', alpha=0.5)
ax1.scatter(x2, ang_after_min/1e4, s=30, color='k', alpha=0.5)
ax1.plot(x3, data, color='k', alpha=0.3)
#
ax2.scatter(x1, mass_before_min, s=30, color='b', alpha=0.5)
ax2.scatter(x2, mass_after_min, s=30, color='k', alpha=0.5)
ax2.plot(x3, data2, color='k', alpha=0.3)
#
my_xticks = ['200 Myr before','200 Myr after']
ax1.set_xticks(np.array([1, 2]), my_xticks, rotation=0)
ax2.set_xticks(np.array([1, 2]), my_xticks, rotation=0)
ax1.set_ylabel('$L_{\\rm tot}$ [10$^4$ kpc km s$^{-1}$]', fontsize=22)
ax2.set_ylabel('$M_{\\rm 200m,sat}$ [$M_{\\odot}$]', fontsize=24)
ax1.set_xlim(0.5, 2.5)
ax1.set_ylim(0, 1.9)
ax2.set_xlim(0.5, 2.5)
ax2.set_ylim(3e7, 2e10)
ax2.set_yscale('log')
ax1.tick_params(axis='both', which='both', labelsize=16, labelbottom=False)
ax2.tick_params(axis='both', which='both', labelsize=16)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/before_and_after_dmin.pdf')



ang_rec = np.array([7117.44, 17233.41, 17683.15, 19841.72, 16320.29, 17488.14, 13663.38, 15630.56, 11433.86, 12121.72, 18266.92])
ang_min = np.array([7114.41, 15088.30, 15951.40, 14094.80, 11386.72, 14638.49, 3773.35, 5246.58, 3246.79, 2113.77, 1800.51])
times = [2.09, 6.64, 3.85, 7.95, 6.81, 6.85, 10.76, 10.13, 7.59, 5.91, 4.36]
#
f, ax1 = plt.subplots(1, 1, figsize=(8,6))
#
# PLOTTING
sc = ax1.scatter(ang_rec/1e4, ang_min/1e4, s=50, c=times, cmap=plt.cm.inferno, alpha=0.9)
#
ax1.set_xlabel('$L_{\\rm tot}$ (at $d_{\\rm peri,recent}$)', fontsize=22)
ax1.set_ylabel('$L_{\\rm tot}$ (at $d_{\\rm peri,min}$)', fontsize=22)
ax1.plot([0,2], [0,2], linestyle='--', color='k', alpha=0.3)
ax1.set_xlim(0.5, 2.1)
ax1.set_ylim(0, 1.75)
cb = plt.colorbar(sc)
cb.set_label('$t_{\\rm peri,min} - t_{\\rm peri,recent}$ [Gyr]', fontsize=16)
cb.ax.tick_params(labelsize=12)
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/L_tot_comparison.pdf')



"""
    Figure A1:
        OLD
        Infall time versus Mstar, for Isolated versus Paired satellites
"""
### Generate all of the data for the plots below
data_total_iso = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='baryon')
masks_infall_iso = summary.data_mask(data_total_iso, peri_sim=False, peri_model=False, hosts='iso_no_z')
masks_infall_iso['m12f'][57] = False
#
t_in_iso = summary.first_infall(data_total_iso, masks_infall_iso, oversample=True, hosts='iso_no_z', sim_type='baryon')
Mstar_z0_iso = summary.mstar(data_total_iso, masks_infall_iso, selection='z0', oversample=True, hosts='iso_no_z', sim_type='baryon')
#
data_total_lg = summary.data_read(directory=sim_data.home_dir, hosts='lg', sim_type='baryon')
masks_infall_lg = summary.data_mask(data_total_lg, peri_sim=False, peri_model=False, hosts='lg')
#
t_in_lg = summary.first_infall(data_total_lg, masks_infall_lg, oversample=True, hosts='lg', sim_type='baryon')
Mstar_z0_lg = summary.mstar(data_total_lg, masks_infall_lg, selection='z0', oversample=True, hosts='lg', sim_type='baryon')
#
summary_plot.median_plot_mult(x=[Mstar_z0_iso, Mstar_z0_lg], y=[t_in_iso, t_in_lg], xtype=['M.star.z0','M.star.z0'], ytype=['t.infall.text','t.infall.text'], labels=['Isolated', 'Paired'], binsize=1, binedges=(4,10), limits=((4,9.5),(0,11)), file_path_and_name=directory+'/iso_vs_lg_infall.pdf')




"""
    Figure 3:
        Pericenter time and Infall time vs Mstar
"""
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_any_tot = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
f, axs = plt.subplots(2, 2, figsize=(24,16))
colorss = ['#000080', '#006400']
binedges = (4.5, 9.5)
binsize = 0.5
limits_1 = ((4,9.5),(0,13.8))
limits_2 = ((4,9.5),(0,8.8))
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
axs[0,0].fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
axs[0,0].fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
axs[0,0].fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
axs[0,0].fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
axs[0,0].plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='MW/M31-mass halo')
axs[0,0].plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Any halo')
#
axs[0,0].set_xscale('log')
axs[0,0].set_xlim(10**(limits_1[0][0]), 10**(limits_1[0][1]))
axs[0,0].set_ylim(limits_1[1])
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
    ax3 = axs[0,0].twinx()
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
axs[1,0].fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
axs[1,0].fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
axs[1,0].fill_between(10**(binss[1][:-1]+half_bins[1]), uppers[1], lowers[1], color=colorss[0], alpha=0.3)
axs[1,0].fill_between(10**(binss[1][:-1]+half_bins[1]), highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
axs[1,0].plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Recent')
axs[1,0].plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='Minimum')
#
axs[1,0].set_xscale('log')
axs[1,0].set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
axs[1,0].set_ylim(limits_2[1])
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
    ax4 = axs[1,0].twinx()
    ax4.set_xscale('log')
    ax4.set_yscale('linear')
    ax4.set_yticks(axis_4_tick_locations)
    ax4.set_yticklabels(axis_4_tick_labels, fontsize=24)
    ax4.set_ylim(limits_2[1])
    ax4.set_ylabel(axis_4_label, labelpad=9)
    ax4.tick_params(pad=3)
#
axs[1,0].set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=32)
axs[0,0].set_ylabel('Infall Lookback Time [Gyr]', fontsize=30)
axs[0,0].get_yaxis().set_label_coords(-0.075,0.5)
axs[1,0].set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=28)
axs[1,0].get_yaxis().set_label_coords(-0.075,0.5)
axs[1,0].legend(prop={'size': 26}, loc='best')
axs[0,0].legend(prop={'size': 26}, loc='best')
axs[1,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26)
axs[1,0].xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
axs[0,0].xaxis.set_major_locator(LogLocator(base=10))
axs[0,0].xaxis.set_minor_locator(LogLocator(base=10,subs=[2,3,4,5,6,7,8,9]))
axs[0,0].set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
axs[0,0].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=26, labelbottom=False)
"""
        Infall time versus d(z = 0), binned by Mstar
"""
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_any_tot = summary.first_infall_any(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
binedges=None
binsize = 50
limits_1 = ((0, 400),(0,13.5))
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
axs[0,1].fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
axs[0,1].fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
axs[0,1].fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
axs[0,1].fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
axs[0,1].plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, alpha=0.5, label='MW-mass halo')
axs[0,1].plot(binss[4][:-1]+half_bins[4], medians[4], color=colorss[0], markersize=10, alpha=0.5, label='Any halo')
#
axs[0,1].set_xscale('linear')
axs[0,1].set_xlim(limits_1[0][0], limits_1[0][1])
axs[0,1].set_ylim(limits_1[1])
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
    ax3 = axs[0,1].twinx()
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.set_yticks(axis_3_tick_locations)
    ax3.set_yticklabels(axis_3_tick_labels, fontsize=26)
    ax3.set_ylim(limits_1[1])
    ax3.set_ylabel(axis_3_label, labelpad=9)
    ax3.tick_params(pad=3)
#
axs[0,1].set_xlabel('Host distance $d$ [kpc]', fontsize=32)
axs[0,1].set_ylabel(' ', fontsize=24)
axs[0,1].legend(prop={'size': 26}, loc='best')
axs[0,1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
axs[1,1].axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace=0.18, hspace=0)
#plt.show()
plt.savefig(directory+'/times.pdf')







"""
    Figure 2:
        Dynamics versus Mstar
"""
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
#f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5,16))
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5,16))
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
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5)
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
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5)
#
ax2.set_xscale('log')
ax2.set_xlim(10**(limits_2[0][0]), 10**(limits_2[0][1]))
ax2.set_ylim(limits_2[1])
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
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax3.fill_between(10**(binss[0][:-1]+half_bins[0]), highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax3.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5)
#
ax3.set_xscale('log')
ax3.set_xlim(10**(limits_3[0][0]), 10**(limits_3[0][1]))
ax3.set_ylim(limits_3[1])
#
ax3.set_xlabel('$M_{\\rm star} [M_{\\odot}]$', fontsize=28)
ax1.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.14,0.5)
ax2.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.14,0.5)
ax3.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=24)
ax3.get_yaxis().set_label_coords(-0.14,0.5)
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
plt.savefig(directory+'/dynamics_vs_mstar.pdf')


################################################################################


################################################################################

"""
    Figure 3:
        Dynamics vs d(z = 0), binned by Mstar
"""
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5,16))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
limits_1 = ((0,400),(0,350))
limits_2 = ((0,400),(0,4.5))
limits_3 = ((0,400),(-6,0.5))
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
potential_tot = summary.potential(data_potentials, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, masks_infall, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_energy', sim_type='baryon')
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
ax3.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax3.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax3.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5)
ax3.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], linestyle='--', markersize=10, alpha=0.5)
#
ax3.set_xscale('linear')
ax3.set_xlim(limits_3[0])
ax3.set_ylim(limits_3[1])
#
ax3.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax1.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.14,0.5)
ax2.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.14,0.5)
ax3.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=28)
ax3.get_yaxis().set_label_coords(-0.14,0.5)
ax1.legend(prop={'size': 24}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax3.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/dynamics_vs_dz0.pdf')


################################################################################


################################################################################


"""
    Figure 5:
        Dynamics vs Infall time
"""
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
vz0_tot = summary.v_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
L_tot = summary.L_z0(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
#
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5,16))
colorss = ['#000080', '#006400']
binedges = None
binsize = 1
limits_1 = ((0,13),(0,300))
limits_2 = ((0,13),(0,4.5))
limits_3 = ((0,13),(-4,0.5))
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.5)
#
ax1.set_xlim(limits_1[0])
ax1.set_ylim(limits_1[1])
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
ax2.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.5)
#
ax2.set_xlim(limits_2[0])
ax2.set_ylim(limits_2[1])
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
ax3.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax3.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax3.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.5)
#
ax3.set_xlim(limits_3[0])
ax3.set_ylim(limits_3[1])
#
ax3.set_xlabel('Infall lookback time [Gyr]', fontsize=28)
ax1.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.14,0.5)
ax2.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=24)
ax2.get_yaxis().set_label_coords(-0.14,0.5)
ax3.set_ylabel('$E$ [10$^4$ km$^2$ s$^{-2}$]', fontsize=24)
ax3.get_yaxis().set_label_coords(-0.14,0.5)
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24, labelbottom=False)
ax3.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/dynamics_vs_t_infall.pdf')



###################################################################


"""
    Figure 5 :
        Pericenter distance and number versus Mstar.
"""
d_sim_tot = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_z', sim_type='baryon')
d_min_tot = summary.dperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall_peri, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = (4.5, 9.5)
binsize = 0.5
limits = ((4,9.5),(0,200))
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
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
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
plt.savefig(directory+'/peri_dn_vs_mstar.pdf')


################################################################################


################################################################################


"""
    Figure 6:
        Pericenter number vs d(z = 0), binned by Mstar
"""
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, masks_infall, selection='z0', oversample=True, hosts='all_no_z', sim_type='baryon')
#
mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, ax1 = plt.subplots(1, 1, figsize=(10,8))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
limits = ((0,400),(0,6.5))
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5, label='$M_{\\rm star} < 10^7 M_{\\odot}$')
ax1.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], linestyle='--', markersize=10, alpha=0.5, label='$M_{\\rm star} > 10^7 M_{\\odot}$')
#
ax1.set_xscale('linear')
ax1.set_xlim(limits[0])
ax1.set_ylim(limits[1])
#
ax1.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax1.set_ylabel('Pericenter Number', fontsize=28)
ax1.legend(prop={'size': 20}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/nperi_vs_dz0.pdf')


################################################################################


################################################################################


"""
    Figure 7:
        Pericenter distance and number versus t_infall
"""
d_min_tot = summary.dperi_min(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
d_1st_tot = summary.dperi_first(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall_peri, oversample=True, hosts='all_no_z', sim_type='baryon')
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = None
binsize = 1
limits_1 = ((0,13),(0,200))
limits_2 = ((0,13),(-0.5,9.8))
#
x = [t_in_tot, t_in_tot]
y = [d_min_tot, d_1st_tot]
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
ax1.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax1.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax1.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Minimum')
ax1.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[0], markersize=10, alpha=0.5, label='First')
#
ax1.set_xlim(limits_1[0])
ax1.set_ylim(limits_1[1])
#
N_sim_tot = summary.nperi(data_total, masks_infall, oversample=True, selection='sim', hosts='all_no_z', sim_type='baryon')
t_in_tot = summary.first_infall(data_total, masks_infall, oversample=True, hosts='all_no_z', sim_type='baryon')
x = [t_in_tot]
y = [N_sim_tot]
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
ax2.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
ax2.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
ax2.plot(binss[0][:-1]+half_bins[0], medians[0], color=colorss[1], markersize=10, alpha=0.5)
#
ax2.set_xlim(limits_2[0])
ax2.set_ylim(limits_2[1])
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
    ax3.set_xticklabels(axis_3_tick_labels, fontsize=24)
    ax3.set_xlim(limits_1[0])
    ax3.set_xlabel(axis_3_label, labelpad=9)
    ax3.tick_params(pad=3)
#
ax2.set_xlabel('Infall lookback time [Gyr]', fontsize=28)
ax1.set_ylabel('Pericenter distance [kpc]', fontsize=28)
ax1.get_yaxis().set_label_coords(-0.09,0.5)
ax2.set_ylabel('Pericenter Number', fontsize=28)
ax2.get_yaxis().set_label_coords(-0.09,0.5)
ax1.legend(prop={'size': 24}, loc='best')
ax1.tick_params(axis='both', which='both', bottom=True, top=False, labelsize=24, labelbottom=False)
ax2.tick_params(axis='both', which='both', bottom=True, top=True, labelsize=24)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
plt.savefig(directory+'/peri_dn_vs_t_infall.pdf')



################################################################################



d_sim_tot = summary.dperi_recent(data_total_all, masks_infall_all_peri, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total_all, masks_infall_all_peri, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
d_sim_tot_dmo = summary.dperi_recent(data_total_dmo, masks_infall_dmo_peri, oversample=True, hosts='iso_no_z', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, masks_infall_dmo_peri, selection='peak', oversample=True, hosts='iso_no_z', sim_type='dmo')
#
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5,12))
colorss = ['#000080', '#006400']
binedges = (8,11.5)
binsize = 0.5
limits_1 = ((8,11.5),(0,150))
limits_2 = ((8,11.5),(0,200))
#
x = [Mhalo_peak_tot, Mhalo_peak_tot_dmo]
y = [d_sim_tot, d_sim_tot_dmo]
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
ax1.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Baryon')
ax1.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='DMO')
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
d_min_tot = summary.dperi_min(data_total_all, masks_infall_all, oversample=True, hosts='iso_no_z', sim_type='baryon_all')
Mhalo_peak_tot = summary.mhalo(data_total_all, masks_infall_all, selection='peak', oversample=True, hosts='iso_no_z', sim_type='baryon_all')
d_min_tot_dmo = summary.dperi_min(data_total_dmo, masks_infall_dmo, oversample=True, hosts='iso_no_z', sim_type='dmo')
Mhalo_peak_tot_dmo = summary.mhalo(data_total_dmo, masks_infall_dmo, selection='peak', oversample=True, hosts='iso_no_z', sim_type='dmo')
#
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
ax2.plot(10**(binss[0][:-1]+half_bins[0]), medians[0], color=colorss[1], markersize=10, alpha=0.5, label='Baryon')
ax2.plot(10**(binss[1][:-1]+half_bins[1]), medians[1], color=colorss[0], markersize=10, alpha=0.5, label='DMO')
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
ax1.set_ylabel('Recent', fontsize=24)
ax1.get_yaxis().set_label_coords(-0.075,0.5)
ax2.set_ylabel('Minimum', fontsize=24)
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
plt.savefig(directory+'/dperi_dmo.pdf')


















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
