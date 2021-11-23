


"""
    Plotting properties vs d(z = 0) in different mass bins
        - This is for the baryonic simulation satellites
"""

mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, ax = plt.subplots(figsize=(11, 8))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
limits = ((0,400),(0,350))
#
#x = [dz0_tot, dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high], dz0_tot[mass_low], dz0_tot[mass_high]]
x = [dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high]]
#x = [t_in_tot, t_in_tot[mass_low], t_in_tot[mass_high]]
#y = [d_sim_tot, d_min_tot, d_sim_tot[mass_low], d_sim_tot[mass_high], d_min_tot[mass_low], d_min_tot[mass_high]]
#y = [t_sim_tot, t_min_tot, t_sim_tot[mass_low], t_sim_tot[mass_high], t_min_tot[mass_low], t_min_tot[mass_high]]
#y = [t_in_tot, t_in_any_tot, t_in_tot[mass_low], t_in_tot[mass_high], t_in_any_tot[mass_low], t_in_any_tot[mass_high]]
#y = [N_sim_tot, N_sim_tot[mass_low], N_sim_tot[mass_high]]
#y = [vz0_tot, vz0_tot[mass_low], vz0_tot[mass_high]]
#y = [L_tot/1e4, (L_tot/1e4)[mass_low], (L_tot/1e4)[mass_high]]
y = [vtan_tot, vtan_tot[mass_low], vtan_tot[mass_high]]
#
#xtype = ['d.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0']
#ytype = ['t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text','t.infall.text']
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
        #med[i] = np.nanmedian(y[j][mask])
        med[i] = np.nanmean(y[j][mask])
        scatter[i] = np.nanstd(y[j][mask])
        #upper[i] = np.nanpercentile(y[j][mask], onesigp)
        #lower[i] = np.nanpercentile(y[j][mask], onesigm)
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
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
#plt.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
#plt.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#plt.fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[0], alpha=0.3)
#plt.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
#plt.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, alpha=0.5, label='MW/M31-mass halo') # Recent, M < 1e7
#plt.plot(binss[3][:-1]+half_bins[3], medians[3], color=colorss[1], markersize=10, linestyle='--', alpha=0.5) #, label='MW/M31-mass host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Recent M > 1e7
#plt.plot(binss[4][:-1]+half_bins[4], medians[4], color=colorss[0], markersize=10, alpha=0.5, label='Any halo') # minimim, M < 1e7
#plt.plot(binss[5][:-1]+half_bins[5], medians[5], color=colorss[0], markersize=10, linestyle='--', alpha=0.5) #, label='Any host ($M_{\\rm star} > 10^{7} M_{\\odot}$)') # Minimum M > 1e7
#
plt.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
plt.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
plt.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5, label='$M_{\\rm star} < 10^{7} M_{\\odot}$')
plt.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, linestyle='--', alpha=0.5, label='$M_{\\rm star} > 10^{7} M_{\\odot}$')
#
plt.xlim(limits[0])
plt.ylim(limits[1])
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
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.set_yticks(axis_2_tick_locations)
    ax2.set_yticklabels(axis_2_tick_labels, fontsize=28)
    ax2.set_ylim(limits[1])
    ax2.set_ylabel(axis_2_label, labelpad=9)
    ax2.tick_params(pad=3)
if 't.' in xtype:
    # Instantiate the cosmology class and run this method first to set up scalefactors
    cc = ut.cosmology.CosmologyClass()
    red = np.array([0, 1])
    cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
    #
    axis_2_label = 'redshift'
    axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
    axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
    axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
    ax2 = ax.twiny()
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.set_xticks(axis_2_tick_locations)
    ax2.set_xticklabels(axis_2_tick_labels, fontsize=28)
    ax2.set_xlim(limits[0])
    ax2.set_xlabel(axis_2_label, labelpad=9)
    ax2.tick_params(pad=3)
ax.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
#ax.set_xlabel('Infall Lookback Time [Gyr]', fontsize=28)
#ax.set_ylabel('Pericenter Distance [kpc]', fontsize=28)
#ax.set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=28)
#ax.set_ylabel('Infall Lookback Time [Gyr]', fontsize=28)
#ax.set_ylabel('Pericenter Number', fontsize=28)
#ax.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=28)
#ax.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
ax.set_ylabel('Tangential velocity [km s$^{-1}$]', fontsize=28)
#ax.legend(prop={'size': 20}, loc='best')
ax.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.show()




############
############
"""
    Plotting E vs d(z = 0)
"""
potential_tot = summary.potential(data_potentials, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon', norm='kinetic')
ke_z0_tot = summary.kinetic_energy(data_total, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon')
#
Mhalo_peak_tot = summary.mhalo(data_total, mask_selection, selection='peak', oversample=True, hosts='all_energy', sim_type='baryon')
Mstar_z0_tot = summary.mstar(data_total, mask_selection, selection='z0', oversample=True, hosts='all_energy', sim_type='baryon')
dz0_tot = summary.d_z0(data_total, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon')

###

mass_high = (Mstar_z0_tot > 10**(7))
mass_low = (Mstar_z0_tot < 10**(7))
#
f, ax = plt.subplots(figsize=(11, 8))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
#
x = [dz0_tot, dz0_tot[mass_low], dz0_tot[mass_high]]
y = [(potential_tot+ke_z0_tot)/1e4, ((potential_tot+ke_z0_tot)/1e4)[mass_low], ((potential_tot+ke_z0_tot)/1e4)[mass_high]]
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
    #twosigp = 97.72
    #twosigm = 2.28
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
        #
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
plt.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
plt.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
plt.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5, label='$M_{\\rm star} < 10^{7} M_{\\odot}$')
plt.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, linestyle='--', alpha=0.5, label='$M_{\\rm star} > 10^{7} M_{\\odot}$')
#
plt.xlim(0, 400)
plt.ylim(-6, 1)
ax.set_xlabel('Host distance $d$ [kpc]', fontsize=28)
ax.set_ylabel('$E$ [10$^4$ km$^{2}$ s$^{-2}$]', fontsize=28)
#ax.legend(prop={'size': 20}, loc='lower right')
ax.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.show()

############################################################################
############################################################################
############################################################################

"""
    Plotting properties vs d(z = 0) in different mass bins
        - This is for ALL satellites in the baryonic sims, light and dark
"""

mass_high = (Mhalo_peak_tot_all > 10**(9.5))
mass_low = (Mhalo_peak_tot_all < 10**(9.5))
#
f, ax = plt.subplots(figsize=(11, 8))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
limits = ((0,400),(0,4))
#
#x = [dz0_tot_all, dz0_tot_all, dz0_tot_all[mass_low], dz0_tot_all[mass_high], dz0_tot_all[mass_low], dz0_tot_all[mass_high]]
x = [dz0_tot_all, dz0_tot_all[mass_low], dz0_tot_all[mass_high]]
#y = [d_sim_tot_all, d_min_tot_all, d_sim_tot_all[mass_low], d_sim_tot_all[mass_high], d_min_tot_all[mass_low], d_min_tot_all[mass_high]]
#y = [t_sim_tot_all, t_min_tot_all, t_sim_tot_all[mass_low], t_sim_tot_all[mass_high], t_min_tot_all[mass_low], t_min_tot_all[mass_high]]
#y = [t_in_tot_all, t_in_any_tot_all, t_in_tot_all[mass_low], t_in_tot_all[mass_high], t_in_any_tot_all[mass_low], t_in_any_tot_all[mass_high]]
#y = [N_sim_tot_all, N_sim_tot_all[mass_low], N_sim_tot_all[mass_high]]
#y = [vz0_tot_all, vz0_tot_all[mass_low], vz0_tot_all[mass_high]]
y = [L_tot_all/1e4, (L_tot_all/1e4)[mass_low], (L_tot_all/1e4)[mass_high]]
#
#xtype = ['d.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0', 'd.z0']
#ytype = ['t.infall.text', 't.infall.text', 't.infall.text', 't.infall.text', 't.infall.text', 't.infall.text']
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
    twosigp = 97.72
    twosigm = 2.28
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
        #
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
#plt.fill_between(binss[1][:-1]+half_bins[1], uppers[1], lowers[1], color=colorss[1], alpha=0.3)
#plt.fill_between(binss[1][:-1]+half_bins[1], highests[1], lowests[1], color=colorss[1], alpha=0.15)
#plt.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[0], alpha=0.3)
#plt.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[0], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
#plt.plot(binss[4][:-1]+half_bins[4], medians[4], color=colorss[1], markersize=10, alpha=0.5, label='Any halo ($M_{\\rm star} < 10^{7} M_{\\odot}$)')
#plt.plot(binss[5][:-1]+half_bins[5], medians[5], color=colorss[1], markersize=10, linestyle='--', alpha=0.5, label='Any halo ($M_{\\rm star} > 10^{7} M_{\\odot}$)')
#plt.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[0], markersize=10, alpha=0.5, label='MW/M31-mass halo ($M_{\\rm star} < 10^{7} M_{\\odot}$)')
#plt.plot(binss[3][:-1]+half_bins[3], medians[3], color=colorss[0], markersize=10, linestyle='--', alpha=0.5, label='MW/M31-mass halo ($M_{\\rm star} > 10^{7} M_{\\odot}$)')
#
#
plt.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
plt.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
plt.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5, label='$M_{\\rm star} < 10^{7} M_{\\odot}$')
plt.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, linestyle='--', alpha=0.5, label='$M_{\\rm star} > 10^{7} M_{\\odot}$')
#
plt.xlim(limits[0])
plt.ylim(limits[1])
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
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.set_yticks(axis_2_tick_locations)
    ax2.set_yticklabels(axis_2_tick_labels, fontsize=28)
    ax2.set_ylim(limits[1])
    ax2.set_ylabel(axis_2_label, labelpad=9)
    ax2.tick_params(pad=3)
ax.set_xlabel('d(z = 0) [kpc]', fontsize=28)
#ax.set_ylabel('Pericenter Distance [kpc]', fontsize=28)
#ax.set_ylabel('Pericenter Lookback Time [Gyr]', fontsize=28)
#ax.set_ylabel('Infall Lookback Time [Gyr]', fontsize=28)
#ax.set_ylabel('Pericenter Number', fontsize=28)
#ax.set_ylabel('$v(z = 0)$ [km s$^{-1}$]', fontsize=28)
ax.set_ylabel('$L(z = 0)$ [10$^4$ kpc km s$^{-1}$]', fontsize=28)
ax.legend(prop={'size': 18}, loc='best')
ax.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.show()


########
########
"""
    Plotting E vs d(z = 0)
"""
potential_tot_all = summary.potential(data_potentials_all, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon_all', norm='kinetic')
ke_z0_tot_all = summary.kinetic_energy(data_total_all, mask_selection, ke_type='z0', oversample=True, hosts='all_energy', sim_type='baryon_all')
#
Mhalo_peak_tot_all = summary.mhalo(data_total_all, mask_selection, selection='peak', oversample=True, hosts='all_energy', sim_type='baryon_all')
dz0_tot_all = summary.d_z0(data_total_all, mask_selection, oversample=True, hosts='all_energy', sim_type='baryon_all')

###

mass_high = (Mhalo_peak_tot_all > 10**(9.5))
mass_low = (Mhalo_peak_tot_all < 10**(9.5))
#
f, ax = plt.subplots(figsize=(11, 8))
colorss = ['#000080', '#006400']
binedges = None
binsize = 50
#
x = [dz0_tot_all, dz0_tot_all[mass_low], dz0_tot_all[mass_high]]
y = [(potential_tot_all+ke_z0_tot_all)/1e4, ((potential_tot_all+ke_z0_tot_all)/1e4)[mass_low], ((potential_tot_all+ke_z0_tot_all)/1e4)[mass_high]]
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
    twosigp = 97.72
    twosigm = 2.28
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
        #
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
plt.fill_between(binss[0][:-1]+half_bins[0], uppers[0], lowers[0], color=colorss[1], alpha=0.3)
plt.fill_between(binss[0][:-1]+half_bins[0], highests[0], lowests[0], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
plt.plot(binss[1][:-1]+half_bins[1], medians[1], color=colorss[1], markersize=10, alpha=0.5, label='$M_{\\rm star} < 10^{7} M_{\\odot}$')
plt.plot(binss[2][:-1]+half_bins[2], medians[2], color=colorss[1], markersize=10, linestyle='--', alpha=0.5, label='$M_{\\rm star} > 10^{7} M_{\\odot}$')
#
plt.xlim(0, 400)
plt.ylim(-6, 1)
ax.set_xlabel('d(z = 0) [kpc]', fontsize=28)
ax.set_ylabel('$E(z = 0)$ [10$^4$ km$^{2}$ s$^{-2}$]', fontsize=28)
ax.legend(prop={'size': 18}, loc='best')
ax.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.show()




######################################################################
######################################################################
######################################################################

"""
    Plotting the Mstar - Mhalo relation for early and late infalling satellites
"""
# Simple median selection
# median infall time ~ 6.9 Gyr
mask_early = (t_in_tot > np.median(t_in_tot))
mask_late = (t_in_tot < np.median(t_in_tot))

summary_plot.median_plot_mult(x=[Mhalo_peak_tot[mask_early], Mhalo_peak_tot[mask_late]], y=[Mstar_z0_tot[mask_early], Mstar_z0_tot[mask_late]], xtype=['M.halo.peak', 'M.halo.peak'], ytype=['M.star.z0', 'M.star.z0'], labels=['Early Infall', 'Late Infall'], binsize=0.5, file_path_and_name=directory+'/mstar_mhalo_infall_compare.pdf')


"""
    Another way of plotting Mstar vs Mhalo for different infall time sats
"""

mask_1 = (Mhalo_peak_tot > 1e7)*(Mhalo_peak_tot < 1e8)
mask_2 = (Mhalo_peak_tot > 1e8)*(Mhalo_peak_tot < 1e9)
mask_3 = (Mhalo_peak_tot > 1e9)*(Mhalo_peak_tot < 1e10)
mask_4 = (Mhalo_peak_tot > 1e10)*(Mhalo_peak_tot < 1e11)
mask_5 = (Mhalo_peak_tot > 1e11)*(Mhalo_peak_tot < 1e12)
masks = [mask_1, mask_2, mask_3, mask_4, mask_5]
#
print(np.median(t_in_tot[mask_1]))
print(np.median(t_in_tot[mask_2]))
print(np.median(t_in_tot[mask_3]))
print(np.median(t_in_tot[mask_4]))
print(np.median(t_in_tot[mask_5]))
#
Mstar_early = []
Mstar_late = []
Mstar_peak_early = []
Mstar_peak_late = []
Mhalo_early = []
Mhalo_late = []
#
for i in range(0, len(masks)):
    Mstar_early.append(Mstar_z0_tot[masks[i]][t_in_tot[masks[i]] > np.median(t_in_tot[masks[i]])])
    Mstar_late.append(Mstar_z0_tot[masks[i]][t_in_tot[masks[i]] <= np.median(t_in_tot[masks[i]])])
    #
    Mstar_peak_early.append(Mstar_peak_tot[masks[i]][t_in_tot[masks[i]] > np.median(t_in_tot[masks[i]])])
    Mstar_peak_late.append(Mstar_peak_tot[masks[i]][t_in_tot[masks[i]] <= np.median(t_in_tot[masks[i]])])
    #
    Mhalo_early.append(Mhalo_peak_tot[masks[i]][t_in_tot[masks[i]] > np.median(t_in_tot[masks[i]])])
    Mhalo_late.append(Mhalo_peak_tot[masks[i]][t_in_tot[masks[i]] <= np.median(t_in_tot[masks[i]])])
#
Mstar_early = np.hstack(Mstar_early)
Mstar_late = np.hstack(Mstar_late)
Mstar_peak_early = np.hstack(Mstar_peak_early)
Mstar_peak_late = np.hstack(Mstar_peak_late)
Mhalo_early = np.hstack(Mhalo_early)
Mhalo_late = np.hstack(Mhalo_late)

f, ax = plt.subplots(figsize=(11, 8))
colorss = ['#000080', '#006400']
binedges = None
binsize = 0.5
binedges = (8,12)
limits = ((8,11.5),(4,10))
#
x = [Mhalo_peak_tot, Mhalo_peak_tot, Mhalo_early, Mhalo_late, Mhalo_early, Mhalo_late]
y = [Mstar_z0_tot, Mstar_peak_tot, Mstar_early, Mstar_late, Mstar_peak_early, Mstar_peak_late]
#
xtype = ['M.halo.peak', 'M.halo.peak', 'M.halo.peak', 'M.halo.peak', 'M.halo.peak', 'M.halo.peak']
ytype = ['M.star.z0', 'M.star.peak', 'M.star.z0', 'M.star.z0', 'M.star.peak', 'M.star.peak']
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
    twosigp = 97.72
    twosigm = 2.28
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
        #
    medians.append(med)
    lowers.append(lower)
    uppers.append(upper)
    lowests.append(lowest)
    highests.append(highest)
    binss.append(bins)
    half_bins.append(half_bin)
# PLOTTING
# Plot the scatter for the recent and minimum pericenters
plt.fill_between(10**(binss[0][:-1]+half_bins[0]), 10**uppers[0], 10**lowers[0], color=colorss[0], alpha=0.3)
plt.fill_between(10**(binss[0][:-1]+half_bins[0]), 10**highests[0], 10**lowests[0], color=colorss[0], alpha=0.15)
plt.fill_between(10**(binss[1][:-1]+half_bins[1]), 10**uppers[1], 10**lowers[1], color=colorss[1], alpha=0.3)
plt.fill_between(10**(binss[1][:-1]+half_bins[1]), 10**highests[1], 10**lowests[1], color=colorss[1], alpha=0.15)
#
# Plot the medians for the two mass bins (low-mass)
plt.plot(10**(binss[2][:-1]+half_bins[2]), 10**medians[2], color=colorss[0], markersize=10, alpha=0.5, label='M$_{\\rm star}(z = 0)$, Early')
plt.plot(10**(binss[3][:-1]+half_bins[3]), 10**medians[3], color=colorss[0], markersize=10, linestyle='--', alpha=0.5, label='M$_{\\rm star}(z = 0)$, Late')
#
plt.plot(10**(binss[4][:-1]+half_bins[4]), 10**medians[4], color=colorss[1], markersize=10, alpha=0.5, label='M$_{\\rm star, peak}$, Early')
plt.plot(10**(binss[5][:-1]+half_bins[5]), 10**medians[5], color=colorss[1], markersize=10, linestyle='--', alpha=0.5, label='M$_{\\rm star, peak}$, Late')
#
plt.xlim(10**limits[0][0], 10**limits[0][1])
plt.ylim(10**limits[1][0], 10**limits[1][1])
plt.xscale('log')
plt.yscale('log')
ax.set_xlabel('M$_{\\rm halo,peak}$', fontsize=28)
ax.set_ylabel('M$_{\\rm star}$', fontsize=28)
ax.legend(prop={'size': 24}, loc='best')
ax.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.show()
