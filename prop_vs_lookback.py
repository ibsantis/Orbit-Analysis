



# For when you need to make plots comparing some of the properties from the simulation to the model
# THIS NEEDS A LOT OF WORK




mask = (data_total['m12b']['dtot.sim'][masks_infall['m12b']][0] != -1)

data_total['m12b']['dtot.sim'][masks_infall['m12b']][0][mask]/data_total['m12b']['dtot.galpy'][masks_infall['m12b']][0][:len(data_total['m12b']['dtot.sim'][masks_infall['m12b']][0])][mask]


halo_dist_1 = data_total['m12b']['dtot.sim'][masks_infall['m12b']][1]
mask_1 = (halo_dist_1 != -1)

print(halo_dist_1)
print(halo_dist_1[mask_1])

halo_model_1 = data_total['m12b']['dtot.galpy'][masks_infall['m12b']][1][:len(halo_dist_1)]

halo_dist_1[mask_1]/halo_model_1[mask_1]



for i in range(0, len(data_total['m12b']['Ltot.sim'][mask_selection['m12b']])):
    mask = np.isfinite(data_total['m12b']['Ltot.sim'][mask_selection['m12b']][i])*(data_total['m12b']['Ltot.sim'][mask_selection['m12b']][i] != -1)
    L_data = data_total['m12b']['Ltot.sim'][mask_selection['m12b']][i][mask]
    L_model = np.linalg.norm(data_total['m12b']['L.galpy'][mask_selection['m12b']][i][:len(L_data)], axis=1)
    lookback_time = np.flip(snaps['time'][-1] - snaps['time'])
    times = lookback_time[:len(L_data)]
    #
    # Set up the figure
    f, ax = plt.subplots(figsize=(10, 8))
    plt.plot(times, (L_model/L_data))
    plt.hlines(1, times[0], times[-1], colors=summary_plot.colors[3], linestyles=':')
    plt.xlim(times[-1], times[0])
    plt.ylabel('$L_{\\rm tot,model}/L_{\\rm tot,sim}$', fontsize=20)
    plt.xlabel('lookback time [Gyr]', fontsize=32)
    plt.tight_layout()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/subhalo_compare/m12b/m12b_sub_'+str(i+1)+'.pdf')
    plt.close()
