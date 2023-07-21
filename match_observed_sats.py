




import pandas as pd
import orbit_io
import summary_io
import numpy as np
import matplotlib
from matplotlib import pyplot as plt



### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')


# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all', sim_type='baryon')
#
lgd = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies.csv')



sat_ind = np.where(lgd['Name'] == 'Sculptor')[0][0]
mass = lgd['Mstar [Msun]'][sat_ind]
dist = lgd['d_host [kpc]'][sat_ind]
vrad = lgd['Velocity Radial wrt Nearest Host [km/s]'][sat_ind]
vtan = lgd['Velocity Tangential wrt Host [km/s]'][sat_ind]
#
#mass_err = lgd['Mstar [Msun]'][sat_ind]
dist_err = 3*lgd['Uncertainty.6'][sat_ind]
vrad_err = 3*lgd['Uncertainty.8'][sat_ind]
vtan_err = 3*lgd['Uncertainty.9'][sat_ind]

count = 0
sats = []
weights = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['indices.z0'])):
        if (np.log10(data_total[name]['M.star.z0'][i]) > (np.log10(mass)-1)) and (np.log10(data_total[name]['M.star.z0'][i]) < (np.log10(mass)+1)):
            if (data_total[name]['d.tot.sim'][i][0] > (dist-dist_err)) and (data_total[name]['d.tot.sim'][i][0] < (dist+dist_err)):
                if (data_total[name]['v.rad.sim'][i][0] > (vrad-vrad_err)) and (data_total[name]['v.rad.sim'][i][0] < (vrad+vrad_err)):
                    if (data_total[name]['v.tan.sim'][i][0] > (vtan-vtan_err)) and (data_total[name]['v.tan.sim'][i][0] < (vtan+vtan_err)):
                        count += 1
                        sats.append((name, i))
                        w1 = ((data_total[name]['d.tot.sim'][i][0] - dist)/dist_err/3)**2
                        w2 = ((data_total[name]['v.rad.sim'][i][0] - vrad)/vrad_err/3)**2
                        w3 = ((data_total[name]['v.tan.sim'][i][0] - vtan)/vtan_err/3)**2
                        w4 = ((np.log10(data_total[name]['M.star.z0'][i]) - np.log10(mass))/1)**2
                        weights.append(np.sqrt(w1 + w2 + w3 + w4))
print(count)


def matching(name, threshold=3):
    #
    sat_ind = np.where(lgd['Name'] == str(name))[0][0]
    mass = lgd['Mstar [Msun]'][sat_ind]
    dist = lgd['d_host [kpc]'][sat_ind]
    vrad = lgd['Velocity Radial wrt Nearest Host [km/s]'][sat_ind]
    vtan = lgd['Velocity Tangential wrt Host [km/s]'][sat_ind]
    #
    dist_err = threshold*lgd['Uncertainty.6'][sat_ind]
    vrad_err = threshold*lgd['Uncertainty.8'][sat_ind]
    vtan_err = threshold*lgd['Uncertainty.9'][sat_ind]
    #
    count = 0
    sats = []
    weights = []
    for name in summary.host_names['all']:
        for i in range(0, len(data_total[name]['indices.z0'])):
            if (np.log10(data_total[name]['M.star.z0'][i]) > (np.log10(mass)-1)) and (np.log10(data_total[name]['M.star.z0'][i]) < (np.log10(mass)+1)):
                if (data_total[name]['d.tot.sim'][i][0] > (dist-dist_err)) and (data_total[name]['d.tot.sim'][i][0] < (dist+dist_err)):
                    if (data_total[name]['v.rad.sim'][i][0] > (vrad-vrad_err)) and (data_total[name]['v.rad.sim'][i][0] < (vrad+vrad_err)):
                        if (data_total[name]['v.tan.sim'][i][0] > (vtan-vtan_err)) and (data_total[name]['v.tan.sim'][i][0] < (vtan+vtan_err)):
                            count += 1
                            sats.append((name, i))
                            w1 = ((data_total[name]['d.tot.sim'][i][0] - dist)/dist_err/3)**2
                            w2 = ((data_total[name]['v.rad.sim'][i][0] - vrad)/vrad_err/3)**2
                            w3 = ((data_total[name]['v.tan.sim'][i][0] - vtan)/vtan_err/3)**2
                            w4 = ((np.log10(data_total[name]['M.star.z0'][i]) - np.log10(mass))/1)**2
                            weights.append(np.sqrt(w1 + w2 + w3 + w4))










peri_dist = []
infall_time = []
ws_p = []
ws_i = []
#
for i in range(0, len(sats)):
    pm = (data_total[sats[i][0]]['pericenter.dist.sim'][sats[i][1]] != -1)
    im = (data_total[sats[i][0]]['all.infall.time.lb'][sats[i][1]] != -1)
    peri_dist.append(data_total[sats[i][0]]['pericenter.dist.sim'][sats[i][1]][pm])
    infall_time.append(data_total[sats[i][0]]['all.infall.time.lb'][sats[i][1]][im])
    ws_p.append(np.repeat(weights[i], np.sum(pm)))
    ws_i.append(np.repeat(weights[i], np.sum(im)))
#
peri_dist = np.hstack(peri_dist)
infall_time = np.hstack(infall_time)
ws_p = np.hstack(ws_p)
ws_i = np.hstack(ws_i)


# Create the bins to use in finding the median + scatter and for plotting
binss, half_binss = summary_plot.binning_scheme(x=peri_dist, xtype='d.sim', binedges=None, binsize=5)
#
# Plot the data
plt.figure(figsize=(10, 8))
#
# Calculate the scatter
sigma_one_op = np.nanpercentile(peri_dist, summary_plot.onesigp)
sigma_one_om = np.nanpercentile(peri_dist, summary_plot.onesigm)
#
y_med = np.max(np.histogram(peri_dist, binss, density=True, weights=ws_p)[0])*1.1
#
plt.hist(peri_dist, binss, density=True, weights=ws_p, linestyle='solid', linewidth=2, histtype='stepfilled', color=summary_plot.colors[3], alpha=0.4)
plt.errorbar(np.median(peri_dist), y_med, xerr=np.array([[np.median(peri_dist)-sigma_one_om],[sigma_one_op-np.median(peri_dist)]]), color='k', lw=5, capsize=0)
plt.scatter(np.median(peri_dist), y_med, s=250, marker='s', c='k')
#
plt.xlim(0,100)
plt.xlabel('Pericenter distance [kpc]', fontsize=28)
plt.ylabel('Probability', fontsize=34)
plt.title('Sculptor', fontsize=34)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/Sculptor_dperi.pdf')
plt.close()





# Create the bins to use in finding the median + scatter and for plotting
binss, half_binss = summary_plot.binning_scheme(x=infall_time, xtype='d.sim', binedges=None, binsize=1)
#
# Plot the data
plt.figure(figsize=(10, 8))
#
# Calculate the scatter
sigma_one_op = np.nanpercentile(infall_time, summary_plot.onesigp)
sigma_one_om = np.nanpercentile(infall_time, summary_plot.onesigm)
#
y_med = np.max(np.histogram(infall_time, binss, density=True, weights=ws_i)[0])*1.1
#
plt.hist(infall_time, binss, density=True, weights=ws_i, linestyle='solid', linewidth=2, histtype='stepfilled', color=summary_plot.colors[3], alpha=0.4)
plt.errorbar(np.median(infall_time), y_med, xerr=np.array([[np.median(infall_time)-sigma_one_om],[sigma_one_op-np.median(infall_time)]]), color='k', lw=5, capsize=0)
plt.scatter(np.median(infall_time), y_med, s=250, marker='s', c='k')
#
plt.xlim(0,13)
plt.xlabel('Infall Lookback time [Gyr]', fontsize=28)
plt.ylabel('Probability', fontsize=34)
plt.title('Sculptor', fontsize=34)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/Sculptor_infall.pdf')
plt.close()
