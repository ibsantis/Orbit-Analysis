#!/usr/bin/python3

"""
    =================================
    = Recent vs Minimum Pericenters =
    =================================

    Find the number of cases where:
        - Satellite has more than 1 pericenter
        - Minimum pericenter = recent
        - Satellite has peri in sim, but not model
        - Satellite has peri in model, but not sim

        These values are saved in:
        '/Users/isaiahsantistevan/simulation/orbit_data/tables/satellite_statistics'


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
data_total = summary.data_read(directory=sim_data.home_dir)
masks_1 = summary.data_mask(data_total) # for cases where there are pericenters in sim, but not required in model
summary_plot = summary_io.SummaryDataPlot()


# Counting the number of satellites with more than 1 pericenter
for name in summary.host_names:
    count = 0
    for n in range(0, len(data_total[name]['pericenter.dist.sim'])):
        mask = (data_total[name]['pericenter.dist.sim'][n] != -1)
        if np.sum(mask) > 1:
            count += 1
    print(name, count)


# Count how many satellites have the recent pericenter as the minimum
for name in summary.host_names:
    count = 0
    for n in range(0, len(data_total[name]['pericenter.dist.sim'])):
        mask = (data_total[name]['pericenter.dist.sim'][n] != -1)
        if np.sum(mask) > 1:
            if data_total[name]['pericenter.dist.sim'][n][mask][0] == np.min(data_total[name]['pericenter.dist.sim'][n][mask]):
                count += 1
    print(name, count)


# Count how many satellites have peris in sims but not model
for name in summary.host_names:
    count = np.sum(data_total[name]['infall.check']*data_total[name]['pericenter.check.sim']*~data_total[name]['pericenter.check.galpy'])
    print(name, count)


# Count how many satellites have peris in model but not sim
for name in summary.host_names:
    count = np.sum(data_total[name]['infall.check']*~data_total[name]['pericenter.check.sim']*data_total[name]['pericenter.check.galpy'])
    print(name, count)








for name in summary.host_names:
    galaxy = name
    data_recent = []
    data_min = []
    for i in range(0, len(data_total[galaxy]['pericenter.dist.sim'][masks_1[galaxy]])):
        mask = (data_total[galaxy]['pericenter.dist.sim'][masks_1[galaxy]][i] != -1)
        if np.sum(mask) != 1:
            data_recent.append(data_total[galaxy]['pericenter.dist.sim'][masks_1[galaxy]][i][mask][0])
            data_min.append(np.min(data_total[galaxy]['pericenter.dist.sim'][masks_1[galaxy]][i][mask]))
    d_peri_recent = np.hstack(data_recent)
    d_peri_min = np.hstack(data_min)


    print(np.sum((d_peri_recent-d_peri_min)/d_peri_recent > 0.1))



data_recent = []
data_min = []
for name in summary.host_names:
    for i in range(0, len(data_total[name]['pericenter.dist.sim'][masks_1[name]])):
        mask = (data_total[name]['pericenter.dist.sim'][masks_1[name]][i] != -1)
        if np.sum(mask) != 1:
            data_recent.append(data_total[name]['pericenter.dist.sim'][masks_1[name]][i][mask][0])
            data_min.append(np.min(data_total[name]['pericenter.dist.sim'][masks_1[name]][i][mask]))
d_peri_recent = np.hstack(data_recent)
d_peri_min = np.hstack(data_min)


f, ax = plt.subplots(figsize=(10, 8))
ax.scatter(d_peri_recent, d_peri_recent-d_peri_min, color='k', s=50, marker='x', alpha=0.5)
#plt.xlim(limits[0])
#plt.ylim(-0.5,50)
plt.xlabel('d$_{\\rm peri,recent}$ [kpc]', fontsize=28)
plt.ylabel('d$_{\\rm peri,recent}$ - d$_{\\rm peri,min}$ [kpc]', fontsize=28)
plt.title('N$_{\\rm peri} \\geq 2$', fontsize=28)
plt.text(20, 180, 'Total: 264', fontsize=14)
plt.text(20, 170, 'd$_{\\rm peri, recent}$ = d$_{\\rm peri,min}$: 90', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/dperi_recent_vs_min.pdf')
plt.close()
#
f, ax = plt.subplots(figsize=(10, 8))
ax.scatter(d_peri_recent, (d_peri_recent-d_peri_min)/d_peri_recent, color='k', s=50, marker='x', alpha=0.5)
#plt.xlim(limits[0])
#plt.ylim(-0.001,0.05)
plt.xlabel('d$_{\\rm peri,recent}$ [kpc]', fontsize=28)
plt.ylabel('(d$_{\\rm peri,recent}$ - d$_{\\rm peri,min}$)/d$_{\\rm peri,recent}$', fontsize=22)
plt.title('N$_{\\rm peri} \\geq 2$', fontsize=28)
plt.text(150, 0.90, 'Total: 264', fontsize=14)
plt.text(150, 0.85, 'd$_{\\rm peri, recent}$ = d$_{\\rm peri,min}$: 90', fontsize=14)
plt.text(150, 0.80, 'fractions > 10%: 155', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/dperi_recent_vs_min_frac.pdf')
plt.close()
