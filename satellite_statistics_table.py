#!/usr/bin/python3

"""
    ==============================
    = Satellite Statistics Table =
    ==============================

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
#data_total = summary.data_read(directory=sim_data.home_dir, hosts='all', sim_type='baryon')
#data_total = summary.data_read(directory=sim_data.home_dir, hosts='all', sim_type='all_baryon')
data_total = summary.data_read(directory=sim_data.home_dir, hosts='iso_no_z', sim_type='dmo')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='iso_no_z')
summary_plot = summary_io.SummaryDataPlot()

# Count the number of satellites that have ever passed within R200
for name in summary.host_names['iso_no_z']:
    print(np.sum(masks_infall[name]))

# Count the number of satellites that have ever experienced pericenter
for name in summary.host_names['iso_no_z']:
    print(np.sum(data_total[name]['pericenter.check.sim']))

# Count the number of satellites within R200, but not experienced peri
for name in summary.host_names['iso_no_z']:
    print(np.sum(~data_total[name]['pericenter.check.sim']*masks_infall[name]))
    print(data_total[name]['dtot.sim'][~data_total[name]['pericenter.check.sim']*masks_infall[name]][:,0])

# Counting the number of satellites with more than 1 pericenter
for name in summary.host_names['iso_no_z']:
    count = 0
    for n in range(0, len(data_total[name]['pericenter.dist.sim'])):
        mask = (data_total[name]['pericenter.dist.sim'][n] != -1)
        if np.sum(mask) > 1:
            count += 1
    print(name, count)

# Count how many satellites have the recent pericenter as the minimum
for name in summary.host_names['iso_no_z']:
    count = 0
    for n in range(0, len(data_total[name]['pericenter.dist.sim'])):
        mask = (data_total[name]['pericenter.dist.sim'][n] != -1)
        if np.sum(mask) > 1:
            if data_total[name]['pericenter.dist.sim'][n][mask][0] == np.min(data_total[name]['pericenter.dist.sim'][n][mask]):
                count += 1
    print(name, count)

# Count how many satellites have peris in sims but not model
for name in summary.host_names['iso_no_z']:
    count = np.sum(data_total[name]['infall.check']*data_total[name]['pericenter.check.sim']*~data_total[name]['pericenter.check.galpy'])
    print(name, count)

# Count how many satellites have peris in model but not sim
for name in summary.host_names['iso_no_z']:
    count = np.sum(data_total[name]['infall.check']*~data_total[name]['pericenter.check.sim']*data_total[name]['pericenter.check.galpy'])
    print(name, count)

# For the satellites that have multiple pericenters, count how many have fractional differences > 10% (between recent and minimum)
for name in summary.host_names['iso_no_z']:
    data_recent = []
    data_min = []
    for i in range(0, len(data_total[name]['pericenter.dist.sim'][masks_infall[name]])):
        mask = (data_total[name]['pericenter.dist.sim'][masks_infall[name]][i] != -1)
        if np.sum(mask) > 1:
            data_recent.append(data_total[name]['pericenter.dist.sim'][masks_infall[name]][i][mask][0])
            data_min.append(np.min(data_total[name]['pericenter.dist.sim'][masks_infall[name]][i][mask]))
    d_peri_recent = np.hstack(data_recent)
    d_peri_min = np.hstack(data_min)
    #
    print(np.sum((d_peri_recent-d_peri_min)/d_peri_recent > 0.1))

# Print out the most massive sat stellar mass + halo mass, and see if they're the same subhalo
for i in summary.host_names['all']:
    print('The maximum stellar mass in {0} is {1} with index {2}'.format(i, data_total[i]['Mstar.z0'][masks_infall[i]].max(), np.where(data_total[i]['Mstar.z0'][masks_infall[i]].max() == data_total[i]['Mstar.z0'][masks_infall[i]])[0][0]))
    #
    print('The maximum peak halo mass in {0} is {1} with index {2}'.format(i, data_total[i]['Mhalo.peak'][masks_infall[i]].max(), np.where(data_total[i]['Mhalo.peak'][masks_infall[i]].max() == data_total[i]['Mhalo.peak'][masks_infall[i]])[0][0]))
    #
    print('\n')

# Recent vs Minimum pericenter plotting stuff
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
