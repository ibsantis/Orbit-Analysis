#!/usr/bin/python3

"""
    =================
    = Summary Plots =
    =================

"""

## Import all of the tools for analysis
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import orbit_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')

# Read the data in
data_i = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/data_m12i', verbose=True)
data_f = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/data_m12f', verbose=True)
data_m = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/data_m12m', verbose=True)

delta_i = data_i['N.peri.sim']-data_i['N.peri.galpy']
delta_f = data_f['N.peri.sim']-data_f['N.peri.galpy']
delta_m = data_m['N.peri.sim']-data_m['N.peri.galpy']

# Number of pericenters
plt.figure(figsize=(10, 8))
plt.hist(delta_m, bins=np.linspace(-3.5, 3.5, 8), density=True, linestyle='solid', linewidth=2, histtype='stepfilled', label='m12m', alpha=0.5)
plt.hist(delta_f, bins=np.linspace(-3.5, 3.5, 8), density=True, linestyle='solid', linewidth=2, histtype='stepfilled', label='m12f', alpha=0.5)
plt.hist(delta_i, bins=np.linspace(-3.5, 3.5, 8), density=True, linestyle='solid', linewidth=2, histtype='stepfilled', label='m12i', alpha=0.5)
plt.xlabel('N$_{\\rm sim}$ - N$_{\\rm galpy}$', fontsize=28)
plt.ylabel('PDF', fontsize=28)
plt.title('Pericenters', fontsize=24)
plt.legend(prop={'size': 14})
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/peri_histogram_pdf.pdf')
plt.close()


# Get most recent pericenters from the simulations and the models
peri_recent_sim = []
peri_recent_galpy = []
for i in range(0, len(data_i['pericenter.dist.sim'])):
    if (data_i['pericenter.dist.sim'][i][0] != -1):
        peri_recent_sim.append(data_i['pericenter.dist.sim'][i][0])
        peri_recent_galpy.append(data_i['pericenter.dist.galpy'][i][0])
for i in range(0, len(data_f['pericenter.dist.sim'])):
    if (data_f['pericenter.dist.sim'][i][0] != -1):
        peri_recent_sim.append(data_f['pericenter.dist.sim'][i][0])
        peri_recent_galpy.append(data_f['pericenter.dist.galpy'][i][0])
for i in range(0, len(data_m['pericenter.dist.sim'])):
    if (data_m['pericenter.dist.sim'][i][0] != -1):
        peri_recent_sim.append(data_m['pericenter.dist.sim'][i][0])
        peri_recent_galpy.append(data_m['pericenter.dist.galpy'][i][0])
peri_recent_sim = np.asarray(peri_recent_sim)
peri_recent_galpy = np.asarray(peri_recent_galpy)
#
f, ax = plt.subplots(figsize=(10, 8))
ax.scatter(peri_recent_sim, peri_recent_galpy, s=50, marker='x', alpha=0.5)
ax.plot([0, 1], [0, 1], linestyle=':', color='k', transform=ax.transAxes)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.xlabel('d$_{\\rm peri, sim}$ [kpc]', fontsize=28)
plt.ylabel('d$_{\\rm peri, model}$ [kpc]', fontsize=28)
plt.title('Most Recent Pericenter', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/recent_peris.pdf')
plt.close()


# Plot distribution of delta d_peris
mask = (peri_recent_galpy > -1)
delta_d_peris = (peri_recent_sim[mask] - peri_recent_galpy[mask])/peri_recent_sim[mask]
#
plt.figure(figsize=(10, 8))
plt.hist(delta_d_peris, bins=np.linspace(-2, 2, 41), linestyle='solid', linewidth=2, histtype='stepfilled', alpha=0.5)
plt.xlabel('(d$_{\\rm peri,sim}$ - d$_{\\rm peri,model}$)/d$_{\\rm peri,sim}$', fontsize=28)
plt.ylabel('N', fontsize=28)
plt.title('Pericenter Distances', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/peri_diff_frac_histogram.pdf')
plt.close()
#
#
delta_d_peris = (peri_recent_sim[mask] - peri_recent_galpy[mask])
#
plt.figure(figsize=(10, 8))
plt.hist(delta_d_peris, bins=np.linspace(-130, 150, 29), linestyle='solid', linewidth=2, histtype='stepfilled', alpha=0.5)
plt.xlabel('(d$_{\\rm peri,sim}$ - d$_{\\rm peri,model}$) [kpc]', fontsize=28)
plt.ylabel('N', fontsize=28)
plt.title('Pericenter Distances', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/peri_diff_histogram.pdf')
plt.close()


# Get most recent pericenter times from the simulations and the models
peri_recent_sim = []
peri_recent_galpy = []
for i in range(0, len(data_i['pericenter.time.lb.sim'])):
    if (data_i['pericenter.time.lb.sim'][i][0] != -1):
        peri_recent_sim.append(data_i['pericenter.time.lb.sim'][i][0])
        peri_recent_galpy.append(data_i['pericenter.time.lb.galpy'][i][0])
for i in range(0, len(data_f['pericenter.time.lb.sim'])):
    if (data_f['pericenter.time.lb.sim'][i][0] != -1):
        peri_recent_sim.append(data_f['pericenter.time.lb.sim'][i][0])
        peri_recent_galpy.append(data_f['pericenter.time.lb.galpy'][i][0])
for i in range(0, len(data_m['pericenter.time.lb.sim'])):
    if (data_m['pericenter.time.lb.sim'][i][0] != -1):
        peri_recent_sim.append(data_m['pericenter.time.lb.sim'][i][0])
        peri_recent_galpy.append(data_m['pericenter.time.lb.galpy'][i][0])
peri_recent_sim = np.asarray(peri_recent_sim)
peri_recent_galpy = np.asarray(peri_recent_galpy)
#
f, ax = plt.subplots(figsize=(10, 8))
ax.scatter(peri_recent_sim, peri_recent_galpy, s=50, marker='x', alpha=0.5)
ax.plot([0, 1], [0, 1], linestyle=':', color='k', transform=ax.transAxes)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.xlabel('t$_{\\rm lb, sim}$ [Gyr]', fontsize=28)
plt.ylabel('t$_{\\rm lb, model}$ [Gyr]', fontsize=28)
plt.title('Most Recent Pericenter', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/recent_peri_times.pdf')
plt.close()


# Plot distribution of delta t_peris
mask = (peri_recent_galpy > -1)
delta_t_peris = (peri_recent_sim[mask] - peri_recent_galpy[mask])/peri_recent_sim[mask]
#
plt.figure(figsize=(10, 8))
plt.hist(delta_t_peris, bins=np.linspace(-4, 1, 51), linestyle='solid', linewidth=2, histtype='stepfilled', alpha=0.5)
plt.xlabel('(t$_{\\rm peri,sim}$ - t$_{\\rm peri,model}$)/t$_{\\rm peri,sim}$', fontsize=28)
plt.ylabel('N', fontsize=28)
plt.xlim(-3.5, 1)
plt.title('Pericenter Times', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/peri_tlb_diff_frac_histogram.pdf')
plt.close()
#
#
delta_t_peris = (peri_recent_sim[mask] - peri_recent_galpy[mask])
#
plt.figure(figsize=(10, 8))
plt.hist(delta_t_peris, bins=np.linspace(-9, 3, 49), linestyle='solid', linewidth=2, histtype='stepfilled', alpha=0.5)
plt.xlabel('(t$_{\\rm peri,sim}$ - t$_{\\rm peri,model}$) [Gyr]', fontsize=28)
plt.ylabel('N', fontsize=28)
plt.title('Pericenter Times', fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/peri_tlb_diff_histogram.pdf')
plt.close()
