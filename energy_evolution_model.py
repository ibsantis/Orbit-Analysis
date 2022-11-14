#!/usr/bin/python3

"""
    ==========================
    = Energy Evolution Model =
    ==========================

    For a single host, plot the total energy vs lookback time for the model
    satellites

"""

import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import orbit_io
import sys
import summary_io
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from astropy import units as u
import pandas as pd
print('Read in the tools')


### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2_fix/energy'
print('Set paths')

# Set up snapshot array to loop through
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_res7100')
tlb = snaps['time'][-1] - np.flip(snaps['time'])

# Read in the potential data
data = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='iso_dmo', selection='model') # These potentials already start at z = 0 and go backward
data_host = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='iso_dmo', selection='sim')
# Read in the mass profile data
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='iso_dmo')
# Read in all of the other data
data_total = summary.data_read(directory=sim_data.home_dir, hosts='iso_dmo', sim_type='baryon')
# Set up distance array (probably don't need it though...)
rs = data_mp['rs']



fitting_data = pd.read_csv(sim_data.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
#
from galpy.potential import DoubleExponentialDiskPotential # For disks
from galpy.potential import TwoPowerSphericalPotential # For DM halos
from galpy.potential import evaluatePotentials
#
disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][sim_data.galaxy]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][sim_data.galaxy]*u.kpc, hz=fitting_data['h_z'][sim_data.galaxy]*u.kpc)
halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][sim_data.galaxy]*u.solMass, a=fitting_data['a_halo'][sim_data.galaxy]*u.kpc, alpha=fitting_data['alpha'][sim_data.galaxy], beta=fitting_data['beta'][sim_data.galaxy])
potential_two_power = disk_inner+disk_outer+halo_2p


phi_host_100kpc = (-1)*evaluatePotentials(potential_two_power, 100*u.kpc, 100*u.kpc)
phi_host_z0 = (-1)*evaluatePotentials(potential_two_power, 100*u.kpc, 100*u.kpc) - (-1)*evaluatePotentials(potential_two_power, data_total[sim_data.galaxy]['host.radius'][0]*u.kpc, data_total[sim_data.galaxy]['host.radius'][0]*u.kpc) - data_host[sim_data.galaxy]['KE.at.Rvir']
phi_host_z = phi_host_z0*np.ones(data['m12i']['model.potential'].shape[1])


# Set up null array to save the normalized subhalo potentials to
sub_pot = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape)
sub_energy = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape)
sub_pot_snaps = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape, int)
sub_pot_tlb = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape)

# Loop through all of the satellites
for i in range(0, sub_pot.shape[0]):
    # Create a mask for the subhalo data
    #mask_sub = (data[sim_data.galaxy]['model.potential'][i] != -1)*np.isfinite(data[sim_data.galaxy]['model.potential'][i])*(data_total[sim_data.galaxy]['v.tot.model'][i][:len(data[sim_data.galaxy]['model.potential'][i])] != -1)*np.isfinite(data_total[sim_data.galaxy]['v.tot.model'][i][:len(data[sim_data.galaxy]['model.potential'][i])])
    #
    # Create a mask for the host data
    #mask_host = (phi_host_z[:len(data[sim_data.galaxy]['model.potential'][i])] != 0)
    #
    # Calculate the normalized subhalo energy
    #sub_pot[i][mask_sub*mask_host] = data[sim_data.galaxy]['model.potential'][i][mask_sub*mask_host] - phi_host_100kpc + phi_host_z[:len(data[sim_data.galaxy]['model.potential'][i])][mask_sub*mask_host]
    sub_pot[i] = (-1)*data[sim_data.galaxy]['model.potential'][i] - phi_host_100kpc + phi_host_z
    #
    # Keep which snapshots it has data for
    sub_pot_snaps[i] = np.flip(snaps['index'])
    sub_pot_tlb[i] = tlb
    #
    # Calculate the total orbital energy
    sub_energy[i] = sub_pot[i] + 0.5*(data_total[sim_data.galaxy]['v.tot.model'][i])**2

# Plot the data
mask_low = (data_total[sim_data.galaxy]['infall.check'])*(data_total[sim_data.galaxy]['M.star.z0'] < 3e6)
mask_high = (data_total[sim_data.galaxy]['infall.check'])*(data_total[sim_data.galaxy]['M.star.z0'] > 3e6)
#
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 1, figsize=(10,12))
#
for i in range(0, np.sum(mask_low)):
    m = (sub_energy[mask_low][i] != -1)*(sub_pot_tlb[mask_low][i] < data_total[sim_data.galaxy]['first.infall.time.lb'][mask_low][i])
    axs[0].plot(sub_pot_tlb[mask_low][i][m], sub_energy[mask_low][i][m]/1e4, 'k', alpha=0.2)
for i in range(0, np.sum(mask_high)):
    m = (sub_energy[mask_high][i] != -1)*(sub_pot_tlb[mask_high][i] < data_total[sim_data.galaxy]['first.infall.time.lb'][mask_high][i])
    axs[1].plot(sub_pot_tlb[mask_high][i][m], sub_energy[mask_high][i][m]/1e4, 'k', alpha=0.2)
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=26)
axz.set_xlim(0,13)
axz.set_xlabel(axis_z_label, fontsize=30, labelpad=9)
axz.tick_params(pad=3)
#
r1 = mpatches.Rectangle(xy=(1,4),width=3.5,height=1, color='k', alpha=0.1)
r2 = mpatches.Rectangle(xy=(1,4),width=3.5,height=1, color='k', alpha=0.1)
axs[0].add_patch(r1)
axs[1].add_patch(r2)
axs[0].text(1.2,4.25,'$M_{\\rm star} < 10^{6.5} M_{\\odot}$',fontsize=20)
axs[1].text(1.2,4.25,'$M_{\\rm star} > 10^{6.5} M_{\\odot}$',fontsize=20)
#
axs[0].set_xlim(0,13)
#axs[0].set_ylim(ymax=5.5)
axs[1].set_xlim(0,13)
#axs[1].set_ylim(ymax=5.5)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
#
axs[0].set_ylabel('Energy [$10^4$ km$^2$ s$^{-2}$]', fontsize=30)
axs[0].get_yaxis().set_label_coords(-0.12,0.5)
axs[1].set_ylabel('Energy [$10^4$ km$^2$ s$^{-2}$]', fontsize=30)
axs[1].get_yaxis().set_label_coords(-0.12,0.5)
#
axs[1].set_xlabel('Lookback time [Gyr]', fontsize=32)
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)
#plt.show()
plt.savefig(directory+'/infall_satellites/'+sim_data.galaxy+'_E_vs_tlb_all_model.pdf')
plt.close()






plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(2, 1, figsize=(10,6))
#
m = (sub_energy[-1] != -1)*(sub_pot_tlb[-1] < data_total[sim_data.galaxy]['first.infall.time.lb'][-1])
axs[0].plot(sub_pot_tlb[-1][m], sub_energy[-1][m]/1e4, 'k', alpha=0.2)
axs[1].plot(sub_pot_tlb[-1][m], data_total['m12i']['d.tot.model'][-1][m], 'k', alpha=0.2, label='Model')
#
msim = (data_total['m12i']['d.tot.sim'][-1] != -1)
mtime = (sub_pot_tlb[-1][:len(msim)] < data_total[sim_data.galaxy]['first.infall.time.lb'][-1])
axs[1].plot(sub_pot_tlb[-1][:len(data_total['m12i']['d.tot.sim'][-1])][msim*mtime], data_total['m12i']['d.tot.sim'][-1][msim*mtime], 'b', alpha=0.2, label='Simulation')
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs[0].twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=26)
axz.set_xlim(0,13)
axz.set_xlabel(axis_z_label, fontsize=30, labelpad=9)
axz.tick_params(pad=3)
#
axs[0].set_xlim(0,13)
#axs[0].set_ylim(ymax=5.5)
axs[1].set_xlim(0,13)
#axs[1].set_ylim(ymax=5.5)
#
axs[0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False)
axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
#
axs[0].set_ylabel('E [$10^4$ km$^2$ s$^{-2}$]', fontsize=16)
axs[0].get_yaxis().set_label_coords(-0.12,0.5)
axs[1].set_ylabel('D [kpc]', fontsize=16)
axs[1].get_yaxis().set_label_coords(-0.12,0.5)
#
axs[1].set_xlabel('Lookback time [Gyr]', fontsize=32)
axs[1].legend(prop={'size': 24}, loc='best')
#
plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)
#plt.show()
plt.savefig(directory+'/infall_satellites/'+sim_data.galaxy+'_single_sat_check.pdf')
plt.close()
