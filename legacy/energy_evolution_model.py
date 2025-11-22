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
from orbit_analysis import orbit_io
import sys
from orbit_analysis import summary_io
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from astropy import units as u
import pandas as pd
print('Read in the tools')


### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Thelma', location='mac')
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2'
print('Set paths')

sim_data.galaxy = sim_data.gal_2

# Set up snapshot array to loop through
#snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12w_res7100')
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/R_J')
tlb = snaps['time'][-1] - np.flip(snaps['time'])

# Read in the potential data
data = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', selection='model') # These potentials already start at z = 0 and go backward
data_host = summary.data_read_potential_full(directory=sim_data.home_dir, hosts='all_energy_new', selection='sim')
# Read in the mass profile data
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_energy_new')
# Read in all of the other data
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_energy_new', sim_type='baryon')
# Set up distance array (probably don't need it though...)
rs = data_mp['rs']


# Set up the potential model
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

# Set the global potential at 100 kpc
#phi_host_100kpc = (-1)*evaluatePotentials(potential_two_power, 100*u.kpc, 100*u.kpc)
#phi_host_100kpc = evaluatePotentials(potential_two_power, 100*u.kpc, 100*u.kpc)
d100 = 100/np.sqrt(2)
d200m = data_total[sim_data.galaxy]['host.radius'][0]/np.sqrt(2)
phi_host_100kpc = evaluatePotentials(potential_two_power, d100*u.kpc, d100*u.kpc)


# Set the potential at z = 0
#phi_host_z0 = (-1)*evaluatePotentials(potential_two_power, 100*u.kpc, 100*u.kpc) - (-1)*evaluatePotentials(potential_two_power, data_total[sim_data.galaxy]['host.radius'][0]*u.kpc, data_total[sim_data.galaxy]['host.radius'][0]*u.kpc) - data_host[sim_data.galaxy]['KE.at.Rvir']
phi_host_z0 = evaluatePotentials(potential_two_power, d100*u.kpc, d100*u.kpc) - evaluatePotentials(potential_two_power, d200m*u.kpc, d200m*u.kpc) - data_host[sim_data.galaxy]['KE.at.Rvir']
# Set the potential across all time as just the potential at z = 0
phi_host_z = phi_host_z0*np.ones(data[sim_data.galaxy]['model.potential'].shape[1])


# Set up null array to save the normalized subhalo potentials to
sub_pot = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape)
sub_energy = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape)
sub_pot_snaps = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape, int)
sub_pot_tlb = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape)
sub_kin = (-1)*np.ones(data[sim_data.galaxy]['model.potential'].shape)
#
# Loop through all of the satellites and calculate the potential the same way that I do in the simulations
for i in range(0, sub_pot.shape[0]):
    # Create a mask for the subhalo data
    #mask_sub = (data[sim_data.galaxy]['model.potential'][i] != -1)*np.isfinite(data[sim_data.galaxy]['model.potential'][i])*(data_total[sim_data.galaxy]['v.tot.model'][i][:len(data[sim_data.galaxy]['model.potential'][i])] != -1)*np.isfinite(data_total[sim_data.galaxy]['v.tot.model'][i][:len(data[sim_data.galaxy]['model.potential'][i])])
    #
    # Create a mask for the host data
    #mask_host = (phi_host_z[:len(data[sim_data.galaxy]['model.potential'][i])] != 0)
    #
    # Calculate the normalized subhalo energy
    #sub_pot[i][mask_sub*mask_host] = data[sim_data.galaxy]['model.potential'][i][mask_sub*mask_host] - phi_host_100kpc + phi_host_z[:len(data[sim_data.galaxy]['model.potential'][i])][mask_sub*mask_host]
    sub_pot[i] = data[sim_data.galaxy]['model.potential'][i] - phi_host_100kpc + phi_host_z
    #
    # Keep which snapshots it has data for
    sub_pot_snaps[i] = np.flip(snaps['index'])
    sub_pot_tlb[i] = tlb
    #
    # Calculate the total orbital energy
    sub_energy[i] = sub_pot[i] + 0.5*(data_total[sim_data.galaxy]['v.tot.model'][i])**2
    sub_kin[i] =  0.5*(data_total[sim_data.galaxy]['v.tot.model'][i])**2

# Calculate the energy in the sims
# Set the host potential at 100 kpc at z = 0
phi_host_z0 = data_host[sim_data.galaxy]['host.pot.100kpc'][-1] - data_host[sim_data.galaxy]['host.pot.R200m'] - data_host[sim_data.galaxy]['KE.at.Rvir']
#
# Set up the enclosed mass ratio within 100-ish kpc
M_enc_100kpc_z0 = data_mp[sim_data.galaxy][-1][16]
M_enc_100kpc_z = data_mp[sim_data.galaxy][:,16]
#
M_enc_ratio = M_enc_100kpc_z/M_enc_100kpc_z0

# Define the host potential at 100 kpc across all snapshots
phi_host_z = np.flip(phi_host_z0*M_enc_ratio) # Goes from z = 0 to some other z

# Set up null array to save the normalized subhalo potentials to
sub_pot_sim = (-1)*np.ones(data_host[sim_data.galaxy]['subhalo.pot'].shape)
sub_energy_sim = (-1)*np.ones(data_host[sim_data.galaxy]['subhalo.pot'].shape)
sub_pot_snaps_sim = (-1)*np.ones(data_host[sim_data.galaxy]['subhalo.pot'].shape, int)
sub_pot_tlb_sim = (-1)*np.ones(data_host[sim_data.galaxy]['subhalo.pot'].shape)
sub_kin_sim = (-1)*np.ones(data_host[sim_data.galaxy]['subhalo.pot'].shape)

# Loop through all of the satellites
for i in range(0, sub_pot_sim.shape[0]):
    # Create a mask for the subhalo data
    mask_sub = (np.flip(data_host[sim_data.galaxy]['subhalo.pot'][i]) != -1)*np.isfinite(np.flip(data_host[sim_data.galaxy]['subhalo.pot'][i]))*(data_total[sim_data.galaxy]['v.tot.sim'][i][:len(data_host[sim_data.galaxy]['subhalo.pot'][i])] != -1)*np.isfinite(data_total[sim_data.galaxy]['v.tot.sim'][i][:len(data_host[sim_data.galaxy]['subhalo.pot'][i])])*np.isfinite(np.flip(data_host[sim_data.galaxy]['host.pot.100kpc']))
    #print(np.sum(mask_sub))
    #
    # Create a mask for the host data
    mask_host = (phi_host_z[:len(data_host[sim_data.galaxy]['subhalo.pot'][i])] != 0)
    #
    # Calculate the normalized subhalo energy
    sub_pot_sim[i][mask_sub*mask_host] = np.flip(data_host[sim_data.galaxy]['subhalo.pot'][i])[mask_sub*mask_host] - np.flip(data_host[sim_data.galaxy]['host.pot.100kpc'])[mask_sub*mask_host] + phi_host_z[:len(data_host[sim_data.galaxy]['subhalo.pot'][i])][mask_sub*mask_host]
    # Keep which snapshots it has data for
    sub_pot_snaps_sim[i][mask_sub*mask_host] = np.flip(snaps['index'])[:len(data_host[sim_data.galaxy]['subhalo.pot'][i])][mask_sub*mask_host]
    sub_pot_tlb_sim[i][mask_sub*mask_host] = tlb[:len(data_host[sim_data.galaxy]['subhalo.pot'][i])][mask_sub*mask_host]
    #
    # Calculate the total orbital energy
    sub_energy_sim[i][mask_sub*mask_host] = sub_pot_sim[i][mask_sub*mask_host] + 0.5*(data_total[sim_data.galaxy]['v.tot.sim'][i][:len(data_host[sim_data.galaxy]['subhalo.pot'][i])][mask_sub*mask_host])**2
    sub_kin_sim[i][mask_sub*mask_host] = 0.5*(data_total[sim_data.galaxy]['v.tot.sim'][i][:len(data_host[sim_data.galaxy]['subhalo.pot'][i])][mask_sub*mask_host])**2



# Plot the two components of the energy on the same plot
for i in range(0, len(data_total[sim_data.galaxy]['infall.check'])):
    if data_total[sim_data.galaxy]['infall.check'][i]:
        plt.rcParams["font.family"] = "serif"
        f, axs = plt.subplots(2, 1, figsize=(10,8))
        sat_ind = i
        #
        # Mask to make sure the potential is real and to get the times after infall
        m = (sub_pot[sat_ind] != -1)*(sub_pot_tlb[sat_ind] < data_total[sim_data.galaxy]['first.infall.time.lb'][sat_ind])
        #
        # Plot the model data
        axs[0].plot(sub_pot_tlb[sat_ind][m], sub_pot[sat_ind][m]/1e4, 'k', alpha=0.2, label='Model')
        axs[0].plot(sub_pot_tlb[sat_ind][m], (sub_pot[sat_ind][m]+sub_kin[sat_ind][m])/1e4, '--k', alpha=0.2)
        axs[0].plot(sub_pot_tlb[sat_ind][m], sub_kin[sat_ind][m]/1e4, ':k', alpha=0.2)
        #
        # Mask to make sure the potential and orbit is real and to get the times after infall
        ms = (sub_pot_sim[sat_ind] != -1)*(sub_pot_tlb_sim[sat_ind] < data_total[sim_data.galaxy]['first.infall.time.lb'][sat_ind])*(data_total[sim_data.galaxy]['d.tot.sim'][sat_ind][:len(sub_pot_sim[sat_ind])] != -1)
        #
        # Plot a null array in the top subpanel for the legend
        axs[0].plot(sub_pot_tlb_sim[sat_ind][ms], 100+sub_pot_sim[sat_ind][ms]/1e4, 'b', alpha=0.2, label='Simulation')
        #
        # Plot the actual data
        axs[1].plot(sub_pot_tlb_sim[sat_ind][ms], sub_pot_sim[sat_ind][ms]/1e4, 'b', alpha=0.2, label='Potential')
        axs[1].plot(sub_pot_tlb_sim[sat_ind][ms], sub_kin_sim[sat_ind][ms]/1e4, ':b', alpha=0.2, label='Kinetic')
        axs[1].plot(sub_pot_tlb_sim[sat_ind][ms], (sub_kin_sim[sat_ind][ms]+sub_pot_sim[sat_ind][ms])/1e4, '--b', alpha=0.2, label='Total')
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
        axs[0].set_ylim(ymin=-7, ymax=7)
        axs[1].set_xlim(0,13)
        axs[1].set_ylim(ymin=-7, ymax=7)
        #
        axs[0].tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=False)
        axs[1].tick_params(axis='both', which='both', bottom=True, top=True, labelsize=28, labelbottom=True)
        #
        axs[0].set_ylabel('E [$10^4$ km$^2$ s$^{-2}$]', fontsize=16)
        axs[0].get_yaxis().set_label_coords(-0.12,0.5)
        axs[1].set_ylabel('E [$10^4$ km$^2$ s$^{-2}$]', fontsize=16)
        axs[1].get_yaxis().set_label_coords(-0.12,0.5)
        #
        axs[1].set_xlabel('Lookback time [Gyr]', fontsize=32)
        axs[0].legend(prop={'size': 24}, loc='best')
        axs[1].legend(prop={'size': 24}, loc='best')
        #
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.12, hspace=0)
        #plt.show()
        plt.savefig(directory+'/energy_conservation/model/'+sim_data.galaxy+'/components/'+sim_data.galaxy+'_single_sat_e_components_'+str(i+1)+'.pdf')
        plt.close()


# Plot the energies and orbits in a 2-panel plot
for i in range(0, len(data_total[sim_data.galaxy]['infall.check'])):
    if data_total[sim_data.galaxy]['infall.check'][i]:
        plt.rcParams["font.family"] = "serif"
        f, axs = plt.subplots(2, 1, figsize=(10,8))
        sat_ind = i
        #
        m = (sub_energy[sat_ind] != -1)*(sub_pot_tlb[sat_ind] < data_total[sim_data.galaxy]['first.infall.time.lb'][sat_ind])
        ms = (sub_energy_sim[sat_ind] != -1)*(sub_pot_tlb_sim[sat_ind] < data_total[sim_data.galaxy]['first.infall.time.lb'][sat_ind])*(data_total[sim_data.galaxy]['d.tot.sim'][sat_ind][:len(sub_pot_sim[sat_ind])] != -1)
        axs[0].plot(sub_pot_tlb[sat_ind][m], sub_energy[sat_ind][m]/1e4, 'k', alpha=0.2)
        axs[0].plot(sub_pot_tlb_sim[sat_ind][ms], sub_energy_sim[sat_ind][ms]/1e4, 'b', alpha=0.2)
        #
        axs[1].plot(sub_pot_tlb[sat_ind][m], data_total[sim_data.galaxy]['d.tot.model'][sat_ind][:len(m)][m], 'k', alpha=0.2, label='Model')
        axs[1].plot(sub_pot_tlb_sim[sat_ind][ms], data_total[sim_data.galaxy]['d.tot.sim'][sat_ind][:len(ms)][ms], 'b', alpha=0.2, label='Simulation')
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
        plt.savefig(directory+'/energy_conservation/model/'+sim_data.galaxy+'/e_and_orbit/'+sim_data.galaxy+'_single_sat_check_'+str(i+1)+'.pdf')
        plt.close()




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
plt.savefig(directory+'/energy_conservation/model/'+sim_data.galaxy+'/'+sim_data.galaxy+'_E_vs_tlb_all_model.pdf')
plt.close()
