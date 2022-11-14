

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
print('Read in the tools')


sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2_fix/energy'
print('Set paths')

# Set up snapshot array to loop through
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_res7100')
tlb = snaps['time'][-1] - np.flip(snaps['time'])

# Read in all of the other data
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
# Set up distance array (probably don't need it though...)
rs = np.logspace(np.log10(5), np.log10(500), 25)

# Set path and initial parameters
sims = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise']


test = []

for name in sims:
    # Read in the potential data
    data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/potentials/all_snapshots/'+name+'_potentials_all', verbose=True)
    # Read in the mass profile data
    mp = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/mass_profiles/'+name+'_mass_profile_all', verbose=True)

    # Set the host potential at 100 kpc at z = 0
    phi_host_z0 = data['host.pot.100kpc'][-1] - data['host.pot.R200m'] - data['KE.at.Rvir']

    # Set up the enclosed mass ratio within 100-ish kpc
    M_enc_100kpc_z0 = mp['mass.profile'][-1][16]
    M_enc_100kpc_z = mp['mass.profile'][:,16]
    #
    M_enc_ratio = M_enc_100kpc_z/M_enc_100kpc_z0

    # Define the host potential at 100 kpc across all snapshots
    phi_host_z = np.flip(phi_host_z0*M_enc_ratio) # Goes from z = 0 to some other z

    # Set up null array to save the normalized subhalo potentials to
    sub_pot = (-1)*np.ones(data['subhalo.pot'].shape)
    sub_energy = (-1)*np.ones(data['subhalo.pot'].shape)
    sub_pot_snaps = (-1)*np.ones(data['subhalo.pot'].shape, int)
    sub_pot_tlb = (-1)*np.ones(data['subhalo.pot'].shape)

    # Loop through all of the satellites
    for i in range(0, sub_pot.shape[0]):
        # Create a mask for the subhalo data
        mask_sub = (np.flip(data['subhalo.pot'][i]) != -1)*np.isfinite(np.flip(data['subhalo.pot'][i]))*(data_total[name]['v.tot.sim'][i][:len(data['subhalo.pot'][i])] != -1)*np.isfinite(data_total[name]['v.tot.sim'][i][:len(data['subhalo.pot'][i])])*np.isfinite(np.flip(data['host.pot.100kpc']))
        #print(np.sum(mask_sub))
        #
        # Create a mask for the host data
        mask_host = (phi_host_z[:len(data['subhalo.pot'][i])] != 0)
        #
        # Extra mask to ensure post infall
        mask_infall = (tlb[:len(mask_sub)] < data_total[name]['first.infall.time.lb'][i])
        #
        # Calculate the normalized subhalo energy
        sub_pot[i][mask_sub*mask_host*mask_infall] = np.flip(data['subhalo.pot'][i])[mask_sub*mask_host*mask_infall] - np.flip(data['host.pot.100kpc'])[mask_sub*mask_host*mask_infall] + phi_host_z[:len(data['subhalo.pot'][i])][mask_sub*mask_host*mask_infall]
        # Keep which snapshots it has data for
        sub_pot_snaps[i][mask_sub*mask_host*mask_infall] = np.flip(snaps['index'])[:len(data['subhalo.pot'][i])][mask_sub*mask_host*mask_infall]
        sub_pot_tlb[i][mask_sub*mask_host*mask_infall] = tlb[:len(data['subhalo.pot'][i])][mask_sub*mask_host*mask_infall]
        #
        # Calculate the total orbital energy
        sub_energy[i][mask_sub*mask_host*mask_infall] = sub_pot[i][mask_sub*mask_host*mask_infall] + 0.5*(data_total[name]['v.tot.sim'][i][:len(data['subhalo.pot'][i])][mask_sub*mask_host*mask_infall])**2
    test.append(sub_energy)

binss, half_binss = summary_plot.binning_scheme(x=tlb, xtype='t.lb', binsize=1)
max_bin = int(binss[-1])
#
meds = np.zeros(max_bin)
upper68 = np.zeros(max_bin)
lower68 = np.zeros(max_bin)
upper95 = np.zeros(max_bin)
lower95 = np.zeros(max_bin)
upper100 = np.zeros(max_bin)
lower100 = np.zeros(max_bin)
#
onesigp = 84.13
onesigm = 15.87
twosigp = 97.72
twosigm = 2.28
thrsigp = 100
thrsigm = 0
#
# Loop through each bin in time
for i in range(0, max_bin):
    mask_time = (tlb[:596] > binss[i])*(tlb[:596] < binss[i+1])
    temp = []
    #
    # Loop through each host
    for n in range(0, len(test)):
        #temp = []
        #
        # Loop through each subhalo
        for j in range(0, len(test[n])):
            mask_sub = (test[n][j][:596] != -1)*np.isfinite(test[n][j][:596])*mask_time
            temp.append(test[n][j][:596][mask_sub])
    #
    meds[i] = np.nanmedian(np.hstack(temp))
    upper68[i] = np.nanpercentile(np.hstack(temp), onesigp)
    lower68[i] = np.nanpercentile(np.hstack(temp), onesigm)
    upper95[i] = np.nanpercentile(np.hstack(temp), twosigp)
    lower95[i] = np.nanpercentile(np.hstack(temp), twosigm)
    upper100[i] = np.nanpercentile(np.hstack(temp), thrsigp)
    lower100[i] = np.nanpercentile(np.hstack(temp), thrsigm)

plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
axs.plot(binss[:max_bin]+half_binss, meds/1e4, 'g', alpha=0.7)
axs.fill_between(binss[:max_bin]+half_binss, upper68/1e4, lower68/1e4, color='g', alpha=0.3)
axs.fill_between(binss[:max_bin]+half_binss, upper95/1e4, lower95/1e4, color='g', alpha=0.15)
axs.fill_between(binss[:max_bin]+half_binss, upper100/1e4, lower100/1e4, color='g', alpha=0.05)
#
cc = ut.cosmology.CosmologyClass()
red = np.array([0, 1])
cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
#
axis_z_label = 'redshift'
axis_z_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
axis_z_tick_values = [float(v) for v in axis_z_tick_labels]
axis_z_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_z_tick_values)
axz = axs.twiny()
axz.set_xscale('linear')
axz.set_yscale('linear')
axz.set_xticks(axis_z_tick_locations)
axz.set_xticklabels(axis_z_tick_labels, fontsize=26)
axz.set_xlim(0,max_bin)
axz.set_xlabel(axis_z_label, fontsize=30, labelpad=9)
axz.tick_params(pad=3)
#
axs.set_xlim(0,max_bin)
#axs.set_ylim(-2.5,2)
#
axs.tick_params(axis='both', which='both', bottom=True, top=False, labelsize=28, labelbottom=True)
#
axs.set_ylabel('Energy [$10^4$ km$^2$ s$^{-2}$]', fontsize=30)
axs.get_yaxis().set_label_coords(-0.12,0.5)
#
axs.set_xlabel('Lookback time [Gyr]', fontsize=32)
#
plt.tight_layout()
#plt.show()
plt.savefig(directory+'/E_vs_tlb_median_all.pdf')
plt.close()
