#!/usr/bin/python3

"""

  ================================
  = Plotting mass profile ratios =
  ================================

  Plot the ratio of the disk, halo, and full mass profiles from the
  model to the disk

"""
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from scipy import special
import orbit_io
import model_io
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='Romulus', location='mac')
sim_data.galaxy = 'Remus'
print('Set paths')

# Read in the data
data_disk_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+sim_data.galaxy+'_disk_radial_profile_fitting')
density_disk_rad = data_disk_rad['density']
mass_disk_rad = data_disk_rad['mass']
rs_disk = data_disk_rad['rs']
#
data_halo = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+sim_data.galaxy+'_halo_fitting')
density_halo = data_halo['density']
mass_halo = data_halo['mass']
rs_halo = data_halo['rs']
#
masses = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/full_profile/'+sim_data.galaxy+'_spherical_mass')
rs = np.logspace(np.log10(0.1), np.log10(500), 81)
#

# Create the radial mass profile
profiles = model_io.Profiles(sim_data.home_dir)
disk_radial_mass_model = profiles.disk_radial_mass(rs_disk, sim_data.galaxy, custom_param=(4.7882851408084E+09, 0.900256692005083, 1.63884986956179E+08, 6.49877777811088, 0.542531341209232))
halo_mass_model = profiles.halo_2p_nfw_mass(rs_halo, sim_data.galaxy)
#
disk_full_mass = profiles.disk_radial_mass(rs, sim_data.galaxy)
halo_full_mass = profiles.halo_2p_nfw_mass(rs, sim_data.galaxy)
total_mass = disk_full_mass+halo_full_mass

# Make the plot of the disk mass ratios
plt.figure(figsize=(10,8))
plt.plot(rs_disk[1:], disk_radial_mass_model[1:]/np.cumsum(mass_disk_rad), '-', label=sim_data.galaxy)
plt.hlines(y=1,xmin=0,xmax=20.1,linestyles='dotted', color='k', alpha=0.5)
plt.xlim(xmin=0, xmax=20.1)
plt.ylim(ymin=0.95, ymax=1.05)
plt.xlabel('R [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<R)/M_{\\rm sim}(<R)$', fontsize=28)
plt.title('Enclosed Disk Mass Ratio', fontsize=28)
plt.legend(prop={'size': 24}, loc='best')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_fits/'+sim_data.galaxy+'_enclosed_disk_mass_ratio.pdf')
plt.close()


# Make the plot of the halo mass ratios
plt.figure(figsize=(10,8))
plt.plot(rs_halo[1:], halo_mass_model[1:]/np.cumsum(mass_halo), '-', label=sim_data.galaxy)
plt.hlines(y=1,xmin=0,xmax=500,linestyles='dotted')
plt.xscale('log')
plt.xlim(xmin=5, xmax=500)
plt.ylim(ymin=0.9, ymax=1.15)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<r)/M_{\\rm sim}(<r)$', fontsize=28)
plt.title('Enclosed Halo Mass Ratio', fontsize=28)
plt.legend(prop={'size': 24}, loc='best')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_fits/'+sim_data.galaxy+'_enclosed_halo_mass_ratio.pdf')
plt.close()


# Make the plot of the full mass ratios
plt.figure(figsize=(10,8))
plt.plot(rs[1:], total_mass[1:]/masses['mass.enclosed'], '-', label=sim_data.galaxy)
plt.hlines(y=1,xmin=0,xmax=500,linestyles='dotted')
plt.xscale('log')
plt.xlim(xmin=5, xmax=500)
plt.ylim(ymin=0.9, ymax=1.1)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<r)/M_{\\rm sim}(<r)$', fontsize=28)
plt.title('Enclosed Total Mass Ratio', fontsize=28)
plt.legend(prop={'size': 24}, loc='best')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_fits/'+sim_data.galaxy+'_enclosed_total_mass_ratio.pdf')
plt.close()


# Plot them all on the same plot
plt.figure(figsize=(10,8))
plt.plot(rs_disk[1:], disk_radial_mass_model[1:]/np.cumsum(mass_disk_rad), '--b', label='Disk')
plt.plot(rs_halo[1:], halo_mass_model[1:]/np.cumsum(mass_halo), '--g', label='Halo')
plt.plot(rs[1:], total_mass[1:]/masses['mass.enclosed'], '-k', label='Disk + Halo')
plt.hlines(y=1,xmin=0,xmax=500,linestyles='dotted')
plt.xscale('log')
plt.xlim(xmin=5, xmax=500)
plt.ylim(ymin=0.9, ymax=1.1)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<r)/M_{\\rm sim}(<r)$', fontsize=28)
plt.title('Enclosed Total Mass Ratio', fontsize=28)
plt.legend(prop={'size': 24}, loc='best')
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_fits/'+sim_data.galaxy+'_enclosed_mass_ratio_components.pdf')
plt.close()





"""
    Print all of the profiles on the same plot
"""

import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from scipy import special
import orbit_io
import model_io
import summary_io
print('Read in the tools')

sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()

mass_profs = []
mass_sims = []
rs = np.logspace(np.log10(0.1), np.log10(500), 81)
#
for name in summary.host_names['all_no_r']:
    # Loop through each galaxy, read in the data, and append the profiles
    data_disk_rad = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/disk/'+name+'_disk_radial_profile_fitting')
    density_disk_rad = data_disk_rad['density']
    mass_disk_rad = data_disk_rad['mass']
    rs_disk = data_disk_rad['rs']
    #
    data_halo = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/halo/complete/'+name+'_halo_fitting')
    density_halo = data_halo['density']
    mass_halo = data_halo['mass']
    rs_halo = data_halo['rs']
    #
    masses = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/fitting_data/full_profile/'+name+'_spherical_mass')
    #
    # Create the radial mass profile
    profiles = model_io.Profiles(sim_data.home_dir)
    disk_full_mass = profiles.disk_radial_mass(rs, name)
    halo_full_mass = profiles.halo_2p_nfw_mass(rs, name)
    total_mass = disk_full_mass+halo_full_mass
    #
    mass_sims.append(masses)
    mass_profs.append(total_mass)

# Plot them all on the same plot
plt.figure(figsize=(10,8))
for i in range(0, len(mass_profs)):
    plt.plot(rs[1:], mass_profs[i][1:]/mass_sims[i]['mass.enclosed'], summary_plot.colors[3], alpha=0.15)
plt.hlines(y=1, xmin=0, xmax=500, linestyles='dotted', colors='k')
plt.xscale('log')
plt.xlim(xmin=5, xmax=500)
plt.ylim(ymin=0.8, ymax=1.2)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<r)\ /\ M_{\\rm sim}(<r)$', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_fits/all_enclosed_mass_ratios.pdf')
plt.close()


# Plot the median and 3 scatters
mass_ratio = np.zeros((len(mass_profs), len(mass_profs[0][1:])))
for i in range(0, len(mass_profs)):
    for j in range(1, len(mass_profs[0])):
        mass_ratio[i,j-1] = mass_profs[i][j]/mass_sims[i]['mass.enclosed'][j-1]
#
onesigp = 84.13
onesigm = 15.87
#
twosigp = 97.72
twosigm = 2.28
#
thrsigp = 100
thrsigm = 0
#
upper_one = np.percentile(mass_ratio, onesigp, axis=0)
lower_one = np.percentile(mass_ratio, onesigm, axis=0)
upper_two = np.percentile(mass_ratio, twosigp, axis=0)
lower_two = np.percentile(mass_ratio, twosigm, axis=0)
upper_thr = np.percentile(mass_ratio, thrsigp, axis=0)
lower_thr = np.percentile(mass_ratio, thrsigm, axis=0)
#
mass_ratio_med = np.median(mass_ratio, axis=0)
#
plt.figure(figsize=(10,8))
plt.fill_between(rs[1:], upper_thr, lower_thr, color='#9966cc', alpha=0.15)
plt.fill_between(rs[1:], upper_two, lower_two, color='#9966cc', alpha=0.3)
plt.fill_between(rs[1:], upper_one, lower_one, color='#9966cc', alpha=0.5)
plt.plot(rs[1:], mass_ratio_med, color='k', alpha=1)
plt.hlines(y=1, xmin=0, xmax=500, linestyles='dotted', colors='k', alpha=0.5)
plt.xscale('log')
plt.xlim(xmin=5, xmax=500)
plt.ylim(ymin=0.8, ymax=1.2)
plt.xlabel('r [kpc]', fontsize=28)
plt.ylabel('$M_{\\rm model}(<r)\ /\ M_{\\rm sim}(<r)$', fontsize=28)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_2/mass_profile_fits/mass_ratio_fit_median.pdf')
plt.close()
