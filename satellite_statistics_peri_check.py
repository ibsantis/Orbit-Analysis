





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

data_total = dict()
for name in summary.host_names['all']:
    data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/peri_check/data_'+name, verbose=True)
    data_total[name] = data


# The infall info from the particle definition isn't really reliable
for name in summary.host_names['all']:
    print('N_infall based on tree {0}: {1}'.format(name, np.sum(data_total[name]['check.halt'])))
    print('N_infall based on particles {0}: {1}'.format(name, np.sum(data_total[name]['check.part'])))

# Count the number of satellites that have ever experienced pericenter
for name in summary.host_names['all']:
    print('Number of pericenters from halo tree position in {0} is: {1}'.format(name, np.sum(data_total[name]['pericenter.check.halt'])))
    print('Number of pericenters from particle position in {0} is: {1}'.format(name, np.sum(data_total[name]['pericenter.check.part'])))

# Count the number of satellites within R200, but not experienced peri
for name in summary.host_names['all']:
    print(np.sum(~data_total[name]['pericenter.check.halt']*data_total[name]['check.halt']))
    print(data_total[name]['dtot.halt'][~data_total[name]['pericenter.check.halt']*data_total[name]['check.halt']][:,0])
for name in summary.host_names['all']:
    print(np.sum(~data_total[name]['pericenter.check.part']*data_total[name]['check.part']))
    print(data_total[name]['dtot.part'][~data_total[name]['pericenter.check.part']*data_total[name]['check.part']][:,0])


# Counting the number of satellites with more than 1 pericenter
for name in summary.host_names['all']:
    count = 0
    count2 = 0
    for n in range(0, len(data_total[name]['pericenter.dist.halt'])):
        mask = (data_total[name]['pericenter.dist.halt'][n] != -1)
        if np.sum(mask) > 1:
            count += 1
    for n in range(0, len(data_total[name]['pericenter.dist.part'])):
        mask = (data_total[name]['pericenter.dist.part'][n] != -1)
        if np.sum(mask) > 1:
            count2 += 1
    print(name, count, count2)

# Count how many satellites have the recent pericenter as the minimum
for name in summary.host_names['all']:
    count = 0
    count2 = 0
    #
    for n in range(0, len(data_total[name]['pericenter.dist.halt'])):
        mask = (data_total[name]['pericenter.dist.halt'][n] != -1)
        if np.sum(mask) > 1:
            if data_total[name]['pericenter.dist.halt'][n][mask][0] == np.min(data_total[name]['pericenter.dist.halt'][n][mask]):
                count += 1
    #
    for n in range(0, len(data_total[name]['pericenter.dist.part'])):
        mask = (data_total[name]['pericenter.dist.part'][n] != -1)
        if np.sum(mask) > 1:
            if data_total[name]['pericenter.dist.part'][n][mask][0] == np.min(data_total[name]['pericenter.dist.part'][n][mask]):
                count2 += 1
    #
    print(name, count, count2)


# For the satellites that have multiple pericenters, count how many have fractional differences > 10% (between recent and minimum)
for name in summary.host_names['all']:
    data_recent_halt = []
    data_min_halt = []
    #
    data_recent_part = []
    data_min_part = []
    #
    for i in range(0, len(data_total[name]['pericenter.dist.halt'])):
        mask = (data_total[name]['pericenter.dist.halt'][i] != -1)
        if np.sum(mask) > 1:
            data_recent_halt.append(data_total[name]['pericenter.dist.halt'][i][mask][0])
            data_min_halt.append(np.min(data_total[name]['pericenter.dist.halt'][i][mask]))
    #
    for i in range(0, len(data_total[name]['pericenter.dist.part'])):
        mask = (data_total[name]['pericenter.dist.part'][i] != -1)
        if np.sum(mask) > 1:
            data_recent_part.append(data_total[name]['pericenter.dist.part'][i][mask][0])
            data_min_part.append(np.min(data_total[name]['pericenter.dist.part'][i][mask]))
    #
    d_peri_recent_halt = np.hstack(data_recent_halt)
    d_peri_min_halt = np.hstack(data_min_halt)
    #
    d_peri_recent_part = np.hstack(data_recent_part)
    d_peri_min_part = np.hstack(data_min_part)
    #
    print(np.sum((d_peri_recent_halt-d_peri_min_halt)/d_peri_recent_halt > 0.05), np.sum((d_peri_recent_part-d_peri_min_part)/d_peri_recent_part > 0.05))


# Recent vs Minimum pericenter plotting stuff
data_recent_halt = []
data_min_halt = []
#
data_recent_part = []
data_min_part = []
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    temp_array = data_total[name]['pericenter.dist.halt'][mask_infall][:,0]
    mask_temp = (temp_array == -1)
    temp_array[mask_temp] = data_total[name]['dtot.halt'][mask_infall][:,0][mask_temp]
    data_recent_halt.append(np.repeat(temp_array, summary.oversample['baryon'][name]))
    #data_recent_halt.append(temp_array)
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    for i in range(0, len(data_total[name]['pericenter.dist.halt'][mask_infall])):
        mask = (data_total[name]['pericenter.dist.halt'][mask_infall][i] != -1)
        if np.sum(mask) > 0:
            data_min_halt.append(np.repeat(np.min(data_total[name]['pericenter.dist.halt'][mask_infall][i][mask]), summary.oversample['baryon'][name]))
            #data_min_halt.append(np.min(data_total[name]['pericenter.dist.halt'][mask_infall][i][mask]))
        else:
            data_min_halt.append(np.repeat(data_total[name]['dtot.halt'][mask_infall][i][0], summary.oversample['baryon'][name]))
            #data_min_halt.append(data_total[name]['dtot.halt'][mask_infall][i][0])
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    temp_array = data_total[name]['pericenter.dist.part'][mask_infall][:,0]
    mask_temp = (temp_array == -1)
    temp_array[mask_temp] = data_total[name]['dtot.part'][mask_infall][:,0][mask_temp]
    data_recent_part.append(np.repeat(temp_array, summary.oversample['baryon'][name]))
    #data_recent_part.append(temp_array)
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    for i in range(0, len(data_total[name]['pericenter.dist.part'][mask_infall])):
        mask = (data_total[name]['pericenter.dist.part'][mask_infall][i] != -1)
        if np.sum(mask) > 0:
            data_min_part.append(np.repeat(np.min(data_total[name]['pericenter.dist.part'][mask_infall][i][mask]), summary.oversample['baryon'][name]))
            #data_min_part.append(np.min(data_total[name]['pericenter.dist.part'][i][mask]))
        else:
            data_min_part.append(np.repeat(data_total[name]['dtot.part'][mask_infall][i][0], summary.oversample['baryon'][name]))
            #data_min_part.append(data_total[name]['dtot.part'][mask_infall][i][0])
#
d_peri_recent_halt = np.hstack(data_recent_halt)
d_peri_min_halt = np.hstack(data_min_halt)
#
d_peri_recent_part = np.hstack(data_recent_part)
d_peri_min_part = np.hstack(data_min_part)


summary_plot = summary_io.SummaryDataPlot()
summary_plot.median_plot_mult(x=[d_peri_recent_halt, d_peri_recent_part], y=[d_peri_min_halt, d_peri_min_part], xtype=['d.peri.recent', 'd.peri.recent'], ytype=['d.peri.min', 'd.peri.min'], labels=['Halo Tree', 'Particle Catalog'], binsize=50, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/recent_vs_min_halt_vs_part.pdf')
mask_delta_d_halt = (np.abs((d_peri_min_halt-d_peri_recent_halt)/d_peri_recent_halt) > 0.05)
mask_delta_d_part = (np.abs((d_peri_min_part-d_peri_recent_part)/d_peri_recent_part) > 0.05)
summary_plot.median_plot_mult(x=[d_peri_recent_halt[mask_delta_d_halt], d_peri_recent_part[mask_delta_d_part]], y=[((d_peri_min_halt-d_peri_recent_halt)/d_peri_recent_halt)[mask_delta_d_halt], ((d_peri_min_part-d_peri_recent_part)/d_peri_recent_part)[mask_delta_d_part]], xtype=['d.peri.recent', 'd.peri.recent'], ytype=['delta_d_frac', 'delta_d_frac'], labels=['Halo Tree', 'Particle Catalog'], binsize=50, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/recent_vs_min_frac_halt_vs_part.pdf')

f, ax = plt.subplots(figsize=(10, 8))
ax.scatter(d_peri_recent_halt, (d_peri_recent_halt-d_peri_min_halt)/d_peri_recent_halt, color='k', s=50, marker='x', alpha=0.5)
plt.xlabel('d$_{\\rm peri,recent}$ [kpc]', fontsize=28)
plt.ylabel('(d$_{\\rm peri,recent}$ - d$_{\\rm peri,min}$)/d$_{\\rm peri,recent}$', fontsize=22)
plt.title('N$_{\\rm peri} \\geq 2$', fontsize=28)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tight_layout()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/dperi_recent_vs_min_frac.pdf')
plt.close()











data_total = dict()
for name in summary.host_names['all']:
    data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/peri_check/data_'+name+'_host_star_position', verbose=True)
    data_total[name] = data


# The infall info from the particle definition isn't really reliable
for name in summary.host_names['all']:
    print('N_infall based on tree and stars in {0}: {1} {2}'.format(name, np.sum(data_total[name]['check.halt']), np.sum(data_total[name]['check.star'])))

# Count the number of satellites that have ever experienced pericenter
for name in summary.host_names['all']:
    print('N_peri from tree and star position in {0} is: {1} {2}'.format(name, np.sum(data_total[name]['pericenter.check.halt']), np.sum(data_total[name]['pericenter.check.star'])))

# Count the number of satellites within R200, but not experienced peri
for name in summary.host_names['all']:
    print(np.sum(~data_total[name]['pericenter.check.halt']*data_total[name]['check.halt']))
    print(data_total[name]['dtot.halt'][~data_total[name]['pericenter.check.halt']*data_total[name]['check.halt']][:,0])
for name in summary.host_names['all']:
    print(np.sum(~data_total[name]['pericenter.check.star']*data_total[name]['check.star']))
    print(data_total[name]['dtot.star'][~data_total[name]['pericenter.check.star']*data_total[name]['check.star']][:,0])


# Counting the number of satellites with more than 1 pericenter
for name in summary.host_names['all']:
    count = 0
    count2 = 0
    for n in range(0, len(data_total[name]['pericenter.dist.halt'])):
        mask = (data_total[name]['pericenter.dist.halt'][n] != -1)
        if np.sum(mask) > 1:
            count += 1
    for n in range(0, len(data_total[name]['pericenter.dist.star'])):
        mask = (data_total[name]['pericenter.dist.star'][n] != -1)
        if np.sum(mask) > 1:
            count2 += 1
    print(name, count, count2)

# Count how many satellites have the recent pericenter as the minimum
for name in summary.host_names['all']:
    count = 0
    count2 = 0
    #
    for n in range(0, len(data_total[name]['pericenter.dist.halt'])):
        mask = (data_total[name]['pericenter.dist.halt'][n] != -1)
        if np.sum(mask) > 1:
            if data_total[name]['pericenter.dist.halt'][n][mask][0] == np.min(data_total[name]['pericenter.dist.halt'][n][mask]):
                count += 1
    #
    for n in range(0, len(data_total[name]['pericenter.dist.star'])):
        mask = (data_total[name]['pericenter.dist.star'][n] != -1)
        if np.sum(mask) > 1:
            if data_total[name]['pericenter.dist.star'][n][mask][0] == np.min(data_total[name]['pericenter.dist.star'][n][mask]):
                count2 += 1
    #
    print(name, count, count2)


# For the satellites that have multiple pericenters, count how many have fractional differences > 10% (between recent and minimum)
for name in summary.host_names['all']:
    data_recent_halt = []
    data_min_halt = []
    #
    data_recent_star = []
    data_min_star = []
    #
    for i in range(0, len(data_total[name]['pericenter.dist.halt'])):
        mask = (data_total[name]['pericenter.dist.halt'][i] != -1)
        if np.sum(mask) > 1:
            data_recent_halt.append(data_total[name]['pericenter.dist.halt'][i][mask][0])
            data_min_halt.append(np.min(data_total[name]['pericenter.dist.halt'][i][mask]))
    #
    for i in range(0, len(data_total[name]['pericenter.dist.star'])):
        mask = (data_total[name]['pericenter.dist.star'][i] != -1)
        if np.sum(mask) > 1:
            data_recent_star.append(data_total[name]['pericenter.dist.star'][i][mask][0])
            data_min_star.append(np.min(data_total[name]['pericenter.dist.star'][i][mask]))
    #
    d_peri_recent_halt = np.hstack(data_recent_halt)
    d_peri_min_halt = np.hstack(data_min_halt)
    #
    d_peri_recent_star = np.hstack(data_recent_star)
    d_peri_min_star = np.hstack(data_min_star)
    #
    print(np.sum((d_peri_recent_halt-d_peri_min_halt)/d_peri_recent_halt > 0.05), np.sum((d_peri_recent_star-d_peri_min_star)/d_peri_recent_star > 0.05))








#### Plot all three checks...

data_total = dict()
for name in summary.host_names['all']:
    data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/peri_check/data_'+name, verbose=True)
    data_total[name] = data
#
data_total2 = dict()
for name in summary.host_names['all']:
    data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/peri_check/data_'+name+'_host_star_position', verbose=True)
    data_total2[name] = data
#
# Recent vs Minimum pericenter plotting stuff
data_recent_halt = []
data_min_halt = []
#
data_recent_part = []
data_min_part = []
#
data_recent_star = []
data_min_star = []
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    temp_array = data_total[name]['pericenter.dist.halt'][mask_infall][:,0]
    mask_temp = (temp_array == -1)
    temp_array[mask_temp] = data_total[name]['dtot.halt'][mask_infall][:,0][mask_temp]
    data_recent_halt.append(np.repeat(temp_array, summary.oversample['baryon'][name]))
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    for i in range(0, len(data_total[name]['pericenter.dist.halt'][mask_infall])):
        mask = (data_total[name]['pericenter.dist.halt'][mask_infall][i] != -1)
        if np.sum(mask) > 0:
            data_min_halt.append(np.repeat(np.min(data_total[name]['pericenter.dist.halt'][mask_infall][i][mask]), summary.oversample['baryon'][name]))
        else:
            data_min_halt.append(np.repeat(data_total[name]['dtot.halt'][mask_infall][i][0], summary.oversample['baryon'][name]))
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    temp_array = data_total[name]['pericenter.dist.part'][mask_infall][:,0]
    mask_temp = (temp_array == -1)
    temp_array[mask_temp] = data_total[name]['dtot.part'][mask_infall][:,0][mask_temp]
    data_recent_part.append(np.repeat(temp_array, summary.oversample['baryon'][name]))
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total[name]['check.halt']
    for i in range(0, len(data_total[name]['pericenter.dist.part'][mask_infall])):
        mask = (data_total[name]['pericenter.dist.part'][mask_infall][i] != -1)
        if np.sum(mask) > 0:
            data_min_part.append(np.repeat(np.min(data_total[name]['pericenter.dist.part'][mask_infall][i][mask]), summary.oversample['baryon'][name]))
        else:
            data_min_part.append(np.repeat(data_total[name]['dtot.part'][mask_infall][i][0], summary.oversample['baryon'][name]))
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total2[name]['check.halt']
    temp_array = data_total2[name]['pericenter.dist.star'][mask_infall][:,0]
    mask_temp = (temp_array == -1)
    temp_array[mask_temp] = data_total2[name]['dtot.star'][mask_infall][:,0][mask_temp]
    data_recent_star.append(np.repeat(temp_array, summary.oversample['baryon'][name]))
#
for name in summary.host_names['all_no_z']:
    mask_infall = data_total2[name]['check.halt']
    for i in range(0, len(data_total2[name]['pericenter.dist.star'][mask_infall])):
        mask = (data_total2[name]['pericenter.dist.star'][mask_infall][i] != -1)
        if np.sum(mask) > 0:
            data_min_star.append(np.repeat(np.min(data_total2[name]['pericenter.dist.star'][mask_infall][i][mask]), summary.oversample['baryon'][name]))
        else:
            data_min_star.append(np.repeat(data_total2[name]['dtot.star'][mask_infall][i][0], summary.oversample['baryon'][name]))
#
d_peri_recent_halt = np.hstack(data_recent_halt)
d_peri_min_halt = np.hstack(data_min_halt)
#
d_peri_recent_part = np.hstack(data_recent_part)
d_peri_min_part = np.hstack(data_min_part)
#
d_peri_recent_star = np.hstack(data_recent_star)
d_peri_min_star = np.hstack(data_min_star)


summary_plot = summary_io.SummaryDataPlot()
summary_plot.median_plot_mult(x=[d_peri_recent_halt, d_peri_recent_part, d_peri_recent_star], y=[d_peri_min_halt, d_peri_min_part, d_peri_min_star], xtype=['d.peri.recent', 'd.peri.recent', 'd.peri.recent'], ytype=['d.peri.min', 'd.peri.min', 'd.peri.min'], labels=['Halo Tree', 'Particle Catalog', 'Halo Tree (star position)'], binsize=50, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/recent_vs_min_checks.pdf')
mask_delta_d_halt = (np.abs((d_peri_min_halt-d_peri_recent_halt)/d_peri_recent_halt) > 0.05)
mask_delta_d_part = (np.abs((d_peri_min_part-d_peri_recent_part)/d_peri_recent_part) > 0.05)
mask_delta_d_star = (np.abs((d_peri_min_star-d_peri_recent_star)/d_peri_recent_star) > 0.05)
summary_plot.median_plot_mult(x=[d_peri_recent_halt[mask_delta_d_halt], d_peri_recent_part[mask_delta_d_part], d_peri_recent_star[mask_delta_d_star]], y=[((d_peri_min_halt-d_peri_recent_halt)/d_peri_recent_halt)[mask_delta_d_halt], ((d_peri_min_part-d_peri_recent_part)/d_peri_recent_part)[mask_delta_d_part], ((d_peri_min_star-d_peri_recent_star)/d_peri_recent_star)[mask_delta_d_star]], xtype=['d.peri.recent', 'd.peri.recent', 'd.peri.recent'], ytype=['delta_d_frac', 'delta_d_frac', 'delta_d_frac'], labels=['Halo Tree', 'Particle Catalog', 'Halo Tree (star position)'], binsize=50, file_path_and_name=sim_data.home_dir+'/orbit_data/plots/summary/recent_vs_min_frac_checks.pdf')
