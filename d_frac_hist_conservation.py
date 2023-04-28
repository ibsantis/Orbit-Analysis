import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import orbit_io
import summary_io
import model_io
from scipy import interpolate
from scipy import stats
import pandas as pd
from matplotlib import patches
from matplotlib import gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from astropy import units as u
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')


# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_no_r', new=True)
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2'
snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/m12i_r7100')
tlb = snaps['time'][-1] - np.flip(snaps['time'])



threshold = 0.05
times_all = []
for name in summary.host_names['all_no_r']:
    times = (-1)*np.ones(len(masks_infall[name]))
    for i in range(0, len(masks_infall[name])):
        if masks_infall[name][i]:
            mask = (data_total[name]['d.tot.sim'][i] != -1)*np.isfinite(data_total[name]['d.tot.sim'][i] != -1)
            r_sim = data_total[name]['d.tot.sim'][i][mask]
            r_model = data_total[name]['d.tot.model'][i][:len(r_sim)]
            for j in range(0, len(r_sim)):
                if (np.abs(r_model[j]-r_sim[j])/r_sim[j] > threshold):
                    times[i] = tlb[j]
                    break
    times_all.append(times)
times_all = np.hstack(times_all)

summary_plot.plot_hist(x=times_all[times_all != -1], xtype='t.infall', x_labels='time when model and sim within 5% [Gyr]', binsize=0.5, pdf=False, file_path_and_name=directory+'/something.pdf')
