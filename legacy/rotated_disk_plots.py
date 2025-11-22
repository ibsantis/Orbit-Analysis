import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from orbit_analysis import orbit_io
from orbit_analysis import summary_io
from orbit_analysis import model_io
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
#
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False
#
data_total_rot_90x = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', rotated=True, rotation=(0,90))
masks_infall_rot_90x = summary.data_mask(data_total_rot_90x, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_rot_90x = summary.data_mask(data_total_rot_90x, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_rot_90x = summary.data_mask_apo(data_total_rot_90x, hosts='all_no_r')
masks_infall_rot_90x['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_rot_90x['m12f'][59] = False
masks_infall_apo_rot_90x['m12f'][59] = False
#
data_total_rot_90y = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', rotated=True, rotation=(1,90))
masks_infall_rot_90y = summary.data_mask(data_total_rot_90y, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_rot_90y = summary.data_mask(data_total_rot_90y, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_rot_90y = summary.data_mask_apo(data_total_rot_90y, hosts='all_no_r')
masks_infall_rot_90y['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_rot_90y['m12f'][59] = False
masks_infall_apo_rot_90y['m12f'][59] = False
#
data_total_rot_180x = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon', rotated=True, rotation=(0,180))
masks_infall_rot_180x = summary.data_mask(data_total_rot_180x, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri_rot_180x = summary.data_mask(data_total_rot_180x, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo_rot_180x = summary.data_mask_apo(data_total_rot_180x, hosts='all_no_r')
masks_infall_rot_180x['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri_rot_180x['m12f'][59] = False
masks_infall_apo_rot_180x['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2/rotated_disk'



def width_of_68(x_array):
    onesigp = 84.13
    onesigm = 15.87
    #
    upper = np.nanpercentile(x_array, onesigp)
    lower = np.nanpercentile(x_array, onesigm)
    #
    return (upper-lower)/2

def combine(x1, x2, n=False):
    y = []
    #
    for i in range(0, len(x1)):
        if (x1[i] == 0) and (x2[i] == 0):
            y.append(0)
        elif (x1[i] == 0) and (x2[i] != 0):
            y.append(x2[i])
        else:
            y.append((x2[i] - x1[i])/x1[i])
    #
    onesigp = 84.13
    onesigm = 15.87
    #
    upper = np.nanpercentile(y, onesigp)
    lower = np.nanpercentile(y, onesigm)
    #
    if n:
        return np.nanmean(y), np.std(y)
    else:
        return np.nanmedian(y), (upper-lower)/2

# Recent pericenter distance
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_rot_90x = summary.dperi_recent(data_total_rot_90x, masks_infall_peri_rot_90x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_rot_90y = summary.dperi_recent(data_total_rot_90y, masks_infall_peri_rot_90y, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_rot_180x = summary.dperi_recent(data_total_rot_180x, masks_infall_peri_rot_180x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
mask = (d_rec_mod != 0)#*(d_rec_mod < 30)
print(np.nanmedian((d_rec_mod_rot_90x[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(np.nanmedian((d_rec_mod_rot_90y[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(np.nanmedian((d_rec_mod_rot_180x[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_rot_90x[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_rot_90y[mask]-d_rec_mod[mask])/d_rec_mod[mask]))
print(width_of_68((d_rec_mod_rot_180x[mask]-d_rec_mod[mask])/d_rec_mod[mask]))


# Minimum pericenter distance
# SAME AS THE RECENT PERICENTER IN THE MODEL


# Recent pericenter time
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_rot_90x = summary.tperi_recent(data_total_rot_90x, masks_infall_peri_rot_90x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_rot_90y = summary.tperi_recent(data_total_rot_90y, masks_infall_peri_rot_90y, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod_rot_180x = summary.tperi_recent(data_total_rot_180x, masks_infall_peri_rot_180x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
mask = (t_rec_mod != 0)*(d_rec_mod < 30)
print(np.nanmedian((t_rec_mod_rot_90x[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(np.nanmedian((t_rec_mod_rot_90y[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(np.nanmedian((t_rec_mod_rot_180x[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_rot_90x[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_rot_90y[mask]-t_rec_mod[mask])/t_rec_mod[mask]))
print(width_of_68((t_rec_mod_rot_180x[mask]-t_rec_mod[mask])/t_rec_mod[mask]))


# Pericenter number
n_mod = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_rot_90x = summary.nperi_model(data_total_rot_90x, masks_infall_rot_90x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_rot_90y = summary.nperi_model(data_total_rot_90y, masks_infall_rot_90y, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_rot_180x = summary.nperi_model(data_total_rot_180x, masks_infall_rot_180x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
d_rec_mod = summary.dperi_recent(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (n_mod != 0)*(d_rec_mod != 0)#*(d_rec_mod < 30)
print(np.nanmedian((n_mod_rot_90x[mask]-n_mod[mask])/n_mod[mask]))
print(np.nanmedian((n_mod_rot_90y[mask]-n_mod[mask])/n_mod[mask]))
print(np.nanmedian((n_mod_rot_180x[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_rot_90x[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_rot_90y[mask]-n_mod[mask])/n_mod[mask]))
print(width_of_68((n_mod_rot_180x[mask]-n_mod[mask])/n_mod[mask]))

n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_rot_90x = summary.nperi_model(data_total_rot_90x, masks_infall_rot_90x, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_rot_90y = summary.nperi_model(data_total_rot_90y, masks_infall_rot_90y, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200_rot_180x = summary.nperi_model(data_total_rot_180x, masks_infall_rot_180x, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
#
mask = (n_mod_r200!= 0)*(d_rec_mod != 0)#*(d_rec_mod < 30)
print(np.nanmedian((n_mod_r200_rot_90x[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(np.nanmedian((n_mod_r200_rot_90y[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(np.nanmedian((n_mod_r200_rot_180x[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_rot_90x[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_rot_90y[mask]-n_mod_r200[mask])/n_mod_r200[mask]))
print(width_of_68((n_mod_r200_rot_180x[mask]-n_mod_r200[mask])/n_mod_r200[mask]))


# Recent pericenter distance
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_rot_90x = summary.vperi_recent(data_total_rot_90x, masks_infall_peri_rot_90x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_rot_90y = summary.vperi_recent(data_total_rot_90y, masks_infall_peri_rot_90y, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod_rot_180x = summary.vperi_recent(data_total_rot_180x, masks_infall_peri_rot_180x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (d_rec_mod != 0)*(d_rec_mod < 30)
print(np.nanmedian((v_rec_mod_rot_90x[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(np.nanmedian((v_rec_mod_rot_90y[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(np.nanmedian((v_rec_mod_rot_180x[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_rot_90x[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_rot_90y[mask]-v_rec_mod[mask])/v_rec_mod[mask]))
print(width_of_68((v_rec_mod_rot_180x[mask]-v_rec_mod[mask])/v_rec_mod[mask]))


# Recent apocenter distance
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_rot_90x = summary.dapo_recent(data_total_rot_90x, masks_infall_apo_rot_90x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_rot_90y = summary.dapo_recent(data_total_rot_90y, masks_infall_apo_rot_90y, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod_rot_180x = summary.dapo_recent(data_total_rot_180x, masks_infall_apo_rot_180x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
d_rec_mod = summary.dperi_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (d_rec_mod != 0)*(d_rec_mod < 30)
print(np.nanmedian((dapo_rec_mod_rot_90x[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(np.nanmedian((dapo_rec_mod_rot_90y[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(np.nanmedian((dapo_rec_mod_rot_180x[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_rot_90x[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_rot_90y[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))
print(width_of_68((dapo_rec_mod_rot_180x[mask]-dapo_rec_mod[mask])/dapo_rec_mod[mask]))


# Infall time with evolving R200m
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_rot_90x = summary.first_infall(data_total_rot_90x, masks_infall_rot_90x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_rot_90y = summary.first_infall(data_total_rot_90y, masks_infall_rot_90y, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_rot_180x = summary.first_infall(data_total_rot_180x, masks_infall_rot_180x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
d_rec_mod = summary.dperi_recent(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (t_in_mod != 0)*(d_rec_mod != 0)*(d_rec_mod < 30)
print(np.nanmedian((t_in_mod_rot_90x[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(np.nanmedian((t_in_mod_rot_90y[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(np.nanmedian((t_in_mod_rot_180x[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_rot_90x[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_rot_90y[mask]-t_in_mod[mask])/t_in_mod[mask]))
print(width_of_68((t_in_mod_rot_180x[mask]-t_in_mod[mask])/t_in_mod[mask]))

t_in_mod_R200m = summary.infall_fixed(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m_rot_90x = summary.infall_fixed(data_total_rot_90x, masks_infall_rot_90x, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m_rot_90y = summary.infall_fixed(data_total_rot_90y, masks_infall_rot_90y, oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m_rot_180x = summary.infall_fixed(data_total_rot_180x, masks_infall_rot_180x, oversample=True, hosts='all_no_r', sim_type='baryon')
#
mask = (t_in_mod_R200m != 0)*(d_rec_mod != 0)*(d_rec_mod < 30)
print(np.nanmedian((t_in_mod_R200m_rot_90y[mask]-t_in_mod_R200m[mask])/t_in_mod_R200m[mask]))
print(np.nanmedian((t_in_mod_R200m_rot_90x[mask]-t_in_mod_R200m[mask])/t_in_mod_R200m[mask]))
print(np.nanmedian((t_in_mod_R200m_rot_180x[mask]-t_in_mod_R200m[mask])/t_in_mod_R200m[mask]))
print(width_of_68((t_in_mod_R200m_rot_90x[mask]-t_in_mod_R200m[mask])/t_in_mod_R200m[mask]))
print(width_of_68((t_in_mod_R200m_rot_90y[mask]-t_in_mod_R200m[mask])/t_in_mod_R200m[mask]))
print(width_of_68((t_in_mod_R200m_rot_180x[mask]-t_in_mod_R200m[mask])/t_in_mod_R200m[mask]))



ecc_model1 = []
ecc_model2 = []
ecc_model3 = []
d_rec_mod1 = []
d_rec_mod2 = []
d_rec_mod3 = []
ecc_model_rot_90x = []
ecc_model_rot_90y = []
ecc_model_rot_180x = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['infall.check'])):
        if data_total[name]['infall.check'][i]:
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_rot_90x[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model1.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_rot_90x.append(np.repeat(data_total_rot_90x[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod1.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_rot_90y[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model2.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_rot_90y.append(np.repeat(data_total_rot_90y[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod2.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['eccentricity.model.apsis'][i][0] != -1) and (data_total_rot_180x[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc_model3.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                ecc_model_rot_180x.append(np.repeat(data_total_rot_180x[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod3.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
ecc_model1 = np.hstack(ecc_model1)
ecc_model2 = np.hstack(ecc_model2)
ecc_model3 = np.hstack(ecc_model3)
ecc_model_rot_90x = np.hstack(ecc_model_rot_90x)
ecc_model_rot_90y = np.hstack(ecc_model_rot_90y)
ecc_model_rot_180x = np.hstack(ecc_model_rot_180x)
d_rec_mod1 = np.hstack(d_rec_mod1)
d_rec_mod2 = np.hstack(d_rec_mod2)
d_rec_mod3 = np.hstack(d_rec_mod3)

mask1 = (d_rec_mod1 != 0)*(d_rec_mod1 < 30)
mask2 = (d_rec_mod2 != 0)*(d_rec_mod2 < 30)
mask3 = (d_rec_mod3 != 0)*(d_rec_mod3 < 30)
print(np.nanmedian((ecc_model_rot_90x[mask1]-ecc_model1[mask1])/ecc_model1[mask1]))
print(np.nanmedian((ecc_model_rot_90y[mask2]-ecc_model2[mask2])/ecc_model2[mask2]))
print(np.nanmedian((ecc_model_rot_180x[mask3]-ecc_model3[mask3])/ecc_model3[mask3]))
print(width_of_68((ecc_model_rot_90x[mask1]-ecc_model1[mask1])/ecc_model1[mask1]))
print(width_of_68((ecc_model_rot_90y[mask2]-ecc_model2[mask2])/ecc_model2[mask2]))
print(width_of_68((ecc_model_rot_180x[mask3]-ecc_model3[mask3])/ecc_model3[mask3]))


per_model1 = []
per_model2 = []
per_model3 = []
d_rec_mod1 = []
d_rec_mod2 = []
d_rec_mod3 = []
per_model_rot_90x = []
per_model_rot_90y = []
per_model_rot_180x = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['infall.check'])):
        if data_total[name]['infall.check'][i]:
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_rot_90x[name]['orbit.period.peri.model'][i][0] != -1):
                per_model1.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_rot_90x.append(np.repeat(data_total_rot_90x[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod1.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #    
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_rot_90y[name]['orbit.period.peri.model'][i][0] != -1):
                per_model2.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_rot_90y.append(np.repeat(data_total_rot_90y[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod2.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
            if (data_total[name]['orbit.period.peri.model'][i][0] != -1) and (data_total_rot_180x[name]['orbit.period.peri.model'][i][0] != -1):
                per_model3.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                per_model_rot_180x.append(np.repeat(data_total_rot_180x[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
                d_rec_mod3.append(np.repeat(data_total[name]['pericenter.dist.model'][i][0], summary.oversample['baryon'][name]))
            #
per_model1 = np.hstack(per_model1)
per_model2 = np.hstack(per_model2)
per_model3 = np.hstack(per_model3)
per_model_rot_90x = np.hstack(per_model_rot_90x)
per_model_rot_90y = np.hstack(per_model_rot_90y)
per_model_rot_180x = np.hstack(per_model_rot_180x)
d_rec_mod1 = np.hstack(d_rec_mod1)
d_rec_mod2 = np.hstack(d_rec_mod2)
d_rec_mod3 = np.hstack(d_rec_mod3)

mask1 = (d_rec_mod1 != 0)*(d_rec_mod1 < 30)
mask2 = (d_rec_mod2 != 0)*(d_rec_mod2 < 30)
mask3 = (d_rec_mod3 != 0)*(d_rec_mod3 < 30)

print(np.nanmedian((per_model_rot_90x[mask1]-per_model1[mask1])/per_model1[mask1]))
print(np.nanmedian((per_model_rot_90y[mask2]-per_model2[mask2])/per_model2[mask2]))
print(np.nanmedian((per_model_rot_180x[mask3]-per_model3[mask3])/per_model3[mask3]))
print(width_of_68((per_model_rot_90x[mask1]-per_model1[mask1])/per_model1[mask1]))
print(width_of_68((per_model_rot_90y[mask2]-per_model2[mask2])/per_model2[mask2]))
print(width_of_68((per_model_rot_180x[mask3]-per_model3[mask3])/per_model3[mask3]))


# da/dr
data_mp = summary.data_read_mass_profile(directory=sim_data.home_dir, hosts='all_no_r', new=True)
data_dadr = summary.data_read_dadr(mass_profile=data_mp, hosts='all_no_r')
#
dadr_max_mod = summary.da_dr(data_total, masks_infall_peri, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('Regular model done')
dadr_max_mod_rot_90x = summary.da_dr(data_total_rot_90x, masks_infall_peri_rot_90x, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('90x model done')
dadr_max_mod_rot_90y = summary.da_dr(data_total_rot_90y, masks_infall_peri_rot_90y, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('90y model done')
dadr_max_mod_rot_180x = summary.da_dr(data_total_rot_180x, masks_infall_peri_rot_180x, data_mp, data_dadr, selection='model', oversample=True, hosts='all_no_r')
print('180x model done')

d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
mask = (d_rec_mod != 0)#*(d_rec_mod < 30)

print(np.nanmedian((dadr_max_mod_rot_90x['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(np.nanmedian((dadr_max_mod_rot_90y['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(np.nanmedian((dadr_max_mod_rot_180x['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_rot_90x['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_rot_90y['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))
print(width_of_68((dadr_max_mod_rot_180x['dadr'][mask]-dadr_max_mod['dadr'][mask])/dadr_max_mod['dadr'][mask]))



"""
    Comparing pericenter distances
"""
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_rot_90x = summary.dperi_recent(data_total_rot_90x, masks_infall_peri_rot_90x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_rot_90y = summary.dperi_recent(data_total_rot_90y, masks_infall_peri_rot_90y, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod_rot_180x = summary.dperi_recent(data_total_rot_180x, masks_infall_peri_rot_180x, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
#
summary_plot.plot_hist_mult(x=[(d_rec_mod_rot_90x-d_rec_mod)/d_rec_mod, (d_rec_mod_rot_90y-d_rec_mod)/d_rec_mod, (d_rec_mod_rot_180x-d_rec_mod)/d_rec_mod], xtype=['d.peri.text','d.peri.text','d.peri.text'], labels=['90 deg about x-axis', '90 deg about y-axis', '180 deg about x-axis'], binsize=0.01, xlimits=(-0.1, 0.1), pdf=True, x_label='$(d_{\\rm peri,rec,rot}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$', file_path_and_name=directory+'/d_peri_rec_hist.pdf')
summary_plot.plot_hist_mult(x=[(d_rec_mod_rot_90x-d_rec_mod)/d_rec_mod, (d_rec_mod_rot_90y-d_rec_mod)/d_rec_mod, (d_rec_mod_rot_180x-d_rec_mod)/d_rec_mod], xtype=['d.peri.text','d.peri.text','d.peri.text'], labels=['90 deg about x-axis', '90 deg about y-axis', '180 deg about x-axis'], binsize=0.01, xlimits=(-0.03, 0.03), pdf=True, x_label='$(d_{\\rm peri,rec,rot}-d_{\\rm peri,rec,disk})/d_{\\rm peri,rec,disk}$', file_path_and_name=directory+'/d_peri_rec_hist_zoom.pdf')

