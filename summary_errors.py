import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib.ticker import LogLocator
from matplotlib.ticker import AutoLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt
import orbit_io
import summary_io
from scipy import interpolate
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths.')


# Initialize the classes, read in the data, and create data masks
summary = summary_io.SummaryDataSort()
summary_plot = summary_io.SummaryDataPlot()
data_total = summary.data_read(directory=sim_data.home_dir, hosts='all_no_r', sim_type='baryon')
masks_infall = summary.data_mask(data_total, peri_sim=False, peri_model=False, hosts='all_no_r')
masks_infall_peri = summary.data_mask(data_total, peri_sim=True, peri_model=False, hosts='all_no_r')
masks_infall_apo = summary.data_mask_apo(data_total, hosts='all_no_r')
masks_infall['m12f'][59] = False # used to be satellite 57 in the older data
masks_infall_peri['m12f'][59] = False
masks_infall_apo['m12f'][59] = False

# Select which mask you want to use and the corresponding directory
directory = sim_data.home_dir+'/orbit_data/plots/summary/paper_2'

d_rec_sim = summary.dperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_sim = summary.dperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
d_rec_mod = summary.dperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
d_min_mod = summary.dperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_sim = summary.tperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_sim = summary.tperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_rec_mod = summary.tperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_min_mod = summary.tperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_sim = summary.nperi(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_mod_infall = summary.nperi_model(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
n_mod_r200 = summary.nperi_model(data_total, masks_infall, selection='model.R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_sim = summary.vperi_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_sim = summary.vperi_min(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
v_rec_mod = summary.vperi_recent(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
v_min_mod = summary.vperi_min(data_total, masks_infall_peri, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_sim = summary.dapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dapo_rec_mod = summary.dapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_sim = summary.dmax(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
dmax_mod = summary.dmax(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
tapo_rec_sim = summary.tapo_recent(data_total, masks_infall_apo, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
tapo_rec_mod = summary.tapo_recent(data_total, masks_infall_apo, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_sim = summary.first_infall(data_total, masks_infall, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod = summary.first_infall(data_total, masks_infall, selection='model', oversample=True, hosts='all_no_r', sim_type='baryon')
t_in_mod_R200m = summary.infall_fixed(data_total, masks_infall, oversample=True, hosts='all_no_r', sim_type='baryon')
mask_in_mod = np.isfinite(t_in_mod)
mask_in_mod_R200m = np.isfinite(t_in_mod_R200m)
#ecc = summary.eccentricity_recent(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
#ecc_model = summary.eccentricity_recent(data_total, masks_infall_peri, selection='model.apsis', oversample=True, hosts='all_no_r', sim_type='baryon')
#mask_finite_sim = np.isfinite(ecc)
#mask_finite_mod = np.isfinite(ecc_model)
#
ecc = []
ecc_model = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['infall.check'])):
        if data_total[name]['infall.check'][i]:
            if (data_total[name]['eccentricity.sim'][i][0] != -1) and (data_total[name]['eccentricity.model.apsis'][i][0] != -1):
                ecc.append(np.repeat(data_total[name]['eccentricity.sim'][i][0], summary.oversample['baryon'][name]))
                ecc_model.append(np.repeat(data_total[name]['eccentricity.model.apsis'][i][0], summary.oversample['baryon'][name]))
ecc = np.hstack(ecc)
ecc_model = np.hstack(ecc_model)
#
per = []
per_model = []
for name in summary.host_names['all_no_r']:
    for i in range(0, len(data_total[name]['infall.check'])):
        if data_total[name]['infall.check'][i]:
            if (data_total[name]['orbit.period.peri.sim'][i][0] != -1) and (data_total[name]['orbit.period.peri.model'][i][0] != -1):
                per.append(np.repeat(data_total[name]['orbit.period.peri.sim'][i][0], summary.oversample['baryon'][name]))
                per_model.append(np.repeat(data_total[name]['orbit.period.peri.model'][i][0], summary.oversample['baryon'][name]))
per = np.hstack(per)
per_model = np.hstack(per_model)



def width_of_68(x_array):
    onesigp = 84.13
    onesigm = 15.87
    #
    upper = np.percentile(x_array, onesigp)
    lower = np.percentile(x_array, onesigm)
    #
    #return (upper-lower, (upper-lower)/2)
    return (upper-lower)/2

def width_of_95(x_array):
    onesigp = 97.72
    onesigm = 2.28
    #
    upper = np.percentile(x_array, onesigp)
    lower = np.percentile(x_array, onesigm)
    #
    #return (upper-lower, (upper-lower)/2)
    return (upper-lower)/2

print('The width of the 68-ile for d_rec is {}'.format(width_of_68(x_array=(d_rec_mod-d_rec_sim))))
print('The width of the 68-ile for d_min is {}'.format(width_of_68(x_array=(d_min_mod-d_min_sim))))
print('The width of the 68-ile for t_rec is {}'.format(width_of_68(x_array=(t_rec_mod-t_rec_sim))))
print('The width of the 68-ile for N_r is {}'.format(width_of_68(x_array=(n_sim-n_mod_mod_infall))))
print('The width of the 68-ile for N_r200 is {}'.format(width_of_68(x_array=(n_sim-n_mod_r200))))
print('The width of the 68-ile for v_rec is {}'.format(width_of_68(x_array=(v_rec_mod-v_rec_sim))))
print('The width of the 68-ile for v_min is {}'.format(width_of_68(x_array=(v_min_mod-v_min_sim))))
print('The width of the 68-ile for d_apo is {}'.format(width_of_68(x_array=(dapo_rec_mod-dapo_rec_sim))))
print('The width of the 68-ile for t_infall is {}'.format(width_of_68(x_array=(t_in_mod[mask_in_mod]-t_in_sim[mask_in_mod]))))
print('The width of the 68-ile for t_infall,R200m is {}'.format(width_of_68(x_array=(t_in_mod_R200m[mask_in_mod_R200m]-t_in_sim[mask_in_mod_R200m]))))
#print('The width of the 68-ile for eccentricity is {}'.format(width_of_68(x_array=(ecc_model-ecc))))
#print('The width of the 68-ile for period is {}'.format(width_of_68(x_array=(per_model-per))))

print('The width of the 68-ile for d_rec is {}'.format(width_of_68(x_array=(d_rec_mod-d_rec_sim)/d_rec_sim)))
print('The width of the 68-ile for d_min is {}'.format(width_of_68(x_array=(d_min_mod-d_min_sim)/d_min_sim)))
print('The width of the 68-ile for t_rec is {}'.format(width_of_68(x_array=(t_rec_mod-t_rec_sim)/t_rec_sim)))
print('The width of the 68-ile for N_r is {}'.format(width_of_68(x_array=(n_sim-n_mod_mod_infall)/n_mod_mod_infall)))
print('The width of the 68-ile for N_r200 is {}'.format(width_of_68(x_array=(n_sim-n_mod_r200)/n_mod_r200)))
print('The width of the 68-ile for v_rec is {}'.format(width_of_68(x_array=(v_rec_mod-v_rec_sim)/v_rec_sim)))
print('The width of the 68-ile for v_min is {}'.format(width_of_68(x_array=(v_min_mod-v_min_sim)/v_min_sim)))
print('The width of the 68-ile for d_apo is {}'.format(width_of_68(x_array=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim)))
print('The width of the 68-ile for t_infall is {}'.format(width_of_68(x_array=(t_in_mod[mask_in_mod]-t_in_sim[mask_in_mod])/t_in_sim[mask_in_mod])))
print('The width of the 68-ile for t_infall,R200m is {}'.format(width_of_68(x_array=(t_in_mod_R200m[mask_in_mod_R200m]-t_in_sim[mask_in_mod_R200m])/t_in_sim[mask_in_mod_R200m])))

print('The width of the 95-ile for d_rec is {}'.format(width_of_95(x_array=(d_rec_mod-d_rec_sim))))
print('The width of the 95-ile for d_min is {}'.format(width_of_95(x_array=(d_min_mod-d_min_sim))))
print('The width of the 95-ile for t_rec is {}'.format(width_of_95(x_array=(t_rec_mod-t_rec_sim))))
print('The width of the 95-ile for N_r is {}'.format(width_of_95(x_array=(n_sim-n_mod_mod_infall))))
print('The width of the 95-ile for N_r200 is {}'.format(width_of_95(x_array=(n_sim-n_mod_r200))))
print('The width of the 95-ile for v_rec is {}'.format(width_of_95(x_array=(v_rec_mod-v_rec_sim))))
print('The width of the 95-ile for v_min is {}'.format(width_of_95(x_array=(v_min_mod-v_min_sim))))
print('The width of the 95-ile for d_apo is {}'.format(width_of_95(x_array=(dapo_rec_mod-dapo_rec_sim))))
print('The width of the 95-ile for t_infall is {}'.format(width_of_95(x_array=(t_in_mod[mask_in_mod]-t_in_sim[mask_in_mod]))))
print('The width of the 95-ile for t_infall,R200m is {}'.format(width_of_95(x_array=(t_in_mod_R200m[mask_in_mod_R200m]-t_in_sim[mask_in_mod_R200m]))))

print('The width of the 95-ile for d_rec is {}'.format(width_of_95(x_array=(d_rec_mod-d_rec_sim)/d_rec_sim)))
print('The width of the 95-ile for d_min is {}'.format(width_of_95(x_array=(d_min_mod-d_min_sim)/d_min_sim)))
print('The width of the 95-ile for t_rec is {}'.format(width_of_95(x_array=(t_rec_mod-t_rec_sim)/t_rec_sim)))
print('The width of the 95-ile for N_r is {}'.format(width_of_95(x_array=(n_sim-n_mod_mod_infall)/n_mod_mod_infall)))
print('The width of the 95-ile for N_r200 is {}'.format(width_of_95(x_array=(n_sim-n_mod_r200)/n_mod_r200)))
print('The width of the 95-ile for v_rec is {}'.format(width_of_95(x_array=(v_rec_mod-v_rec_sim)/v_rec_sim)))
print('The width of the 95-ile for v_min is {}'.format(width_of_95(x_array=(v_min_mod-v_min_sim)/v_min_sim)))
print('The width of the 95-ile for d_apo is {}'.format(width_of_95(x_array=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim)))
print('The width of the 95-ile for t_infall is {}'.format(width_of_95(x_array=(t_in_mod[mask_in_mod]-t_in_sim[mask_in_mod])/t_in_sim[mask_in_mod])))
print('The width of the 95-ile for t_infall,R200m is {}'.format(width_of_95(x_array=(t_in_mod_R200m[mask_in_mod_R200m]-t_in_sim[mask_in_mod_R200m])/t_in_sim[mask_in_mod_R200m])))
