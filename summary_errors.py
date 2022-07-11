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
print('Set paths')


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
t_in_mod_R200m = summary.infall_diagnostics(data_total, masks_infall, selection='R200m', oversample=True, hosts='all_no_r', sim_type='baryon')
mask_in_mod = np.isfinite(t_in_mod)
mask_in_mod_R200m = np.isfinite(t_in_mod_R200m)
ecc = summary.eccentricity(data_total, masks_infall_peri, selection='sim', oversample=True, hosts='all_no_r', sim_type='baryon')
ecc_model = summary.eccentricity(data_total, masks_infall_peri, selection='model.apsis', oversample=True, hosts='all_no_r', sim_type='baryon')
mask_finite_sim = np.isfinite(ecc)
mask_finite_mod = np.isfinite(ecc_model)




def rmse(sim, mod):
    return np.sqrt(np.sum((sim-mod)**2)/len(sim))

print(rmse(d_rec_sim, d_rec_mod))
print(rmse(d_min_sim, d_min_mod))
print(rmse(t_rec_sim, t_rec_mod))
print(rmse(t_min_sim, t_min_mod))
print(rmse(n_sim, n_mod_mod_infall))
print(rmse(n_sim, n_mod_r200))
print(rmse(v_rec_sim, v_rec_mod))
print(rmse(v_min_sim, v_min_mod))
print(rmse(dapo_rec_sim, dapo_rec_mod))
print(rmse(dmax_sim, dmax_mod))
print(rmse(tapo_rec_sim, tapo_rec_mod))
print(rmse(t_in_sim[mask_in_mod], t_in_mod[mask_in_mod]))
print(rmse(t_in_sim[mask_in_mod_R200m], t_in_mod_R200m[mask_in_mod_R200m]))
print(rmse(ecc[mask_finite_sim*mask_finite_mod], ecc_model[mask_finite_sim*mask_finite_mod]))

def rmse_norm(sim, mod):
    results = []
    for i in range(0, len(sim)):
        if (sim[i] == 0) & (mod[i] == 0):
            results.append(1)
        elif (sim[i] == 0) & (mod[i] != 0):
            results.append(((1e-5-mod[i])/1e-5)**2)
        else:
            results.append(((sim[i]-mod[i])/sim[i])**2)
    results = np.asarray(results)
    return np.sqrt(np.sum(results)/len(sim))

print(rmse_norm(d_rec_sim, d_rec_mod))
print(rmse_norm(d_min_sim, d_min_mod))
print(rmse_norm(t_rec_sim, t_rec_mod))
print(rmse_norm(t_min_sim, t_min_mod))
print(rmse_norm(n_sim, n_mod_mod_infall))
print(rmse_norm(n_sim, n_mod_r200))
print(rmse_norm(v_rec_sim, v_rec_mod))
print(rmse_norm(v_min_sim, v_min_mod))
print(rmse_norm(dapo_rec_sim, dapo_rec_mod))
print(rmse_norm(dmax_sim, dmax_mod))
print(rmse_norm(tapo_rec_sim, tapo_rec_mod))
print(rmse_norm(t_in_sim[mask_in_mod], t_in_mod[mask_in_mod]))
print(rmse_norm(t_in_sim[mask_in_mod_R200m], t_in_mod_R200m[mask_in_mod_R200m]))
print(rmse_norm(ecc[mask_finite_sim*mask_finite_mod], ecc_model[mask_finite_sim*mask_finite_mod]))


def width_of_68(x_array):
    onesigp = 84.13
    onesigm = 15.87
    #
    upper = np.percentile(x_array, onesigp)
    lower = np.percentile(x_array, onesigm)
    #
    return (upper-lower, (upper-lower)/2)

print(width_of_68(x_array=(d_rec_mod-d_rec_sim)/d_rec_sim))
print(width_of_68(x_array=(d_min_mod-d_min_sim)/d_min_sim))
print(width_of_68(x_array=(t_rec_mod-t_rec_sim)))
print(width_of_68(x_array=(t_min_mod-t_min_sim)))
print(width_of_68(x_array=(n_sim-n_mod_mod_infall)))
print(width_of_68(x_array=(n_sim-n_mod_r200)))
print(width_of_68(x_array=(v_rec_mod-v_rec_sim)/v_rec_sim))
print(width_of_68(x_array=(v_min_mod-v_min_sim)/v_min_sim))
print(width_of_68(x_array=(dapo_rec_mod-dapo_rec_sim)/dapo_rec_sim))
print(width_of_68(x_array=(dmax_mod-dmax_sim)/dmax_sim))
print(width_of_68(x_array=(tapo_rec_mod-tapo_rec_sim)))
print(width_of_68(x_array=(t_in_mod[mask_in_mod]-t_in_sim[mask_in_mod])))
print(width_of_68(x_array=(t_in_mod_R200m[mask_in_mod_R200m]-t_in_sim[mask_in_mod_R200m])))
print(width_of_68(x_array=(ecc_model[mask_finite_sim*mask_finite_mod]-ecc[mask_finite_sim*mask_finite_mod])/ecc[mask_finite_sim*mask_finite_mod]))
