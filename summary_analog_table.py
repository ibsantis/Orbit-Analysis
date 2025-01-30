#!/usr/bin/python3

"""
    =============================
    = Paper III Summary Table =
    =============================

    ...
"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import pandas as pd
import satellite_io
import matplotlib
from matplotlib import pyplot as plt
import time
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
sat_analysis = satellite_io.SatelliteAnalysis(gal1='m12i', location=loc)
#
print('Set paths')

# Read in the snapshot dictionary and the entire tree
lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus', 'm12n']
                
mw_sats_1Mpc =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
                    'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
                    'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
                    'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus III', 'Eridanus IV', \
                    'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
                    'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
                    'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
                    'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
                    'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sagittarius II', 'Sculptor', 'Segue 1', \
                    'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
                    'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
                    'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']

# Work on master plots
sat_mstar = []
sat_dist = []
v_tan = []
#
first_infall = []
nperi = []
tperi_rec = []
dperi_rec = []
vperi_rec = []
tperi_min = []
dperi_min = []
vperi_min = []
tapo_rec = []
dapo_rec = []
elltot = []
ketot = []
mhalo = []
#
for galaxy in mw_sats_1Mpc:
    #
    satellite_name = galaxy.replace(' ', '_')
    file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
    #
    if len(gal_data['Host']) == 0:
        first_infall.append((-1, -1, -1))
        nperi.append((-1, -1, -1))
        tperi_rec.append((-1, -1, -1))
        dperi_rec.append((-1, -1, -1))
        vperi_rec.append((-1, -1, -1))
        tperi_min.append((-1, -1, -1))
        dperi_min.append((-1, -1, -1))
        vperi_min.append((-1, -1, -1))
        tapo_rec.append((-1, -1, -1))
        dapo_rec.append((-1, -1, -1))
        elltot.append((-1, -1, -1))
        ketot.append((-1, -1, -1))
        continue
    #
    sat_mstar.append(lg_data[galaxy]['mass.star'])
    sat_dist.append(lg_data[galaxy]['host.distance.total'])
    #
    orbit_dictionary = dict()
    orbit_dictionary['first.infall.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.num'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.vel'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['pericenter.min.vel'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['apocenter.time.lb'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['apocenter.dist'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['halo.mass.peak'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['L.tot.sim'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['distance'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.rad'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['velocity.tan'] = np.zeros(gal_data.shape[0])
    orbit_dictionary['v.tot.sim'] = np.zeros(gal_data.shape[0])
    #
    for sim_name in galaxies:
        if sim_name in np.array(gal_data['Host']):
            # Read in the mini data and snapshot information
            mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=True)
            snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)
            #
            orbit_history = sat_analysis.orbit_property_distribution(sim_name, mini_data, gal_data, snaps)
            mask = np.where(sim_name == gal_data['Host'])[0]
            for key in orbit_history.keys():
                orbit_dictionary[key][mask] = orbit_history[key]
    if len(orbit_history['distance']) == 0:
        first_infall.append((-1, -1, -1))
        nperi.append((-1, -1, -1))
        tperi_rec.append((-1, -1, -1))
        dperi_rec.append((-1, -1, -1))
        vperi_rec.append((-1, -1, -1))
        tperi_min.append((-1, -1, -1))
        dperi_min.append((-1, -1, -1))
        vperi_min.append((-1, -1, -1))
        tapo_rec.append((-1, -1, -1))
        dapo_rec.append((-1, -1, -1))
        elltot.append((-1, -1, -1))
        ketot.append((-1, -1, -1))
        continue
    #
    # Infall times
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        first_infall.append((x_med, x_lower, x_upper))
    else:
        first_infall.append((-1, -1, -1))
    #
    # Pericenter number
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.num'][m]
        x_med = np.sum(x*gal_data['Weight'][m])
        x_std = np.sqrt(np.sum((x-x_med)**2*gal_data['Weight'][m])/np.sum(gal_data['Weight'][m])/np.sum(gal_data['Weight'][m]))
        nperi.append((x_med, x_std, -1))
    else:
        nperi.append((-1, -1, -1))
    #
    # Recent pericenter time
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        tperi_rec.append((x_med, x_lower, x_upper))
    else:
        tperi_rec.append((-1, -1, -1))
    #
    # Recent pericenter distance
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        dperi_rec.append((x_med, x_lower, x_upper))
    else:
        dperi_rec.append((-1, -1, -1))
    #
    # Recent pericenter velocity
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        vperi_rec.append((x_med, x_lower, x_upper))
    else:
        vperi_rec.append((-1, -1, -1))
    #
    # Minimum pericenter time
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        tperi_min.append((x_med, x_lower, x_upper))
    else:
        tperi_min.append((-1, -1, -1))
    #
    # Minimum pericenter distance
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        dperi_min.append((x_med, x_lower, x_upper))
    else:
        dperi_min.append((-1, -1, -1))
    #
    # Minimum pericenter velocity
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        vperi_min.append((x_med, x_lower, x_upper))
    else:
        vperi_min.append((-1, -1, -1))
    #
    # Recnet apocenter time
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        tapo_rec.append((x_med, x_lower, x_upper))
    else:
        tapo_rec.append((-1, -1, -1))
    #
    # Recent apocenter distance
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['apocenter.dist'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        dapo_rec.append((x_med, x_lower, x_upper))
    else:
        dapo_rec.append((-1, -1, -1))
    #
    # Angular momentum at match
    m = (orbit_dictionary['L.tot.sim'] != -1)
    if np.sum(m) != 0:
        x = orbit_dictionary['L.tot.sim'][m]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        elltot.append((x_med, x_lower, x_upper))
    else:
        elltot.append((-1, -1, -1))
    #
    # Specific Kinetic Energy at match
    m = (orbit_dictionary['v.tot.sim'] != -1)
    if np.sum(m) != 0:
        x = 0.5*orbit_dictionary['v.tot.sim'][m]**2
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m])
        ketot.append((x_med, x_lower, x_upper))
    else:
        ketot.append((-1, -1, -1))
    

"""
    Infall time plots
"""
first_infall = np.asarray(first_infall)
mask = (first_infall[:,0] != -1)
meds = first_infall[:,0]
lowers = first_infall[:,1]
uppers = first_infall[:,2]


"""
    Pericenter number plots
"""
orbit_prop = np.asarray(nperi)
mask = (orbit_prop[:,0] != -1)
means = orbit_prop[:,0]
stds = orbit_prop[:,1]



"""
    Recent pericenter time plots
"""
orbit_prop = np.asarray(tperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]


"""
    Recent pericenter distance plots
"""
orbit_prop = np.asarray(dperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]




"""
    Recent pericenter velocity plots
"""
orbit_prop = np.asarray(vperi_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]




"""
    Minimum pericenter time plots
"""
orbit_prop = np.asarray(tperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]




"""
    Minimum pericenter distance plots
"""
orbit_prop = np.asarray(dperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]



"""
    Minimum pericenter velocity plots
"""
orbit_prop = np.asarray(vperi_min)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]





"""
    Recent apocenter time plots
"""
orbit_prop = np.asarray(tapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]



"""
    Recent apocenter distance plots
"""
orbit_prop = np.asarray(dapo_rec)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]




"""
    Specific Angular momentum plots
"""
orbit_prop = np.asarray(elltot)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]




"""
    Kinetic Energy plots
"""
orbit_prop = np.asarray(ketot)
mask = (orbit_prop[:,0] != -1)
meds = orbit_prop[:,0]
lowers = orbit_prop[:,1]
uppers = orbit_prop[:,2]








