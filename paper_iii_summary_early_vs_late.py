
#!/usr/bin/python3

"""
    =============================
    = Paper III Summary Figures =
    =============================

    Create plots showing various orbit history properties 
    as a function of either distance or stellar mass. This will
    plot each MW satellite as a point, and the error-bars will
    show the 68% scatter among the subhalo analogs for that
    MW satellite.
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

# Based on Horta et al
galaxies_early = ['m12f', 'm12i', 'Romeo', 'Juliet', 'Thelma', 'Louise']
galaxies_later = ['m12b', 'm12c', 'm12m', 'm12w','Romulus', 'Remus']


# mw_sats_1Mpc =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
#                     'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
#                     'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
#                     'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus III', 'Eridanus IV', \
#                     'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
#                     'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
#                     'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
#                     'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
#                     'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sagittarius II', 'Sculptor', 'Segue 1', \
#                     'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
#                     'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
#                     'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']

mw_sats_1Mpc_no_GC =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
                    'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
                    'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
                    'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus IV', \
                    'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
                    'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
                    'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
                    'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
                    'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sculptor', 'Segue 1', \
                    'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
                    'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
                    'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']


# Work on master plots
sat_mstar = []
sat_dist = []
v_tan = []
v_rad = []
#
first_infall_early = []
nperi_early = []
tperi_rec_early = []
dperi_rec_early = []
vperi_rec_early = []
tperi_min_early = []
dperi_min_early = []
vperi_min_early = []
tapo_rec_early = []
dapo_rec_early = []
ell_early = []
ketot_early = []
mhalo_early = []
dreion_early = []
#
first_infall_later = []
nperi_later = []
tperi_rec_later = []
dperi_rec_later = []
vperi_rec_later = []
tperi_min_later = []
dperi_min_later = []
vperi_min_later = []
tapo_rec_later = []
dapo_rec_later = []
ell_later = []
ketot_later = []
mhalo_later = []
dreion_later = []
#
n_555 = [12, 330, 378, 83, 131, 3, 2, 305, 42, 213, 15, 13, 75, 89, 125, 5, 243, 153, 43, 13, 26, 57, 125, 4, 99, 99, 111, 163, 301, 212, 49, 52, 0, 25, 167, 255, 347, 18, 67, 517, 362, 178, 13, 387, 127, 77, 363, 104, 954, 2, 4, 103, 20, 12, 42, 17, 15, 151, 0, 183, 113, 193, 261, 20, 62, 87, 45, 53]

#galaxy = 'Sculptor'
for sat_idx, galaxy in enumerate(mw_sats_1Mpc_no_GC):
    #
    satellite_name = galaxy.replace(' ', '_')
    if n_555[sat_idx] < 10:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_10_10_10/weights_{satellite_name}.txt'
    else:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
    #
    if len(gal_data['Host']) == 0:
        continue
    #
    sat_mstar.append(lg_data[galaxy]['mass.star'])
    sat_dist.append(lg_data[galaxy]['host.distance.total'])
    v_rad.append(lg_data[galaxy]['host.velocity.rad'])
    v_tan.append(lg_data[galaxy]['host.velocity.tan'])
    #
    orbit_dictionary = dict()
    orbit_dictionary['first.infall.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.num'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.dist'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.rec.vel'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.min.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.min.dist'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['pericenter.min.vel'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['apocenter.time.lb'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['apocenter.dist'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['L.tot.sim'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['halo.mass.peak'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['distance'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['velocity.rad'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['velocity.tan'] = (-1)*np.ones(gal_data.shape[0])
    orbit_dictionary['v.tot.sim'] = (-1)*np.ones(gal_data.shape[0])
    reionization_distances = np.zeros(gal_data.shape[0])
    #
    maskEarly = np.array([True if gal_data['Host'][i] in galaxies_early else False for i in range(len(gal_data['Host']))])
    maskLater = np.array([True if gal_data['Host'][i] in galaxies_later else False for i in range(len(gal_data['Host']))])
    #
    for sim_name in galaxies:
        if sim_name in np.array(gal_data['Host']):
            # Read in the mini data and snapshot information
            mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=True)
            snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)
            #
            orbit_history = sat_analysis.orbit_property_distribution(sim_name, mini_data, gal_data, snaps)
            reionDists = sat_analysis.reionization_distance(sim_name, mini_data, gal_data, snaps)
            mask = np.where(sim_name == gal_data['Host'])[0]
            for key in orbit_history.keys():
                orbit_dictionary[key][mask] = orbit_history[key]
            reionization_distances[mask] = reionDists[mask]
    if len(orbit_history['distance']) == 0:
        continue
    #
    # Reionization distances
    m = (reionization_distances >= 0)
    if np.sum(m * maskEarly) != 0:
        x = reionization_distances[m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        dreion_early.append((x_med, x_lower, x_upper))
        #
    else:
        dreion_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = reionization_distances[m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        dreion_later.append((x_med, x_lower, x_upper))
    else:
        dreion_later.append((-1, -1, -1))
    #
    # Infall times
    m = (orbit_dictionary['first.infall.time.lb'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        first_infall_early.append((x_med, x_lower, x_upper))
        #
    else:
        first_infall_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['first.infall.time.lb'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        first_infall_later.append((x_med, x_lower, x_upper))
    else:
        first_infall_later.append((-1, -1, -1))
    #
    # Pericenter number
    m = (orbit_dictionary['pericenter.num'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['pericenter.num'][m * maskEarly]
        x_med = np.sum(x*gal_data['Weight'][m * maskEarly])
        x_std = np.sqrt(np.sum((x-x_med)**2*gal_data['Weight'][m * maskEarly])/np.sum(gal_data['Weight'][m * maskEarly])/np.sum(gal_data['Weight'][m * maskEarly]))
        nperi_early.append((x_med, x_std, -1))
    else:
        nperi_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['pericenter.num'][m * maskLater]
        x_med = np.sum(x*gal_data['Weight'][m * maskLater])
        x_std = np.sqrt(np.sum((x-x_med)**2*gal_data['Weight'][m * maskLater])/np.sum(gal_data['Weight'][m * maskLater])/np.sum(gal_data['Weight'][m * maskLater]))
        nperi_later.append((x_med, x_std, -1))
    else:
        nperi_later.append((-1, -1, -1))
    #
    # Recent pericenter time
    m = (orbit_dictionary['pericenter.rec.time.lb'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        tperi_rec_early.append((x_med, x_lower, x_upper))
    else:
        tperi_rec_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['pericenter.rec.time.lb'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        tperi_rec_later.append((x_med, x_lower, x_upper))
    else:
        tperi_rec_later.append((-1, -1, -1))
    #
    # Recent pericenter distance
    m = (orbit_dictionary['pericenter.rec.dist'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        dperi_rec_early.append((x_med, x_lower, x_upper))
    else:
        dperi_rec_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['pericenter.rec.dist'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        dperi_rec_later.append((x_med, x_lower, x_upper))
    else:
        dperi_rec_later.append((-1, -1, -1))
    #
    # Recent pericenter velocity
    m = (orbit_dictionary['pericenter.rec.vel'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        vperi_rec_early.append((x_med, x_lower, x_upper))
    else:
        vperi_rec_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['pericenter.rec.vel'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        vperi_rec_later.append((x_med, x_lower, x_upper))
    else:
        vperi_rec_later.append((-1, -1, -1))
    #
    # Minimum pericenter time
    m = (orbit_dictionary['pericenter.min.time.lb'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        tperi_min_early.append((x_med, x_lower, x_upper))
    else:
        tperi_min_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['pericenter.min.time.lb'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        tperi_min_later.append((x_med, x_lower, x_upper))
    else:
        tperi_min_later.append((-1, -1, -1))
    #
    # Minimum pericenter distance
    m = (orbit_dictionary['pericenter.min.dist'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        dperi_min_early.append((x_med, x_lower, x_upper))
    else:
        dperi_min_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['pericenter.min.dist'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        dperi_min_later.append((x_med, x_lower, x_upper))
    else:
        dperi_min_later.append((-1, -1, -1))
    #
    # Minimum pericenter velocity
    m = (orbit_dictionary['pericenter.min.vel'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        vperi_min_early.append((x_med, x_lower, x_upper))
    else:
        vperi_min_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['pericenter.min.vel'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        vperi_min_later.append((x_med, x_lower, x_upper))
    else:
        vperi_min_later.append((-1, -1, -1))
    #
    # Recnet apocenter time
    m = (orbit_dictionary['apocenter.time.lb'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        tapo_rec_early.append((x_med, x_lower, x_upper))
    else:
        tapo_rec_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['apocenter.time.lb'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        tapo_rec_later.append((x_med, x_lower, x_upper))
    else:
        tapo_rec_later.append((-1, -1, -1))
    #
    # Recent apocenter distance
    m = (orbit_dictionary['apocenter.dist'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['apocenter.dist'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        dapo_rec_early.append((x_med, x_lower, x_upper))
    else:
        dapo_rec_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['apocenter.dist'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        dapo_rec_later.append((x_med, x_lower, x_upper))
    else:
        dapo_rec_later.append((-1, -1, -1))
    #
    # Specific Angular Momentum
    m = (orbit_dictionary['L.tot.sim'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = orbit_dictionary['L.tot.sim'][m * maskEarly]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        ell_early.append((x_med, x_lower, x_upper))
    else:
        ell_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = orbit_dictionary['L.tot.sim'][m * maskLater]
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        ell_later.append((x_med, x_lower, x_upper))
    else:
        ell_later.append((-1, -1, -1))
    #
    # Kinetic energy
    m = (orbit_dictionary['v.tot.sim'] != -1)
    if np.sum(m * maskEarly) != 0:
        x = 0.5*orbit_dictionary['v.tot.sim'][m * maskEarly]**2
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskEarly])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskEarly])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskEarly])
        ketot_early.append((x_med, x_lower, x_upper))
    else:
        ketot_early.append((-1, -1, -1))
    #
    if np.sum(m * maskLater) != 0:
        x = 0.5*orbit_dictionary['v.tot.sim'][m * maskLater]**2
        x_med = ut.math.percentile_weighted(x, 50, gal_data['Weight'][m * maskLater])
        x_lower = ut.math.percentile_weighted(x, 15.87, gal_data['Weight'][m * maskLater])
        x_upper = ut.math.percentile_weighted(x, 84.13, gal_data['Weight'][m * maskLater])
        ketot_later.append((x_med, x_lower, x_upper))
    else:
        ketot_later.append((-1, -1, -1))

sat_mstar = np.asarray(sat_mstar)
sat_dist = np.asarray(sat_dist)
v_rad = np.asarray(v_rad)
v_tan = np.asarray(v_tan)

"""
    Infall time plots
"""
orbit_prop = np.asarray(first_infall_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(first_infall_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/infall_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/infall_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
axs.set_xlim(0, 420)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/infall_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/infall_vs_vrad.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/infall_vs_vtan.pdf')
plt.close()




"""
    Pericenter number plots
"""
orbit_prop = np.asarray(nperi_early)
mask_early = (orbit_prop[:,0] != -1)
means_early = orbit_prop[:,0]
stds_early = orbit_prop[:,1]
#
orbit_prop = np.asarray(nperi_later)
mask_later = (orbit_prop[:,0] != -1)
means_later = orbit_prop[:,0]
stds_later = orbit_prop[:,1]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], means_early[mask_early][0], yerr=stds_early[mask_early][0], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], means_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], means_later[mask_later][0], yerr=stds_later[mask_later][0], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], means_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], means_early[mask_early][i], yerr=stds_early[mask_early][i], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], means_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
#
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], means_later[mask_later][i], yerr=stds_later[mask_later][i], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], means_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(ymin=0)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/nperi_vs_mstar.pdf')
plt.close()

# Vs Mstar (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], means_early[mask_early][0], yerr=stds_early[mask_early][0], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], means_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], means_later[mask_later][0], yerr=stds_later[mask_later][0], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], means_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], means_early[mask_early][i], yerr=stds_early[mask_early][i], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], means_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
#
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], means_later[mask_later][i], yerr=stds_later[mask_later][i], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], means_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xscale('log')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(0,8)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/nperi_vs_mstar_zoom.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], means_early[mask_early][0], yerr=stds_early[mask_early][0], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], means_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], means_later[mask_later][0], yerr=stds_later[mask_later][0], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], means_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], means_early[mask_early][i], yerr=stds_early[mask_early][i], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], means_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
#
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], means_later[mask_later][i], yerr=stds_later[mask_later][i], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], means_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(ymin=0)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/nperi_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], means_early[mask_early][0], yerr=stds_early[mask_early][0], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], means_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], means_later[mask_later][0], yerr=stds_later[mask_later][0], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], means_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], means_early[mask_early][i], yerr=stds_early[mask_early][i], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], means_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
#
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], means_later[mask_later][i], yerr=stds_later[mask_later][i], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], means_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Host distance [kpc]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
axs.set_ylim(0, 8)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/nperi_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], means_early[mask_early][0], yerr=stds_early[mask_early][0], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], means_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], means_later[mask_later][0], yerr=stds_later[mask_later][0], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], means_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], means_early[mask_early][i], yerr=stds_early[mask_early][i], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], means_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
#
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], means_later[mask_later][i], yerr=stds_later[mask_later][i], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], means_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(ymin=0)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/nperi_vs_vrad.pdf')
plt.close()

# Vs tan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], means_early[mask_early][0], yerr=stds_early[mask_early][0], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], means_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], means_later[mask_later][0], yerr=stds_later[mask_later][0], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], means_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], means_early[mask_early][i], yerr=stds_early[mask_early][i], color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], means_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
#
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], means_later[mask_later][i], yerr=stds_later[mask_later][i], color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], means_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$N_{\\rm peri}$', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_ylim(ymin=0)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/nperi_vs_vtan.pdf')
plt.close()



"""
    Recent pericenter time plots
"""
orbit_prop = np.asarray(tperi_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(tperi_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_rec_vs_mstar.pdf')
plt.close()

# Vs Mstar (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.set_ylim(0, 5)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_rec_vs_mstar_zoom.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
axs.set_ylim(0, 5)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_rec_vs_vtan.pdf')
plt.close()



"""
    Recent pericenter distance plots
"""
orbit_prop = np.asarray(dperi_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(dperi_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_rec_vs_mstar.pdf')
plt.close()

# Vs Mstar (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.set_ylim(0, 150)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_rec_vs_mstar_zoom.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_rec_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
axs.set_ylim(0, 150)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_rec_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_rec_vs_vtan.pdf')



"""
    Minimum pericenter time plots
"""
orbit_prop = np.asarray(tperi_min_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(tperi_min_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_min_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_min_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm peri, min}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tperi_min_vs_vtan.pdf')
plt.close()



"""
    Minimum pericenter distance plots
"""
orbit_prop = np.asarray(dperi_min_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(dperi_min_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_min_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_min_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_min_vs_dist_zoom.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_min_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm peri, min}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dperi_min_vs_vtan.pdf')
plt.close()



"""
    Recent apocenter time plots
"""
orbit_prop = np.asarray(tapo_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(tapo_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tapo_rec_vs_dist.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tapo_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$t_{\\rm apo, rec}$ [Gyr]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/tapo_rec_vs_vtan.pdf')
plt.close()


"""
    Recent apocenter distance plots
"""
orbit_prop = np.asarray(dapo_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(dapo_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dapo_rec_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dapo_rec_vs_dist.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dapo_rec_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d_{\\rm apo, rec}$ [kpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dapo_rec_vs_vtan.pdf')
plt.close()



"""
    Reionization distance plots
"""
orbit_prop = np.asarray(dreion_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]/1000
lowers_early = orbit_prop[:,1]/1000
uppers_early = orbit_prop[:,2]/1000
#
orbit_prop = np.asarray(dreion_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]/1000
lowers_later = orbit_prop[:,1]/1000
uppers_later = orbit_prop[:,2]/1000

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$d(z = 7)$ [Mpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dreion_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$d(z = 7)$ [Mpc]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dreion_vs_dist.pdf')
plt.close()

# Vs vrad
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_rad[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_rad[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_rad[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_rad[mask_early])):
    # axs.errorbar(v_rad[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_rad[mask_later])):
    # axs.errorbar(v_rad[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_rad[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm rad}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d(z = 7)$ [Mpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dreion_vs_vrad.pdf')
plt.close()

# Vs vtan
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(v_tan[mask_early][0], meds_early[mask_early][0], yerr=np.array([[meds_early[mask_early][0]-lowers_early[mask_early][0]],[uppers_early[mask_early][0]-meds_early[mask_early][0]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_early][0], meds_early[mask_early][0], s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(v_tan[mask_later][0], meds_later[mask_later][0], yerr=np.array([[meds_later[mask_later][0]-lowers_later[mask_later][0]],[uppers_later[mask_later][0]-meds_later[mask_later][0]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(v_tan[mask_later][0], meds_later[mask_later][0], s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(v_tan[mask_early])):
    # axs.errorbar(v_tan[mask_early][i], meds_early[mask_early][i], yerr=np.array([[meds_early[mask_early][i]-lowers_early[mask_early][i]],[uppers_early[mask_early][i]-meds_early[mask_early][i]]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_early][i], meds_early[mask_early][i], s=50, c='#c76438', alpha=0.75)
for i in range(1, len(v_tan[mask_later])):
    # axs.errorbar(v_tan[mask_later][i], meds_later[mask_later][i], yerr=np.array([[meds_later[mask_later][i]-lowers_later[mask_later][i]],[uppers_later[mask_later][i]-meds_later[mask_later][i]]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(v_tan[mask_later][i], meds_later[mask_later][i], s=50, c='#389BC7', alpha=0.75)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$v_{\\rm tan}$ [km s$^{-1}$]', fontsize=24)
axs.set_ylabel('$d(z = 7)$ [Mpc]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/dreion_vs_vtan.pdf')
plt.close()



"""
    Specific Angular momentum plots
"""
orbit_prop = np.asarray(ell_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(ell_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0]/1e4, yerr=np.array([[meds_early[mask_early][0]/1e4-lowers_early[mask_early][0]/1e4],[uppers_early[mask_early][0]/1e4-meds_early[mask_early][0]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0]/1e4, s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0]/1e4, yerr=np.array([[meds_later[mask_later][0]/1e4-lowers_later[mask_later][0]/1e4],[uppers_later[mask_later][0]/1e4-meds_later[mask_later][0]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0]/1e4, s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i]/1e4, yerr=np.array([[meds_early[mask_early][i]/1e4-lowers_early[mask_early][i]/1e4],[uppers_early[mask_early][i]/1e4-meds_early[mask_early][i]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i]/1e4, s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i]/1e4, yerr=np.array([[meds_later[mask_later][i]/1e4-lowers_later[mask_later][i]/1e4],[uppers_later[mask_later][i]/1e4-meds_later[mask_later][i]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i]/1e4, s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/ell_vs_mstar.pdf')
plt.close()

# Vs Mstar (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0]/1e4, yerr=np.array([[meds_early[mask_early][0]/1e4-lowers_early[mask_early][0]/1e4],[uppers_early[mask_early][0]/1e4-meds_early[mask_early][0]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0]/1e4, s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0]/1e4, yerr=np.array([[meds_later[mask_later][0]/1e4-lowers_later[mask_later][0]/1e4],[uppers_later[mask_later][0]/1e4-meds_later[mask_later][0]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0]/1e4, s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i]/1e4, yerr=np.array([[meds_early[mask_early][i]/1e4-lowers_early[mask_early][i]/1e4],[uppers_early[mask_early][i]/1e4-meds_early[mask_early][i]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i]/1e4, s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i]/1e4, yerr=np.array([[meds_later[mask_later][i]/1e4-lowers_later[mask_later][i]/1e4],[uppers_later[mask_later][i]/1e4-meds_later[mask_later][i]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i]/1e4, s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('$\\ell$ [10$^4$ kpc km s$^{-1}$]', fontsize=24)
axs.set_ylim(0, 5)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/ell_vs_mstar_zoom.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, yerr=np.array([[meds_early[mask_early][0]/1e4-lowers_early[mask_early][0]/1e4],[uppers_early[mask_early][0]/1e4-meds_early[mask_early][0]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, yerr=np.array([[meds_later[mask_later][0]/1e4-lowers_later[mask_later][0]/1e4],[uppers_later[mask_later][0]/1e4-meds_later[mask_later][0]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, yerr=np.array([[meds_early[mask_early][i]/1e4-lowers_early[mask_early][i]/1e4],[uppers_early[mask_early][i]/1e4-meds_early[mask_early][i]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, yerr=np.array([[meds_later[mask_later][i]/1e4-lowers_later[mask_later][i]/1e4],[uppers_later[mask_later][i]/1e4-meds_later[mask_later][i]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$\\ell$ [kpc km s$^{-1}$]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/ell_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, yerr=np.array([[meds_early[mask_early][0]/1e4-lowers_early[mask_early][0]/1e4],[uppers_early[mask_early][0]/1e4-meds_early[mask_early][0]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, yerr=np.array([[meds_later[mask_later][0]/1e4-lowers_later[mask_later][0]/1e4],[uppers_later[mask_later][0]/1e4-meds_later[mask_later][0]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, yerr=np.array([[meds_early[mask_early][i]/1e4-lowers_early[mask_early][i]/1e4],[uppers_early[mask_early][i]/1e4-meds_early[mask_early][i]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, yerr=np.array([[meds_later[mask_later][i]/1e4-lowers_later[mask_later][i]/1e4],[uppers_later[mask_later][i]/1e4-meds_later[mask_later][i]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('$\\ell$ [kpc km s$^{-1}$]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0, 420)
axs.set_ylim(0, 5)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/ell_vs_dist_zoom.pdf')
plt.close()



"""
    Kinetic Energy plots
"""
orbit_prop = np.asarray(ketot_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(ketot_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]

# Vs Mstar
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_mstar[mask_early][0], meds_early[mask_early][0]/1e4, yerr=np.array([[meds_early[mask_early][0]/1e4-lowers_early[mask_early][0]/1e4],[uppers_early[mask_early][0]/1e4-meds_early[mask_early][0]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_early][0], meds_early[mask_early][0]/1e4, s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_mstar[mask_later][0], meds_later[mask_later][0]/1e4, yerr=np.array([[meds_later[mask_later][0]/1e4-lowers_later[mask_later][0]/1e4],[uppers_later[mask_later][0]/1e4-meds_later[mask_later][0]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_mstar[mask_later][0], meds_later[mask_later][0]/1e4, s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_mstar[mask_early][i], meds_early[mask_early][i]/1e4, yerr=np.array([[meds_early[mask_early][i]/1e4-lowers_early[mask_early][i]/1e4],[uppers_early[mask_early][i]/1e4-meds_early[mask_early][i]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_early][i], meds_early[mask_early][i]/1e4, s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_mstar[mask_later][i], meds_later[mask_later][i]/1e4, yerr=np.array([[meds_later[mask_later][i]/1e4-lowers_later[mask_later][i]/1e4],[uppers_later[mask_later][i]/1e4-meds_later[mask_later][i]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_mstar[mask_later][i], meds_later[mask_later][i]/1e4, s=50, c='#389BC7', alpha=0.75)
axs.set_xscale('log')
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
axs.set_ylabel('Specific Kinetic Energy [10$^4$ km$^2$ s$^{-2}$]', fontsize=24)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/ke_vs_mstar.pdf')
plt.close()

# Vs distance
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, yerr=np.array([[meds_early[mask_early][0]/1e4-lowers_early[mask_early][0]/1e4],[uppers_early[mask_early][0]/1e4-meds_early[mask_early][0]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, yerr=np.array([[meds_later[mask_later][0]/1e4-lowers_later[mask_later][0]/1e4],[uppers_later[mask_later][0]/1e4-meds_later[mask_later][0]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, yerr=np.array([[meds_early[mask_early][i]/1e4-lowers_early[mask_early][i]/1e4],[uppers_early[mask_early][i]/1e4-meds_early[mask_early][i]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, yerr=np.array([[meds_later[mask_later][i]/1e4-lowers_later[mask_later][i]/1e4],[uppers_later[mask_later][i]/1e4-meds_later[mask_later][i]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Specific Kinetic Energy [10$^4$ km$^2$ s$^{-2}$]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/ke_vs_dist.pdf')
plt.close()

# Vs distance (zoom)
plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(10,8))
#
# axs.errorbar(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, yerr=np.array([[meds_early[mask_early][0]/1e4-lowers_early[mask_early][0]/1e4],[uppers_early[mask_early][0]/1e4-meds_early[mask_early][0]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_early][0], meds_early[mask_early][0]/1e4, s=50, c='#c76438', alpha=0.5, label='Early-forming hosts')
#
# axs.errorbar(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, yerr=np.array([[meds_later[mask_later][0]/1e4-lowers_later[mask_later][0]/1e4],[uppers_later[mask_later][0]/1e4-meds_later[mask_later][0]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
axs.scatter(sat_dist[mask_later][0], meds_later[mask_later][0]/1e4, s=50, c='#389BC7', alpha=0.5, label='Late-forming hosts')
#
for i in range(1, len(sat_mstar[mask_early])):
    # axs.errorbar(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, yerr=np.array([[meds_early[mask_early][i]/1e4-lowers_early[mask_early][i]/1e4],[uppers_early[mask_early][i]/1e4-meds_early[mask_early][i]/1e4]]), color='#c76438', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_early][i], meds_early[mask_early][i]/1e4, s=50, c='#c76438', alpha=0.75)
for i in range(1, len(sat_mstar[mask_later])):
    # axs.errorbar(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, yerr=np.array([[meds_later[mask_later][i]/1e4-lowers_later[mask_later][i]/1e4],[uppers_later[mask_later][i]/1e4-meds_later[mask_later][i]/1e4]]), color='#389BC7', alpha=0.5, lw=1.5, capsize=0)
    axs.scatter(sat_dist[mask_later][i], meds_later[mask_later][i]/1e4, s=50, c='#389BC7', alpha=0.75)
#
axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
axs.set_ylabel('Specific Kinetic Energy [10$^4$ km$^2$ s$^{-2}$]', fontsize=24)
axs.legend(prop={'size': 16}, loc='best')
axs.set_xlim(0,420)
plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/ke_vs_dist_zoom.pdf')
plt.close()



def making_plots(num, den, propname):
    #
    print('Some useful stuff:')
    outliers = (num/den)[num/den > 1]
    outliers_sats = np.asarray(mw_sats_1Mpc)[mask_early*mask_later][num/den > 1]
    print(f'    * Values above 1: {outliers}')
    print(f'    * Sats with values above 1: {outliers_sats}')
    # Vs Mstar
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    axs.scatter(sat_mstar[mask_early*mask_later], num/den, s=50, c='#c76438', alpha=0.75)
    axs.set_xscale('log')
    axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
    axs.set_ylabel('|$\\Delta$ median| / Average width of scatters', fontsize=24)
    axs.hlines(0, np.min(sat_mstar[mask_early*mask_later]), np.max(sat_mstar[mask_early*mask_later]), color='k', linestyle='dashed', alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/'+propname+'_vs_mass_comparison.pdf')
    plt.close()

    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    axs.scatter(sat_mstar[mask_early*mask_later], num/den, s=50, c='#c76438', alpha=0.75)
    axs.set_xscale('log')
    axs.set_xlabel('$M_{\\rm star}$ [$M_{\odot}$]', fontsize=24)
    axs.set_ylabel('|$\\Delta$ median| / Average width of scatters', fontsize=24)
    axs.set_ylim(-0.05,1.05)
    axs.hlines(0, np.min(sat_mstar[mask_early*mask_later]), np.max(sat_mstar[mask_early*mask_later]), color='k', linestyle='dashed', alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/'+propname+'_vs_mass_comparison_zoom.pdf')
    plt.close()

    # Vs distance
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    axs.scatter(sat_dist[mask_early*mask_later], num/den, s=50, c='#c76438', alpha=0.75)
    axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
    axs.set_ylabel('|$\\Delta$ median| / Average width of scatters', fontsize=24)
    axs.hlines(0, np.min(sat_dist[mask_early*mask_later]), np.max(sat_dist[mask_early*mask_later]), color='k', linestyle='dashed', alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/'+propname+'_vs_dist_comparison.pdf')
    plt.close()

    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    axs.scatter(sat_dist[mask_early*mask_later], num/den, s=50, c='#c76438', alpha=0.75)
    axs.set_xlabel('Distance from MW [kpc]', fontsize=24)
    axs.set_ylabel('|$\\Delta$ median| / Average width of scatters', fontsize=24)
    axs.set_ylim(-0.05,1.05)
    axs.set_xlim(0, 420)
    axs.hlines(0, np.min(sat_dist[mask_early*mask_later]), np.max(sat_dist[mask_early*mask_later]), color='k', linestyle='dashed', alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+'/orbit_data/plots/summary/paper_3/histories/early_vs_late/'+propname+'_vs_dist_comparison_zoom.pdf')
    plt.close()

# Infall time
first_infall_early = np.asarray(first_infall_early)
mask_early = (first_infall_early[:,0] != -1)
meds_early = first_infall_early[:,0]
lowers_early = first_infall_early[:,1]
uppers_early = first_infall_early[:,2]
#
first_infall_later = np.asarray(first_infall_later)
mask_later = (first_infall_later[:,0] != -1)
meds_later = first_infall_later[:,0]
lowers_later = first_infall_later[:,1]
uppers_later = first_infall_later[:,2]
#
numerator = np.abs(meds_early[mask_early*mask_later] - meds_later[mask_early*mask_later])
denominator = 0.5*((uppers_early[mask_early*mask_later] - lowers_early[mask_early*mask_later]) + (uppers_later[mask_early*mask_later] - lowers_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'infall')

# Pericenter number
orbit_prop = np.asarray(nperi_early)
mask_early = (orbit_prop[:,0] != -1)
means_early = orbit_prop[:,0]
stds_early = orbit_prop[:,1]
#
orbit_prop = np.asarray(nperi_later)
mask_later = (orbit_prop[:,0] != -1)
means_later = orbit_prop[:,0]
stds_later = orbit_prop[:,1]
#
numerator = np.abs(means_early[mask_early*mask_later] - means_later[mask_early*mask_later])
denominator = 0.5*((stds_early[mask_early*mask_later] - stds_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'nperi')

# Recent Pericenter Time
orbit_prop = np.asarray(tperi_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(tperi_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]
#
numerator = np.abs(meds_early[mask_early*mask_later] - meds_later[mask_early*mask_later])
denominator = 0.5*((uppers_early[mask_early*mask_later] - lowers_early[mask_early*mask_later]) + (uppers_later[mask_early*mask_later] - lowers_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'tperi_rec')

# Recent pericenter distance
orbit_prop = np.asarray(dperi_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(dperi_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]
#
numerator = np.abs(meds_early[mask_early*mask_later] - meds_later[mask_early*mask_later])
denominator = 0.5*((uppers_early[mask_early*mask_later] - lowers_early[mask_early*mask_later]) + (uppers_later[mask_early*mask_later] - lowers_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'dperi_rec')

# Minimum pericenter time
orbit_prop = np.asarray(tperi_min_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(tperi_min_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]
#
numerator = np.abs(meds_early[mask_early*mask_later] - meds_later[mask_early*mask_later])
denominator = 0.5*((uppers_early[mask_early*mask_later] - lowers_early[mask_early*mask_later]) + (uppers_later[mask_early*mask_later] - lowers_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'tperi_min')

# Minimum pericenter distance
orbit_prop = np.asarray(dperi_min_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(dperi_min_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]
#
numerator = np.abs(meds_early[mask_early*mask_later] - meds_later[mask_early*mask_later])
denominator = 0.5*((uppers_early[mask_early*mask_later] - lowers_early[mask_early*mask_later]) + (uppers_later[mask_early*mask_later] - lowers_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'dperi_min')

# Recent apocenter time
orbit_prop = np.asarray(tapo_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(tapo_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]
#
numerator = np.abs(meds_early[mask_early*mask_later] - meds_later[mask_early*mask_later])
denominator = 0.5*((uppers_early[mask_early*mask_later] - lowers_early[mask_early*mask_later]) + (uppers_later[mask_early*mask_later] - lowers_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'tapo_rec')

# Recent apocenter time
orbit_prop = np.asarray(dapo_rec_early)
mask_early = (orbit_prop[:,0] != -1)
meds_early = orbit_prop[:,0]
lowers_early = orbit_prop[:,1]
uppers_early = orbit_prop[:,2]
#
orbit_prop = np.asarray(dapo_rec_later)
mask_later = (orbit_prop[:,0] != -1)
meds_later = orbit_prop[:,0]
lowers_later = orbit_prop[:,1]
uppers_later = orbit_prop[:,2]
#
numerator = np.abs(meds_early[mask_early*mask_later] - meds_later[mask_early*mask_later])
denominator = 0.5*((uppers_early[mask_early*mask_later] - lowers_early[mask_early*mask_later]) + (uppers_later[mask_early*mask_later] - lowers_later[mask_early*mask_later]))
#
making_plots(numerator, denominator, 'tapo_rec')


