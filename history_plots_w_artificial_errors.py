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
galaxies_early = ['m12f', 'm12i', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise']
galaxies_later = ['m12b', 'm12c', 'm12m', 'Romulus', 'Remus']


mw_sats_1Mpc = ['Antlia 2', 'Aquarius 2', 'Bootes 1', 'Bootes 2', 'Bootes 3', \
                'Canes Venatici 1', 'Canes Venatici 2', 'Carina', 'Carina 2', \
                'Carina 3', 'Cetus 2', 'Cetus 3', 'Columba 1', 'Coma Berenices', \
                'Crater 2', 'DES J0225+0304', 'Draco', 'Draco 2', 'Eridanus 2', \
                'Eridanus 3', 'Fornax', 'Grus 1', 'Grus 2', 'Hercules', \
                'Horologium 1', 'Horologium 2', 'Hydra 2', 'Hydrus 1', 'Indus 1', \
                'Indus 2', 'Leo 1', 'Leo 2', 'Leo 4', 'Leo 5', 'Leo A', 'Leo T', \
                'Pegasus 3', 'Phoenix', 'Phoenix 2', 'Pictor 1', 'Pictor 2', \
                'Pisces 2', 'Reticulum 2', 'Reticulum 3', 'Sagittarius 2', \
                'Sculptor', 'Segue 1', 'Segue 2', 'Sextans 1', 'Triangulum 2', \
                'Tucana', 'Tucana 2', 'Tucana 3', 'Tucana 4', 'Tucana 5', \
                'Ursa Major 1', 'Ursa Major 2', 'Ursa Minor', 'Virgo 1', \
                'Willman 1']

err_type = 'mass'
galaxy = 'Bootes_3'
err_tweak_array = [0.01, 0.1, 0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 1.75, 1.9, 1.99]
color_array = ['#278b46', '#278b46', '#278b46', '#278b46', '#278b46', '#389bc7', '#c76438', '#c76438', '#c76438', '#c76438', '#c76438']

for galaxy in mw_sats_1Mpc:
    galaxy_name = galaxy.replace(' ', '_')

    listInfall = []
    listDperiRec = []
    listDperiMin = []
    listTperiRec = []
    listTperiMin = []
    listVperiRec = []
    listVperiMin = []
    listNperi = []
    listWeights = []
    listDapoRec = []
    listTapoRec = []

    for i in range(len(err_tweak_array)):
        if err_tweak_array[i] == 1.0:
            file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/fiducial/weights_{galaxy_name}.txt'
        else:
            file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/mass_tweaks/err_tweak_{err_tweak_array[i]}/weights_{galaxy_name}.txt'
        #
        gal_data = sat_analysis.read_subhalo_matches(galaxy_name, file_path=file_path_read)
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
        orbit_dictionary['L.tot.sim'] = np.zeros(gal_data.shape[0])
        orbit_dictionary['v.tot.sim'] = np.zeros(gal_data.shape[0])
        orbit_dictionary['halo.mass.peak'] = np.zeros(gal_data.shape[0])
        orbit_dictionary['distance'] = np.zeros(gal_data.shape[0])
        orbit_dictionary['velocity.rad'] = np.zeros(gal_data.shape[0])
        orbit_dictionary['velocity.tan'] = np.zeros(gal_data.shape[0])
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
        #
        listInfall.append(orbit_dictionary['first.infall.time.lb'])
        listDperiRec.append(orbit_dictionary['pericenter.rec.dist'])
        listTperiRec.append(orbit_dictionary['pericenter.rec.time.lb'])
        listVperiRec.append(orbit_dictionary['pericenter.rec.vel'])
        listDperiMin.append(orbit_dictionary['pericenter.min.dist'])
        listTperiMin.append(orbit_dictionary['pericenter.min.time.lb'])
        listVperiMin.append(orbit_dictionary['pericenter.min.vel'])
        listDapoRec.append(orbit_dictionary['apocenter.dist'])
        listTapoRec.append(orbit_dictionary['apocenter.time.lb'])
        listNperi.append(orbit_dictionary['pericenter.num'])
        listWeights.append(np.asarray(gal_data['Weight']))



    # Infall times
    first_infall = []
    for i in range(len(listInfall)):
        m = (listInfall[i] != -1)
        if np.sum(m) != 0:
            x = listInfall[i][m]
            x_med = ut.math.percentile_weighted(x, 50, listWeights[i][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, listWeights[i][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, listWeights[i][m])
            first_infall.append((x_med, x_lower, x_upper))
        else:
            first_infall.append((-1, -1, -1))
    #
    first_infall = np.asarray(first_infall)
    mask = (first_infall[:,0] != -1)
    meds = first_infall[:,0]
    lowers = first_infall[:,1]
    uppers = first_infall[:,2]
    #
    xtickArray = np.arange(len(listInfall))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], meds[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    #plt.xticks(xtickArray[mask], ['-99% error', '-90% error', '-75% error', '-50% error', '-10% error', 'Fiducial', '+10% error', '+50% error', '+75% error', '+90% error', '+99% error'], rotation=45)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Lookback infall time [Gyr]', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_infall_mass_errors.pdf')
    plt.close()



    # Recent Pericenter distance
    dperi_rec = []
    for i in range(len(listDperiRec)):
        m = (listDperiRec[i] != -1)
        if np.sum(m) != 0:
            x = listDperiRec[i][m]
            x_med = ut.math.percentile_weighted(x, 50, listWeights[i][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, listWeights[i][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, listWeights[i][m])
            dperi_rec.append((x_med, x_lower, x_upper))
        else:
            dperi_rec.append((-1, -1, -1))
    #
    dperi_rec = np.asarray(dperi_rec)
    mask = (dperi_rec[:,0] != -1)
    meds = dperi_rec[:,0]
    lowers = dperi_rec[:,1]
    uppers = dperi_rec[:,2]
    #
    xtickArray = np.arange(len(listDperiRec))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], meds[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Recent pericenter distance [kpc]', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_dperirec_mass_errors.pdf')
    plt.close()



    # Recent Pericenter lookback time
    tperi_rec = []
    for i in range(len(listTperiRec)):
        m = (listTperiRec[i] != -1)
        if np.sum(m) != 0:
            x = listTperiRec[i][m]
            x_med = ut.math.percentile_weighted(x, 50, listWeights[i][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, listWeights[i][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, listWeights[i][m])
            tperi_rec.append((x_med, x_lower, x_upper))
        else:
            tperi_rec.append((-1, -1, -1))
    #
    tperi_rec = np.asarray(tperi_rec)
    mask = (tperi_rec[:,0] != -1)
    meds = tperi_rec[:,0]
    lowers = tperi_rec[:,1]
    uppers = tperi_rec[:,2]
    #
    xtickArray = np.arange(len(listTperiRec))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], meds[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Recent pericenter lookback time [Gyr]', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_tperirec_mass_errors.pdf')
    plt.close()



    # Pericenter number
    nperi = []
    for i in range(len(listNperi)):
        m = (listNperi[i] != -1)
        if np.sum(m) != 0:
            x = listNperi[i][m]
            x_med = np.sum(x * listWeights[i][m])
            x_std = np.sqrt(np.sum((x - x_med)**2 * listWeights[i][m]) / np.sum(listWeights[i][m])/np.sum(listWeights[i][m]))
            nperi.append((x_med, x_std, -1))
        else:
            nperi.append((-1, -1, -1))
    #
    nperi = np.asarray(nperi)
    mask = (nperi[:,0] != -1)
    means = nperi[:,0]
    stds = nperi[:,1]
    #
    xtickArray = np.arange(len(listNperi))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], means[mask][i], yerr=stds[mask][i], color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], means[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Number of pericentric passages', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_nperi_mass_errors.pdf')
    plt.close()



    # Minimum Pericenter distance
    dperi_min = []
    for i in range(len(listDperiMin)):
        m = (listDperiMin[i] != -1)
        if np.sum(m) != 0:
            x = listDperiMin[i][m]
            x_med = ut.math.percentile_weighted(x, 50, listWeights[i][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, listWeights[i][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, listWeights[i][m])
            dperi_min.append((x_med, x_lower, x_upper))
        else:
            dperi_min.append((-1, -1, -1))
    #
    dperi_min = np.asarray(dperi_min)
    mask = (dperi_min[:,0] != -1)
    meds = dperi_min[:,0]
    lowers = dperi_min[:,1]
    uppers = dperi_min[:,2]
    #
    xtickArray = np.arange(len(listDperiMin))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], meds[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Minimum pericenter distance [kpc]', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_dperimin_mass_errors.pdf')
    plt.close()



    # Minimum Pericenter lookback time
    tperi_min = []
    for i in range(len(listTperiMin)):
        m = (listTperiMin[i] != -1)
        if np.sum(m) != 0:
            x = listTperiMin[i][m]
            x_med = ut.math.percentile_weighted(x, 50, listWeights[i][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, listWeights[i][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, listWeights[i][m])
            tperi_min.append((x_med, x_lower, x_upper))
        else:
            tperi_min.append((-1, -1, -1))
    #
    tperi_min = np.asarray(tperi_min)
    mask = (tperi_min[:,0] != -1)
    meds = tperi_min[:,0]
    lowers = tperi_min[:,1]
    uppers = tperi_min[:,2]
    #
    xtickArray = np.arange(len(listTperiMin))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], meds[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Minimum pericenter lookback time [Gyr]', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_tperimin_mass_errors.pdf')
    plt.close()



    # Recent Apocenter distance
    dapo_rec = []
    for i in range(len(listDapoRec)):
        m = (listDapoRec[i] != -1)
        if np.sum(m) != 0:
            x = listDapoRec[i][m]
            x_med = ut.math.percentile_weighted(x, 50, listWeights[i][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, listWeights[i][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, listWeights[i][m])
            dapo_rec.append((x_med, x_lower, x_upper))
        else:
            dapo_rec.append((-1, -1, -1))
    #
    dapo_rec = np.asarray(dapo_rec)
    mask = (dapo_rec[:,0] != -1)
    meds = dapo_rec[:,0]
    lowers = dapo_rec[:,1]
    uppers = dapo_rec[:,2]
    #
    xtickArray = np.arange(len(listDapoRec))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], meds[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Recent apocenter distance [kpc]', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_daporec_mass_errors.pdf')
    plt.close()



    # Recent Pericenter lookback time
    tapo_rec = []
    for i in range(len(listTapoRec)):
        m = (listTapoRec[i] != -1)
        if np.sum(m) != 0:
            x = listTapoRec[i][m]
            x_med = ut.math.percentile_weighted(x, 50, listWeights[i][m])
            x_lower = ut.math.percentile_weighted(x, 15.87, listWeights[i][m])
            x_upper = ut.math.percentile_weighted(x, 84.13, listWeights[i][m])
            tapo_rec.append((x_med, x_lower, x_upper))
        else:
            tapo_rec.append((-1, -1, -1))
    #
    tapo_rec = np.asarray(tapo_rec)
    mask = (tapo_rec[:,0] != -1)
    meds = tapo_rec[:,0]
    lowers = tapo_rec[:,1]
    uppers = tapo_rec[:,2]
    #
    xtickArray = np.arange(len(listTapoRec))
    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(10,8))
    plt.title(f'{galaxy_name} - {err_type} tweaking')
    #
    for i in range(len(xtickArray[mask])):
        axs.errorbar(xtickArray[mask][i], meds[mask][i], yerr=np.array([[meds[mask][i]-lowers[mask][i]],[uppers[mask][i]-meds[mask][i]]]), color=np.asarray(color_array)[mask][i], alpha=0.7, lw=4.5, capsize=0)
        axs.scatter(xtickArray[mask][i], meds[mask][i], s=75, c=np.asarray(color_array)[mask][i], alpha=0.7)
    xtickNames = [str(np.round(np.asarray(err_tweak_array[i])*0.35, decimals=3)) for i in range(len(err_tweak_array))]
    plt.xticks(xtickArray[mask], np.asarray(xtickNames)[mask], rotation=45)
    axs.tick_params(axis='x', which='minor', bottom=False, top=False)
    axs.set_ylabel('Recent apocenter lookback time [Gyr]', fontsize=24)
    axs.set_xlabel('Mass bin width [dex]', fontsize=24)
    plt.tight_layout()
    #plt.show()
    plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/artificial_errors/{galaxy_name}_taporec_mass_errors.pdf')
    plt.close()








