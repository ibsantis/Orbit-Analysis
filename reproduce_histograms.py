#!/usr/bin/python3
# pyright: reportOperatorIssue=false

"""
    =========================================
    = Paper III Orbit History Distributions =
    =========================================

    Create the multi-panel orbit history PDFs for each satellite.

    This will create a figure that shows the differential and cumulative PDFs of:
        - Infall time
        - Apocenter time and distance (recent)
        - Pericenter time, distance, and velocity (recent and minimum)
        - Disance, radial velocity, and tangential velocity
"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import satellite_io
import matplotlib
from matplotlib import pyplot as plt
import h5py
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
sat_analysis = satellite_io.SatelliteAnalysis(gal1='m12i', location=loc)
#
print('Set paths')

def load_subhalo_orbit_properties(filename):
    """
    Load all hosts' orbit-property data from one satellite HDF5 file.

    Returns
    -------
    data : dict
        {
          host_name: {
            "meta": {
                ... attrs on the host group ...,
                "z_reion": float or None,
            },
            "subhalos": {
                "tree_ids":       np.ndarray (N_host,),
                "weights":        np.ndarray (N_host,) or None,
                "snapshot_match": np.ndarray (N_host,) or None,
            },
            "orbit_props": {
                "values": {
                    'first.infall.time.lb': np.ndarray (N_host,),
                    'pericenter.num':       np.ndarray (N_host,),
                    'pericenter.rec.time.lb': ...,
                    'pericenter.rec.dist':    ...,
                    'pericenter.rec.vel':     ...,
                    'pericenter.min.time.lb': ...,
                    'pericenter.min.dist':    ...,
                    'pericenter.min.vel':     ...,
                    'apocenter.time.lb':      ...,
                    'apocenter.dist':         ...,
                    'd.reion':                np.ndarray (N_host,),
                },
                "units": {
                    <same keys>: str or None
                }
            },
          },
          ...,
          "_z_reion": float or None   # global convenience copy
        }
    """
    out = {}

    # Keys we expect (but we’ll be tolerant if some are missing)
    prop_keys = [
        "first.infall.time.lb",
        "pericenter.num",
        "pericenter.rec.time.lb",
        "pericenter.rec.dist",
        "pericenter.rec.vel",
        "pericenter.min.time.lb",
        "pericenter.min.dist",
        "pericenter.min.vel",
        "apocenter.time.lb",
        "apocenter.dist",
        "d.reion",
    ]

    with h5py.File(filename, "r") as f:
        g_hosts = f.get("hosts")
        if g_hosts is None:
            raise KeyError(f"No 'hosts' group found in {filename}")
    
        file_attrs = dict(f.attrs)

        for host_name, g_host in g_hosts.items():
            # --- meta ---
            meta = dict(g_host.attrs)
            meta.update(file_attrs)

            # --- subhalos ---
            g_sub = g_host["subhalos"]
            subhalos = {
                "tree.ids": g_sub["tree.ids"][:],
                "weights": g_sub["weights"][:] if "weights" in g_sub else None,
                "snapshot.at.match": (
                    g_sub["snapshot.at.match"][:] if "snapshot.at.match" in g_sub else None
                ),
            }

            # --- orbit_props ---
            g_props = g_host["orbit.props"]
            orbit_values = {}
            orbit_units = {}

            for key in prop_keys:
                if key not in g_props:
                    continue
                ds = g_props[key]
                orbit_values[key] = ds[:]
                orbit_units[key] = ds.attrs.get("units", None)

            out[host_name] = {
                "meta": meta,
                "subhalos": subhalos,
                "orbit.props": {
                    "values": orbit_values,
                    "units": orbit_units,
                },
            }

    return out

fname = "Carina_II_subhalo_orbit_properties.h5"
props = load_subhalo_orbit_properties(fname)

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12n', 'm12q', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']

binWidths = [0.25, 1, 0.25, 5, 10, 0.25, 5, 10, 0.25, 5, 8]
orbitPropString = ['t.infall', 
                   'N.peri',
                   't.peri.rec', 
                   'd.peri.rec', 
                   'v.peri.rec', 
                   't.peri.min', 
                   'd.peri.min', 
                   'v.peri.min', 
                   't.apo', 
                   'd.apo',
                   'd.reion']

def orbit_property_array(analog_data, orbit_property):
    temp_array = []
    temp_weight = []
    for host in analog_data.keys():
        mask = (analog_data[host]['orbit.props']['values'][orbit_property] != -1)
        if np.sum(mask) == 0:
            continue
        temp_array.append(analog_data[host]['orbit.props']['values'][orbit_property][mask])
        temp_weight.append(analog_data[host]['subhalos']['weights'][mask])
    prop_array = np.hstack(temp_array)
    weight_array = np.hstack(temp_weight)
    #
    return prop_array, weight_array

first_host = list(props.keys())[0]
for idx, propString in enumerate(props[first_host]['orbit.props']['values'].keys()):

    propValues, propWeights = orbit_property_array(props, propString)

    binss, half_binss = sat_analysis.binning_scheme(propValues, orbitPropString[idx], binWidths[idx])

    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(1, 1, figsize=(8,8))
    #
    p = np.histogram(propValues, binss, density=True, weights=propWeights)
    axs.bar(p[1][:-1]+half_binss, p[0]/np.max(p[0]), width=binWidths[idx], color='k', alpha=0.4, edgecolor=None)
    x_med = ut.math.percentile_weighted(propValues, 50, propWeights)
    y_med = 1.1
    sigma_one_om = ut.math.percentile_weighted(propValues, 15.87, propWeights)
    sigma_one_op = ut.math.percentile_weighted(propValues, 84.13, propWeights)
    axs.errorbar(x_med, y_med, xerr=np.array([[x_med-sigma_one_om],[sigma_one_op-x_med]]), color='k', lw=3.5, capsize=0)
    axs.scatter(x_med, y_med, s=75, marker='s', c='k')
    axs.axhline(0.5, 0, 1, linestyle='dotted', linewidth=2, color='k')
    axs.hist(propValues, binss, density=True, weights=propWeights, cumulative=True, linestyle='dashed', linewidth=2, histtype='step', color='b', alpha=0.4)
    axs.set_xlabel(propString, fontsize=18)
    axs.tick_params(axis='both', which='major', labelsize=14)
plt.show()


## REPRODUCES THE RIGHT THING!