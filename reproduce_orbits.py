#!/usr/bin/python3
# pyright: reportOperatorIssue=false

"""
    ===========================
    = Paper III Analog Orbits =
    ===========================

    Create the plot of analog orbits for each satellite.

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

def load_subhalo_orbit_data(filename):
    """
    Load all hosts and their data from one satellite HDF5 file.

    Returns
    -------
    data : dict
        {
          host_name: {
            "meta": {
                ... attrs on the host group ...
            },
            "host_props": {
                "host_mass_ratio": float,
                "host_mass":       np.ndarray (nSnap,),
                "host_radius":     np.ndarray (nSnap,),
                "units": {
                    "host_mass":   str or None,
                    "host_radius": str or None,
                },
                # optional:
                # "fill_value_mass":   float (if present in attrs, else -1.0),
                # "fill_value_radius": float (if present in attrs, else -1.0),
            },
            "time": {
                "time":     np.ndarray (nSnap,),
                "snapshot": np.ndarray (nSnap,),
                "redshift": np.ndarray (nSnap,),
            },
            "subhalos": {
                "tree_ids":       np.ndarray (nAnalogs,),
                "weights":        np.ndarray (nAnalogs,) or None,
                "snapshot_match": np.ndarray (nAnalogs,) or None,
            },
            "orbits": {
                "d":     np.ndarray (nAnalogs, nSnap),
                "v_rad": np.ndarray (nAnalogs, nSnap),
                "v_tan": np.ndarray (nAnalogs, nSnap),
                "units": {
                    "d":     str or None,
                    "v_rad": str or None,
                    "v_tan": str or None,
                },
                # fill_value used for padding; defaults to -1.0 if not set
                "fill_value": float,
            },
          },
          ...
        }
    """
    out = {}

    with h5py.File(filename, "r") as f:
        if "hosts" not in f:
            raise KeyError(f"No 'hosts' group found in {filename}")

        g_hosts = f["hosts"]

        for host_name, g_host in g_hosts.items():
            # --- host-level attrs ---
            meta = dict(g_host.attrs)

            # --- host_props ---
            g_props = g_host["host_props"]
            host_mass_ratio = g_props["host.mass.ratio"][()]      # scalar
            host_mass       = g_props["host.mass"][:]             # (nSnap,)
            host_radius     = g_props["host.radius"][:]           # (nSnap,)

            host_props = {
                "host.mass.ratio": host_mass_ratio,
                "host.mass":       host_mass,
                "host.radius":     host_radius,
                "units": {
                    "host.mass":   g_props["host.mass"].attrs.get("units", None),
                    "host.radius": g_props["host.radius"].attrs.get("units", None),
                },
            }

            # Optionally expose fill_values for host props if you set them
            if "fill.value" in g_props["host.mass"].attrs:
                host_props["fill.value.mass"] = g_props["host.mass"].attrs["fill.value"]
            if "fill.value" in g_props["host.radius"].attrs:
                host_props["fill.value.radius"] = g_props["host.radius"].attrs["fill.value"]

            # --- time ---
            g_time = g_host["time"]
            time_data = {
                "time":     g_time["time"][:],
                "snapshot": g_time["snapshot"][:],
                "redshift": g_time["redshift"][:],
            }

            # --- subhalos ---
            g_sub = g_host["subhalos"]
            subhalos = {
                "tree.ids": g_sub["tree.ids"][:],
            }

            # New: optional weights + snapshot_match
            if "weights" in g_sub:
                subhalos["weights"] = g_sub["weights"][:]
            else:
                subhalos["weights"] = None

            if "snapshot.at.match" in g_sub:
                subhalos["snapshot.at.match"] = g_sub["snapshot.at.match"][:]
            else:
                subhalos["snapshot.at.match"] = None

            # --- orbits ---
            g_orb = g_host["orbits"]
            d_ds     = g_orb["d"]
            v_rad_ds = g_orb["v.rad"]
            v_tan_ds = g_orb["v.tan"]

            orbits = {
                "d":     d_ds[:],
                "v.rad": v_rad_ds[:],
                "v.tan": v_tan_ds[:],
                "units": {
                    "d":     d_ds.attrs.get("units", None),
                    "v.rad": v_rad_ds.attrs.get("units", None),
                    "v.tan": v_tan_ds.attrs.get("units", None),
                },
                "fill.value": d_ds.attrs.get("fill.value", -1.0),
            }

            out[host_name] = {
                "meta": meta,
                "host.props": host_props,
                "time": time_data,
                "subhalos": subhalos,
                "orbits": orbits,
            }

    return out

fname = "Carina_II_subhalo_orbit_data.h5"
props = load_subhalo_orbit_data(fname)

plt.rcParams["font.family"] = "serif"
f, axs = plt.subplots(1, 1, figsize=(8,8))
for host in props.keys():
    time_array = props[host]['time']['time'][0] - props[host]['time']['time']
    for i in range(props[host]['orbits']['d'].shape[0]):
        starting_idx = np.where(props[host]['subhalos']['snapshot.at.match'][i] == props[host]['time']['snapshot'])[0][0]
        starting_time = props[host]['time']['time'][0] - props[host]['time']['time'][starting_idx]
        mask = (props[host]['orbits']['d'][i][starting_idx:] != -1)
        axs.plot(time_array[starting_idx:][mask] - starting_time, props[host]['orbits']['d'][i][starting_idx:][mask], color='k', linewidth=0.5, alpha=0.1)
axs.set_xlim(0, 13.78)
axs.set_ylim(-5, 400)
axs.set_xlabel('Lookback Time [Gyr]', fontsize=18)
axs.set_ylabel('Distance from MW [kpc]', fontsize=18)
axs.tick_params(axis='both', which='major', labelsize=14)
plt.show()


## REPRODUCES THE RIGHT THING!