#!/usr/bin/env python3

"""

    ========================
    = Integrating subhalos =
    ========================

    Save the mini-data files that are used for Paper III

"""

# Import packages
import orbit_io
import satellite_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import pandas as pd
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
sat_analysis = satellite_io.SatelliteAnalysis(gal1='m12i', location=loc)
#
print('Set paths')

lg_data = pd.read_csv(sim_data.home_dir+'/orbit_data/paper_III/localgroup_galaxies_condensed.csv', index_col=0)

galaxies = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12n', 'm12q', 'm12w', 'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']

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

propArray = ['first.infall.time.lb',\
             'pericenter.num',\
             'pericenter.rec.time.lb',\
             'pericenter.rec.dist',\
             'pericenter.rec.vel',\
             'pericenter.min.time.lb',\
             'pericenter.min.dist',\
             'pericenter.min.vel',\
             'apocenter.time.lb',\
             'apocenter.dist']

n_555 = [12, 330, 378, 83, 131, 3, 2, 305, 42, 213, 15, 13, 75, 89, 125, 5, 243, 153, 43, 13, 26, 57, 4, 125, 4, 99, 99, 111, 163, 301, 212, 49, 52, 0, 25, 167, 255, 347, 18, 67, 517, 362, 178, 13, 387, 127, 77, 363, 104, 954, 2, 107, 4, 103, 20, 12, 42, 17, 15, 151, 0, 183, 113, 193, 261, 20, 62, 87, 45, 53]

z_reion = 7.0


for sat_idx, galaxy in enumerate(mw_sats_1Mpc):
    satellite_name = galaxy.replace(' ', '_')

    # Choose which weights file to use
    if n_555[sat_idx] < 10:
        file_path_read = (
            sim_data.home_dir
            + f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/'
              f'floor_10_10_10/weights_{satellite_name}.txt'
        )
    else:
        file_path_read = (
            sim_data.home_dir
            + f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/'
              f'floor_5_5_5/weights_{satellite_name}.txt'
        )

    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)

    # Skip if no analogs at all for this satellite
    if len(gal_data['Host']) == 0:
        continue

    outname = f"{satellite_name}_subhalo_orbit_properties.h5"

    with h5py.File(outname, "w") as f:
        f.attrs["z.reion"] = float(z_reion)
        f.attrs["z.reion.description"] = "Redshift at which d.reion is evaluated"
        g_hosts = f.create_group("hosts")

        for sim_name in galaxies:
            mask_host = (gal_data['Host'] == sim_name)
            if not np.any(mask_host):
                continue

            # 
            tree_ids       = np.asarray(gal_data['Halo tree index'][mask_host])
            weights        = np.asarray(gal_data['Weight'][mask_host])
            snapshot_match = np.asarray(gal_data['Snapshot at match'][mask_host])
            N_host = tree_ids.size

            # --- Load host-specific sim data ---
            mini_data = ut.io.file_hdf5(
                sim_data.home_dir
                + f'/orbit_data/hdf5_files/summary_data/data_{sim_name}_all_subhalos',
                verbose=False,
            )
            snaps = ut.simulation.read_snapshot_times(
                directory=sim_data.home_dir + '/galaxies/snapshot_times/' + sim_name
            )

            # --- Orbit properties for this host's analogs ---
            # Assumed to return arrays of length N_host aligned with gal_data[mask_host]
            orbit_history = sat_analysis.orbit_property_distribution(
                sim_name, mini_data, gal_data, snaps
            )

            # --- Map tree_ids -> mini_data rows ---
            mini_data_match_inds = np.asarray([
                np.where(mini_data['indices.z0'][:, 0] == tid)[0][0]
                for tid in tree_ids
            ])

            # --- Distance at reionization (d.reion) ---
            # Find index of snapshot closest to z_reion in the full snapshot list
            reion_snap_global = int(
                np.argmin(np.abs(snaps['redshift'] - z_reion))
            )

            full_nsnap = len(snaps['time'])
            sub_nsnap  = mini_data['d.tot.sim'].shape[1]
            # Offset between full snapshot timeline and truncated mini_data arrays
            offset = full_nsnap - sub_nsnap
            reion_idx_in_mini = reion_snap_global - offset

            d_reion = -1.0 * np.ones(N_host)

            if 0 <= reion_idx_in_mini < sub_nsnap:
                for ii, m_idx in enumerate(mini_data_match_inds):
                    # Match your previous logic:
                    # use flipped d.tot.sim to align with how you index in time
                    d_series = mini_data['d.tot.sim'][m_idx]
                    d_reion[ii] = np.flip(d_series)[reion_idx_in_mini]
            # else: leave as -1.0 (no data at reionization for this host/analog)

            # Add to orbit_history dict so it gets written with everything else
            orbit_history["d.reion"] = d_reion

            # --- Create host group in output file ---
            g_host = g_hosts.create_group(sim_name)
            g_host.attrs["host.name"] = sim_name
            g_host.attrs["mw.satellite"] = galaxy

            # subhalos (ensure consistency with orbit_data file)
            g_sub = g_host.create_group("subhalos")
            g_sub.create_dataset("tree.ids",       data=tree_ids)
            g_sub.create_dataset("weights",        data=weights)
            g_sub.create_dataset("snapshot.at.match", data=snapshot_match)

            # orbit_props
            g_props = g_host.create_group("orbit.props")

            # Only write the selected orbit properties
            for key in propArray:
                if key not in orbit_history:
                    continue

                values = orbit_history[key]
                dset = g_props.create_dataset(key, data=values)

                # Units / descriptions
                if key.endswith(".dist"):
                    dset.attrs["units"] = "kpc"
                elif key.endswith(".time.lb"):
                    dset.attrs["units"] = "Gyr"
                elif key.endswith(".vel"):
                    dset.attrs["units"] = "km/s"
                elif key == "pericenter.num":
                    dset.attrs["description"] = "Number of resolved pericentric passages"

            # Add reionization distance
            dset_reion = g_props.create_dataset("d.reion", data=d_reion)
            dset_reion.attrs["units"] = "kpc"
            dset_reion.attrs["description"] = "Satellite distance from host at z_reion"



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
                "tree.ids":       np.ndarray (N_host,),
                "weights":        np.ndarray (N_host,) or None,
                "snapshot.at.match": np.ndarray (N_host,) or None,
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


fname = "Bootes_V_subhalo_orbit_properties.h5"
props = load_subhalo_orbit_properties(fname)

print(props.keys())                 # host names + "_z_reion"
m12i = props["m12i"]
