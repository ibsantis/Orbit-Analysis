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

n_555 = [12, 330, 378, 83, 131, 3, 2, 305, 42, 213, 15, 13, 75, 89, 125, 5, 243, 153, 43, 13, 26, 57, 4, 125, 4, 99, 99, 111, 163, 301, 212, 49, 52, 0, 25, 167, 255, 347, 18, 67, 517, 362, 178, 13, 387, 127, 77, 363, 104, 954, 2, 107, 4, 103, 20, 12, 42, 17, 15, 151, 0, 183, 113, 193, 261, 20, 62, 87, 45, 53]


for sat_idx, galaxy in enumerate(mw_sats_1Mpc):
    #
    satellite_name = galaxy.replace(' ', '_')
    #
    if n_555[sat_idx] < 10:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_10_10_10/weights_{satellite_name}.txt'
    else:
        file_path_read = sim_data.home_dir+f'/orbit_data/hdf5_files/satellite_matching/combined_physical_tweaks/floor_5_5_5/weights_{satellite_name}.txt'
    gal_data = sat_analysis.read_subhalo_matches(galaxy, file_path_read)
    #
    if len(gal_data['Host']) == 0:
        continue
    #
    outname = f"{satellite_name}_subhalo_orbit_data.h5"
    with h5py.File(outname, "w") as f:
        g_hosts = f.create_group("hosts") 
        f.attrs["selection.Mhalo.peak.bin.width.dex"] = 0.7
        f.attrs["selection.phase.space.sigma"] = 3.0
        f.attrs["selection.gaussian.fraction"] = 0.99
        #
        for sim_name in galaxies:
            mask_host = (gal_data['Host'] == sim_name)
            if not np.any(mask_host):
                continue
            #
            # Read in the mini data and snapshot information
            mini_data = ut.io.file_hdf5(sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_'+sim_name+'_all_subhalos', verbose=False)
            snaps = ut.simulation.read_snapshot_times(directory=sim_data.home_dir+'/galaxies/snapshot_times/'+sim_name)
            ###
            # Things to save before opening the file to write out
            host_name = sim_name
            nSnap = snaps['index'].shape[0]
            nAnalogs = np.sum(mask_host)
            timeArray = np.flip(snaps['time'])
            snapArray = np.flip(snaps['index'])
            redsArray = np.flip(snaps['redshift'])
            #
            subhaloTreeIndices = np.asarray(gal_data['Halo tree index'][mask_host])
            subhaloMiniIndices = np.asarray([np.where(mini_data['indices.z0'][:,0] == subhaloTreeIndices[i])[0][0] for i in range(len(subhaloTreeIndices))])
            weights = np.asarray(gal_data['Weight'][mask_host])
            snap_match = np.asarray(gal_data['Snapshot at match'][mask_host])

            distanceArray = (-1)*np.ones((nAnalogs, nSnap))
            vradArray = (-1)*np.ones((nAnalogs, nSnap))
            vtanArray = (-1)*np.ones((nAnalogs, nSnap))
            haloArray = (-1)*np.ones(nAnalogs)
            #
            hostMass = (-1)*np.ones(nSnap)
            hostRadius = (-1)*np.ones(nSnap)

            for i in range(nAnalogs):
                idx = subhaloMiniIndices[i]
                distanceArray[i][:len(mini_data['d.tot.sim'][idx])] = mini_data['d.tot.sim'][idx] ###### Check these later by plotting the orbits and see waht they look like
                vradArray[i][:len(mini_data['v.rad.sim'][idx])] = mini_data['v.rad.sim'][idx]
                vtanArray[i][:len(mini_data['v.tan.sim'][idx])] = mini_data['v.tan.sim'][idx]
                haloArray[i] = mini_data['M.halo.peak'][idx]

            host_mass_ratio = mini_data['host.mass.ratio'].item()
            hostMass[:len(mini_data['host.mass'])] = mini_data['host.mass']
            hostRadius[:len(mini_data['host.radius'])] = mini_data['host.radius']
            #
            # Host information
            g_host = g_hosts.create_group(host_name)
            g_host.attrs["host.name"] = host_name
            #
            g_props = g_host.create_group("host_props")
            g_props.create_dataset("host.mass.ratio", data=host_mass_ratio)
            g_props.create_dataset("host.mass", data=hostMass)
            g_props.create_dataset("host.radius", data=hostRadius)
            #
            g_props["host.mass"].attrs["units"] = "Msun"
            g_props["host.radius"].attrs["units"] = "kpc"
            #
            g_props["host.mass"].attrs["fill.value"]   = -1.0
            g_props["host.radius"].attrs["fill.value"] = -1.0

            # time properties
            g_time = g_host.create_group("time")
            g_time.create_dataset("time", data=timeArray)
            g_time.create_dataset("snapshot", data=snapArray)
            g_time.create_dataset("redshift", data=redsArray)

            # subhalos
            g_sub = g_host.create_group("subhalos")
            g_sub.create_dataset("tree.ids", data=subhaloTreeIndices)
            g_sub.create_dataset("weights", data=weights)
            g_sub.create_dataset("snapshot.at.match", data=snap_match)

            # orbits: shape (Nsub_1, Nsnap_1)
            g_orb = g_host.create_group("orbits")
            dset_d    = g_orb.create_dataset("d", data=distanceArray, compression="gzip")
            dset_vr   = g_orb.create_dataset("v.rad", data=vradArray, compression="gzip")
            dset_vtan = g_orb.create_dataset("v.tan", data=vtanArray, compression="gzip")
            #
            dset_d.attrs["units"] = "kpc"
            dset_vr.attrs["units"] = "km/s"
            dset_vtan.attrs["units"] = "km/s"
            #
            dset_d.attrs["fill.value"]    = -1.0
            dset_vr.attrs["fill.value"]   = -1.0
            dset_vtan.attrs["fill.value"] = -1.0
###




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



satellite_name = 'Bootes_V'
fname = f"{satellite_name}_subhalo_orbit_data.h5"

data = load_subhalo_orbit_data(fname)

print(data.keys())
