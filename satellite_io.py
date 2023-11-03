#!/usr/bin/python3

"""

    Intended for use with the FIRE-2 simulations

    @author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

        TBD

"""

import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker
from scipy.interpolate import splrep, splev, interp1d
import pandas as pd
import galpy
import os

# Have a class that reads in the data
class SatelliteRead:

    def __init__(self, gal1, location, dmo=False):
        """
        Set the home directory, simulation directory, and number of galaxies

        VARIABLES:
            - gal1     : string
                         Name of the MW-mass galaxy you are interested in.
                         If analyzing the LG-pairs, this is the name of the
                         first host (Romeo, Thelma, Romulus).

            - location : string
                         Name of where you are working (peloton, stampede, or on
                         my mac).

            - dmo      : boolean
                         True/False of whether analyzing DMO or Baryonic
                         simulations.

        NOTES:
            - Depending on the variables you enter, sets the number of galaxies,
              the simulation directory, the home directory, and the galaxy name.
        """
        # Set up a dictionary of galaxy information
        galaxy_info = {
            'Romeo': ('Juliet', 'm12_elvis_RomeoJuliet', '_r3500', 2),
            'Thelma': ('Louise', 'm12_elvis_ThelmaLouise', '_r4000', 2),
            'Romulus': ('Remus', 'm12_elvis_RomulusRemus', '_r4000', 2),
            'm12z': (None, 'm12z', '_r4200', 1),
            'm12i_lr': (None, 'm12i', '_r57000', 1),
            'm12i_hr': (None, 'm12i', '_r880', 1)
        }
        # Depending on the galaxy name, set up a few variables
        if gal1 in galaxy_info:
            gal2, self.galaxy, resolution, self.num_gal = galaxy_info[gal1]
        else:
            gal2 = None
            self.galaxy = gal1
            resolution = '_r7100'
            self.num_gal = 1
        #
        # Set up the important paths
        if location == 'mac':
            self.home_dir = '/Users/isaiahsantistevan/simulation'
            if self.num_gal == 2:
                self.gal_1 = gal1
                self.gal_2 = gal2
            else:
                self.simulation_dir = self.home_dir+'/galaxies/'+self.galaxy+resolution
        #
        elif location == 'peloton':
            self.home_dir = '/home/ibsantis/scripts'
            if self.num_gal == 2:
                self.simulation_dir = '/group/awetzelgrp/m12_elvis/'+self.galaxy+resolution
                self.gal_1 = gal1
                self.gal_2 = gal2
            else:
                self.simulation_dir = '/group/awetzelgrp/'+self.galaxy+'/'+self.galaxy+resolution
        #
        elif location == 'stampede':
            self.home_dir = '/home1/05400/ibsantis/scripts'
            self.simulation_dir = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'+self.galaxy+resolution
            if self.num_gal == 2:
                self.gal_1 = gal1
                self.gal_2 = gal2
        #
        if dmo:
            self.simulation_dir += '_dm'

# Have a class that matches the satellites
class SatelliteMatch:

    def __init__(self, gal1, location, tree=None, mini=None, host=1, minimum_mass=1e8):
        """
        DESCRIPTION:
            Returns the indices of satellites along with their progenitor
            indices.

        VARIABLES:
            tree         : dictionary
                           This is the halo merger tree, read in by Andrew's function
                           "halo.io.IO.read_tree" from halo_io.py

            gal1         : string
                           Name of the MW-mass galaxy you are interested in.
                           If analyzing the LG-pairs, this is the name of the
                           first host (Romeo, Thelma, Romulus).

            location     : string
                           Name of where you are working (peloton, stampede, or on
                           my mac).

            host         : integer (1 or 2)
                           Host number. This is 1 for the 'm12' hosts, and could be
                           1 or 2 for the LG-pair hosts.
            
            minimum_mass : integer
                           The minimum mass to select satellites down to.

        NOTES:
            - Returns a 2D array:
                - Each row corresponds to a luminous subhalo
                - The first element in a row is the index of the luminous
                  subhalo at z = 0
                - Each other element in a row corresponds to the subhalo's main
                  progenitor
            - Elements that are negative correspond to times when it did not
              exist
            - For each subhalo (row), the arrays are ordered from
              z = 0 to z = z_form (i.e., from present-day to the past)
            - Each row has a length of 597. There are no halos that exist in
              snapshots 0,1,2,3.
        """
        # Want to inherit the OrbitRead class so that I can adapt pipeline for LG runs
        SatelliteRead.__init__(self, gal1, location, dmo=False)
        #
        if tree and not mini:
            # Selection criteria for the DMO simulations or for non-luminous satellites in the baryonic simulations
            if host == 2:
                hindex = 'host2'
            else:
                hindex = 'host'
            #
            # Select the subhalo indices at z = 0
            z0_inds = ut.array.get_indices(tree['snapshot'], 600)
            z0_inds = z0_inds[z0_inds != tree[hindex+'.index'][0]]
            z0_inds = ut.array.get_indices(tree.prop('lowres.mass.frac'), [0,0.02], z0_inds)
            #
            z0_inds = z0_inds[ut.array.get_indices(tree.prop('mass.peak',z0_inds), [minimum_mass, np.inf])]
            z0_inds_w_prog = tree.prop('progenitor.main.indices', z0_inds)
            #
            self.sub_inds = z0_inds_w_prog
            self.shape = self.sub_inds.shape
        elif mini and not tree:
            self.sub_inds = mini['indices.z0']
            self.shape = mini['indices.z0'].shape
        else:
            raise AssertionError('Either no data input or two many datasets.')
        #
        self.smhm_constant = -15.21177826 # From fitting the SMHMR from paper I, in mstar_mhalo_fitting.py
        self.smhm_slope = 2.2111824

    def satellite_mhalo(self, mstar):
        """
        TBD

        Want to add more criteria, like a minimum mass, particle number, etc
        """
        mhalo = (mstar - self.smhm_constant)/self.smhm_slope
        return mhalo # this is the log of the halo mass actually...
    
    def lg_satellite_properties(self, lg_data, galaxy_name, mass_err=0.35):
        """
        TBD

        Create a dictionary of properties for a given LG satellite from the CSV table
        """
        #
        satellite_dict = {}
        #
        if galaxy_name in lg_data.keys():
            print('* galaxy name = {0}'.format(galaxy_name))
        else:
            raise ValueError('* galaxy name = {0} not in the input catalog!'.format(galaxy_name))
        #
        for prop_name in lg_data[galaxy_name].keys():
            satellite_dict[prop_name] = lg_data[galaxy_name][prop_name]
        #
        log_mass_halo = self.satellite_mhalo(np.log10(satellite_dict['mass.star']))
        mass_halo = 10**(log_mass_halo)
        satellite_dict['mass.peak'] = mass_halo
        satellite_dict['mass.peak.err'] = mass_err
        #
        return satellite_dict

    def subhalo_data(self, tree=None, mini=None, snapshot_data=None):
        """
        TBD
        """
        # Create a sub-dictionary for the properties of interest
        sim_sats = {}
        #
        # Set up empty arrays to save to
        distances = (-1)*np.ones(self.shape)
        velocity_rad = (-1)*np.ones(self.shape)
        velocity_tan = (-1)*np.ones(self.shape)
        sat_snapshots = (-1)*np.ones(self.shape, int)
        masses = (-1)*np.ones(self.shape[0])
        #
        if mini:
            # Loop through the number of satellites and save values to the empty arrays
            for i in range(0, self.shape[0]):
                mask = (self.sub_inds[i] >= 0)
                distances[i][mask] = mini['d.tot.sim'][i][mask]
                velocity_rad[i][mask] = mini['v.rad.sim'][i][mask]
                velocity_tan[i][mask] = mini['v.tan.sim'][i][mask]
                sat_snapshots[i][mask] = snapshot_data['index'][:self.shape[1]][mask]
                masses[i] = mini['M.halo.peak'][i]
        else:
            # Loop through the number of satellites and save values to the empty arrays
            for i in range(0, self.shape[0]):
                mask = (self.sub_inds[i] >= 0)
                distances[i][mask] = tree.prop('host.distance.total', self.sub_inds[i][mask])
                velocity_rad[i][mask] = tree.prop('host.velocity.rad', self.sub_inds[i][mask])
                velocity_tan[i][mask] = tree.prop('host.velocity.tan', self.sub_inds[i][mask])
                sat_snapshots[i][mask] = tree.prop('snapshot', self.sub_inds[i][mask])
                masses[i] = tree.prop('mass.peak', self.sub_inds[i,0])
        distances[np.isnan(distances)] = -1 # This is to take care of instances in which the subhalos existed before the host
        velocity_rad[np.isnan(velocity_rad)] = -1
        velocity_tan[np.isnan(velocity_tan)] = -1
        #
        sim_sats['host.distance.total'] = distances
        sim_sats['host.velocity.rad'] = velocity_rad
        sim_sats['host.velocity.tan'] = velocity_tan
        sim_sats['snapshot'] = sat_snapshots
        sim_sats['mass.peak'] = masses
        #
        return sim_sats
    
    def subhalo_match(self, indices, subhalos, satellite, snapshot_data, lookback_window=1, max_sigma=3, probability_max=99):
        """
        DESCRIPTION:
            TBD

        VARIABLES:
            indices         : 2D array
                              The halo tree indices of subhalos in the simulations

            subhalos        : dictionary
                              This is a subset of simulation data that includes 
                              - total distance from host
                              - radial velocity
                              - tangential velocity
                              - peak subhalo mass
                              - snapshot numbers that it existed in

            satellite       : dictionary
                              This is data for the actual satellite we want to find matches of. 
                              Created by "lg_satellite_properties()" and includes:
                              - total distance from host + error
                              - radial velocity + error
                              - tangential velocity + error
                              - stellar mass
                              - peak subhalo mass using the SMHM relation from Paper I and "satellite_mhalo()"

            lookback_window : integer
                              Lookback time (Gyr) to search for satellite analogs
            
            max_sigma       : integer
                              Threshold of how much error we allow in selecting satellites

        NOTES:
            - Temporarily adding notes so that I can get VS Code to accept another commit
        """
        # Figure out how many snapshots to search for satellites from the snapshot file
        max_time_window = snapshot_data['time'][-1] - lookback_window
        n_snapshots = snapshot_data['index'][-1] - np.where(np.min(np.abs(max_time_window - snapshot_data['time'])) == np.abs(max_time_window - snapshot_data['time']))[0][0]
        #
        # Set up an empty dictionary to save the actual matches to for a given observed satellite
        sub_match = {}
        sub_match['mass.index'] = (-1)*np.ones(indices.shape[0], int)
        sub_match['mass.peak'] = (-1)*np.ones(indices.shape[0], int)
        sub_match['mass.peak.log'] = (-1)*np.ones(indices.shape[0], int)
        sub_match['tree.index'] = (-1)*np.ones(indices.shape[0], int)
        sub_match['weight'] = (-1)*np.ones((indices.shape[0], n_snapshots))
        sub_match['sigma.dif'] = (-1)*np.ones((indices.shape[0], n_snapshots))
        sub_match['snapshot'] = (-1)*np.ones((indices.shape[0], n_snapshots), int)
        #
        properties = [prop_name for prop_name in sorted(satellite.keys()) if '.star' not in prop_name and '.err' not in prop_name]
        #
        dof_number = int(len(properties))
        if dof_number == 1:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.0, 2.0, 3.0
        elif dof_number == 2:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.36, 2.27, 3.206
        elif dof_number == 3:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.56, 2.42, 3.32
        elif dof_number == 4:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.69, 2.52, 3.4
        #
        if probability_max == 68:
            sigma_dif_max = sigma_dif_68
        elif probability_max == 95:
            sigma_dif_max = sigma_dif_95
        elif probability_max == 99:
            sigma_dif_max = sigma_dif_99
        else:
            sigma_dif_max = sigma_dif_95
        #
        # Get subhalos within +/- N sigma * 0.35 dex of M_halo,peak 
        mass_kind = 'mass.peak'
        mass_halo_log = np.log10(satellite[mass_kind])
        mass_inds = ut.array.get_indices(subhalos[mass_kind], [10**(mass_halo_log - max_sigma*satellite[mass_kind+'.err']), 10**(mass_halo_log + max_sigma*satellite[mass_kind+'.err'])])
        #
        # Create a list of coordinate names and property names to loop through
        coord_names = [prop_name for prop_name in sorted(subhalos.keys()) if prop_name != 'mass.peak' and prop_name != 'snapshot']
        properties = [prop_name for prop_name in sorted(subhalos.keys()) if prop_name != 'snapshot']
        #
        # Loop through snapshots
        for snap_ind in range(0, n_snapshots):
            #
            # Use satellites already selected by mass
            match_inds = mass_inds
            # Loop through the 6D coordinates
            for prop_name in coord_names:
                # Get the bin limits for a given property based on the observed satellite values and max error
                prop_limits = ut.binning.get_bin_limits([satellite[prop_name], max_sigma*satellite[prop_name+'.err']], 'error')
                # Set up another 2D array for the subhalo coordinates at a given snapshot
                prop_values = subhalos[prop_name][:,snap_ind]
                # Get the indices of the subhalos that are within the bin limits
                match_inds = ut.array.get_indices(prop_values, prop_limits, match_inds)
            #
            # If there are matches, continue!
            if len(match_inds) != 0:
                # Set up null array to save to
                sigma_difs_z = np.zeros(len(match_inds))
                # Loop through the 6D + mass properties
                for prop_name in properties:
                    # If mass is the property, take the log of the values and calculate sigma_dif
                    if 'mass' in prop_name:
                        prop_values = np.log10(subhalos[prop_name][match_inds])
                        match_prop = np.log10(satellite[prop_name])
                        sigma_difs_z += (
                        (prop_values - match_prop) / satellite[prop_name+'.err']
                        ) **2
                    # If 6D coords, do the same thing without the log
                    else:
                        prop_values = subhalos[prop_name][match_inds, snap_ind]
                        sigma_difs_z += (
                            (prop_values - satellite[prop_name]) / satellite[prop_name+'.err']
                            ) **2
                #
                # Finally take the square root of sigmas
                sigma_difs_z = np.sqrt(sigma_difs_z)
                # Only keep cases that are within our max allowed error
                masks = ut.array.get_indices(sigma_difs_z, [0, sigma_dif_max])
                sigma_difs_z = sigma_difs_z[masks]
                match_inds = match_inds[masks]
                #
                # If there are are still matches, continue!
                if len(sigma_difs_z) != 0:
                    # calculate the weights from the gaussian arguments (sigma_difs)
                    weights_z = ut.math.Function.gaussian_normalized(sigma_difs_z)
                    # Save all of the data to the arrays
                    sub_match['mass.index'][match_inds] = match_inds
                    sub_match['mass.peak'][match_inds] = subhalos['mass.peak'][match_inds]
                    sub_match['mass.peak.log'][match_inds] = np.log10(subhalos['mass.peak'][match_inds])
                    sub_match['tree.index'][match_inds] = self.sub_inds[:,0][match_inds]
                    sub_match['snapshot'][match_inds, snap_ind] = np.flip(snapshot_data['index'])[:n_snapshots][snap_ind]
                    sub_match['weight'][match_inds, snap_ind] = weights_z
                    sub_match['sigma.dif'][match_inds, snap_ind] = sigma_difs_z
                    # Print out the satellites that are matches for a given snapshot
                    print('* Satellite(s) {0} are a match at snapshot {1}'.format(match_inds, np.flip(snapshot_data['index'])[:n_snapshots][snap_ind]))
                    # Print out how many of them are within the max errors allowed
                    print('* {0}, {1} within 68 percent, 95 percent limits'.format(np.sum(sigma_difs_z < sigma_dif_68), np.sum(sigma_difs_z < sigma_dif_95)))
                #
                # If there are no matches, print that out for the current snapshot
                else:
                    print('! no subhalos within {0} percent limits at snapshot {1}'.format(probability_max, np.flip(snapshot_data['index'])[snap_ind]))
                #
                # Now re-weight the subhalos so that the centroid is near the middle of the bin
                # If I don't, then I will likely be assigning more weight to lower mass subhalos
            #
            # If there are no matches, print that out
            else:
                print('! no subhalos match at snapshot {0}'.format(np.flip(snapshot_data['index'])[snap_ind]))
        #
        return sub_match
    
    def mass_weighting(self, weights, mass_array_subs, mass_sat, SMHM_slope=0.44):
        """
        DESCRIPTION:
            TBD

        VARIABLES:
            TBD

        NOTES:
            - TBD
        """
        # Take the log of the masses
        mass_log_sub = np.log10(mass_array_subs)
        mass_log_sat = np.log10(mass_sat)
        #
        # Calculate the mass weights
        weights_m = 10 ** (SMHM_slope * (mass_log_sub - mass_log_sat)) 
        weights_m /= weights_m.sum()  # normalize
        #
        weights *= weights_m 
        weights /= weights.sum() 
        #
        return weights
    
    def write_subhalo_matches(self, satellite, hosts, indices, weights, snapshots):
        """
        DESCRIPTION:
            Want this to be the middle step where I save files that include:
                - Host
                - Tree index
                - Weight (after re-weighting by mass)
                - Snapshot at match
                - sigma_dif ( I actually don't think I need this one... ) 

        VARIABLES:
            TBD

        NOTES:
            - TBD
        """
        # If the file exists, then append to it, otherwise create it
        file_path = self.home_dir+'/orbit_data/hdf5_files/satellite_matching/'
        if os.path.isfile(file_path+'weights_'+satellite+'.txt'):
            print('File exists. Delete or move it elsewhere.')
            return
        else:
            file_name = 'weights_'+satellite+'.txt'
            file_object = open(file_path+file_name, 'w')
            #
            file_object.write('# {0}\n'.format(satellite)) ############################# add more header info like the snapshot window I used and other parameters
            file_object.write('# Host, Halo tree index, Weight, Snapshot at match\n')
            for i in range(0, len(weights)):
                file_object.write('{0}, {1}, {2}, {3} \n'.format(hosts[i], indices[i], weights[i], snapshots[i]))
            file_object.close()
            print('Finished writing to file.')
