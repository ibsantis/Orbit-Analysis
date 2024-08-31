#!/usr/bin/python3

"""

    Intended for use with the FIRE-2 simulations

    @author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

    This package contains several classes intended to read data, 
    match simulated subhalos to satellite galaxies around the MW, 
    and some analysis functions. Below is a brief description of the
    various classes included:

    SatelliteRead:
        - Defines the home and simulation directories and the number 
          of host galaxies
    
    SatelliteMatch:
        - Functions to find subhalo analogs for a given satellite, and
          calculate weights based on how good of a "match" they are;
          can save data.
    
    SatelliteAnalysis:
        - Read in analog data and use it to make a dictionary of
          orbit properties for the analogs; some plotting functions that
          might be useful.

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


class SatelliteRead:

    def __init__(self, gal1, location, dmo=False):
        """
        DESCRIPTION
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


class SatelliteMatch:

    def __init__(self, gal1, location, tree=None, mini=None, host=1, minimum_mass=1e8):
        """
        DESCRIPTION:
            Returns the indices of satellites along with their progenitor
            indices.

        VARIABLES:
            - tree         : dictionary
                             This is the halo merger tree, read in by Andrew's function
                             "halo.io.IO.read_tree" from halo_io.py

            - gal1         : string
                             Name of the MW-mass galaxy you are interested in.
                             If analyzing the LG-pairs, this is the name of the
                             first host (Romeo, Thelma, Romulus).

            - location     : string
                             Name of where you are working (peloton, stampede, or on
                             my mac).

            - host         : integer (1 or 2)
                             Host number. This is 1 for the 'm12' hosts, and could be
                             1 or 2 for the LG-pair hosts.
            
            - minimum_mass : integer
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
        DESCRIPTION:
            Approximates the halo mass of a satellite galaxy based on
            the SMHM fitting data above.

        VARIABLES:
            - mstar : float
                      This is the stellar mass of a satellite galaxy

        NOTES:
            - Returns the log of the halo mass for a satellite of a 
              given stellar mass.
        """
        mhalo = (mstar - self.smhm_constant)/self.smhm_slope
        return mhalo 
    
    def lg_satellite_properties(self, lg_data, galaxy_name, mass_err=0.35):
        """
        DESCRIPTION:
            Using a CSV file, create a dictionary of satellite properties
            and uncertainties

        VARIABLES:
            - lg_data     : pandas dataframe
                            This is the stellar mass of a satellite galaxy.
            
            - galaxy_name : string
                            The name of the satellite galaxy
            
            - mass_err    : float
                            The log of the error in halo mass that we accept.

        NOTES:
            - Must already read in a CSV that contains various properties about
              the satellite galaxies, such as: stellar mass, host distance (and error),
              radial velocity with respect to the host (and error), and tangential
              velocity with respect ot the host (and error); the host is the MW
              in the file that I use.
            - We choose a halo mass error of 0.35 dex, but did not affect end results
              for choices from 0.25 - 0.45; see Paper III for details.
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
        DESCRIPTION:
            Uses either the halo tree or the mini-data files that I previously
            created to make a dictionary of properties that we care about for
            the subhalos.

        VARIABLES:
            - tree          : dictionary
                              Halo tree from the FIRE simualtions.
            
            - mini          : dictionary
                              A smaller version of the halo tree from the FIRE simulations.
                              Contains data useful from Papers I, II, and III.
            
            - snapshot_data : dictionary
                              A dictionary that contains snapshot indices, times, 
                              and redshifts.

        NOTES:
            - Saves a dictionary that contains:
                - Total distance with respect to the center of the host
                - Radial velocity with respect to the center of the host
                - Tangential velocity with respect to the center of the host
                - Peak halo mass
                - Snapshot indices
            
            - For all dictionary elements beside the peak halo mass, the element
              is a 2D array:
                - Each row in the array corresponds to a particular subhalo
                - Each element in that row corresponds to a particular snapshot
                    - These elements correspond to z = 0, and go backward in time, i.e.
                      element 0, is for z = 0
            
            - The snapshot array is more for indexing corresponding to the same convention in
              the original FIRE simulations. Most simulations have snapshots from 0, which is the
              beginning of the simulation, to 600 (sometimes 500), which is the end of the
              simulation, i.e. z = 0.

            - If a satellite existed before the host, we set its properties to -1.
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
            Given a real satellite's distance, velocity (radial and/or tangential), and 
            halo mass, find analogs in the simulations to within the specified limits. Does
            not require all 4 properties to find matches, can mix and match.

        VARIABLES:
            indices         : 2D array
                              The halo tree indices of subhalos in the simulations

            subhalos        : dictionary
                              This is a subset of simulation data created with 
                              "subhalo_data()" and includes:
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

            snapshot_data   : dictionary
                              A dictionary that contains snapshot indices, times, 
                              and redshifts.

            lookback_window : integer
                              Lookback time (Gyr) to search for satellite analogs
            
            max_sigma       : integer
                              Threshold of how much error we allow in selecting satellites
            
            probability_max : integer
                              A "sigma" for the N-dimensional Gaussian for the selection properties (distance
                              velocity (radial and tangential), and mass)

        NOTES:
            - Produces a dictionary that contains:
                - The peak halo mass (normal and log) for the subhalo matches
                - Indices to point to the analogs in the mass array (created in the function) and in
                  the original halo tree from the FIRE simulations (at z = 0)
                - The snapshot that the subhalo analog is matched with the satellite
                - The weight, which is a measure of how "good" of a match the subhalo is
                - The value of the N-dimensional Gaussian with the given weights; also
                  kind of a measure of how "good" a match is.
            - We don't have to worry about double counting a match across snapshots.
            - Lots of code previously written by Andrew Wetzel.
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
        properties = [prop_name for prop_name in sorted(satellite.keys()) if '.star' not in prop_name and '.err' not in prop_name and ~np.isnan(satellite[prop_name])]
        #
        dof_number = len([i for i in range(0, len(properties)) if ~np.isnan(satellite[properties[i]])])
        #
        for prop_name in properties:
            if ~np.isnan(satellite[prop_name]) and np.isnan(satellite[prop_name+'.err']):
                if 'dist' in prop_name:
                    satellite[prop_name+'.err'] = 5
                if 'vel' in prop_name:
                    satellite[prop_name+'.err'] = 5
        #
        if dof_number == 1:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.0, 2.0, 3.0
        elif dof_number == 2:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.36, 2.27, 3.206
        elif dof_number == 3:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.56, 2.42, 3.32
        elif dof_number == 4:
            sigma_dif_68, sigma_dif_95, sigma_dif_99 = 1.69, 2.52, 3.4
        else:
            raise AssertionError('* DOF number must be between 1-4!')
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
        coord_names = [prop_name for prop_name in properties if prop_name != 'mass.peak']
        #properties = [prop_name for prop_name in sorted(subhalos.keys()) if prop_name != 'snapshot']
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
                    #print('* Satellite(s) {0} are a match at snapshot {1}'.format(match_inds, np.flip(snapshot_data['index'])[:n_snapshots][snap_ind]))
                    # Print out how many of them are within the max errors allowed
                    #print('* {0}, {1} within 68 percent, 95 percent limits'.format(np.sum(sigma_difs_z < sigma_dif_68), np.sum(sigma_difs_z < sigma_dif_95)))
                #
                # If there are no matches, print that out for the current snapshot
                #else:
                    #print('! no subhalos within {0} percent limits at snapshot {1}'.format(probability_max, np.flip(snapshot_data['index'])[snap_ind]))
                #
                # Now re-weight the subhalos so that the centroid is near the middle of the bin
                # If I don't, then I will likely be assigning more weight to lower mass subhalos
            #
            # If there are no matches, print that out
            #else:
                #print('! no subhalos match at snapshot {0}'.format(np.flip(snapshot_data['index'])[snap_ind]))
        #
        return sub_match
    
    def mass_weighting(self, weights, mass_array_subs, mass_sat, SMHM_slope=0.44):
        """
        DESCRIPTION:
            Re-normalizes an array of weights based on a mass array. This is so that
            the resultant orbit property results aren't skewed toward values more
            represented by lower-mass subhalos (the mass function is steep and we
            have many more low-mass subhalos).

        VARIABLES:
            - weights         : array or list
                                An array or list of weights for subhalos analogs of a given satellite.

            - mass_array_subs : array or list
                                An array or list of subhalo masses for the analogs.

            - mass_sat        : float
                                The peak halo mass for the satellite we are interested in.

            - SMHM_slope      : float
                                Value for the slope of the stellar mass - halo mass relation.

        NOTES:
            - Returns an array of equal length to the "weights" variable.
            - This only re-normalizes the weights based on their peak halo mass.
        """
        # Take the log of the masses
        mass_log_sub = np.log10(mass_array_subs)
        mass_log_sat = np.log10(mass_sat)
        #
        # Calculate the mass weights
        weights_m = 10 ** (SMHM_slope * (mass_log_sub - mass_log_sat)) 
        weights_m /= weights_m.sum()  # normalize
        #
        weights_new = weights*weights_m 
        weights_new /= weights_new.sum() 
        #
        return weights_new
    
    def write_subhalo_matches(self, satellite, hosts, indices, weights, snapshots, params):
        """
        DESCRIPTION:
            After finding matches based on the methods above, "subhalo_match()" 
            and "mass_weighting()", save the data to a text file.
            Want this to be the middle step where I save files that include:
                - Host
                - Tree index
                - Weight (after re-weighting by mass)
                - Snapshot at match
                - sigma_dif ( I actually don't think I need this one... ) 

        VARIABLES:
            - satellite : string
                          Name of the satellite of interest

            - hosts     : list or array
                          List of strings, where each element is the name of a host
                          galaxy from the FIRE simulations

            - indices   : list or array
                          List of the subhalo analog indices in the halo tree from
                          the FIRE simulations

            - weights   : array or list
                          Array of weights that were calculated from "subhalo_match()"
                          and "mass_weighting()". These are the subhalo analog weights.

            - snapshots : list or array
                          List of snapshots that the subhalo analogs were matched at.
                          Correspond to the original snapshots in the FIRE simulations.

            - params    : list
                          List of parameters used in the matching schemes above. 
                          Contains: [error in halo dex, the sigma error we choose in
                          the phase-space selection, the max sigma (probability) in the
                          N-dimensional Gaussian]

        NOTES:
            - (Could improve the documentation above)
            - Writes some header info and the following data to a text file:
                - FIRE simulation host galaxy name
                - Halo tree index (at z = 0)
                - Subhalo analog weight
                - Snapshot that the analog was found at
        """
        # If the file exists, then append to it, otherwise create it
        file_path = self.home_dir+'/orbit_data/hdf5_files/satellite_matching/'
        satellite_name = satellite.replace(' ', '_')
        if os.path.isfile(file_path+'weights_'+satellite_name+'.txt'):
            print('File exists. Delete or move it elsewhere.')
            return
        else:
            file_name = 'weights_'+satellite_name+'.txt'
            file_object = open(file_path+file_name, 'w')
            #
            file_object.write('# {0}\n'.format(satellite)) ############################# add more header info like the snapshot window I used and other parameters
            file_object.write('# M_halo,peak bin width: {0} [dex]\n'.format(2*params[0]))
            file_object.write('# {0} sigma error in phase-space coordinates\n'.format(params[1]))
            file_object.write('# {0}% selection in N-D Gaussian\n'.format(params[2]))
            file_object.write('# Host, Halo tree index, Weight, Snapshot at match\n')
            for i in range(0, len(weights)):
                file_object.write('{0}, {1}, {2}, {3} \n'.format(hosts[i], indices[i], weights[i], snapshots[i]))
            file_object.close()
            print('Finished writing to file.')


class SatelliteAnalysis(SatelliteRead):

    def __init__(self, gal1, location):
        """
        DESCRIPTION:
            Mainly used for reading subhalo analog data, creating orbit property distributions,
            and other schemes for plotting.

        VARIABLES:
            - gal1 : string
                     Name of the MW-mass galaxy you are interested in.
                     If analyzing the LG-pairs, this is the name of the
                     first host (Romeo, Thelma, Romulus).
            
            - location : string
                         Name of where you are working (peloton, stampede, or on
                         my mac).

        NOTES:
            Instatiates the SatelliteRead class so that we can use its methods too.
        """
        SatelliteRead.__init__(self, gal1, location, dmo=False)

    def read_subhalo_matches(self, satellite, home_dir=None):
        """
        DESCRIPTION:
            Read in the files with the weights and create a Pandas dataframe object.

        VARIABLES:
            - satellite : string
                          Name of the MW satellite that you want to read in.
            
            - home_dir  : string
                          Path to where the weight files are stored for a satellite.

        NOTES:
            - Returns a Pandas dataframe that contains:
                - The MW-mass host name that the analog lives in.
                - The z = 0 halo tree index for reference.
                - The "weight" of a subhalo for the satellite in question. This
                  is calculated with the methods in the SatelliteMatch class
                  above.
                - The snapshot that the subhalo is matched at. We want this so that
                  we can set that time to be "t = 0" and look at it's orbit and 
                  orbit properties before this instace. We do not care about the analog
                  after this new "t = 0". This would be akin to seeing what a 
                  satellite would do in the "future".
        """
        # If the file exists, then open it
        satellite_name = satellite.replace(' ', '_')
        if home_dir:
            file_path = home_dir+'/weights_'+satellite_name+'.txt'
        else:
            file_path = self.home_dir+'/orbit_data/hdf5_files/satellite_matching/weights_'+satellite_name+'.txt'
        header_info = ['Host', 'Halo tree index', 'Weight', 'Snapshot at match']
        data = pd.read_csv(file_path, skiprows=5, names=header_info)
        #
        return data
    
    def orbit_property_distribution(self, sim_name, mini_sim_data, sat_match_data, time_array):
        """
        DESCRIPTION:
            Using the satellite data, and the subhalo analog data, compile the interesting
            orbit history properties together in a dictionary.

        VARIABLES:
            - sim_name       : string
                               The name of the MW-mass host in the FIRE sims.
                               Limited to m12b, m12c, m12f, m12i, m12m, m12w, m12z, Romeo,
                               Juliet, Thelma, Louise, Romulus, & Remus.

            - mini_sim_data  : dictionary
                               A dictionary containing a lot of information for all of the 
                               subhalos within a FIRE simulation. Includes: distances, velocities,
                               infall times, and TONS of other information.

            - sat_match_data : dictionary
                               A dictionary containing: z = 0 halo tree indices for each subhalo analog
                               to the satellite in question, the host galaxy that it belongs to, 
                               the weight of the analog to the satellite, and the snapshot that it is 
                               matched at.
                               This is what "read_subhalo_matches()" returns.
            
            - time_array     : dictionary
                               Contains all of the snapshots, times, redshifts, and other time-
                               related information from the FIRE simulation. 

        NOTES:
            - Returns a dictionary containing all of the orbit properties that we are interested in:
                - First infall lookback times [Gyr]
                - Peak subhalo mass [M_sun]
                - Pericenter number, distance [kpc], velocity [km/s], and lookback time [Gyr] (both minimum 
                  and recent pericenters)
                - Apocenter distance [kpc] and lookback time [Gyr]
                - Orbit distance [kpc]
                - Radial and tangential velocity [km/s]
            - Any values in this array that are "-1" are null values, i.e., there is no value for that
              particular property and subhalo.
        """
        d = dict()
        #
        # for sim_name in galaxy_name_list: ############# Next time incorporate the loop over all hosts into the function
        #
        # Get the matches for a simulation host
        mask = (sat_match_data['Host'] == sim_name)
        # Get an array of the tree indices that are matches in m12m
        subhalo_inds = np.asarray(sat_match_data['Halo tree index'][mask])
        # Get indices within the mini data to select properties
        mini_data_match_inds = np.asarray([np.where(i == mini_sim_data['indices.z0'][:,0])[0][0] for i in subhalo_inds])
        #
        # Find the infall time
        # Snapshot at match
        time_at_match = time_array['time'][sat_match_data['Snapshot at match'][mask]]
        #
        d['first.infall.time.lb'] = (-1)*np.ones(len(mini_data_match_inds))
        for i in range(0, len(mini_data_match_inds)):
            if (mini_sim_data['first.infall.time'][mini_data_match_inds][i] != -1) and (mini_sim_data['first.infall.snap'][mini_data_match_inds][i] < np.asarray(sat_match_data['Snapshot at match'][mask])[i]):
                time_since_infall_and_match = time_at_match[i] - mini_sim_data['first.infall.time'][mini_data_match_inds][i] # First property to save
                d['first.infall.time.lb'][i] = time_since_infall_and_match
        #
        # Get the peak halo mass of the satellite
        d['halo.mass.peak'] = mini_sim_data['M.halo.peak'][mini_data_match_inds]
        d['pericenter.num'] = mini_sim_data['N.peri.sim'][mini_data_match_inds]
        #
        # Find the most recent pericenter
        d['pericenter.rec.time.lb'] = (-1)*np.ones(len(mini_data_match_inds))
        d['pericenter.rec.dist'] = (-1)*np.ones(len(mini_data_match_inds))
        d['pericenter.rec.vel'] = (-1)*np.ones(len(mini_data_match_inds))
        d['pericenter.min.time.lb'] = (-1)*np.ones(len(mini_data_match_inds)) 
        d['pericenter.min.dist'] = (-1)*np.ones(len(mini_data_match_inds))
        d['pericenter.min.vel'] = (-1)*np.ones(len(mini_data_match_inds))
        d['apocenter.time.lb'] = (-1)*np.ones(len(mini_data_match_inds))
        d['apocenter.dist'] = (-1)*np.ones(len(mini_data_match_inds))
        d['L.tot.sim'] = (-1)*np.ones(len(mini_data_match_inds))
        d['distance'] = (-1)*np.ones(len(mini_data_match_inds))
        d['velocity.rad'] = (-1)*np.ones(len(mini_data_match_inds))
        d['velocity.tan'] = (-1)*np.ones(len(mini_data_match_inds))
        d['v.tot.sim'] = (-1)*np.ones(len(mini_data_match_inds))
        #
        for i in range(0, len(mini_data_match_inds)):
            mask_peri = (mini_sim_data['pericenter.time.sim'][mini_data_match_inds][i] != -1)
            if (np.sum(mask_peri) != 0):
                time_since_dperi_rec = time_at_match[i] - mini_sim_data['pericenter.time.sim'][mini_data_match_inds][i][mask_peri]
                mask_peri_t = (time_since_dperi_rec > 0)
                if (np.sum(mask_peri_t) != 0):
                    d['pericenter.rec.time.lb'][i] = time_since_dperi_rec[mask_peri_t][0]
                    d['pericenter.rec.dist'][i] = mini_sim_data['pericenter.dist.sim'][mini_data_match_inds][i][mask_peri][mask_peri_t][0]
                    d['pericenter.rec.vel'][i] = mini_sim_data['pericenter.vel.sim'][mini_data_match_inds][i][mask_peri][mask_peri_t][0]
                    min_ind = np.where(np.min(mini_sim_data['pericenter.dist.sim'][mini_data_match_inds][i][mask_peri][mask_peri_t]) == mini_sim_data['pericenter.dist.sim'][mini_data_match_inds][i][mask_peri][mask_peri_t])[0][0]
                    d['pericenter.min.time.lb'][i] = time_since_dperi_rec[mask_peri_t][min_ind]
                    d['pericenter.min.dist'][i] = mini_sim_data['pericenter.dist.sim'][mini_data_match_inds][i][mask_peri][mask_peri_t][min_ind]
                    d['pericenter.min.vel'][i] = mini_sim_data['pericenter.vel.sim'][mini_data_match_inds][i][mask_peri][mask_peri_t][min_ind]
            #
            mask_apo = (mini_sim_data['apocenter.time.sim'][mini_data_match_inds][i] != -1)
            if (np.sum(mask_apo) != 0):
                time_since_dapo_rec = time_at_match[i] - mini_sim_data['apocenter.time.sim'][mini_data_match_inds][i][mask_apo]
                mask_apo_t = (time_since_dapo_rec > 0)
                if (np.sum(mask_apo_t) != 0):
                    d['apocenter.time.lb'][i] = time_since_dapo_rec[mask_apo_t][0]
                    d['apocenter.dist'][i] = mini_sim_data['apocenter.dist.sim'][mini_data_match_inds][i][mask_apo][mask_apo_t][0]
            #
            # Get the "lookback" snapshot to the match to get the radial and tangential velocity information
            time_ind = time_array['index'][-1] - np.where(np.min(np.abs(time_at_match[i] - time_array['time'])) == np.abs(time_at_match[i] - time_array['time']))[0][0]
            d['distance'][i] = mini_sim_data['d.tot.sim'][mini_data_match_inds][i][time_ind]
            d['velocity.rad'][i] = mini_sim_data['v.rad.sim'][mini_data_match_inds][i][time_ind]
            d['velocity.tan'][i] = mini_sim_data['v.tan.sim'][mini_data_match_inds][i][time_ind]
            d['L.tot.sim'][i] = mini_sim_data['L.tot.sim'][mini_data_match_inds][i][time_ind]
            d['v.tot.sim'][i] = mini_sim_data['v.tot.sim'][mini_data_match_inds][i][time_ind]
            #
        #
        return d

    def binning_scheme(self, x, xtype, binsize, binedges=None):
        """
        DESCRIPTION:
            Function that takes in an array, a bin size (and bin edges if you want)
            and creates an array that contains the bin values, as well as the
            "half bin" value (useful for plotting).

        VARIABLES:
            - x        : 1D array
                         Array that you want to discretize into bins.

            - xtype    : string
                         The property that you are binning. This is important because
                         binning discrete things like pericenter number is handled 
                         differently and binning masses involves logs.
                         Just use the dictionary keys that are output from 
                         "orbit_property_distribution()"

            - binsize  : int or float
                         How big you want the size of the bins to be.

            - binedges : tuple (I think)
                         Specifies the edges of the bin array you want. If left "None"
                         then the function will return a bin array that includes a 
                         bin below the smallest datapoint in x, and above the highest.

        NOTES:
            - Same thing from "summary_io.py". 
        """
        if 'M.' in xtype:
            x = np.log10(x)
        #
        if binedges:
            bin_num = int((binedges[1]-binedges[0])/binsize + 1)
            bins = np.linspace(binedges[0], binedges[1], bin_num)
            half_bin = (bins[1]-bins[0])/2
        #
        else:
            if 'N.' not in xtype:
                minn = binsize*np.floor(np.nanmin(x)/binsize)
                maxx = binsize*np.ceil(np.nanmax(x)/binsize)
                if (minn < 0) and (maxx > 0):
                    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
                elif (minn < 0) and (maxx < 0):
                    bin_num = np.abs(int(np.around((np.abs(maxx)-np.abs(minn))/binsize)))+1
                else:
                    bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
                bins = np.linspace(minn, maxx, bin_num)
                half_bin = (bins[1]-bins[0])/2
            #
            elif 'N.' in xtype:
                minn = int(binsize*np.floor(np.nanmin(x)/binsize))-0.5
                maxx = int(binsize*np.ceil(np.nanmax(x)/binsize))+0.5
                if minn < 0:
                    bin_num = int((np.abs(maxx)+np.abs(minn))/binsize+1)
                else:
                    bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
        #
        return bins, half_bin
    
    def median_and_scatter(self, x, y, xtype, ytype, bins):
        """
        DESCRIPTION:
            Function that calculates what the median trend is for a data array and bin array.

        VARIABLES:
            - x        : 1D array
                         The x-axis property that you are plotting and binning up.

            - y        : 1D array
                         The y-axis property that you are plotting

            - xtype    : string
                         The property that you are binning. This is important because
                         binning discrete things like pericenter number is handled 
                         differently and binning masses involves logs.
                         Just use the dictionary keys that are output from 
                         "orbit_property_distribution()"
            
            - ytype    : string
                         The property that you are binning. This is important because
                         binning discrete things like pericenter number is handled 
                         differently and binning masses involves logs.
                         Just use the dictionary keys that are output from 
                         "orbit_property_distribution()"

            - bins     : int or float
                         The array of bins for the plot.

        NOTES:
            - Same thing from "summary_io.py". 
            - Loops through the bins, takes data within the bin, and calculates the median in
              that bin, along with the 68th and 95th percentile values.
            - Not really used a lot for Paper III, but used a lot in Paper I and Paper II.
        """
        onesigp = 84.13
        onesigm = 15.87
        twosigp = 97.73
        twosigm = 2.27
        #
        if 'M.' in xtype:
            x = np.log10(x)
        if 'M.' in ytype:
            y = np.log10(y)
        #
        if 'N.' not in ytype:
            #
            med = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            lowest = np.zeros(len(bins)-1)
            highest = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                med[i] = np.nanmedian(y[mask])
                upper[i] = np.nanpercentile(y[mask], onesigp)
                lower[i] = np.nanpercentile(y[mask], onesigm)
                highest[i] = np.nanpercentile(y[mask], twosigp)
                lowest[i] = np.nanpercentile(y[mask], twosigm)
        #
        if 'N.' in ytype:
            #
            twosigp = 100
            twosigm = 0
            #
            means = np.zeros(len(bins)-1)
            scatter = np.zeros(len(bins)-1)
            highest = np.zeros(len(bins)-1)
            lowest = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                means[i] = np.nanmean(y[mask])
                scatter[i] = np.nanstd(y[mask])
                highest[i] = np.nanpercentile(y[mask], twosigp)
                lowest[i] = np.nanpercentile(y[mask], twosigm)
                upper[i] = means[i]+scatter[i]
                lower[i] = means[i]-scatter[i]
                if (upper[i] > highest[i]):
                    upper[i] = highest[i]
                if (lower[i] < lowest[i]):
                    lower[i] = lowest[i]
            #
            med = means
        #
        return med, upper, lower, highest, lowest


