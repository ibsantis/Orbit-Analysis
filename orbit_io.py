#!/usr/bin/python3

"""

    Intended for use with the FIRE-2 simulations.

    @author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

    This package contains multiple classes designed to ultimately calculate orbit
    parameters of simulated satellite galaxies. Below is a brief description of the
    various classes included:

    OrbitRead:
        - Defines the home and simulation directories, sets the number of host galaxies,
          and reads in the potential profile fitting data for Galpy.
            [Need to adopt this into my actual code...]

    OrbitAnalysis:
        - Applies a selection function to the satellites. Satellites at z = 0 must
          have:
            - low-resolution DM mass fractions < 2%
            - Mstar > 3e4 (if working with luminous satellites)
            - Mpeak > 1e8 (if working with any satellite)
            - Mpeak * (1-f_baryon) > 1e8 (if working with DMO simulations)
          Then, saves their indices across all time
        - Methods included in this class calculate a variety of orbital properties

    OrbitGalpy:
        - Similar to OrbitAnalysis, includes methods to calculate orbtial properties
          of satellites integrated with Galpy.
        - Selects 6D ICs from a simulation for the satellites of interest, creates
          orbit instances to be used in Galpy.
        - Finally, does a check to see whether or not a satellite nears a "pole",
          which could cause numerical problems in Galpy.

    OrbitPlot:
        - Includes methods to plot different orbital properties.
          [I don't really use this anymore, but this could be improved and useful later.]

"""

import utilities as ut
from scipy.interpolate import interp1d
from scipy import spatial
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from galpy.orbit import Orbit
from astropy import units as u
import pandas as pd
import sys


class OrbitRead:

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
        # Depending on the galaxy name, set up a few variables
        if gal1 == 'Romeo':
            gal2 = 'Juliet'
            self.galaxy = 'm12_elvis_'+gal1+gal2
            if location == 'peloton':
                resolution = '_r3500'
            else:
                resolution = '_res3500'
            self.num_gal = 2
        elif gal1 == 'Thelma':
            gal2 = 'Louise'
            self.galaxy = 'm12_elvis_'+gal1+gal2
            if location == 'peloton':
                resolution = '_r4000'
            else:
                resolution = '_res4000'
            self.num_gal = 2
        elif gal1 == 'Romulus':
            gal2 = 'Remus'
            self.galaxy = 'm12_elvis_'+gal1+gal2
            if location == 'peloton':
                resolution = '_r4000'
            else:
                resolution = '_res4000'
            self.num_gal = 2
        elif gal1 == 'm12z':
            self.galaxy = gal1
            if location == 'peloton':
                resolution = '_r4200'
            else:
                resolution = '_res4200'
            self.num_gal = 1
        else:
            self.galaxy = gal1
            if location == 'peloton':
                resolution = '_r7100'
            else:
                resolution = '_res7100'
            self.num_gal = 1

        # Set up the important paths
        if location == 'mac' and self.num_gal == 1:
            self.home_dir = '/Users/isaiahsantistevan/simulation'
            self.simulation_dir = self.home_dir+'/galaxies/'+self.galaxy+resolution # maybe this is a good place for a try-except statement?
            self.fitting_data = pd.read_csv(self.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
        elif location == 'mac' and self.num_gal == 2:
            self.home_dir = '/Users/isaiahsantistevan/simulation'
            self.gal_1 = gal1
            self.gal_2 = gal2
            self.fitting_data = pd.read_csv(self.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
        #
        elif location == 'peloton' and self.num_gal == 1:
            self.home_dir = '/home/ibsantis/scripts'
            self.simulation_dir = '/share/wetzellab/'+self.galaxy+'/'+self.galaxy+resolution
            self.fitting_data = pd.read_csv(self.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
        elif location == 'peloton' and self.num_gal == 2:
            self.home_dir = '/home/ibsantis/scripts'
            self.simulation_dir = '/share/wetzellab/m12_elvis/'+self.galaxy+resolution
            self.gal_1 = gal1
            self.gal_2 = gal2
            self.fitting_data = pd.read_csv(self.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
        #
        elif location == 'stampede' and self.num_gal == 1:
            self.home_dir = '/home1/05400/ibsantis/scripts'
            self.simulation_dir = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'+self.galaxy+resolution
            self.fitting_data = pd.read_csv(self.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
        elif location == 'stampede' and self.num_gal == 2:
            self.home_dir = '/home1/05400/ibsantis/scripts'
            self.simulation_dir = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'+self.galaxy+resolution
            self.fitting_data = pd.read_csv(self.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
            self.gal_1 = gal1
            self.gal_2 = gal2
        #
        if dmo:
            self.simulation_dir = self.simulation_dir+'_dm'


class OrbitAnalysis:

    def __init__(self, tree, gal1, location, host, dmo=False):
        """
        DESCRIPTION:
            Returns the indices of luminous subhalos along with their progenitor
            indices.

        VARIABLES:
            tree     : dictionary
                       This is the halo merger tree, read in by Andrew's function
                       "halo.io.IO.read_tree" from halo_io.py

            gal1     : string
                       Name of the MW-mass galaxy you are interested in.
                       If analyzing the LG-pairs, this is the name of the
                       first host (Romeo, Thelma, Romulus).

            location : string
                       Name of where you are working (peloton, stampede, or on
                       my mac).

            host     : integer (1 or 2)
                       Host number. This is 1 for the 'm12' hosts, and could be
                       1 or 2 for the LG-pair hosts.

            dmo      : boolean
                       True/False moreso to decide if doing a 'halo'-like selection
                       or 'stellar mass' selection. Set dmo=True for the former.

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
        OrbitRead.__init__(self, gal1, location, dmo=dmo)
        #
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
        # Select subhalos based on their halo mass
        if dmo:
            self.baryon_frac = tree.Cosmology['omega_baryon']/tree.Cosmology['omega_matter']
            z0_inds = z0_inds[ut.array.get_indices(tree.prop('mass.peak',z0_inds)*(1-self.baryon_frac), [1e8,np.inf])]
            z0_inds_w_prog = tree.prop('progenitor.main.indices', z0_inds)
            self.sub_inds = z0_inds_w_prog
        #
        # Select subhalos based on their stellar mass
        else:
            z0_inds_w_star = ut.array.get_indices(tree['star.mass'], [3e4, np.inf], z0_inds)
            z0_inds_w_star_prog = tree.prop('progenitor.main.indices', z0_inds_w_star)
            self.sub_inds = z0_inds_w_star_prog
        #
        self.shape = self.sub_inds.shape

    def halo_distances(self, tree, dist_type='total', host=1):
        """
        DESCRIPTION:
            Reads in the halo tree and subhalo indices, then returns a 2D array,
            where each row contains subhalo distances from the main host galaxy.

        VARIABLES:
            tree      : dictionary
            dist_type : string
            host      : integer (1 or 2)

        NOTES:
            - Returns a 2D (or 3D) array:
                - Each row corresponds to a different subhalo
                - Each element in a row contains the 1D (or 3D) distance from the host
                  galaxy for the subhalo
                - Each row starts at z = 0, and goes back in time
                - Negative elements correspond to times when the subhalo
                  did not exist
            - The 2D array is ordered however the subhalo indices are ordered
            - The default is to create arrays of distances from the first host
                - If using a LG simulation, need to specify the second host to
                  get distances from that host
        """
        # Set string for whichever host you're interested in
        if host == 2:
            hindex = 'host2'
        else:
            hindex = 'host'
        #
        # If you want to save the full 3D distance
        if dist_type == '3d':
            # Set up null 3D array with the same shape as the subhalo index array in the first two dimensions
            distances = (-1)*np.ones((self.shape[0],self.shape[1],3))
            # Loop over the number of subhalos
            for i in range(0, len(self.sub_inds)):
                # Mask only the subhalos that exist (non-negative elements)
                mask = (self.sub_inds[i] >= 0)
                # Loop over the number of snapshots a subhalo exists
                for j, val in enumerate(tree.prop(hindex+'.distance', self.sub_inds[i][mask])):
                    # Fill in the null array with 3D distances
                    distances[i][j] = val
                #
                # There are cases where the subhalo progenitor existed before the host
                # Replace these nan instances with -1s
                nan_mask = np.isnan(distances[i])
                distances[i][nan_mask] = -1
        #
        else:
            # Set up null 2D array with the same shape as the subhalo index array
            distances = (-1)*np.ones(self.shape)
            # Loop over the number of subhalos
            for i in range(0, len(self.sub_inds)):
                # Mask only the subhalos that exist (non-negative elements)
                mask = (self.sub_inds[i] >= 0)
                # Loop over the number of snapshots a subhalo exists
                for j, val in enumerate(tree.prop(hindex+'.distance.total', self.sub_inds[i][mask])):
                    # Fill in the null array with 1D distances
                    distances[i][j] = val
                #
                # There are cases where the subhalo progenitor existed before the host
                # Replace these nan instances with -1s
                nan_mask = np.isnan(distances[i])
                distances[i][nan_mask] = -1
        #
        return distances

    def halo_distances_norm(self, distances, host_halo_radii):
        """
        DESCRIPTION:
            Reads in 1D distances (for each subhalo) and the host radii (at all
            snapshots that it exists), then returns the subhalo distances
            normalized by the host radii (at all snapshots it exists).

        VARIABLES:
            distances       : 2D array (given in kpc physical)
            host_halo_radii : 1D array (given in kpc physical)

        NOTES:
            - Returns a 2D array:
                - Each row corresponds to a different subhalo
                - Each element in a row contains the normalized distance from
                  the host galaxy
                - Each row starts at z = 0, and goes back in time
                - Negative elements correspond to times when the subhalo
                  did not exist
            - Lists are ordered however the subhalo indices are ordered
        """
        # Set up an empty array to save to
        distances_norm = (-1)*np.ones(distances.shape)
        #
        # Loop over the number of subhalos
        for i in range(0, len(distances_norm)):
            # Mask out snapshots that it didn't exist in
            mask = (distances[i] >= 0)
            # Divide the distances by the host radius at each snapshot
            temp = distances[i][mask]/host_halo_radii[:len(distances[i][mask])]
            # Save values to the empty array
            for j, val in enumerate(temp):
                distances_norm[i][j] = val
        return distances_norm

    def halo_velocities(self, tree, host=1, vel_type='total'):
        """
        DESCRIPTION:
            Reads in the halo tree and subhalo indices, then returns a 2D array,
            where each row contains subhalo velocities (in km/s, physical) from
            the main host galaxy.

        VARIABLES:
            tree : dictionary
            host : int

        NOTES:
            - Returns a 2D array:
                - Each row corresponds to a different subhalo
                - Each element in a row contains the 1D velocity from the host
                  galaxy for the subhalo
                - Each row starts at z = 0, and goes back in time
                - Negative elements correspond to times when the subhalo
                  did not exist
            - The 2D array is ordered however the subhalo indices are ordered
            - The default is to create arrays of velocities from the first host
                - If using a LG simulation, need to specify the second host to
                  get velocities from that host
        """
        # Set string for whichever host you're interested in
        if host == 2:
            hindex = 'host2'
        else:
            hindex = 'host'
        #
        # Set up null 2D array with the same shape as the subhalo index array
        velocities = (-1)*np.ones(self.shape)
        # Loop over the number of subhalos
        for i in range(0, len(self.sub_inds)):
            # Mask only the subhalos that exist (non-negative elements)
            mask = (self.sub_inds[i] >= 0)
            # Loop over the number of snapshots a subhalo exists
            for j, val in enumerate(tree.prop(hindex+'.velocity.'+vel_type, self.sub_inds[i][mask])):
                # Fill in the null array with 1D distances
                velocities[i][j] = val
        #
        return velocities

    def infall_times(self, distances_norm, time_array):
        """
        DESCRIPTION:
            Reads in normalized subhalo distances and snapshot information and returns
            the snapshots and times when the subhalos first fell into the host

        VARIABLES:
            distances_norm : 2D array (given in kpc physical)
            time_array     : dictionary
                             This is the snapshot time dictionary stored in the
                             simulations

        NOTES:
            - Returns a dictionary
                - d['check'] is a boolean array that tells you if the subhalo has
                  fallen into the host
                - d['first.infall.snap'] is a 1D array that gives the snapshot at infall
                - d['first.infall.time'] is a 1D array that gives the age of the Universe when
                  a subhalo first fell into the host galaxy
                - d['first.infall.time.lb'] is a 1D array that gives the lookback time when
                  a subhalo first fell into the host galaxy
                - d['all.infall.snap'] is a 2D array that gives the snapshots at infall
                    - The size of these arrays is (number of subhalos) x (max number of infalls)
                - d['all.infall.time'] is a 2D array that gives the ages of the Universe when
                  a subhalo fell into the host galaxy
                    - The size of these arrays is (number of subhalos) x (max number of infalls)
                - d['all.infall.time.lb'] is a 2D array that gives the lookback times when
                  a subhalo fell into the host galaxy
                    - The size of these arrays is (number of subhalos) x (max number of infalls)
            - Negative elements correspond to subhalos that have not fallen into
              the host galaxy
        """
        # Set up a dictionary to store the information you want
        d = dict();
        #
        # Initialize some arrays for the dictionary
        first_infall_snap = (-1)*np.ones(len(distances_norm), int)
        first_infall_times = (-1)*np.ones(len(distances_norm))
        first_infall_times_lookback = (-1)*np.ones(len(distances_norm))
        infall_check = np.zeros(len(distances_norm), bool)
        #
        infall_snaps = []
        infall_times = []
        infall_times_lookback = []
        #
        # Set up lookback time array
        lookback = time_array['time'][-1] - time_array['time']
        #
        # Loop over subhalos (normalized distance arrays)
        for i in range(0, len(distances_norm)):
            temp = []
            # Check to see if the subhalo is within the virial radius of the host
            inds = np.where(np.abs(distances_norm[i]) < 1)[0]
            #
            # If it is, save all indices of when it fell into the host
            if len(inds) != 0:
                for j in range(0, len(inds)-1):
                    if (inds[j+1] > inds[j]+1):
                        temp.append(inds[j])
                temp.append(np.max(inds))
                #
                # Save the infall snapshots and times
                infall_snaps.append(time_array['index'][-1] - temp)
                infall_times.append(time_array['time'][infall_snaps[i]])
                infall_times_lookback.append(lookback[infall_snaps[i]])
                #
                first_infall_snap[i] = time_array['index'][-1]-np.max(np.where(np.abs(distances_norm[i]) < 1)[0])
                first_infall_times[i] = time_array['time'][first_infall_snap[i]]
                first_infall_times_lookback[i] = lookback[first_infall_snap[i]]
                #
                # Save whether or not subhalo fell into host
                if first_infall_snap[i] >= 0:
                    infall_check[i] = True
            else:
                infall_snaps.append(np.array([-1]))
                infall_times.append(np.array([-1]))
                infall_times_lookback.append(np.array([-1]))
        #
        # Find the maximum number of infalls any of the satellites experienced
        N = np.max([len(infall_snaps[i]) for i in range(0, len(infall_snaps))])
        #
        # Set up empty arrays to save all instances of infall
        all_infall_snaps = (-1)*np.ones((len(distances_norm), N))
        all_infall_times = (-1)*np.ones((len(distances_norm), N))
        all_infall_times_lookback = (-1)*np.ones((len(distances_norm), N))
        #
        # Fill in the data
        for i in range(0, len(distances_norm)):
            for j in range(0, len(infall_snaps[i])):
                all_infall_snaps[i,j] = infall_snaps[i][j]
                all_infall_times[i,j] = infall_times[i][j]
                all_infall_times_lookback[i,j] = infall_times_lookback[i][j]
        #
        # Assign arrays to dictionary elements
        d['check'] = infall_check
        d['first.infall.snap'] = first_infall_snap
        d['first.infall.time'] = first_infall_times
        d['first.infall.time.lb'] = first_infall_times_lookback
        d['all.infall.snap'] = all_infall_snaps
        d['all.infall.time'] = all_infall_times
        d['all.infall.time.lb'] = all_infall_times_lookback
        #
        return d

    def first_infall_any(self, tree, time_array, host=1):
        """
        DESCRIPTION:
            Reads in subhalo indices and halo tree, then finds the first instance
            of when a subhalo fell into any other more massive halo.

        VARIABLES:
            tree           : dictionary
            time_array     : dictionary
            host           : integer

        NOTES:
            - Returns a dictionary
                - d['first.infall.snap.any'] is a 1D array that gives the snapshot at infall
                - d['first.infall.time.any'] is a 1D array that gives the age of the Universe when
                  a subhalo first fell into the host galaxy
                - d['first.infall.time.lb.any'] is a 1D array that gives the lookback time when
                  a subhalo first fell into the host galaxy
                - d['infall.check.any'] is a 2D array that gives the snapshots at infall
                - The size of these arrays is (number of subhalos) x (max number of infalls)
            - Negative elements correspond to subhalos that have not fallen into
              the host galaxy
        """
        if host == 1:
            host_str = 'host'
        else:
            host_str = 'host2'
        #
        # Set up a dictionary to store the information you want
        d = dict();
        #
        # Initialize some arrays for the dictionary
        first_infall_snap = (-1)*np.ones(self.shape[0], int)
        first_infall_times = (-1)*np.ones(self.shape[0])
        first_infall_times_lookback = (-1)*np.ones(self.shape[0])
        infall_check = np.zeros(self.shape[0], bool)
        #
        host_infall_check = np.zeros(self.shape[0], bool)
        any_ind = (-1)*np.ones(self.shape[0], int)
        any_star_mass = (-1)*np.ones(self.shape[0])
        any_halo_mass = (-1)*np.ones(self.shape[0])
        #
        # Set up lookback time array
        lookback = time_array['time'][-1] - time_array['time']
        #
        # Loop over subhalos
        host_ind = tree.prop('progenitor.main.indices', tree[host_str+'.index'][0])
        #
        for i in range(0, self.shape[0]):
            # For each subhalo, selects the indices that it existed at
            mask = (self.sub_inds[i] >= 0)
            # Gets the central halo's index for each subhalo
            central_inds = tree.prop('central.index', self.sub_inds[i][mask])
            # If there was a central halo, save some properties to the empty arrays
            if (len(central_inds[central_inds >= 0]) != 0):
                first_infall_snap[i] = tree.prop('snapshot', central_inds[central_inds >= 0][-1])
                first_infall_times[i] = time_array['time'][first_infall_snap[i]]
                first_infall_times_lookback[i] = lookback[first_infall_snap[i]]
                infall_check[i] = True
                #
                if (central_inds[central_inds >= 0][-1] in host_ind):
                    host_infall_check[i] = True
                else:
                    any_ind[i] = central_inds[central_inds >= 0][-1]
                    any_star_mass[i] = tree['star.mass'][central_inds[central_inds >= 0][-1]]
                    any_halo_mass[i] = tree['mass'][central_inds[central_inds >= 0][-1]]
        #
        # Save arrays to a dictionary
        d['first.infall.snap.any'] = first_infall_snap
        d['first.infall.time.any'] = first_infall_times
        d['first.infall.time.lb.any'] = first_infall_times_lookback
        d['infall.check.any'] = infall_check
        #
        d['host.check'] = host_infall_check
        d['any.host.ind'] = any_ind
        d['any.host.star.mass'] = any_star_mass
        d['any.host.halo.mass'] = any_halo_mass
        #
        return d

    # Optmized, but could be improved on sliding window part...
    def pericenter_interp(self, distances, velocities, virial_radii, time_array, infall_array=None, reach=20):
        """
        DESCRIPTION:
            Reads in subhalo distances, velocites, host virial radii across time,
            and snapshot information and returns a dictionary of pericenter
            distances, velocities, and times.

        VARIABLES:
            distances    : 2D array (given in kpc physical)
            velocites    : 2D array (km / s)
            virial radii : 1D array (given in kpc physical)
                           These are the MW-mass HOST halo radii
            time_array   : dictionary
                           This is the snapshot dictionary stored in the
                           simulations
            infall array : dictionary (not used at this time, but will be later)
                           Created from "infall_times()" method above
            reach        : integer

        NOTES:
            - Loops through an array and checks to see if a value is smaller than
              N (defined by 'reach') of its neighbors on either side.
              If True, also checks to see if this distance is within the virial
              radius of the host. If True, saves some values.
            - If a subhalo does not experience pericenter, the distances, velocities,
              times, and host radii values are set to -1
            - Returns a dictionary
                - d['pericenter.check'] is a 1D array of booleans
                  Each element tells you if the subhalo has experienced a pericenter
                - d['pericenter.host.r200'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the virial radius of the host
                    when the subhalo reached pericenter
                - d['pericenter.dist'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the pericenter distance (in kpc physical)
                - d['pericenter.vel'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the pericenter velocity (in kpc physical)
                - d['pericenter.time'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the age of the Universe when the
                    subhalo experienced a pericenter
                - d['pericenter.time.lb'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the lookback time when the
                  subhalo experienced a pericenter
            **[Want to add more to this to have the sliding window on the other
               side of the distance array]**
        """
        # Set up a dictionary and lists to save values to
        d = dict();
        host_peri_rad = []
        check = []
        peri_spl = []
        peri_vel_spl = []
        time_spl = []
        #
        # Loop over the number of subhalos
        for k in range(0, len(distances)):
            temp_halo_d = distances[k] # Now goes from z = 0 to z_form (un-normalized)
            temp_halo_v = velocities[k] # Same as above
            peri_rad_list = []
            # Want initial element to be this because we check neighbors on each side
            temp_peri = temp_halo_d[reach]
            temp_check = np.zeros(len(temp_halo_d))
            temp_peri_spl = []
            temp_peri_vel_spl = []
            temp_time_spl = []
            #
            # Loop through each subhalo
            start_ind = 4
            for i in range(start_ind, len(temp_halo_d)-reach):
                # These if-else statements allow for a "sliding", non-symmetric window to look for pericenters
                if (i-reach < 0):
                    left_ind = 0
                else:
                    left_ind = i-reach
                #
                # NEEDS WORK HERE. This would also allow for a non-symmetric window on the other side of the array...
                #if (i+1+reach > (600-infall_array['first.infall.snap'][k])):
                #    right_ind = 600-infall_array['first.infall.snap'][k]
                #else:
                #    right_ind = i+1+reach
                ##
                #if (right_ind-left_ind > 10):
                #...
                #
                # Check its neighbors and if it is within virial radius
                if (all(temp_peri < temp_halo_d[left_ind:i])) and (all(temp_peri < temp_halo_d[i+1:i+1+reach])) and (temp_peri/virial_radii[i] < 1):
                    temp_check[i] = 1
                    peri_rad_list.append(virial_radii[i])
                    temp_peri_spl.append(temp_halo_d[left_ind:i+reach])
                    temp_peri_vel_spl.append(temp_halo_v[left_ind:i+reach])
                    temp_time_spl.append(np.flip(time_array['time'])[left_ind:i+reach])
                    temp_peri = temp_halo_d[i+1]
                else:
                    temp_peri = temp_halo_d[i+1]
            #
            host_peri_rad.append(peri_rad_list)
            check.append(temp_check)
            peri_spl.append(temp_peri_spl)
            peri_vel_spl.append(temp_peri_vel_spl)
            time_spl.append(temp_time_spl)
        #
        # Create a mask that tells you whether or not a subhalo experienced a pericenter
        peri_bool = np.zeros(len(check), bool)
        for i in range(0, len(check)):
            if (np.sum(check[i]) > 0):
                peri_bool[i] = True
        d['pericenter.check'] = peri_bool
        #
        # Find maximum number of pericenter events
        N = np.max([len(host_peri_rad[i]) for i in range(0, len(host_peri_rad))])
        #
        # Initialize array with size (number subhalos) x (number of pericenter events)
        host_peri_rad_array = (-1)*np.ones((len(distances), N))
        #
        # Store host radii in 2D array
        for i in range(0, len(host_peri_rad)):
            if len(host_peri_rad[i]) != 0:
                for j in range(0, len(host_peri_rad[i])):
                    host_peri_rad_array[i,j] = host_peri_rad[i][j]
        # Save the 2D array to dictionary
        d['pericenter.host.r200'] = host_peri_rad_array
        #
        # Set up empty lists for spline fitting
        pericenter_spline = []
        pericenter_vel_spline = []
        time_spline = []
        #
        # Loop over all of the subhalos
        for i in range(0, len(peri_spl)):
            # Check if subhalo experienced pericenter. If so, continue.
            if (len(peri_spl[i]) != 0):
                temp_peri_new_spl = []
                temp_peri_vel_new_spl = []
                temp_time_new_spl = []
                # Loop over the number of pericenter events
                for j in range(0, len(peri_spl[i])):
                    temp_dist = peri_spl[i][j]
                    temp_vel = peri_vel_spl[i][j]
                    temp_time = time_spl[i][j]
                    # Work on distance
                    f = interp1d(temp_time, temp_dist, kind='cubic')
                    f2 = interp1d(temp_time, temp_vel, kind='cubic')
                    x_new = np.linspace(temp_time[0], temp_time[-1], 100)
                    temp_peri_new_spl.append(np.min(f(x_new)))
                    temp_time_new_spl.append(x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
                    temp_peri_vel_new_spl.append(f2(x_new)[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
                pericenter_spline.append(temp_peri_new_spl)
                pericenter_vel_spline.append(temp_peri_vel_new_spl)
                time_spline.append(temp_time_new_spl)
            else:
                temp_peri_new_spl = []
                temp_peri_vel_new_spl = []
                temp_time_new_spl = []
                pericenter_spline.append(temp_peri_new_spl)
                pericenter_vel_spline.append(temp_peri_vel_new_spl)
                time_spline.append(temp_time_new_spl)
        #
        # Initialize arrays with size (number subhalos) x (number of pericenter events)
        pericenter_spline_array = (-1)*np.ones((len(distances), N))
        pericenter_vel_spline_array = (-1)*np.ones((len(distances), N))
        time_spline_array = (-1)*np.ones((len(distances), N))
        #
        # Store the data in 2D arrays
        for i in range(0, len(pericenter_spline)):
            if len(pericenter_spline[i]) != 0:
                for j in range(0, len(pericenter_spline[i])):
                    pericenter_spline_array[i,j] = pericenter_spline[i][j]
                    pericenter_vel_spline_array[i,j] = pericenter_vel_spline[i][j]
                    time_spline_array[i,j] = time_spline[i][j]
        #
        # Save 2D arrays to dictionary
        d['pericenter.dist'] = pericenter_spline_array
        d['pericenter.vel'] = pericenter_vel_spline_array
        d['pericenter.time'] = time_spline_array
        #
        # Save the number of pericenters a subhalo experiences
        d['pericenter.num'] = np.zeros(len(distances),int)
        for i in range(0, len(distances)):
            d['pericenter.num'][i] = np.sum(d['pericenter.dist'][i] > -1)
        #
        # Find lookback time and save to 2D array
        time_lb_spline_array = (-1)*np.ones((len(distances), N))
        mask = (time_spline_array > 0)
        time_lb_spline_array[mask] = (time_array['time'][-1] - time_spline_array[mask])
        d['pericenter.time.lb'] = time_lb_spline_array
        #
        return d

    # Optimized, but could also be improved on sliding window part.
    def apocenter_interp(self, distances, velocities, time_array, infall_array, reach=20):
        """
        DESCRIPTION:
            Reads in a list of subhalo distances and velocities, as well as
            snapshot information, and returns a dictionary of apocenter distances,
            velocities, and times, as well as maximum distances a subhalo
            experiences, and the times this happens.

        VARIABLES:
            distances    : 2D array (given in kpc physical)
            velocites    : 2D array (km / s)
            time_array   : dictionary
                           This is the snapshot dictionary stored in the
                           simulations
            infall_array : dictionary
                           This is created from the "infall_times()" method
                           above
            reach        : integer

        NOTES:
            - Loops through an array and checks to see:
                - If the subhalo has fallen into the host
                - If the subhalo distance at this time is larger than the
                  distances at 10 snapshots on either side of this element.
                - If True, saves the values listed above.
            - Returns a dictionary
                - d['apocenter.check'] is a 1D array of booleans
                  These will tell you if there was an apocenter event for
                  a specific halo.
                - d['apocenter.dist'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the apocenter distance (in kpc physical)
                - d['apocenter.velocity'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the apocenter velocity (in km/s)
                - d['apocenter.time'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the age of the Universe at apocenter
                - d['apocenter.time.lb'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the lookback time at apocenter
                - d['max.dist'] is a 1D array
                  Each element gives the maximum distance that a subhalo reached
                - d['max.dist.time'] is a 1D array
                  Tells you the age of the Universe when the subhalo was at
                  max distance
                - d['max.dist.time.lb'] is a 1D array
                  Tells you the lookback time when the subhalo was at max distance
            - If a subhalo never reaches apocenter, then distances, velocities,
              and times are set to -1
        """
        # Set up some initial variables
        d = dict();
        check = []
        apo_spl = []
        apo_vel_spl = []
        time_spl = []
        max_dist = np.zeros(len(distances))
        max_dist_time = np.zeros(len(distances))
        max_dist_time_lb = np.zeros(len(distances))
        #
        # Loop through the number of subhalos
        for k in range(0, len(distances)):
            temp_halo_d = distances[k] # Now goes from z = 0 to z_form (un-normalized)
            temp_halo_v = velocities[k] # Same as above
            #
            # Save the max distance a subhalo ever experienced and the times this occurred
            max_dist[k] = np.nanmax(distances[k])
            max_dist_time[k] = np.flip(time_array['time'])[np.where(distances[k] == np.nanmax(distances[k]))[0][0]]
            max_dist_time_lb[k] = (time_array['time'][-1] - max_dist_time[k])
            #
            # Want initial element to be this because we check neighbors on each side
            temp_apo = temp_halo_d[reach]
            temp_check = np.zeros(len(temp_halo_d))
            temp_apo_spl = []
            temp_apo_vel_spl = []
            temp_time_spl = []
            #
            # Loop through each subhalo
            start_ind = 4
            for i in range(start_ind, len(temp_halo_d)-reach):
                # These if-else statements allow for a "sliding", non-symmetric window to look for apocenters
                if (i-reach < 0):
                    left_ind = 0
                else:
                    left_ind = i-reach
                #
                temp_apo_time = np.flip(time_array['time'])[i]
                # Check to make sure that this is the local maximum and post-infall
                if (infall_array['first.infall.time'][k] != -1) and (all(temp_apo > temp_halo_d[left_ind:i])) and (all(temp_apo > temp_halo_d[i+1:i+1+reach])) and (temp_apo_time > infall_array['first.infall.time'][k]):
                    temp_check[i] = 1
                    temp_apo_spl.append(temp_halo_d[left_ind:i+reach])
                    temp_apo_vel_spl.append(temp_halo_v[left_ind:i+reach])
                    temp_time_spl.append(np.flip(time_array['time'])[left_ind:i+reach])
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = np.flip(time_array['time'])[i+1]
                else:
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = np.flip(time_array['time'])[i+1]
            #
            check.append(temp_check)
            apo_spl.append(temp_apo_spl)
            apo_vel_spl.append(temp_apo_vel_spl)
            time_spl.append(temp_time_spl)
        #
        # Create a mask that tells you whether or not halo experienced apocenter
        apo_bool = np.zeros(len(check), bool)
        for i in range(0, len(check)):
            if (np.sum(check[i]) > 0):
                apo_bool[i] = True
        d['apocenter.check'] = apo_bool
        #
        # Set up empty lists to save to during the spline fitting
        apocenter_spline = []
        apocenter_vel_spline = []
        time_spline = []
        #
        # Loop over all of the subhalos
        for i in range(0, len(apo_spl)):
            # Check if subhalo experienced apocenter. If so, continue.
            if (len(apo_spl[i]) != 0):
                temp_apo_new_spl = []
                temp_apo_vel_new_spl = []
                temp_time_new_spl = []
                # Loop over the number of apocenter events
                for j in range(0, len(apo_spl[i])):
                    temp_dist = apo_spl[i][j]
                    temp_vel = apo_vel_spl[i][j]
                    temp_time = time_spl[i][j]
                    # Work on distance
                    f = interp1d(temp_time, temp_dist, kind='cubic')
                    f2 = interp1d(temp_time, temp_vel, kind='cubic')
                    x_new = np.linspace(temp_time[0], temp_time[-1], 100)
                    temp_apo_new_spl.append(np.max(f(x_new)))
                    temp_time_new_spl.append(x_new[np.where(f(x_new) == np.max(f(x_new)))[0][0]])
                    temp_apo_vel_new_spl.append(f2(x_new)[np.where(f(x_new) == np.max(f(x_new)))[0][0]])
                apocenter_spline.append(temp_apo_new_spl)
                apocenter_vel_spline.append(temp_apo_vel_new_spl)
                time_spline.append(temp_time_new_spl)
            else:
                temp_apo_new_spl = []
                temp_apo_vel_new_spl = []
                temp_time_new_spl = []
                apocenter_spline.append(temp_apo_new_spl)
                apocenter_vel_spline.append(temp_apo_vel_new_spl)
                time_spline.append(temp_time_new_spl)
        #
        # Create null arrays that are of size (number of subhalos) x (max number of apocenter events any halo experiences)
        N = np.max([len(apocenter_spline[i]) for i in range(0, len(apocenter_spline))])
        apocenter_spline_array = (-1)*np.ones((len(distances), N))
        apocenter_vel_spline_array = (-1)*np.ones((len(distances), N))
        time_spline_array = (-1)*np.ones((len(distances), N))
        #
        # Fill in the 2D arrays with the spline data
        for i in range(0, len(apocenter_spline)):
            if len(apocenter_spline[i]) != 0:
                for j in range(0, len(apocenter_spline[i])):
                    apocenter_spline_array[i,j] = apocenter_spline[i][j]
                    apocenter_vel_spline_array[i,j] = apocenter_vel_spline[i][j]
                    time_spline_array[i,j] = time_spline[i][j]
        #
        # Find lookback time and save to 2D array
        time_lb_spline_array = (-1)*np.ones((len(distances), N))
        mask = (time_spline_array > 0)
        time_lb_spline_array[mask] = (time_array['time'][-1] - time_spline_array[mask])
        #
        # Save everything to a dictionary
        d['apocenter.dist'] = apocenter_spline_array
        d['apocenter.vel'] = apocenter_vel_spline_array
        d['apocenter.time'] = time_spline_array
        d['apocenter.time.lb'] = time_lb_spline_array
        d['max.dist'] = max_dist
        d['max.dist.time'] = max_dist_time
        d['max.dist.time.lb'] = max_dist_time_lb
        #
        return d

    def angular_momentum(self, tree, host=1):
        """
        DESCRIPTION:
            Reads in the tree and subhalo indices and returns a dictionary that contains
            the cylindrical components of the angular momentum vectors and their magnitudes.

        VARIABLES:
            tree : dictionary
            host : int

        NOTES:
            - Returns a dictionary:
                - d['ang.mom.vector'] is a 3D array
                    - Array shape: (number of subhalos) x (total number of snapshots)
                                    x 3 (for each component of the vector)
                    - d['ang.mom.vector'][i,j,k] gives the k-th angular momentum
                      component of the i-th subhalo at snapshot j
                      - j starts at z = 0 and goes back in time
                    - Each vector is ordered (lr, lphi, lz)
                - d['ang.mom.total'] is a 2D array
                    - Array shape: (number of subhalos) x (total number of snapshots)
                    - d['ang.mom.vector'][i,j]: jth angular momentum value for subhalo i (at time j)
                        - j starts at z = 0 and goes back in time
                - The default is to create arrays of values from the first host
                    - If using a LG simulation, need to specify the second host to
                      get values from that host
        """
        # Set which host string you're working with
        if host == 2:
            hindex = 'host2'
        else:
            hindex = 'host'
        #
        # Initialize a dictionary to save values to and some arrays
        d = dict();
        ang_mom_vec_tot = []
        ang_mom_norm_tot = []
        #
        # Loop over all of the subhalos
        for i in range(0, len(self.sub_inds)):
            # Mask out indices where the halo didn't exist (negative indices)
            mask = (self.sub_inds[i] >= 0)
            #
            # Calculate the different components of angular momentum
            if host == 1:
                lr = (-1)*tree.prop(hindex+'.distance.principal.cylindrical', self.sub_inds[i][mask])[:,2]*tree.prop(hindex+'.velocity.principal.cylindrical', self.sub_inds[i][mask])[:,1]
                lphi = (-1)*((tree.prop(hindex+'.distance.principal.cylindrical', self.sub_inds[i][mask])[:,0]*tree.prop(hindex+'.velocity.principal.cylindrical', self.sub_inds[i][mask])[:,2]) - (tree.prop(hindex+'.distance.principal.cylindrical', self.sub_inds[i][mask])[:,2]*tree.prop(hindex+'.velocity.principal.cylindrical', self.sub_inds[i][mask])[:,0]))
                lz = tree.prop(hindex+'.distance.principal.cylindrical', self.sub_inds[i][mask])[:,0]*tree.prop(hindex+'.velocity.principal.cylindrical', self.sub_inds[i][mask])[:,1]
            #
            # Save the values to arrays
            ang_mom_vec_subhalo = np.asarray([(lr[j], lphi[j], lz[j]) for j in range(0, len(lr))])
            ang_mom_norm_subhalo = np.linalg.norm(ang_mom_vec_subhalo,axis=1)
            ang_mom_vec_tot.append(ang_mom_vec_subhalo)
            ang_mom_norm_tot.append(ang_mom_norm_subhalo)
        #
        # Create an array that will store the 3D angular momentum of each subhalo across time
        angular_momentum_3d = (-1)*np.ones((len(self.sub_inds), len(self.sub_inds[0]), 3))
        #
        # Loop over the number of subhalos
        for i in range(0, len(ang_mom_vec_tot)):
            # Loop over the total number of snapshots
            for j in range(0, len(ang_mom_vec_tot[i])):
                angular_momentum_3d[i][j] = ang_mom_vec_tot[i][j]
        #
        # Create an array that will store the total angular momentum of each subhalo across time
        angular_momentum_1d = (-1)*np.ones((len(self.sub_inds), len(self.sub_inds[0])))
        #
        # Loop over the number of subhalos
        for i in range(0, len(ang_mom_norm_tot)):
            for j in range(0, len(ang_mom_norm_tot[i])):
                angular_momentum_1d[i][j] = ang_mom_norm_tot[i][j]
        #
        # Save arrays to dictionary
        d['ang.mom.vector'] = angular_momentum_3d
        d['ang.mom.total'] = angular_momentum_1d
        #
        return d

    def orbit_period(self, distances, velocities, virial_radii, time_array, infall_array):
        """
        DESCRIPTION:
            Calculate orbital period of satellite based on pericenters or
            apocenters.

        VARIABLES:
            distances    : 2D array (given in kpc physical)
            velocities   : 2D array (km / s)
            virial_radii : 1D array (given in kpc physical)
                           These are the MW-mass HOST halo radii
            time_array   : dictionary
                           This is the snapshot dictionary stored in the
                           simulations
            infall_array : dictionary (not used at this time, but will be later)
                           Created from "infall_times()" method above

        NOTES:
            - Returns a dictionary:
                - d['pericenter.orbit.period'] is a 2D array
                    - Each row corresponds to a different subhalo
                    - Each element corresponds to a different period
                    - Calculated by taking the pericenter to pericenter times,
                      which means there are N_peri -1 period calculations
                - d['apocenter.orbit.check'] is a 2D array
                    - Similar to d['pericenter.orbit.period']
                - For a given subhalo, the derived periods refer to orbits going
                  back in time, i.e. the first element corresponds to the most
                  recent period, the second is the second-most recent, etc.
                - Periods are given in Gyr
        """
        # Find the pericenter and apocenter times
        peri_dict = self.pericenter_interp(distances=distances, velocities=velocities, virial_radii=virial_radii, time_array=time_array)
        apo_dict = self.apocenter_interp(distances=distances, velocities=velocities, time_array=time_array, infall_array=infall_array)
        #
        # Set up empty dictionary to save to
        d = dict()
        #
        # Pericenter periods
        # Set up empty array to save to
        peri_period = (-1)*np.ones((len(peri_dict['pericenter.dist']), np.max(peri_dict['pericenter.num'])-1))
        # Loop over each subhalo
        for i in range(0, len(peri_dict['pericenter.time.lb'])):
            # Check if there are at least 2 pericenters
            if (peri_dict['pericenter.num'][i] > 1):
                # Loop over the number of pericenters - 1, given how period is calculated
                for j in range(0, peri_dict['pericenter.num'][i]-1):
                    peri_period[i,j] = peri_dict['pericenter.time.lb'][i][j+1] - peri_dict['pericenter.time.lb'][i][j]
        #
        # Apocenter periods
        # Set up empty array to save to
        N = np.max([len(apo_dict['apocenter.time.lb'][i]) for i in range(0, len(apo_dict['apocenter.time.lb']))])
        apo_period = (-1)*np.ones((len(apo_dict['apocenter.time.lb']), N-1))
        # Loop over each subhalo
        for i in range(0, len(apo_dict['apocenter.time.lb'])):
            # Check if there are apocenter events and more than one of them
            mask = (apo_dict['apocenter.time.lb'][i] != -1)
            if (np.sum(mask) > 1):
                # Loop over the number of apocenters
                for j in range(0, np.sum(mask)-1):
                    apo_period[i,j] = apo_dict['apocenter.time.lb'][i][j+1] - apo_dict['apocenter.time.lb'][i][j]
        #
        # Save to dictionary
        d['pericenter.orbit.period'] = peri_period
        d['apocenter.orbit.period'] = apo_period
        #
        return d

    def eccentricity(self, distances, velocities, virial_radii, time_array, infall_array):
        """
        DESCRIPTION:
            Calculate the eccentricity based on the apocenters and pericenters
            with the relation below:
                e = (d_apo - d_peri) / (d_apo + d_peri)

        VARIABLES:
            distances    : 2D array (given in kpc physical)
            velocities   : 2D array (km / s)
            virial_radii : 1D array (given in kpc physical)
                           These are the MW-mass HOST halo radii
            time_array   : dictionary
                           This is the snapshot dictionary stored in the
                           simulations
            infall_array : dictionary (not used at this time, but will be later)
                           Created from "infall_times()" method above

        NOTES:
            - Returns a 2D array:
                - Each row corresponds to a different subhalo
                - Each element corresponds to a different eccentricity
                - Calculated by taking subsequent peri/apo-center combinations
                - For a given subhalo, the derived eccentricities refer to orbits
                  going back in time, i.e. the first element corresponds to the
                  most recent eccentricity, the second is the second-most
                  recent, etc.
                - Unitless
        """
        # Find the pericenter and apocenter times
        peri_dict = self.pericenter_interp(distances=distances, velocities=velocities, virial_radii=virial_radii, time_array=time_array)
        apo_dict = self.apocenter_interp(distances=distances, velocities=velocities, time_array=time_array, infall_array=infall_array)
        #
        # Calculate the eccentricities
        # Set up empty list to temporarily append to
        ecc = []
        #
        # Loop over the number of subhalos
        for n in range(0, len(peri_dict['pericenter.check'])):
            # Set up temporary array
            ecc_ind = []
            #
            # Check if there are pericenters and apocenters, create masks
            if (peri_dict['pericenter.check'][n] & apo_dict['apocenter.check'][n]):
                mask_peri = (peri_dict['pericenter.dist'][n] != -1)
                mask_apo = (apo_dict['apocenter.dist'][n] != -1)
                #
                # Check if there is at least 1 apocenter event
                if (np.sum(mask_apo) >= 1):
                    # Check if pericenter was more recent than the apocenter
                    if (peri_dict['pericenter.time.lb'][n][mask_peri][0] < apo_dict['apocenter.time.lb'][n][mask_apo][0]):
                        # Loop through the number of pericenters
                        for j in range(0, len(peri_dict['pericenter.dist'][n][mask_peri])):
                            # Loop through the number of apocenters
                            for i in range(0, len(apo_dict['apocenter.dist'][n][mask_apo])):
                                # Make sure they are adjacent
                                if ((j-i == 1) or (j-i == 0)):
                                    # Calculate and append
                                    ecc_ind.append((apo_dict['apocenter.dist'][n][mask_apo][i]-peri_dict['pericenter.dist'][n][mask_peri][j])/(apo_dict['apocenter.dist'][n][mask_apo][i]+peri_dict['pericenter.dist'][n][mask_peri][j]))
                    #
                    # Check if apocenter was more recent than the pericenter
                    if (peri_dict['pericenter.time.lb'][n][mask_peri][0] > apo_dict['apocenter.time.lb'][n][mask_apo][0]):
                        # Loop through the number of pericenters
                        for j in range(0, len(peri_dict['pericenter.dist'][n][mask_peri])):
                            # Loop through the number of apocenters
                            for i in range(0, len(apo_dict['apocenter.dist'][n][mask_apo])):
                                # Make sure they are adjacent
                                if ((i-j == 1) or (i-j == 0)):
                                    # Calculate and append
                                    ecc_ind.append((apo_dict['apocenter.dist'][n][mask_apo][i]-peri_dict['pericenter.dist'][n][mask_peri][j])/(apo_dict['apocenter.dist'][n][mask_apo][i]+peri_dict['pericenter.dist'][n][mask_peri][j]))
            #
            # Append to main array
            ecc.append(ecc_ind)
        #
        # Find the max number of eccentricity values there are in a subhalo population and set up an empty array for the values
        N = np.max([len(ecc[i]) for i in range(0, len(ecc))])
        eccentricity = (-1)*np.ones((len(peri_dict['pericenter.check']), N))
        #
        # Loop through the range of values and add to 2D array
        for i in range(0, len(peri_dict['pericenter.check'])):
            for j in range(0, len(ecc[i])):
                eccentricity[i,j] = ecc[i][j]
        #
        return eccentricity

    def satellite_host_angle(self):
        """
            Calculate the angle between the angular momentum vectors between the
            satellite and host galaxy.

            Never set up or thought of a way to do this, but leaving it in for
            the time being.
        """
        pass

# Pick up here when documenting again...
class OrbitGalpy(OrbitAnalysis):

    def __init__(self, tree, gal1, location, host, dmo=False):
        """
        This is required to inherit the subhalo indices defined from
        OrbitAnalysis.__init__()
        """
        OrbitAnalysis.__init__(self, tree, gal1, location, host, dmo=dmo)

    def galpy_orbit_init(self, tree, host=1):
        """
        DESCRIPTION:
            Reads in the halo tree and subhalo indices, then returns a list of
            orbit instances in Galpy where each element in the list corresponds
            to the initial conditions for a specific subhalo.

        VARIABLES:
            tree      : dictionary
            host      : integer (1 or 2)

        NOTES:
            - Returns a list:
                - Each element in the list is an "Orbit" initialization, based
                  on a subhalo's 6D initial conditions specified at z = 0
            - Initial conditions are specified at z = 0 and include:
                - Cylindrical R, Z, Phi
                - Cylindrical vR, vz
                - Tangential velocity
            - These are then integrated in Galpy to get model orbits.
        """
        # Initialize an empty array to save the Orbit instances to
        sub_orbits = []
        #
        # Loop through each subhalo and save the 6D initial conditions
        if host == 1:
            for i in range(0, len(self.sub_inds)):
                R = tree.prop('host.distance.principal.cylindrical', self.sub_inds[i][0])[0]
                vR = tree.prop('host.velocity.principal.cylindrical', self.sub_inds[i][0])[0]
                vT = tree.prop('host.velocity.principal.cylindrical', self.sub_inds[i][0])[1] # This is really vphi
                z = tree.prop('host.distance.principal.cylindrical', self.sub_inds[i][0])[2]
                vz = tree.prop('host.velocity.principal.cylindrical', self.sub_inds[i][0])[2]
                phi = np.rad2deg(np.arctan(tree.prop('host.distance.principal', self.sub_inds[i][0])[1]/tree.prop('host.distance.principal', self.sub_inds[i][0])[0]))
                #
                sub_orbits.append(Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.deg]))
        #
        elif host == 2:
            for i in range(0, len(self.sub_inds)):
                R = tree.prop('host2.distance.principal.cylindrical', self.sub_inds[i][0])[0]
                vR = tree.prop('host2.velocity.principal.cylindrical', self.sub_inds[i][0])[0]
                vT = tree.prop('host2.velocity.principal.cylindrical', self.sub_inds[i][0])[1]
                z = tree.prop('host2.distance.principal.cylindrical', self.sub_inds[i][0])[2]
                vz = tree.prop('host2.velocity.principal.cylindrical', self.sub_inds[i][0])[2]
                phi = np.rad2deg(np.arctan(tree.prop('host2.distance.principal', self.sub_inds[i][0])[1]/tree.prop('host2.distance.principal', self.sub_inds[i][0])[0]))
                #
                sub_orbits.append(Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.deg]))
        #
        return Orbit(sub_orbits)

    def galpy_potential(self):
        """
        Have some function that defines the potential? This would be pretty cool.
        Have it use the fitting data that is defined inside of OrbitRead.
        """
        pass

    def galpy_velocities(self, vx, vy, vz):
        """
        DESCRIPTION:
            Takes the radial and tangential velocities given by galpy and computes
            the total scalar velocity.

        VARIABLES:
            vx : 2D array
            vy : 2D array
            vz : 2D array

        NOTES:
            - Returns a 2D array
                - Each element in the array corresponds to a given subhalo
                - For each subhalo, the array is the total scalar velocity from
                  Galpy.
            - These are used in galpy_pericenter_interp & galpy_apocenter_interp
              to return pericenter and apocenter velocities.
        """
        return np.sqrt(vx**2 + vy**2 + vz**2)

    def galpy_pericenter_interp(self, distances, velocities, time_array, virial_radii, reach=20):
        """
        DESCRIPTION:
            Reads in subhalo distances, velocites, host virial radii across time,
            and snapshot information and returns a dictionary of pericenter
            distances, velocities, and times.

        VARIABLES:
            distances    : 2D array (given in kpc physical)
            velocites    : 2D array (km / s)
            time_array   : dictionary
            reach        : integer

        NOTES:
            - Same as OrbitAnalysis.pericenter_interp() except this does not
              check to see if the subhalo is within the virial radius since the
              subhalos are integrated in a static potential (the virial radius
              does not change).
            - Loops through an array and checks to see if a value is smaller than
              N (defined by 'reach') of its neighbors on either side.
              If True, also checks to see if this distance is within the virial
              radius of the host. If True, saves some values.
            - If a subhalo does not experience pericenter, the distances, velocities,
              times, and host radii values are set to -1
            - Returns a dictionary
                - d['pericenter.check'] is a 1D array of booleans
                  Each element tells you if the subhalo has experienced a pericenter
                - d['pericenter.dist'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the pericenter distance (in kpc physical)
                - d['pericenter.vel'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the pericenter velocity (in kpc physical)
                - d['pericenter.time'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the age of the Universe when the
                    subhalo experienced a pericenter
                - d['pericenter.time.lb'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the lookback time when the
                  subhalo experienced a pericenter
            **[Want to add more to this to have the sliding window on the other
               side of the distance array]**
        """
        # Set up a dictionary and lists to save values to
        d = dict();
        check = []
        peri_spl = []
        peri_vel_spl = []
        time_spl = []
        #
        # Loop over the number of subhalos
        for k in range(0, len(distances)):
            temp_halo_d = distances[k] # Now goes from z = 0 to z_form (un-normalized)
            temp_halo_v = velocities[k] # Same as above
            # Want initial element to be this because we check neighbors on each side
            temp_peri = temp_halo_d[reach]
            temp_check = np.zeros(len(temp_halo_d))
            temp_peri_spl = []
            temp_peri_vel_spl = []
            temp_time_spl = []
            #
            # Loop through each subhalo
            start_ind = 4
            for i in range(start_ind, len(temp_halo_d)-reach):
                # These if-else statements allow for a "sliding", non-symmetric window to look for pericenters
                if (i-reach < 0):
                    left_ind = 0
                else:
                    left_ind = i-reach
                #
                # Check its neighbors
                if (all(temp_peri < temp_halo_d[left_ind:i])) and (all(temp_peri < temp_halo_d[i+1:i+1+reach])) and (temp_peri/virial_radii[i] < 1):
                    temp_check[i] = 1
                    temp_peri_spl.append(temp_halo_d[left_ind:i+reach])
                    temp_peri_vel_spl.append(temp_halo_v[left_ind:i+reach])
                    temp_time_spl.append(np.flip(time_array['time'])[left_ind:i+reach])
                    temp_peri = temp_halo_d[i+1]
                else:
                    temp_peri = temp_halo_d[i+1]
            #
            check.append(temp_check)
            peri_spl.append(temp_peri_spl)
            peri_vel_spl.append(temp_peri_vel_spl)
            time_spl.append(temp_time_spl)
        #
        # Create a mask that tells you whether or not halo experienced pericenter
        peri_bool = np.zeros(len(check), bool)
        for i in range(0, len(check)):
            if (np.sum(check[i]) > 0):
                peri_bool[i] = True
        d['pericenter.check'] = peri_bool
        #
        # Find maximum number of pericenter events
        N = np.max([len(peri_spl[i]) for i in range(0, len(peri_spl))])
        #
        # Set up empty lists for spline fitting
        pericenter_spline = []
        pericenter_vel_spline = []
        time_spline = []
        #
        # Loop over all of the subhalos
        for i in range(0, len(peri_spl)):
            # Check if subhalo experienced pericenter. If so, continue.
            if (len(peri_spl[i]) != 0):
                temp_peri_new_spl = []
                temp_peri_vel_new_spl = []
                temp_time_new_spl = []
                # Loop over the number of pericenter events
                for j in range(0, len(peri_spl[i])):
                    temp_dist = peri_spl[i][j]
                    temp_vel = peri_vel_spl[i][j]
                    temp_time = time_spl[i][j]
                    # Work on distance
                    f = interp1d(temp_time, temp_dist, kind='cubic')
                    f2 = interp1d(temp_time, temp_vel, kind='cubic')
                    x_new = np.linspace(temp_time[0], temp_time[-1], 100)
                    temp_peri_new_spl.append(np.min(f(x_new)))
                    temp_time_new_spl.append(x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
                    temp_peri_vel_new_spl.append(f2(x_new)[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
                pericenter_spline.append(temp_peri_new_spl)
                pericenter_vel_spline.append(temp_peri_vel_new_spl)
                time_spline.append(temp_time_new_spl)
            else:
                temp_peri_new_spl = []
                temp_peri_vel_new_spl = []
                temp_time_new_spl = []
                pericenter_spline.append(temp_peri_new_spl)
                pericenter_vel_spline.append(temp_peri_vel_new_spl)
                time_spline.append(temp_time_new_spl)
        #
        # Initialize arrays with size (number subhalos) x (number of pericenter events)
        pericenter_spline_array = (-1)*np.ones((len(distances), N))
        pericenter_vel_spline_array = (-1)*np.ones((len(distances), N))
        time_spline_array = (-1)*np.ones((len(distances), N))
        #
        # Store the data in 2D arrays
        for i in range(0, len(pericenter_spline)):
            if len(pericenter_spline[i]) != 0:
                for j in range(0, len(pericenter_spline[i])):
                    pericenter_spline_array[i,j] = pericenter_spline[i][j]
                    pericenter_vel_spline_array[i,j] = pericenter_vel_spline[i][j]
                    time_spline_array[i,j] = time_spline[i][j]
        #
        # Save 2D arrays to dictionary
        d['pericenter.dist'] = pericenter_spline_array
        d['pericenter.vel'] = pericenter_vel_spline_array
        d['pericenter.time'] = time_spline_array
        #
        # Save the number of pericenters a subhalo experiences
        d['pericenter.num'] = np.zeros(len(distances),int)
        for i in range(0, len(distances)):
            d['pericenter.num'][i] = np.sum(d['pericenter.dist'][i] > -1)
        #
        # Find lookback time and save to 2D array
        time_lb_spline_array = (-1)*np.ones((len(distances), N))
        mask = (time_spline_array > 0)
        time_lb_spline_array[mask] = (time_array['time'][-1] - time_spline_array[mask])
        d['pericenter.time.lb'] = time_lb_spline_array
        #
        return d

    def galpy_apocenter_interp(self, distances, velocities, time_array, infall_array, reach=20):
        """
        DESCRIPTION:
            Reads in a list of integrated subhalo distances and velocities, and
            snapshot information, and returns a dictionary of apocenter distances,
            velocities, and times, as well as maximum distances a subhalo
            experiences, and the times this happens.

        VARIABLES:
            distances    : 2D array (given in kpc physical)
            velocites    : 2D array (km / s)
            time_array   : dictionary
            reach        : integer

        NOTES:
            - Similar to OrbitAnalysis.apocenter_interp() except this does not
              check if the subhalo has fallen into the host since the subhalos
              were integrated in a static potential. For the same reason, this
              does not return maximum distances, times, and velocities.
            - Returns a dictionary
                - d['apocenter.check'] is a 1D array of booleans
                  These will tell you if there was an apocenter event for
                  a specific halo.
                - d['apocenter.dist'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the apocenter distance (in kpc physical)
                - d['apocenter.velocity'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the apocenter velocity (in km/s)
                - d['apocenter.time'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the age of the Universe at apocenter
                - d['apocenter.time.lb'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row corresponds to a different subhalo
                  Each element gives the lookback time at apocenter
            - If a subhalo never reaches apocenter, then distances, velocities,
              and times are set to -1
        """
        # Set up some initial variables
        d = dict();
        check = []
        apo_spl = []
        apo_vel_spl = []
        time_spl = []
        #
        # Loop through the number of subhalos
        for k in range(0, len(distances)):
            temp_halo_d = distances[k] # Now goes from z = 0 to z_form (un-normalized)
            temp_halo_v = velocities[k] # Same as above
            # Want initial element to be this because we check +- 10 neighbors on each side
            temp_apo = temp_halo_d[reach]
            temp_check = np.zeros(len(temp_halo_d))
            temp_apo_spl = []
            temp_apo_vel_spl = []
            temp_time_spl = []
            #
            # Loop through each subhalo
            start_ind = 4
            for i in range(start_ind, len(temp_halo_d)-reach):
                # These if-else statements allow for a "sliding", non-symmetric window to look for pericenters
                if (i-reach < 0):
                    left_ind = 0
                else:
                    left_ind = i-reach
                #
                temp_apo_time = np.flip(time_array['time'])[i]
                # Check to make sure that this is the local maximum
                if (infall_array['first.infall.time'][k] != -1) and (all(temp_apo > temp_halo_d[left_ind:i])) and (all(temp_apo > temp_halo_d[i+1:i+1+reach])) and (temp_apo_time > infall_array['first.infall.time'][k]):
                    temp_check[i] = 1
                    temp_apo_spl.append(temp_halo_d[left_ind:i+reach])
                    temp_apo_vel_spl.append(temp_halo_v[left_ind:i+reach])
                    temp_time_spl.append(np.flip(time_array['time'])[left_ind:i+reach])
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = np.flip(time_array['time'])[i+1]
                else:
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = np.flip(time_array['time'])[i+1]
            #
            check.append(temp_check)
            apo_spl.append(temp_apo_spl)
            apo_vel_spl.append(temp_apo_vel_spl)
            time_spl.append(temp_time_spl)
        #
        # Create a mask that tells you whether or not halo experienced apocenter
        apo_bool = np.zeros(len(check), bool)
        for i in range(0, len(check)):
            if (np.sum(check[i]) > 0):
                apo_bool[i] = True
        d['apocenter.check'] = apo_bool
        #
        # Do the spline fitting
        apocenter_spline = []
        apocenter_vel_spline = []
        time_spline = []
        #
        # Loop over all of the subhalos
        for i in range(0, len(apo_spl)):
            # Check if subhalo experienced apocenter. If so, continue.
            if (len(apo_spl[i]) != 0):
                temp_apo_new_spl = []
                temp_apo_vel_new_spl = []
                temp_time_new_spl = []
                # Loop over the number of apocenter events
                for j in range(0, len(apo_spl[i])):
                    temp_dist = apo_spl[i][j]
                    temp_vel = apo_vel_spl[i][j]
                    temp_time = time_spl[i][j]
                    # Work on distance
                    f = interp1d(temp_time, temp_dist, kind='cubic')
                    f2 = interp1d(temp_time, temp_vel, kind='cubic')
                    x_new = np.linspace(temp_time[0], temp_time[-1], 100)
                    temp_apo_new_spl.append(np.max(f(x_new)))
                    temp_time_new_spl.append(x_new[np.where(f(x_new) == np.max(f(x_new)))[0][0]])
                    temp_apo_vel_new_spl.append(f2(x_new)[np.where(f(x_new) == np.max(f(x_new)))[0][0]])
                apocenter_spline.append(temp_apo_new_spl)
                apocenter_vel_spline.append(temp_apo_vel_new_spl)
                time_spline.append(temp_time_new_spl)
            else:
                temp_apo_new_spl = []
                temp_apo_vel_new_spl = []
                temp_time_new_spl = []
                apocenter_spline.append(temp_apo_new_spl)
                apocenter_vel_spline.append(temp_apo_vel_new_spl)
                time_spline.append(temp_time_new_spl)
        #
        # Create null arrays that are of size (number of subhalos) x (max number of apocenter events any halo experiences)
        N = np.max([len(apocenter_spline[i]) for i in range(0, len(apocenter_spline))])
        apocenter_spline_array = (-1)*np.ones((len(distances), N))
        apocenter_vel_spline_array = (-1)*np.ones((len(distances), N))
        time_spline_array = (-1)*np.ones((len(distances), N))
        #
        # Fill in the 2D arrays with the spline data
        for i in range(0, len(apocenter_spline)):
            if len(apocenter_spline[i]) != 0:
                for j in range(0, len(apocenter_spline[i])):
                    apocenter_spline_array[i,j] = apocenter_spline[i][j]
                    apocenter_vel_spline_array[i,j] = apocenter_vel_spline[i][j]
                    time_spline_array[i,j] = time_spline[i][j]
        #
        # Find lookback time and save to 2D array
        time_lb_spline_array = (-1)*np.ones((len(distances), N))
        mask = (time_spline_array > 0)
        time_lb_spline_array[mask] = (time_array['time'][-1] - time_spline_array[mask])
        #
        # Save everything to a dictionary
        d['apocenter.dist'] = apocenter_spline_array
        d['apocenter.vel'] = apocenter_vel_spline_array
        d['apocenter.time'] = time_spline_array
        d['apocenter.time.lb'] = time_lb_spline_array
        #
        return d

    def galpy_pole_check(self, orbits_int, times):
        """
        DESCRIPTION:
            Reads in the integrates orbits from Galpy, along with the time array
            used in integrating them, and checks if any part of the orbit is
            within 1 degree of the orbital poles. This is important because we
            might not want to trust orbits that were near poles because there
            could be numerical issues in the integration method.

        VARIABLES:
            orbits_int : list of orbit instances
            times      : 1D array

        NOTES:
            - Returns a 1D array
                - Each element in the array corresponds to a given subhalo
                - For each subhalo, returns a True/False value of whether or
                  not the subhalo was within 1 degree of either orbital pole.
                - Checks of circular polar orbits  were strange which is why we
                  include this flag.
        """
        # Create empty array to save to
        check = np.zeros(len(orbits_int), bool)
        #
        # Loop through each subhalo and find out if the angle is within 1 degree of a pole
        for i in range(1, len(orbits_int)):
            angle = np.rad2deg(np.arccos(orbits_int[i].z(times)/np.sqrt(orbits_int[i].R(times)**2+orbits_int[i].z(times)**2)))
            if np.sum(angle < 1) or np.sum(angle > 179):
                check[i] = True
        #
        return check

    # Needs dox
    def galpy_infall_times(self, distances, time_array, distance_threshold=300):
        """
        DESCRIPTION:
            Reads in normalized subhalo distances and snapshot information and returns
            the snapshots and times when the subhalos first fell into the host

        VARIABLES:
            distances_norm     : 2D array (given in kpc physical)
            time_array         : dictionary
            distance_threshold : Distance at which you define infall
                                 Default to 300 kpc

        NOTES:
            - Returns a dictionary
                - d['check'] is a boolean array that tells you if the subhalo has
                  fallen into the host
                - d['infall.snap'] is a 1D array that gives the snapshot at infall
                - d['infall.time'] is a 1D array that gives the age of the Universe when
                  a subhalo recently fell into the host galaxy
                - d['infall.time.lb'] is a 1D array that gives the lookback time when
                  a subhalo recently fell into the host galaxy
            - Negative elements correspond to subhalos that have not fallen into
              the host galaxy
            - Very similar to OrbitAnalysis.infall_time(), only I'm saving the
              most recent infall times, and taking in non-normalized distances
        """
        # Set up a dictionary to store the information you want
        d = dict();
        #
        # Initialize some arrays for the dictionary
        first_infall_snap = (-1)*np.ones(len(distances), int)
        first_infall_times = (-1)*np.ones(len(distances))
        first_infall_times_lookback = (-1)*np.ones(len(distances))
        infall_check = np.zeros(len(distances), bool)
        #
        infall_snaps = []
        infall_times = []
        infall_times_lookback = []
        #
        # Set up lookback time array
        lookback = time_array['time'][-1] - time_array['time']
        distances_norm = distances/distance_threshold
        #
        # Loop over subhalos (normalized distance arrays)
        for i in range(0, len(distances_norm)):
            temp = []
            # Check to see if the subhalo is within the virial radius of the host
            inds = np.where(np.abs(distances_norm[i]) < 1)[0]
            # If it is, save all indices of when it fell into the host
            if len(inds) != 0:
                for j in range(0, len(inds)-1):
                    if (inds[j+1] > inds[j]+1):
                        temp.append(inds[j])
                temp.append(np.max(inds))
                #
                # Save the infall snapshots and times
                infall_snaps.append(time_array['index'][-1] - temp)
                infall_times.append(time_array['time'][infall_snaps[i]])
                infall_times_lookback.append(lookback[infall_snaps[i]])
                #
                first_infall_snap[i] = time_array['index'][-1]-np.max(np.where(np.abs(distances_norm[i]) < 1)[0])
                first_infall_times[i] = time_array['time'][first_infall_snap[i]]
                first_infall_times_lookback[i] = lookback[first_infall_snap[i]]
                #
                # Save whether or not subhalo fell into host
                if first_infall_snap[i] >= 0:
                    infall_check[i] = True
            else:
                infall_snaps.append(np.array([-1]))
                infall_times.append(np.array([-1]))
                infall_times_lookback.append(np.array([-1]))
        #
        # Find the maximum number of infalls any of the satellites experienced
        N = np.max([len(infall_snaps[i]) for i in range(0, len(infall_snaps))])
        #
        # Set up empty arrays to save all instances of infall
        all_infall_snaps = (-1)*np.ones((len(distances_norm), N))
        all_infall_times = (-1)*np.ones((len(distances_norm), N))
        all_infall_times_lookback = (-1)*np.ones((len(distances_norm), N))
        #
        # Fill in the data
        for i in range(0, len(distances_norm)):
            for j in range(0, len(infall_snaps[i])):
                all_infall_snaps[i,j] = infall_snaps[i][j]
                all_infall_times[i,j] = infall_times[i][j]
                all_infall_times_lookback[i,j] = infall_times_lookback[i][j]
        #
        # Assign arrays to dictionary elements
        d['check'] = infall_check
        d['infall.snap'] = (-1)*np.ones(len(infall_check))
        d['infall.time'] = (-1)*np.ones(len(infall_check))
        d['infall.time.lb'] = (-1)*np.ones(len(infall_check))
        #
        for i in range(0, len(infall_check)):
            if infall_check[i]:
                d['infall.snap'][i] = all_infall_snaps[i][all_infall_snaps[i] != -1][0]
                d['infall.time'][i] = all_infall_times[i][all_infall_times[i] != -1][0]
                d['infall.time.lb'][i] = all_infall_times_lookback[i][all_infall_times_lookback[i] != -1][0]
        #
        return d

class OrbitTree(OrbitAnalysis):

    def __init__(self, tree, gal1, location, host, particles, subsampling, dmo=False):
        """
        DESCRIPTION:
            Creates a "tree" structure to efficiently select particles within
            the subhalos. Makes use of the KDTree method.

        VARIABLES:
            tree        : dictionary
            gal1        : string
            location    : string
            host        : integer
            particles   : dictionary
            subsampling : integer
            dmo         : boolean

        NOTES:
            - Creates the tree via scipy's method which uses the KDTree method.
            - This is used in conjunction with the "neighbors" method to select
              particles within certain subhalos.
        """
        OrbitAnalysis.__init__(self, tree, gal1, location, host, dmo=dmo)
        #
        # Build the tree
        self.kdtree = spatial.KDTree(data=particles['dark']['position'][::subsampling].astype(np.float32))
        self.subsampling = subsampling

    def neighbors(self, centers, neigh_num_max, neigh_dist_max, workerss=2):
        """
        DESCRIPTION:
            Find the particles within certain subhalos and return their distances
            and indices.

        VARIABLES:
            centers        : 2D array
            neigh_num_max  : int
            neigh_dist_max : int or float
            workerss       : int

        NOTES:
            - centers: the 3D positions of the subhalos you want particles in
            - neigh_num_max : the maximum number of neighbors you want to select
                              in a given halo
            - neigh_dist_max : the maximum distance you want to search for neighbors
            - workerss: the number of threads you can parallelize with
            - Returns a tuple:
                - pos : the 1D scalar distances for each particle from the subhalos
                - ind : the indices of the particles in the particle array you use
                        to generate the tree
            - This is used to select particles to then calculate the potential of
              a given subhalo.
        """
        pos, ind = self.kdtree.query(x=centers, k=neigh_num_max, distance_upper_bound=neigh_dist_max, workers=workerss)
        #
        return pos, ind

class OrbitPlot(OrbitAnalysis):

    def __init__(self, tree, gal1, location, host, dmo=False):
        """
        This is required to inherit the subhalo indices defined from
        OrbitAnalysis.__init__()
        """
        OrbitAnalysis.__init__(self, tree, gal1, location, host, dmo=dmo)

    def multi_plot(self, host_rads, infall_dict, peri_dict, time_dict, sim_dist, sim_vel, sim_ell, model_orbits, model_times, host=1):
        #
        # Set up the model stuff
        d_model = model_orbits.r(model_times)
        v_model = np.sqrt(model_orbits.vx(model_times)**2 + model_orbits.vy(model_times)**2 + model_orbits.vz(model_times)**2)
        L_model = model_orbits.L(model_times)
        #
        # Loop over the number of subhalos
        for i in range(0, self.shape[0]):
            # Check to see if the subhalo fell in and experienced a pericenter
            if (infall_dict['check'][i]):
                #
                # Set up the distances and times to plot
                d_mask = (sim_dist[i] >= 0)
                d_data = sim_dist[i][d_mask]
                #
                lookback_time = np.flip(time_dict['time'][-1] - time_dict['time'])
                times = lookback_time[:len(d_data)]
                #
                v_data = sim_vel[i][:len(times)]
                L_data = sim_ell['ang.mom.total'][i][:len(times)]
                #
                # Set up the figure
                plt.rcParams["font.family"] = "serif"
                plt.figure(figsize=(10, 12))
                ax1 = plt.subplot(311)
                ax2 = plt.subplot(312, sharex=ax1)
                ax3 = plt.subplot(313, sharex=ax2)
                #
                # Plot the distances
                ax1.plot(times, d_data, 'k', label='Simulation')
                ax1.plot(-1*model_times, d_model[i], label='Model', alpha=0.5)
                ax1.plot(times, host_rads[:len(times)], 'k', alpha=0.3)
                ax1.set_xlim(times[-1], times[0])
                #
                # Check to see if there were infall, pericenter, or apocenter events
                infall = infall_dict['check'][i]
                peri = peri_dict['pericenter.check'][i]
                #
                # If there were, plot when they occurred
                if infall:
                    infall_time = infall_dict['first.infall.time.lb'][i]
                    ax1.axvline(x=infall_time, ymin=0, ymax=1, color='k', linestyle=':')
                #
                if peri:
                    for j in peri_dict['pericenter.time.lb'][i][peri_dict['pericenter.time.lb'][i] != -1]:
                        ax1.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
                #
                # Set the labels and save the figure
                ax1.set_ylim(top=np.nanmax(d_data)+100)
                ax1.label_outer()
                ax1.set_ylabel('Host Distance [kpc]', fontsize=20)
                ax1.legend(prop={'size': 16})
                #
                # Plot the velocity data
                ax2.plot(times, v_data, 'k')
                ax2.plot(-1*model_times, v_model[i], alpha=0.5)
                ax2.set_xlim(times[-1], times[0])
                ax2.label_outer()
                if infall == True:
                    infall_time = infall_dict['first.infall.time.lb'][i]
                    ax2.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
                if peri:
                    for j in peri_dict['pericenter.time.lb'][i][peri_dict['pericenter.time.lb'][i] != -1]:
                        ax2.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
                #
                ax2.set_ylabel('Total velocity [km s$^{-1}$]', fontsize=20)
                #
                # Plot the velocity data
                ax3.plot(times, L_data/1000, 'k')
                ax3.plot(-1*model_times, np.linalg.norm(L_model[i], axis=1)/1000, alpha=0.5)
                ax3.set_xlim(times[-1], times[0])
                ax3.set_ylabel('$\\ell$ [$10^3$ kpc km s$^{-1}$]', fontsize=20)
                if infall == True:
                    infall_time = infall_dict['first.infall.time.lb'][i]
                    ax3.axvline(infall_time, ymin=0, ymax=1, color='k', linestyle=':')
                if peri:
                    for j in peri_dict['pericenter.time.lb'][i][peri_dict['pericenter.time.lb'][i] != -1]:
                        ax3.axvline(x=j, ymin=0, ymax=1, color='#9400D3', linestyle=':')
                #
                ax3.set_xlabel('Lookback time [Gyr]', fontsize=32)
                plt.tight_layout()
                plt.subplots_adjust(wspace=0, hspace=0)
                #
                if self.num_gal == 1:
                    plt.savefig(self.home_dir+'/orbit_data/plots/subhalo_integration/'+self.galaxy+'/'+self.galaxy+'_sub_'+str(i+1)+'.pdf')
                if self.num_gal == 2:
                    if host == 1:
                        plt.savefig(self.home_dir+'/orbit_data/plots/subhalo_integration/'+self.gal_1+'/'+self.gal_1+'_sub_'+str(i+1)+'.pdf')
                    if host == 2:
                        plt.savefig(self.home_dir+'/orbit_data/plots/subhalo_integration/'+self.gal_2+'/'+self.gal_2+'_sub_'+str(i+1)+'.pdf')
                plt.close()

# Testing out some of what Shea wrote in elvis_analysis.elvis_plot.OrbitPropertyClass().get_subhalos_match()
class OrbitMatch(OrbitAnalysis):

    def __init__(self, tree, gal1, location, host, dmo=False):
        """
        This is required to inherit the subhalo indices defined from
        OrbitAnalysis.__init__()
        """
        OrbitAnalysis.__init__(self, tree, gal1, location, host, dmo=dmo)
