#!/usr/bin/python3

"""
Intended for use with the FIRE-2 simulations

@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

This package is written to help compute the following subhalo orbital parameters
with the OrbitAnalysis class:
    - Infall times of subhalos around a host halo
    - Pericenter distances, velocities, and times
    - Apocenter distances, velocities, and times
    - Orbit angular momentum
    - Orbit energy

There is also a class named OrbitPlot which can generate the following kinds of
figures:
    - Distance of subhalo vs time
        - r, phi, and z components
        - total distance magnitude
    - Velocity of subhalo vs time
        - r, phi, and z components
        - radial or tangential components
        - total velocity magnitude
    - Angular momentum of subhalo vs time
        - r, phi, and z components
        - total angular momentum magnitude
    - Orbit energy vs time
        - potential
        - kinetic
        - total energy (potential + kinetic)
"""

import utilities as ut
from scipy.interpolate import interp1d
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class OrbitAnalysis:

    def __init__(self, tree):
        """
        DESCRIPTION:
            Returns the indices of luminous subhalos along with their progenitor
            indices.

        VARIABLES:
            tree : dictionary

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
        # Select the subhalo indices at z = 0
        z0_inds = ut.array.get_indices(tree['snapshot'], 600)
        # Select luminous subhalos at z = 0 and find their progenitor indices
        z0_inds_w_star = ut.array.get_indices(tree['star.mass'], [1, np.inf], z0_inds)
        self.z0_inds_w_star_prog = tree.prop('progenitor.main.indices', z0_inds_w_star)

    #def get_luminous_halos(self, tree):
        """
        DESCRIPTION:
            Reads in the halo tree, and returns the indices of luminous subhalos
            along with their progenitor indices.

        VARIABLES:
            tree : dictionary

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
        ## Select the subhalo indices at z = 0
        #z0_inds = ut.array.get_indices(tree['snapshot'], 600)
        ## Select luminous subhalos at z = 0 and find their progenitor indices
        #z0_inds_w_star = ut.array.get_indices(tree['star.mass'], [1, np.inf], z0_inds)
        #z0_inds_w_star_prog = tree.prop('progenitor.main.indices', z0_inds_w_star)
        #return z0_inds_w_star_prog

    def halo_distances(self, tree, sub_inds):
        """
        DESCRIPTION:
            Reads in the halo tree and subhalo indices, then returns a 2D array,
            where each row contains subhalo distances from the main host galaxy.

        VARIABLES:
            tree     : dictionary
            sub_inds : 2D array

        NOTES:
            - Returns a 2D array:
                - Each row corresponds to a different subhalo
                - Each element in a row contains the 1D distance from the host
                  galaxy for the subhalo
                - Each row starts at z = 0, and goes back in time
                - Negative elements correspond to times when the subhalo
                  did not exist
            - The 2D array is ordered however the subhalo indices are ordered
        """
        # Set up null 2D array with the same shape as the subhalo index array
        distances = (-1)*np.ones(sub_inds.shape)
        # Loop over the number of subhalos
        for i in range(0, len(sub_inds)):
            # Mask only the subhalos that exist (non-negative elements)
            mask = (sub_inds[i] >= 0)
            # Loop over the number of snapshots a subhalo exists
            for j, val in enumerate(tree.prop('host.distance.total', sub_inds[i][mask])):
                # Fill in the null array with 1D distances
                distances[i][j] = val
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
        distances_norm = (-1)*np.ones(distances.shape)
        for i in range(0, len(distances_norm)):
            mask = (distances[i] >= 0)
            temp = distances[i][mask]/host_halo_radii[:len(distances[i][mask])]
            for j, val in enumerate(temp):
                distances_norm[i][j] = val
        return distances_norm

    def halo_velocities(self, tree, sub_inds):
        """
        DESCRIPTION:
            Reads in the halo tree and subhalo indices, then returns a 2D array,
            where each row contains subhalo velocities (in km/s, physical) from
            the main host galaxy.

        VARIABLES:
            tree     : dictionary
            sub_inds : 2D array

        NOTES:
            - Returns a 2D array:
                - Each row corresponds to a different subhalo
                - Each element in a row contains the 1D velocity from the host
                  galaxy for the subhalo
                - Each row starts at z = 0, and goes back in time
                - Negative elements correspond to times when the subhalo
                  did not exist
            - The 2D array is ordered however the subhalo indices are ordered
        """
        # Set up null 2D array with the same shape as the subhalo index array
        velocities = (-1)*np.ones(sub_inds.shape)
        # Loop over the number of subhalos
        for i in range(0, len(sub_inds)):
            # Mask only the subhalos that exist (non-negative elements)
            mask = (sub_inds[i] >= 0)
            # Loop over the number of snapshots a subhalo exists
            for j, val in enumerate(tree.prop('host.velocity.total', sub_inds[i][mask])):
                # Fill in the null array with 1D distances
                velocities[i][j] = val
        return velocities

    def first_infall_times(self, distances_norm, time_array):
        """
        DESCRIPTION:
            Reads in normalized subhalo distances and snapshot information and returns
            the snapshots and times when the subhalos first fell into the host

        VARIABLES:
            distances_norm : 2D array (given in kpc physical)
            time_array     : dictionary (given in Gyr)

        NOTES:
            - Returns a dictionary
                - d['check'] is a boolean array that tells you if the halo has
                  fallen into the host
                - d['snapshot'] is a 1D array that gives the snapshot at infall
                - d['time'] is a 1D array that gives the age of the Universe when
                  a subhalo first fell into the host galaxy
                - d['time.lb'] is a 1D array that gives the lookback time when
                  a subhalo first fell into the host galaxy
            - Times given correspond to the age of the Universe (Gyr)
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
        # Set up lookback time array
        lookback = time_array['time'][-1] - time_array['time']
        # Loop over subhalos (normalized distance arrays)
        for i in range(0, len(distances_norm)):
            # Check to see if the subhalo is within the virial radius of the host
            if len(np.where(np.abs(distances_norm[i]) < 1)[0]) != 0:
                # If it is, get the snapshot
                first_infall_snap[i] = time_array['index'][-1]-np.max(np.where(np.abs(distances_norm[i]) < 1)[0])
                first_infall_times[i] = time_array['time'][first_infall_snap[i]]
                first_infall_times_lookback[i] = lookback[first_infall_snap[i]]
                if first_infall_snap[i] >= 0:
                    infall_check[i] = True
        # Assign arrays to dictionary elements
        d['check'] = infall_check
        d['snapshot'] = first_infall_snap
        d['time'] = first_infall_times
        d['time.lb'] = first_infall_times_lookback
        return d

    def pericenter_interp(self, distances, velocities, virial_radii, time_array):
        """
        DESCRIPTION:
            Reads in subhalo distances, velocites, host virial radii across time,
            and snapshot information and returns a dictionary of pericenter
            distances, velocities, and times.

        VARIABLES:
            distances    : 2D array (given in kpc physical)
            velocites    : 2D array (km / s)
            virial radii : 1D array (given in kpc physical)
            time_array   : dictionary

        NOTES:
            - Loops through an array and checks to see if a value is smaller than
              4 of its neighbors on either side. If True, also checks to see if this
              distance is within the virial radius of the host. If True, saves some
              values.
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
                - d['pericenter'] is a 2D array
                  Array shape: (number of subhalos) x (max number of pericenters
                                                       any halo experienced)
                  Each row of the array corresponds to a different subhalo
                  Each element in a row gives the pericenter distance (in kpc physical)
                - d['pericenter.velocity'] is a 2D array
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
            # Want initial element to be this because we check +- 4 neighbors on each side
            temp_peri = temp_halo_d[4]
            temp_check = np.zeros(len(temp_halo_d))
            temp_peri_spl = []
            temp_peri_vel_spl = []
            temp_time_spl = []
            #
            # Loop through each subhalo
            for i in range(4, len(temp_halo_d)-4):
                # Check its neighbors and if it is within virial radius
                if (temp_peri < temp_halo_d[i+1]) and (temp_peri < temp_halo_d[i+2]) and (temp_peri < temp_halo_d[i+3])and (temp_peri < temp_halo_d[i+4]) and (temp_peri < temp_halo_d[i-1]) and (temp_peri < temp_halo_d[i-2]) and (temp_peri < temp_halo_d[i-3])and (temp_peri < temp_halo_d[i-4]) and (temp_peri/virial_radii[i] < 1):
                    temp_check[i] = 1
                    peri_rad_list.append(virial_radii[i])
                    temp_peri_spl.append(temp_halo_d[i-4:i+4])
                    temp_peri_vel_spl.append(temp_halo_v[i-4:i+4])
                    temp_time_spl.append(time_array['time'][600-i-4:600-i+4])
                    temp_peri = temp_halo_d[i+1]
                else:
                    temp_peri = temp_halo_d[i+1]
            host_peri_rad.append(peri_rad_list)
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
        # Find lookback time and save to 2D array
        time_lb_spline_array = (-1)*np.ones((len(distances), N))
        mask = (time_spline_array > 0)
        time_lb_spline_array[mask] = (time_array['time'][-1] - time_spline_array[mask])
        #
        d['pericenter.time.lb'] = time_lb_spline_array
        #
        return d

    def apocenter_interp(self, distances, velocities, time_array, infall_array):
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
            infall_array : dictionary

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
            #
            temp_halo_d = distances[k] # Now goes from z = 0 to z_form (un-normalized)
            temp_halo_v = velocities[k] # Same as above
            #
            # Save the max distance a subhalo ever experienced and the times this happens at
            max_dist[k] = np.nanmax(distances[k])
            max_dist_time[k] = np.flip(time_array['time'])[np.where(distances[k] == np.nanmax(distances[k]))[0][0]]
            max_dist_time_lb[k] = (time_array['time'][-1] - max_dist_time[k])
            #
            # Want initial element to be this because we check +- 10 neighbors on each side
            temp_apo = temp_halo_d[10]
            temp_apo_time = time_array['time'][600-10]
            temp_check = np.zeros(len(temp_halo_d))
            temp_apo_spl = []
            temp_apo_vel_spl = []
            temp_time_spl = []
            #
            # Loop through each subhalo
            for i in range(10, len(temp_halo_d)-10):
                # Check to make sure that this is the local maximum
                if (infall_array['time'][k] != -1) and (temp_apo > temp_halo_d[i+1]) and (temp_apo > temp_halo_d[i+2]) and (temp_apo > temp_halo_d[i+3]) and (temp_apo > temp_halo_d[i+4]) and (temp_apo > temp_halo_d[i+5]) and (temp_apo > temp_halo_d[i+6]) and (temp_apo > temp_halo_d[i+7]) and (temp_apo > temp_halo_d[i+8]) and (temp_apo > temp_halo_d[i+9]) and (temp_apo > temp_halo_d[i+10]) and (temp_apo > temp_halo_d[i-1]) and (temp_apo > temp_halo_d[i-2]) and (temp_apo > temp_halo_d[i-3]) and (temp_apo > temp_halo_d[i-4]) and (temp_apo > temp_halo_d[i-5]) and (temp_apo > temp_halo_d[i-6]) and (temp_apo > temp_halo_d[i-7]) and (temp_apo > temp_halo_d[i-8]) and (temp_apo > temp_halo_d[i-9]) and (temp_apo > temp_halo_d[i-10]) and (temp_apo_time > infall_array['time'][k]):
                    temp_check[i] = 1
                    temp_apo_spl.append(temp_halo_d[i-10:i+10])
                    temp_apo_vel_spl.append(temp_halo_v[i-10:i+10])
                    temp_time_spl.append(time_array['time'][600-i-10:600-i+10])
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = time_array['time'][600-(i+1)]
                else:
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = time_array['time'][600-(i+1)]
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
        d['apocenter.dist'] = apocenter_spline_array
        d['apocenter.vel'] = apocenter_vel_spline_array
        d['apocenter.time'] = time_spline_array
        d['apocenter.time.lb'] = time_lb_spline_array
        d['max.dist'] = max_dist
        d['max.dist.time'] = max_dist_time
        d['max.dist.time.lb'] = max_dist_time_lb
        return d

    def angular_momentum(self, tree, sub_inds):
        """
        DESCRIPTION:
            Reads in the tree and subhalo indices and returns a dictionary that contains
            the angular momentum vectors and their magnitudes.

        VARIABLES:
            tree     : dictionary
            sub_inds : 2D array

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
        """
        # Initialize a dictionary to save values to and some arrays
        d = dict();
        ang_mom_vec_tot = []
        ang_mom_norm_tot = []
        #
        # Loop over all of the subhalos
        for i in range(0, len(sub_inds)):
            # Mask out indices where the halo didn't exist (negative indices)
            mask = (sub_inds[i] >= 0)
            #
            # Calculate the different components of angular momentum
            lr = (-1)*tree.prop('host.distance.principal.cylindrical', sub_inds[i][mask])[:,2]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i][mask])[:,1]
            lphi = (-1)*((tree.prop('host.distance.principal.cylindrical', sub_inds[i][mask])[:,0]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i][mask])[:,2]) - (tree.prop('host.distance.principal.cylindrical', sub_inds[i][mask])[:,2]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i][mask])[:,0]))
            lz = tree.prop('host.distance.principal.cylindrical', sub_inds[i][mask])[:,0]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i][mask])[:,1]
            #
            # Save the values to arrays
            ang_mom_vec_subhalo = np.asarray([(lr[j], lphi[j], lz[j]) for j in range(0, len(lr))])
            ang_mom_norm_subhalo = np.linalg.norm(ang_mom_vec_subhalo,axis=1)
            ang_mom_vec_tot.append(ang_mom_vec_subhalo)
            ang_mom_norm_tot.append(ang_mom_norm_subhalo)
        #
        # Create an array that will store the 3D angular momentum of each subhalo across time
        angular_momentum_3d = (-1)*np.ones((len(sub_inds), len(sub_inds[0]), 3))
        #
        # Loop over the number of subhalos
        for i in range(0, len(ang_mom_vec_tot)):
            # Loop over the total number of snapshots
            for j in range(0, len(ang_mom_vec_tot[i])):
                angular_momentum_3d[i][j] = ang_mom_vec_tot[i][j]
        #
        # Create an array that will store the total angular momentum of each subhalo across time
        angular_momentum_1d = (-1)*np.ones((len(sub_inds), len(sub_inds[0])))
        #
        # Loop over the number of subhalos
        for i in range(0, len(ang_mom_norm_tot)):
            for j in range(0, len(ang_mom_norm_tot[i])):
                angular_momentum_1d[i][j] = ang_mom_norm_tot[i][j]
        #
        # Save arrays to dictionary
        d['ang.mom.vector'] = angular_momentum_3d
        d['ang.mom.total'] = angular_momentum_1d
        return d

    def potential_norm(self, tree, potential, sub_inds):
        """
        DESCRIPTION:
            Normalize the subhalo potentials so that their values at z = 0 are
            equal to -2*KE(z = 0), and apply normalization to all other snapshots

        VARIABLES:
            tree      : dictionary
            potential : dictionary
            sub_inds  : 2D array

        NOTES:
            - Returns a 2D array:
                - Array shape: same as sub_inds
                             (number of subhalos) x (total number snapshots)
                - Each row corresponds to a different subhalo
                - Each element gives the subhalo potential at a different snapshot
        """
        # Set up arrays to save the normalized potentials to
        halo_potential_norm_z0 = (-1)*np.ones((sub_inds.shape))
        #
        # Create a mask for the host potential
        mask_host = (sub_inds[0] >= 0)
        host_potential = potential['halo.potentials'][sub_inds[0][mask_host]]
        # Find the kinetic energy, and what you need to multiply the potential to be virialized (for the host; this is zero...)
        kin_host = (0.5*tree.prop('host.velocity.total', sub_inds[0][mask_host])**2)[0]
        multiplier = (-2)*kin_host/potential['halo.potentials'][sub_inds[0][mask_host]][0]
        # Set the normalized potential for the host
        halo_potential_norm_z0[0][mask_host] = multiplier*potential['halo.potentials'][sub_inds[0][mask_host]]
        #
        # Loop through the number of subhalos
        for i in range(1, len(sub_inds)):
            # Create a mask for the snapshots the subhalo existed
            mask = (sub_inds[i] >= 0)
            #
            # Check to see if the host existed longer
            if len(sub_inds[0][mask_host]) >= len(sub_inds[i][mask]):
                # Subtract the host potential from the subhalo potential
                temp = potential['halo.potentials'][sub_inds[i][mask]] - host_potential[:len(sub_inds[i][mask])]
            if len(sub_inds[0][mask_host]) < len(sub_inds[i][mask]):
                # Only get instances where subhalo and host existed
                mask = mask & mask_host
                # Subtract the host potential from the subhalo potential
                temp = potential['halo.potentials'][sub_inds[i][mask]] - host_potential
            #
            # Find the kinetic energy at z = 0
            kin_z0 = (0.5*tree.prop('host.velocity.total', sub_inds[i][mask])**2)[0]
            #
            # Calculate the multiplier to normalize all of the potentials
            multiplier = (-2)*kin_z0/temp[0]
            #
            # Apply the multiplier to the halo potentials and save to the new array
            halo_potential_norm_z0[i][mask] = multiplier*temp
        return halo_potential_norm_z0

    def orbit_energy(self, tree, potential_norm, sub_inds):
        """
        DESCRIPTION:
            Reads in the tree, an array of subhalo gravitational potentials,
            and a subhalo's index and progenitor indices and calculates the total
            orbital energy for a subhalo and it's progenitor subhalos
            (i.e., the energy across time).

        VARIABLES:
            tree           : dictionary
            potential_norm : 2D array
            sub_inds       : 2D array

        NOTES:
            - Energy is defined as E = (1/2)*velocity**2 + potential
            - Returns a dictionary:
                - d['energy.norm.sub'] is a 2D array
                  Array shape: same as sub_inds
                               (number of subhalos) x (total number snapshots)
                  Each row corresponds to a different subhalo
                  Each element is the total energy of the halo at that snapshot
        """
        # Set up an empty array to save to
        energy = (-1)*np.ones((sub_inds.shape))
        #
        # Loop through each subhalo
        for i in range(0, len(sub_inds)):
            # Create a mask to only select subhalos that exist
            mask = (sub_inds[i] >= 0)
            #
            # Calculate the total energy and save it to the array
            energy[i][mask] = 0.5*tree.prop('host.velocity.total', sub_inds[i][mask])**2 + potential_norm[i][mask]
        return energy

class OrbitPlot:

    def orbit_energy_plot(
        self,
        tree,
        potential_norm,
        energy_tot,
        sub_inds,
        subhalo_num,
        infall_array,
        pericenter_array,
        apocenter_array,
        time_array,
        file_name
    ):
        """
        DESCRIPTION:
            Plots the orbital energy of a subhalo across time.

        VARIABLES:
            tree             : dictionary
            potential_norm   : 2D array
            energy_tot       : 2D array
            sub_inds         : list of arrays
            subhalo_num      : integer
                               The subhalo that you want to plot (starts at zero)
                               This is the element of "sub_inds" you want
            infall_array     : dictionary
            pericenter_array : dictionary
            apocenter_array  : dictionary
            time_array       : dictionary
            file_name        : string

        NOTES:
            - This requires that you use the function "orbit_energy()" to generate
              the orbital energies for all subhalos and save them all into one list.
            - Will plot a black vertical line indicating when the subhalo first
              fell into the host.
            - Will plot a red vertical line indicating when the subhalo experienced
              an apocenter event.
            - Will plot a green vertical line indicating when the subhalo
              experienced a pericenter event.
        """
        # Set up a figure to plot to
        plt.rcParams["font.family"] = "serif"
        plt.figure(figsize=(10, 8))
        #
        # Mask out snapshots where subhalo didn't exist
        mask = (sub_inds[subhalo_num] >= 0)
        #
        # Set up the arrays to be plotted. Divide by 1000 to make the y-axis better
        halo_potential = (potential_norm[subhalo_num][mask])/1000
        halo_kinetic = (0.5*tree.prop('host.velocity.total', sub_inds[subhalo_num][mask])**2)/1000
        halo_total = (energy_tot[subhalo_num][mask])/1000
        #
        # Set up lookback time vector and select the time range to plot
        lookback_time = np.flip(time_array['time'][-1] - time_array['time'])
        times = lookback_time[:len(halo_potential)]
        # Plot the data and set the limits
        plt.plot(times, halo_potential, label='U')
        plt.plot(times, halo_kinetic, label='K')
        plt.plot(times, halo_total, linestyle=':', label='E$_{\\rm tot}$')
        plt.xlim(lookback_time[-1], lookback_time[0])
        plt.ylim(min(np.nanmin(halo_potential), np.nanmin(halo_kinetic), np.nanmin(halo_total))-30, max(np.nanmax(halo_total), np.nanmax(halo_kinetic), np.nanmax(halo_total))+30)
        #
        # Check for infall, pericenter, and apocenter events
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        #
        # If there were, plot when they occurred
        if infall == True:
            infall_time = infall_array['time.lb'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            mask = (pericenter_array['pericenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                peri_times = np.asarray(pericenter_array['pericenter.time.lb'][subhalo_num])
                [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            mask = (apocenter_array['apocenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                apo_times = np.asarray(apocenter_array['apocenter.time.lb'][subhalo_num])
                [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        #
        # Set the labels and save the figure
        plt.xlabel('lookback time [Gyr]', fontsize=28)
        plt.ylabel('E [10$^3$ km$^2$ s$^{-2}$]', fontsize=28)
        plt.title('Subhalo '+str(subhalo_num), fontsize=24)
        plt.legend(prop={'size': 14})
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home/ibsantis/scripts/orbit_data/plots/'+file_name+'.pdf')
        plt.close()

    def angular_momentum_plot(
        self,
        ell,
        subhalo_num,
        comp,
        infall_array,
        pericenter_array,
        apocenter_array,
        time_array,
        file_name
    ):
        """
        DESCRIPTION:
            Plot any component of angular momentumn for a subhalo across time

        VARIABLES:
            ell              : dictionary
            subhalo_num      : integer
                               The subhalo you want to plot (starts from zero)
            comp             : string
                               This is the component of angular momentum that you want to plot
            infall_array     : dictionary
            pericenter_array : dictionary
            apocenter_array  : dictionary
            time_array       : dictionary
            file_name        : string

        NOTES:
            - This requires you to use the function "angular_momentum()" beforehand
            - Will plot a black vertical line indicating when the subhalo first
              fell into the host.
            - Will plot a red vertical line indicating when the subhalo experienced
              an apocenter event.
            - Will plot a green vertical line indicating when the subhalo
              experienced a pericenter event.
        """
        # Set up a figure to plot to
        plt.rcParams["font.family"] = "serif"
        plt.figure(figsize=(10, 8))
        #
        # Select which component you want to plot
        if comp == 'r':
            ls = ell['ang.mom.vector'][subhalo_num][:,0]/1000
            comp_str = '$_{\\rm r}$'
        elif comp == 'phi':
            ls = ell['ang.mom.vector'][subhalo_num][:,1]/1000
            comp_str = '$_{\\rm \phi}$'
        elif comp == 'z':
            ls = ell['ang.mom.vector'][subhalo_num][:,2]/1000
            comp_str = '$_{\\rm z}$'
        elif comp == 'all':
            ls = ell['ang.mom.total'][subhalo_num]/1000
            comp_str = '$_{\\rm tot}$'
        #
        # Mask out the snapshots where the subhalo didn't exist
        mask_nan = np.isfinite(ls) # To mask out nan's
        mask_neg = (ls != -1)      # To mask out -1's
        mask_tot = mask_nan & mask_neg
        #
        # Set up lookback time vector and select the time range to plot
        lookback_time = np.flip(time_array['time'][-1] - time_array['time'])
        times = lookback_time[:len(ls[mask_tot])]
        #
        # Plot the data and set the limits
        plt.plot(times, ls[mask_tot])
        plt.xlim(lookback_time[-1], lookback_time[0])
        plt.ylim(np.nanmin(ls)-5, np.nanmax(ls)+5)
        #
        # Check to see if there were infall, pericenter, or apocenter events
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        #
        # If there were, plot when they occurred
        if infall == True:
            infall_time = infall_array['time.lb'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            mask = (pericenter_array['pericenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                peri_times = np.asarray(pericenter_array['pericenter.time.lb'][subhalo_num])
                [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            mask = (apocenter_array['apocenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                apo_times = np.asarray(apocenter_array['apocenter.time.lb'][subhalo_num])
                [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        #
        # Set the labels and save the figure
        plt.xlabel('lookback time [Gyr]', fontsize=28)
        plt.ylabel('L'+comp_str+' [1000 km s$^{-1}$ kpc]', fontsize=28)
        plt.title('Subhalo '+str(subhalo_num), fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home/ibsantis/scripts/orbit_data/plots/'+file_name+'.pdf')
        plt.close()

    def velocity_plot(
        self,
        tree,
        sub_inds,
        subhalo_num,
        comp,
        infall_array,
        pericenter_array,
        apocenter_array,
        time_array,
        file_name
    ):
        """
        DESCRIPTION:
            Plot any component of velocity for a subhalo across time

        VARIABLES:
            tree             : dictionary
            sub_inds         : 2D array
            subhalo_num      : integer
                               The subhalo you want to plot (starts at zero)
            comp             : string
                               Component of velocity you want to plot.
                               Choose between r, phi, z, radial, tangential, or
                               all (total magnitude)
            infall_array     : dictionary
            pericenter_array : dictionary
            apocenter_array  : dictionary
            time_array       : dictionary
            file_name        : string

        NOTES:
            - Will plot a black vertical line indicating when the subhalo first
              fell into the host.
            - Will plot a red vertical line indicating when the subhalo experienced
              an apocenter event.
            - Will plot a green vertical line indicating when the subhalo
              experienced a pericenter event.
        """
        # Set up a figure to plot to
        plt.rcParams["font.family"] = "serif"
        plt.figure(figsize=(10, 8))
        #
        # Mask out snapshots where subhalo didn't exist
        v_mask = (sub_inds[subhalo_num] >= 0)
        #
        # Select which component you want to plot
        if comp == 'r':
            vs = tree.prop('host.velocity.principal.cylindrical', sub_inds[subhalo_num][v_mask])[:,0]
            comp_str = '$_{\\rm r}$'
        elif comp == 'phi':
            vs = tree.prop('host.velocity.principal.cylindrical', sub_inds[subhalo_num][v_mask])[:,2]
            comp_str = '$_{\\rm \phi}$'
        elif comp == 'z':
            vs = tree.prop('host.velocity.principal.cylindrical', sub_inds[subhalo_num][v_mask])[:,1]
            comp_str = '$_{\\rm z}$'
        elif comp == 'rad':
            vs = tree.prop('host.velocity.rad', sub_inds[subhalo_num][v_mask])
            comp_str = '$_{\\rm rad}$'
        elif comp == 'tan':
            vs = tree.prop('host.velocity.tan', sub_inds[subhalo_num][v_mask])
            comp_str = '$_{\\rm tan}$'
        elif comp == 'all':
            vs = tree.prop('host.velocity.principal.cylindrical.total', sub_inds[subhalo_num][v_mask])
            comp_str = '$_{\\rm tot}$'
        elif comp == 'three':
            vs1 = tree.prop('host.velocity.principal.cylindrical.total', sub_inds[subhalo_num][v_mask])
            vs2 = tree.prop('host.velocity.rad', sub_inds[subhalo_num][v_mask])
            vs3 = tree.prop('host.velocity.tan', sub_inds[subhalo_num][v_mask])
            comp_str = ''
        #
        # Plot the data and set the limits
        if comp != 'three':
            # Set up lookback time
            lookback_time = np.flip(time_array['time'][-1] - time_array['time'])
            times = lookback_time[:len(vs)]
            #
            plt.plot(times, vs)
            plt.xlim(lookback_time[-1], lookback_time[0])
            plt.ylim(np.nanmin(vs), np.nanmax(vs))
        else:
            # Set up lookback time
            lookback_time = np.flip(time_array['time'][-1] - time_array['time'])
            times = lookback_time[:len(vs1)]
            #
            plt.plot(times, vs1, label='v$_{\\rm tot}$')
            plt.plot(times, vs2, label='v$_{\\rm rad}$', alpha=0.5)
            plt.plot(times, vs3, label='v$_{\\rm tan}$', alpha=0.5)
            plt.xlim(lookback_time[-1], lookback_time[0])
            plt.ylim(min(np.nanmin(vs1), np.nanmin(vs2), np.nanmin(vs3)), max(np.nanmax(vs1), np.nanmax(vs2), np.nanmax(vs3)))
            plt.legend(prop={'size': 14})
        #
        # Check to see if there were infall, pericenter, or apocenter events
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        #
        # If there were, plot when they occurred
        if infall == True:
            infall_time = infall_array['time.lb'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            mask = (pericenter_array['pericenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                peri_times = np.asarray(pericenter_array['pericenter.time.lb'][subhalo_num])
                [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            mask = (apocenter_array['apocenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                apo_times = np.asarray(apocenter_array['apocenter.time.lb'][subhalo_num])
                [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        #
        # Set your labels and save the figure
        plt.xlabel('lookback time [Gyr]', fontsize=28)
        plt.ylabel('v'+comp_str+' [km s$^{-1}$]', fontsize=28)
        plt.title('Subhalo '+str(subhalo_num), fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home/ibsantis/scripts/orbit_data/plots/'+file_name+'.pdf')
        plt.close()

    def distance_plot(
        self,
        tree,
        sub_inds,
        subhalo_num,
        comp,
        infall_array,
        pericenter_array,
        apocenter_array,
        time_array,
        file_name
    ):
        """
        DESCRIPTION:
            Plot any component of distance for a subhalo across time.

        VARIABLES:
            tree             : dictionary
            sub_inds         : 2D array
            subhalo_num      : integer
                               The subhalo you want to plot (starts at zero)
            comp             : string
                               Component of distance/position you want to plot.
                               Choose between r, phi, z, or all (total magnitude)
            infall_array     : dictionary
            pericenter_array : dictionary
            apocenter_array  : dictionary
            time_array       : dictionary
            file_name        : string

        NOTES:
            - Will plot a black vertical line indicating when the subhalo first
              fell into the host.
            - Will plot a red vertical line indicating when the subhalo experienced
              an apocenter event.
            - Will plot a green vertical line indicating when the subhalo
              experienced a pericenter event.
        """
        # Open a figure
        plt.rcParams["font.family"] = "serif"
        plt.figure(figsize=(10, 8))
        #
        # Mask out snapshots where subhalo didn't exist
        d_mask = (sub_inds[subhalo_num] >= 0)
        # See which component you want to plot
        if comp == 'r':
            ds = tree.prop('host.distance.principal.cylindrical', sub_inds[subhalo_num][d_mask])[:,0]
            comp_str = '$_{\\rm r}$'
        elif comp == 'phi':
            ds = tree.prop('host.distance.principal.cylindrical', sub_inds[subhalo_num][d_mask])[:,1]
            comp_str = '$_{\\rm \phi}$'
        elif comp == 'z':
            ds = tree.prop('host.distance.principal.cylindrical', sub_inds[subhalo_num][d_mask])[:,2]
            comp_str = '$_{\\rm z}$'
        elif comp == 'all':
            ds = tree.prop('host.distance.principal.cylindrical.total', sub_inds[subhalo_num][d_mask])
            comp_str = '$_{\\rm tot}$'
        #
        # Set up lookback time vector and select the time range to plot
        lookback_time = np.flip(time_array['time'][-1] - time_array['time'])
        times = lookback_time[:len(ds)]
        #
        # Plot the data and set the limits
        plt.plot(times, ds)
        plt.xlim(lookback_time[-1], lookback_time[0])
        plt.ylim(0, np.nanmax(ds))
        #
        # Check to see if there were infall, pericenter, or apocenter events
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        #
        # If there are, plot when they occurred
        if infall == True:
            infall_time = infall_array['time.lb'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            mask = (pericenter_array['pericenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                peri_times = pericenter_array['pericenter.time.lb'][subhalo_num][mask]
                [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            mask = (apocenter_array['apocenter.time.lb'][subhalo_num] > 0)
            if np.sum(mask) > 0:
                apo_times = apocenter_array['apocenter.time.lb'][subhalo_num][mask]
                [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        #
        # Set your labels and save the figure
        plt.xlabel('lookback time [Gyr]', fontsize=28)
        plt.ylabel('d'+comp_str+' [kpc]', fontsize=28)
        plt.title('Subhalo '+str(subhalo_num), fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home/ibsantis/scripts/orbit_data/plots/'+file_name+'.pdf')
        plt.close()
