#!/usr/bin/python3

"""
Intended for use with the FIRE-2 simulations

@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

This package is written to help compute the following subhalo orbital parameters
with the OrbitAnalysis class:
    - Infall times of subhalos around a host halo
    - Pericenter distances, velocities, and times
    - Apoventer distances, velocities, and times
    - Orbit angular momentum
    - Orbit energy

There is also a OrbitPlot class which can generate the following kinds of figures:
    - Distance of subhalo vs time
        - Can plot r, phi, and z components, or the total distance magnitude
    - Velocity of subhalo vs time
        - Can plot r, phi, and z components, or the total velocity magnitude
    - Angular momentum of subhalo vs time
        - Can plot r, phi, and z components, or the total angular
          momentum magnitude
    - Orbit energy vs time
"""

from scipy.interpolate import interp1d
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class OrbitAnalysis:

    def get_luminous_halos(self, tree):
        """
        DESCRIPTION:
            Reads in the halo tree, and returns the indices of luminous subhalos
            along with their progenitor indices.

        VARIABLES:
            tree: dictionary

        NOTES:
            - Returns an array of length equal to the number of luminous subhalos
            - For each subhalo, the length of the array is equal to the number of
              snapshots that is has existed for
            - For each subhalo, the arrays are ordered going from
              z = 0 to z = z_form
        """

        # Select the subhalo indices at z = 0
        z0_inds = np.where(tree['snapshot'] == 600)[0]
        # Create a mask for subhalos with stars
        mask = (tree['star.mass'][z0_inds] > 0)
        # Select luminous subhalos with mask and find their progenitor indices
        z0_inds_w_star = z0_inds[mask]
        z0_inds_w_star_prog = tree.prop('progenitor.main.indices', z0_inds_w_star)
        # Select indices for each subhalo that are non-negative (indices where subhalo exists)
        prog_mask = [(z0_inds_w_star_prog[i] >= 0) for i in range(0, len(z0_inds_w_star_prog))]
        sub_inds_w_prog = [z0_inds_w_star_prog[i][prog_mask[i]] for i in range(0, len(z0_inds_w_star_prog))]
        return sub_inds_w_prog

    def halo_distances(self, tree, sub_inds):
        """
        DESCRIPTION:
            Reads in the subhalo indices and tree, then returns the subhalo distances
            from the tree.

        VARIABLES:
            tree: dictionary
            sub_inds: list of arrays

        NOTES:
            - Returns a list of length equal to the number of subhalos
            - For each subhalo, the length of the list is equal to the number
              of snapshots that it has existed for
            - Lists are ordered however the subhalo indices are ordered
                - If used in conjunction with get_luminous_halos, they go from
                  z = 0 to z = z_form
        """
        distances = [[tree.prop('host.distance.total', sub_inds[i][j]) for j in range(0, len(sub_inds[i]))] for i in range(0, len(sub_inds))]
        return distances

    def halo_distances_norm(self, distances, host_halo_radii):
        """
        DESCRIPTION:
            Reads in distances (for each subhalo) and the host radii (at all
            snapshots that it exists)

        VARIABLES:
            distances: list of lists (given in kpc physical)
            host_halo_radii: array (given in kpc physical)

        NOTES:
            - Returns a list of length equal to the number of subhalos
            - For each subhalo, the length of their array is equal to either
              the number of snapshots the subhalo exists for, or the number of
              snapshots the hosts exists for; takes the smaller value
            - Lists are ordered however the subhalo indices are ordered
                - If used in conjunction with get_luminous_halos, they go from
                z = 0 to z = z_form
        """
        distances_norm = []
        for i in range(0, len(distances)):
            # For when the host has existed longer than the subhalo
            if len(host_halo_radii) > len(distances[i]):
                distances_norm.append(distances[i]/host_halo_radii[:len(distances[i])])
            # For when the subhalo has existed longer than the halo
            else:
                distances_norm.append(distances[i][:len(host_halo_radii)]/host_halo_radii)
        return distances_norm

    def halo_velocities(self, tree, sub_inds):
        """
        DESCRIPTION:
            Reads in the subhalo indices and tree, then returns the subhalo velocities
            from the tree.

        VARIABLES:
            tree: dictionary
            sub_inds: list of arrays

        NOTES:
            - Returns a list of length equal to the number of subhalos
            - For each subhalo, the length of the list is equal to the number
              of snapshots that it has existed for
            - Lists are ordered however the subhalo indices are ordered
                - If used in conjunction with get_luminous_halos, they go from
                  z = 0 to z = z_form
        """
        velocites = [[tree.prop('host.velocity.total', sub_inds[i][j]) for j in range(0, len(sub_inds[i]))] for i in range(0, len(sub_inds))]
        return velocites

    def first_infall_times(self, distances_norm, time_array):
        """
        DESCRIPTION:
            Reads in normalized subhalo distances and snapshot information and returns
            the snapshots and times when the subhalos first fell into the host

        VARIABLES:
            distances_norm: list of arrays (given in kpc physical)
            time_array: dictionary (given in Gyr)

        NOTES:
            - Returns a dictionary
                - d['snapshot'] is an array of length equal to the number of
                  subhalos
                - d['time'] is an array of length equal to the number of subhalos
                  that have fallen into the host
                - d['check'] is a boolean array that tells you if the halo has
                  fallen into the host
            - Times given correspond to the age of the Universe (Gyr)
        """
        # Set up a dictionary to store the information you want
        d = dict();
        first_infall_snap = []
        # Loop over normalized distances
        for i in range(0, len(distances_norm)):
            # Check to see if the subhalo is within the virial radius of the host
            if len(np.where(distances_norm[i] < 1)[0]) != 0:
                first_infall_snap.append(600-np.max(np.where(distances_norm[i] < 1)[0]))
            else:
                first_infall_snap.append(-1)
        # Save the snapshot that this happens at
        d['snapshot'] = np.asarray(first_infall_snap)
        infall_mask = (d['snapshot'] > 0)
        infall_times = []
        for i in range(0, len(d['snapshot'])):
            if d['snapshot'][i] > 0:
                infall_times.append(time_array['time'][d['snapshot'][i]])
            else:
                infall_times.append(-1)
        d['time'] = np.asarray(infall_times)
        d['check'] = infall_mask
        return d

    def pericenter_interp(self, distances, velocities, virial_radii, time_array):
        """
        DESCRIPTION:
            Reads in subhalo distances, velocites, host virial radii across time,
            and snapshot information and returns a dictionary of pericenter distances, times, and a
            boolean array.

        VARIABLES:
            distances: list of lists (given in kpc physical)
            velocites: list of lists (km / s)
            virial radii: array (given in kpc physical)
            time_array: dictionary

        NOTES:
            - Returns a dictionary
                - d['pericenter.check'] is a list of booleans
                  Tells you if the subhalo has experienced a pericenter
                - d['pericenter.host.r200'] is a list of lists
                  Tells you the virial radius of the host at time of subhalo
                  pericenter
                - d['pericenter'] is a list of lists
                  Tells you the pericenter distances (in kpc physical)
                - d['pericenter.velocity'] is a list of lists
                  Tells you the velocity (in km/s) at time of pericenter
                - d['pericenter.time'] is a list of lists
                  Tells you what the age of the Universe was when the subhalo
                  experienced a pericenter
            - Loops through an array and checks to see if a value is smaller than
              4 of its neighbors on either side. If True, also checks to see if this
              distance is within the virial radius of the host. If True, saves some
              values.
        """
        # Set up a dictionary to save values to
        d = dict();
        host_peri_rad = []
        check = []
        peri_spl = []
        peri_vel_spl = []
        time_spl = []
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
        # Create a mask that tells you whether or not halo experienced pericenter
        peri_bool = []
        for i in range(0, len(check)):
            if (np.sum(check[i]) > 0):
                peri_bool.append(True)
            else:
                peri_bool.append(False)
        d['pericenter.check'] = np.asarray(peri_bool)
        # Save the virial radii of the host at pericenter times
        d['pericenter.host.r200'] = host_peri_rad
        # Do the spline fitting
        pericenter_spline = []
        peri_vel_spline = []
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
                peri_vel_spline.append(temp_peri_vel_new_spl)
                time_spline.append(temp_time_new_spl)
            else:
                temp_peri_new_spl = []
                temp_peri_vel_new_spl = []
                temp_time_new_spl = []
                pericenter_spline.append(temp_peri_new_spl)
                peri_vel_spline.append(temp_peri_vel_new_spl)
                time_spline.append(temp_time_new_spl)
        d['pericenter.dist'] = pericenter_spline
        d['pericenter.vel'] = peri_vel_spline
        d['pericenter.time'] = time_spline
        return d

    def apocenter_interp(self, distances, velocities, time_array, infall_array):
        """
        DESCRIPTION:
            Reads in a list of subhalo distances and velocities, as well as
            snapshot information, and returns a dictionary of apocenter distances,
            velocities, and times.

        VARIABLES:
            distances: list of lists (given in kpc physical)
            velocites: list of lists (km / s)
            time_array: dictionary
            infall_array: dictionary

        NOTES:
            - Returns a dictionary
                - d['apocenter.check'] is a list of booleans
                  These will tell you if there was an apocenter event for
                  a specific halo.
                - d['apocenter.dist'] is a list of lists
                  Tells you the apocenter distances (in kpc physical)
                - d['apocenter.velocity'] is a list of lists
                  Tells you the velocites of the subhalos at apocenter
                  (in km/s physical)
                - d['apocenter.time'] is a list of lists
                  Tells you the age of the Universe when a subhalo reached
                  apocenter.
            - Loops through an array and checks to see:
                - If the subhalo has fallen into the host
                - If the subhalo distance at this time is larger than the
                  distances at 4 snapshots on either side of this element.
                If True, saves the values listed above.
        """
        # Set up some initial variables
        d = dict();
        check = []
        apo_spl = []
        apo_vel_spl = []
        time_spl = []
        # Loop through the number of subhalos
        for k in range(0, len(distances)):
            temp_halo_d = distances[k] # Now goes from z = 0 to z_form (un-normalized)
            temp_halo_v = velocities[k] # Same as above
            # Want initial element to be this because we check +- 4 neighbors on each side
            temp_apo = temp_halo_d[4]
            temp_apo_time = time_array['time'][600-4]
            temp_check = np.zeros(len(temp_halo_d))
            temp_apo_spl = []
            temp_apo_vel_spl = []
            temp_time_spl = []
            # Loop through each subhalo
            for i in range(4, len(temp_halo_d)-4):
                if (infall_array['time'][k] != -1) and (temp_apo > temp_halo_d[i+1]) and (temp_apo > temp_halo_d[i+2]) and (temp_apo > temp_halo_d[i+3])and (temp_apo > temp_halo_d[i+4]) and (temp_apo > temp_halo_d[i-1]) and (temp_apo > temp_halo_d[i-2]) and (temp_apo > temp_halo_d[i-3]) and (temp_apo > temp_halo_d[i-4]) and (temp_apo_time > infall_array['time'][k]):
                    temp_check[i] = 1
                    temp_apo_spl.append(temp_halo_d[i-4:i+4])
                    temp_apo_vel_spl.append(temp_halo_v[i-4:i+4])
                    temp_time_spl.append(time_array['time'][600-i-4:600-i+4])
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = time_array['time'][600-(i+1)]
                else:
                    temp_apo = temp_halo_d[i+1]
                    temp_apo_time = time_array['time'][600-(i+1)]
            check.append(temp_check)
            apo_spl.append(temp_apo_spl)
            apo_vel_spl.append(temp_apo_vel_spl)
            time_spl.append(temp_time_spl)
        # Create a mask that tells you whether or not halo experienced apocenter
        apo_bool = []
        for i in range(0, len(check)):
            if (np.sum(check[i]) > 0):
                apo_bool.append(True)
            else:
                apo_bool.append(False)
        d['apocenter.check'] = np.asarray(apo_bool)
        # Do the spline fitting
        apocenter_spline = []
        apo_vel_spline = []
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
                apo_vel_spline.append(temp_apo_vel_new_spl)
                time_spline.append(temp_time_new_spl)
            else:
                temp_apo_new_spl = []
                temp_apo_vel_new_spl = []
                temp_time_new_spl = []
                apocenter_spline.append(temp_apo_new_spl)
                apo_vel_spline.append(temp_apo_vel_new_spl)
                time_spline.append(temp_time_new_spl)
        d['apocenter.dist'] = apocenter_spline
        d['apocenter.vel'] = apo_vel_spline
        d['apocenter.time'] = time_spline
        return d

    def angular_momentum(self, tree, sub_inds):
        """
        DESCRIPTION:
            Reads in the tree and subhalo indices and returns a dictionary that contains
            the angular momentum vectors and their magnitudes.

        VARIABLES:
            tree     : dictionary
            sub_inds : list of arrays

        NOTES:
            - Returns a dictionary:
                - d['ang.mom.vector'] is a list of arrays, where each array contains
                  angular momentum values for each subhalo.
                    - Goes from z = 0 to z = z_form
                    - Each vector is ordered (lr, lphi, lz)
                    - d['ang.mom.vector'][i]: array of angular momentum vectors for subhalo i
                    - d['ang.mom.vector'][i][j]: jth angular momentum vector for subhalo i (at time j)
                - d['ang.mom.total'] is a list of arrays, where each array contains
                  the norm of the angular momentum vector for each subhalo.
                    - Goes from z = 0 to z = z_form
                    - d['ang.mom.total'][i]: array of angular momentum magnitudes for subhalo i
                    - d['ang.mom.vector'][i][j]: jth angular momentum value for subhalo i (at time j)
            - 'tree' is organized as (r, z, phi)
        """
        d = dict();
        ang_mom_vec_tot = []
        ang_mom_norm_tot = []
        for i in range(0, len(sub_inds)):
            lr = (-1)*tree.prop('host.distance.principal.cylindrical', sub_inds[i])[:,1]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i])[:,2]
            lphi = (-1)*((tree.prop('host.distance.principal.cylindrical', sub_inds[i])[:,0]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i])[:,1]) - (tree.prop('host.distance.principal.cylindrical', sub_inds[i])[:,1]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i])[:,0]))
            lz = tree.prop('host.distance.principal.cylindrical', sub_inds[i])[:,0]*tree.prop('host.velocity.principal.cylindrical', sub_inds[i])[:,2]
            ang_mom_vec_subhalo = np.asarray([(lr[j], lphi[j], lz[j]) for j in range(0, len(lr))])
            ang_mom_norm_subhalo = np.linalg.norm(ang_mom_vec_subhalo,axis=1)
            ang_mom_vec_tot.append(ang_mom_vec_subhalo)
            ang_mom_norm_tot.append(ang_mom_norm_subhalo)
        d['ang.mom.vector'] = ang_mom_vec_tot
        d['ang.mom.total'] = ang_mom_norm_tot
        return d

    def orbit_energy(self, tree, potential, sub_inds):
        """
        DESCRIPTION:
            Reads in the tree, a subhalo's index and progenitor indices, and an array
            of subhalo gravitational potentials and calculates the total orbital energy
            for a subhalo and it's progenitor subhalos (i.e., the energy across time).

        VARIABLES:
            tree      : dictionary
            sub_inds  : list of arrays
            potential : array

        NOTES:
            - Energy is defined as E = (1/2)*velocity**2 + potential
            - Returns an array the size of an element of sub_inds
                - To be more explicit, returns values for the subhalo of interest
                  across the entire time range that it existed
            - Only handles ONE subhalo at a time.
        """
        energy = 0.5*tree.prop('host.velocity.total')[sub_inds]**2 + potential['halo.potentials'][sub_inds]
        return energy

class OrbitPlot:

    def orbit_energy_plot(
        self,
        energy_list,
        subhalo_num,
        infall_array,
        pericenter_array,
        apocenter_array,
        time_array,
        file_name
    ):
        """
        DESCRIPTION:
            Plots the orbital energy of a subhalo across time .

        VARIABLES:
            energy_list      : list of lists
            subhalo_num      : integer
                               The subhalo that you want to plot (starts at zero)
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
        plt.figure(figsize=(10, 8))
        halo_energy = energy_list[subhalo_num]
        times = np.flip(time_array['time'], axis=0)[:len(halo_energy)]
        plt.plot(times, halo_energy)
        plt.xlim(0, 13.8)
        plt.ylim(np.nanmin(halo_energy)-300, np.nanmax(halo_energy)+300)
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        if infall == True:
            infall_time = infall_array['time'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            peri_times = np.asarray(pericenter_array['pericenter.time'][subhalo_num])
            [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            apo_times = np.asarray(apocenter_array['apocenter.time'][subhalo_num])
            [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        plt.xlabel('time [Gyr]', fontsize=28)
        plt.ylabel('(1/2)*v$^2$ + U [km$^2$ s$^{-2}$]', fontsize=28)
        plt.title('Subhalo '+str(subhalo_num), fontsize=24)
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
        plt.figure(figsize=(10, 8))
        if comp == 'r':
            ls = ell['ang.mom.vector'][subhalo_num][:,0]
            comp_str = '$_{r}$'
        elif comp == 'phi':
            ls = ell['ang.mom.vector'][subhalo_num][:,1]
            comp_str = '$_{\phi}$'
        elif comp == 'z':
            ls = ell['ang.mom.vector'][subhalo_num][:,2]
            comp_str = '$_{z}$'
        elif comp == 'all':
            ls = ell['ang.mom.total'][subhalo_num]
            comp_str = '$_{tot}$'
        times = np.flip(time_array['time'], axis=0)[:len(ls)]
        plt.plot(times, ls)
        plt.xlim(0, 13.8)
        plt.ylim(np.nanmin(ls)-300, np.nanmax(ls)+300)
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        if infall == True:
            infall_time = infall_array['time'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            peri_times = np.asarray(pericenter_array['pericenter.time'][subhalo_num])
            [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            apo_times = np.asarray(apocenter_array['apocenter.time'][subhalo_num])
            [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        plt.xlabel('time [Gyr]', fontsize=28)
        plt.ylabel('L'+comp_str+' [km s$^{-1}$ kpc]', fontsize=28)
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
            sub_inds         : list of lists
            subhalo_num      : integer
                               The subhalo you want to plot (starts at zero)
            comp             : string
                               Component of velocity you want to plot.
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
        plt.figure(figsize=(10, 8))
        if comp == 'r':
            vs = tree.prop('host.velocity.cylindrical', sub_inds[subhalo_num])[:,0]
            comp_str = '$_{r}$'
        elif comp == 'phi':
            vs = tree.prop('host.velocity.cylindrical', sub_inds[subhalo_num])[:,2]
            comp_str = '$_{\phi}$'
        elif comp == 'z':
            vs = tree.prop('host.velocity.cylindrical', sub_inds[subhalo_num])[:,1]
            comp_str = '$_{z}$'
        elif comp == 'all':
            vs = tree.prop('host.velocity.cylindrical.total', sub_inds[subhalo_num])
            comp_str = '$_{tot}$'
        times = np.flip(time_array['time'], axis=0)[:len(vs)]
        plt.plot(times, vs)
        plt.xlim(0, 13.8)
        plt.ylim(0, np.nanmax(vs))
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        if infall == True:
            infall_time = infall_array['time'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            peri_times = np.asarray(pericenter_array['pericenter.time'][subhalo_num])
            [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            apo_times = np.asarray(apocenter_array['apocenter.time'][subhalo_num])
            [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        plt.xlabel('time [Gyr]', fontsize=28)
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
            sub_inds         : list of lists
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
        plt.figure(figsize=(10, 8))
        if comp == 'r':
            ds = tree.prop('host.distance.cylindrical', sub_inds[subhalo_num])[:,0]
            comp_str = '$_{r}$'
        elif comp == 'phi':
            ds = tree.prop('host.distance.cylindrical', sub_inds[subhalo_num])[:,2]
            comp_str = '$_{\phi}$'
        elif comp == 'z':
            ds = tree.prop('host.distance.cylindrical', sub_inds[subhalo_num])[:,1]
            comp_str = '$_{z}$'
        elif comp == 'all':
            ds = tree.prop('host.distance.cylindrical.total', sub_inds[subhalo_num])
            comp_str = '$_{tot}$'
        times = np.flip(time_array['time'], axis=0)[:len(ds)]
        plt.plot(times, ds)
        plt.xlim(0, 13.8)
        plt.ylim(0, np.nanmax(ds))
        infall = infall_array['check'][subhalo_num]
        peri = pericenter_array['pericenter.check'][subhalo_num]
        apo = apocenter_array['apocenter.check'][subhalo_num]
        if infall == True:
            infall_time = infall_array['time'][subhalo_num]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        if peri == True:
            peri_times = np.asarray(pericenter_array['pericenter.time'][subhalo_num])
            [plt.vlines(peri_times[i], -1000000, 1000000, color='#228833', alpha=0.5, linestyles='dotted') for i in range(0, len(peri_times))]
        if apo == True:
            apo_times = np.asarray(apocenter_array['apocenter.time'][subhalo_num])
            [plt.vlines(apo_times[i], -1000000, 1000000, color='r', alpha=0.8, linestyles='dotted') for i in range(0, len(apo_times))]
        plt.xlabel('time [Gyr]', fontsize=28)
        plt.ylabel('d'+comp_str+' [kpc]', fontsize=28)
        plt.title('Subhalo '+str(subhalo_num), fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home/ibsantis/scripts/orbit_data/plots/'+file_name+'.pdf')
        plt.close()
