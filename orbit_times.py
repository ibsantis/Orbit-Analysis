#!/usr/bin/python3

"""
Intended for use with the FIRE-2 simulations

@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

This package is written to help compute:
    - Infall times of subhalos around a host halo
    - Pericenter distances and times
    - Orbit energy
    - Orbit angular momentum
"""

from scipy.interpolate import interp1d
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class OrbitAnalysis:

    def get_luminous_halos(self, tree):
        """
            Reads in the halo tree, and returns the indices of luminous subhalos
            along with their progenitor indices.

            tree: dictionary

            NOTES:
                - Returns an array of length equal to the number of luminous subhalos
                - For each subhalo, the length of the array is equal to the number of
                  snapshots that is has existed for
                - For each subhalo, the arrays are ordered going from
                  z = 0 to z = z_form

            Returns a list of arrays
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
            Reads in the subhalo indices and tree, then returns the subhalo distances
            from the tree.

            tree: dictionary
            sub_inds: list of arrays

            NOTES:
                - Returns a list of length equal to the number of subhalos
                - For each subhalo, the length of the list is equal to the number
                  of snapshots that it has existed for
                - Lists are ordered however the subhalo indices are ordered
                    - If used in conjunction with get_luminous_halos, they go from
                      z = 0 to z = z_form

            Returns a list of lists
        """
        distances = [[tree.prop('host.distance.total', sub_inds[i][j]) for j in range(0, len(sub_inds[i]))] for i in range(0, len(sub_inds))]
        return distances

    def halo_distances_norm(self, distances, host_halo_radii):
        """
            Reads in distances (for each subhalo) and the host radii (at all
            snapshots that it exists)

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

            Returns a list of arrays
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

    def first_infall_times(self, distances_norm, time_array):
        """
        Reads in normalized subhalo distances and snapshot information and returns
        the snapshots and times when the subhalos first fell into the host

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

        Returns dictionary
        """
        # Set up a dictionary to store the information you want
        d = dict();
        first_infall_snap = []
        # Loop over normalized distances
        for i in range(0, len(distances_norm)):
            # Check to see if the subhalo is within the virial radius of the host
            if len(np.where(distances_norm[i] < 1)[0]) != 0:
                first_infall_snap.append(np.max(np.where(distances_norm[i] < 1)[0]))
            else:
                first_infall_snap.append(-1)
        # Save the snapshot that this happens at (but these snapshots are counted from 600 backward,
        # i.e., if d['snapshot'] = 590, this is really snapshot 10)
        d['snapshot'] = np.asarray(first_infall_snap)
        infall_mask = (d['snapshot'] > 0)
        d['time'] = np.flip(time_array['time'])[d['snapshot'][infall_mask]]
        d['check'] = infall_mask
        return d

    def pericenter_interp(self, distances, distances_norm, virial_radii, time_array):
        """
        Reads in subhalo distances, normalized subhalo distances, host virial radii across time,
        and snapshot information and returns a dictionary of pericenter distances, times, and a
        boolean array.

        distances: list of lists (given in kpc physical)
        distances_norm: list of arrays (given in kpc physical)
        virial radii: array (given in kpc physical)
        time_array: dictionary

        NOTES:
            - Returns a dictionary
                - d['pericenter.check'] is a list of booleans
                  Tells you if the subhalo has experienced a pericenter
                - d['pericenter'] is a list of lists
                  Tells you the pericenter distances (in kpc physical)
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
        pericenters_raw = []
        time_raw = []
        host_peri_rad = []
        check = []
        peri_spl = []
        time_spl = []
        # Loop over the number of subhalos
        for k in range(0, len(distances)):
            temp_halo_1 = distances[k] # Now goes from z = 0 to z_form (un-normalized)
            temp_halo_2 = distances_norm[k] # (normalized)
            peri_list = []
            time_list = []
            peri_rad_list = []
            # Want initial element to be this because we check +- 4 neighbors on each side
            temp_peri = temp_halo_1[4]
            temp_check = np.zeros(len(temp_halo_1))
            temp_peri_spl = []
            temp_time_spl = []
            # Loop through each subhalo
            for i in range(4, len(temp_halo_2)-4):
                # Check its neighbors and if it is within virial radius
                if (temp_peri < temp_halo_1[i+1]) and (temp_peri < temp_halo_1[i+2]) and (temp_peri < temp_halo_1[i+3])and (temp_peri < temp_halo_1[i+4]) and (temp_peri < temp_halo_1[i-1]) and (temp_peri < temp_halo_1[i-2]) and (temp_peri < temp_halo_1[i-3])and (temp_peri < temp_halo_1[i-4]) and (temp_peri/virial_radii[i] < 1):
                    temp_check[i] = 1
                    peri_list.append(temp_halo_1[i])
                    time_list.append(time_array['time'][600-i])
                    peri_rad_list.append(virial_radii[i])
                    temp_peri_spl.append(temp_halo_1[i-4:i+4])
                    temp_time_spl.append(time_array['time'][600-i-4:600-i+4])
                    temp_peri = temp_halo_1[i+1]
                else:
                    temp_peri = temp_halo_1[i+1]
            pericenters_raw.append(peri_list)
            time_raw.append(time_list)
            host_peri_rad.append(peri_rad_list)
            check.append(temp_check)
            peri_spl.append(temp_peri_spl)
            time_spl.append(temp_time_spl)
        # Create a mask that tells you whether or not halo experienced pericenter
        peri_bool = []
        for i in range(0, len(check)):
            if (np.sum(check[i]) > 0):
                peri_bool.append(True)
            else:
                peri_bool.append(False)
        d['pericenter.check'] = peri_bool
        d['pericenter.host.r200'] = host_peri_rad
        # Do the spline fitting
        pericenter_spline = []
        time_spline = []
        for i in range(0, len(peri_spl)):
            if (len(peri_spl[i]) != 0):
                temp_peri_new_spl = []
                temp_time_new_spl = []
                for j in range(0, len(peri_spl[i])):
                    temp_dist = peri_spl[i][j]
                    temp_time = time_spl[i][j]
                    f = interp1d(temp_time, temp_dist, kind='cubic')
                    x_new = np.linspace(temp_time[0], temp_time[-1], 100)
                    temp_peri_new_spl.append(np.min(f(x_new)))
                    temp_time_new_spl.append(x_new[np.where(f(x_new) == np.min(f(x_new)))[0][0]])
                pericenter_spline.append(temp_peri_new_spl)
                time_spline.append(temp_time_new_spl)
            else:
                temp_peri_new_spl = []
                temp_time_new_spl = []
                pericenter_spline.append(temp_peri_new_spl)
                time_spline.append(temp_time_new_spl)
        d['pericenter.dist'] = pericenter_spline
        d['pericenter.time'] = time_spline
        return d

    def angular_momentum(self, tree, sub_inds):
        """
        Reads in the tree and subhalo indices and returns a dictionary that contains
        the angular momentum vectors and their magnitudes.

        tree: dictionary
        sub_inds: list of arrays

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
            - tree is organized as (r, z, phi)
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

    def angular_momentum_plot(self, ell, subhalo_num, infall, infall_array, time_array, comp, file_name):
        """
        Plot any component of angular momentumn for a subhalo

        ell: ?
        subhalo_num: integer
                     The halo in particular you want to plot (starts from 0)
        time_array: dictionary
        comp: string
              This is the component of angular momentum that you want to plot
        file_name: string
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
        if infall == True:
            infall_time = np.flip(time_array['time'])[infall_array['snapshot'][subhalo_num]]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        plt.xlabel('time [Gyr]', fontsize=28)
        plt.ylabel('L'+comp_str+' [km s$^{-1}$ kpc]', fontsize=28)
        plt.title('Halo '+str(subhalo_num+1), fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home1/05400/ibsantis/scripts/orbit_plots/'+file_name+'.pdf')
        plt.close()

    def velocity_plot(self, tree, sub_inds, subhalo_num, comp, infall, infall_array, time_array, file_name):
        """
        Plot any component of velocity for a subhalo
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
        plt.ylim(np.nanmin(vs), np.nanmax(vs))
        if infall == True:
            infall_time = np.flip(time_array['time'])[infall_array['snapshot'][subhalo_num]]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        plt.xlabel('time [Gyr]', fontsize=28)
        plt.ylabel('v'+comp_str+' [km s$^{-1}$]', fontsize=28)
        plt.title('Halo '+str(subhalo_num+1), fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home1/05400/ibsantis/scripts/orbit_plots/'+file_name+'.pdf')
        plt.close()

    def distance_plot(self, tree, sub_inds, subhalo_num, comp, infall, infall_array, time_array, file_name):
        """
        Plot any component of velocity for a subhalo
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
        plt.ylim(np.nanmin(ds), np.nanmax(ds))
        if infall == True:
            infall_time = np.flip(time_array['time'])[infall_array['snapshot'][subhalo_num]]
            plt.vlines(infall_time,-1000000,1000000,color='k',linestyles='dotted')
        plt.xlabel('time [Gyr]', fontsize=28)
        plt.ylabel('d'+comp_str+' [kpc]', fontsize=28)
        plt.title('Halo '+str(subhalo_num+1), fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig('/home1/05400/ibsantis/scripts/orbit_plots/'+file_name+'.pdf')
        plt.close()

    def orbit_energy(self, tree, potential, sub_inds):
        """
        Reads in the tree, a subhalo's index and progenitor indices, and an array
        of subhalo gravitational potentials and calculates the total orbital energy
        for a subhalo and it's progenitor subhalos (i.e., the energy across time).

        Energy is defined as E = (1/2)*velocity**2 + potential

        tree : dictionary
        sub_inds : list of arrays
        potential : array

        NOTES:
            - Returns an array the size of an element of sub_inds
                - To be more explicit, returns values for the subhalo of interest
                  across the entire time range that it existed
            - Only handles ONE subhalo at a time.
        """
        energy = 0.5*tree.prop('host.velocity.total')[sub_inds]**2 + potential['halo.potentials'][sub_inds]
        return energy
