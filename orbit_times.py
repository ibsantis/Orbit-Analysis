#!/usr/bin/python3

"""
This is going to contain all of the functions that I make for my orbital
analysis pipeline.

[Give some important notes here]  
"""


def get_luminous_halos(tree):
    """
        Reads in the halo tree, and returns the indices of luminous subhalos
        along with their progenitor indices.

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

def halo_distances(tree, sub_inds):
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

def halo_distances_norm(distances, host_halo_radii):
    """
        Reads in distances (for each subhalo) and the host radii (at all
        snapshots that it exists)

        distances: list of lists
        host_halo_radii: array

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
        if len(host_halo_radii) > len(distances[i]):
            distances_norm.append(distances[i]/host_halo_radii[:len(distances[i])])
        else:
            distances_norm.append(distances[i][:len(host_halo_radii)]/host_halo_radii)
    return distances_norm

def first_infall_times(distances_norm, time_array):
    """
    Reads in normalized subhalo distancs and snapshot information and returns
    the snapshots and times when the subhalos first fell into the host

    distances_norm: list of arrays
    time_array: dictionary

    NOTES:
        - Returns a dictionary
            - d['snapshot'] is an array of length equal to the number of
              subhalos
            - d['time'] is an array of length equal to the number of subhalos
              that have fallen into the host
        - The times given here correspond to the age of the Universe (Gyr)
    """
    d = dict();
    first_infall_snap = []
    for i in range(0, len(distances_norm)):
        if len(np.where(distances_norm[i] < 1)[0]) != 0:
            first_infall_snap.append(np.max(np.where(distances_norm[i] < 1)[0]))
        else:
            first_infall_snap.append(-1)
    d['snapshot'] = np.asarray(first_infall_snap)
    infall_mask = (d['snapshot'] > 0)
    d['time'] = np.flip(time_array['time'])[d['snapshot'][infall_mask]]
    return d
