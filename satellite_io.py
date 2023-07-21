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
            'Romulus': ('Remus', 'm12_elvis_RomulusRemus', '_r3000', 2),
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
        self.fitting_data = pd.read_csv(self.home_dir+'/orbit_data/fitting_param.csv', index_col=0)
        #
        if dmo:
            self.simulation_dir += '_dm'
        
    def satellite_indices(self, tree, gal1, location, host=1, minimum_mass=1e8):
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
        # Uncomment out these lines if I need to be multiplying the subhalos by the baryon fraction
        # self.baryon_frac = tree.Cosmology['omega_baryon']/tree.Cosmology['omega_matter']
        # z0_inds = z0_inds[ut.array.get_indices(tree.prop('mass.peak',z0_inds)*(1-self.baryon_frac), [minimum_mass, np.inf])]
        #
        z0_inds = z0_inds[ut.array.get_indices(tree.prop('mass.peak',z0_inds), [minimum_mass, np.inf])]
        z0_inds_w_prog = tree.prop('progenitor.main.indices', z0_inds)
        self.sub_inds = z0_inds_w_prog
        #
        self.shape = self.sub_inds.shape

# Have a class that matches the satellites
class SatelliteMatch:

    def __init__(self):
        """
        TBD

        Just constants for now...
        """
        self.smhm_constant = -15.21177826 # From fitting the SMHMR from paper I, in mstar_mhalo_fitting.py
        self.smhm_slope = 2.2111824


    def satellite_select(self, mstar):
        """
        TBD

        Want to add more criteria, like a minimum mass, particle number, etc
        """
        mhalo = self.smhm_constant + mstar*self.smhm_slope
        pass