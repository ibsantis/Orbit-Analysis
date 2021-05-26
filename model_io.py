#!/usr/bin/python3

"""
Intended for use with astropy's modeling packages

@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

    This was written to help compute parameters for different density / mass
    models for:

        - Double exponential disk
            - This accounts for an "inner" and "outer" disk region

        - NFW halo

        - Generalized NFW halo with an inner and outer slope
"""

import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from scipy import special
import pandas as pd
import sys


class MassModelFit:

    def __init__(self):
        """
        Don't really know what to put in here...
        Leaving it empty for now.
        """
        pass

    def disk_vert_mass_model(self, distances, masses, Amp, hz, Amp_bounds, hz_bounds, iters=100000):
        """
        DESCRIPTION:
            Model a single exponential profile for the vertical component of the
            disk. Got this profile by integrating the corresponding density
            profile.

        VARIABLES:
            distances  : 1D array
            masses     : 1D array
            Amp        : float
            hz         : float
            Amp_bounds : tuple
            hz_bounds  : tuple
            iters      : int

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to provide good estimates for the amplitude and scale height
            - Also, need to provide reasonable bounds for these parameters
            - Returns a mass model of the vertical component of the disk
                - Returns an amplitude and scale height
        """
        # Define the mass profile
        @custom_model
        def exponential_vert_mass(z, amp1=Amp, z1=hz):
            mass = amp1*(2*z1)*(1-np.exp(-z/z1))
            return mass
        #
        # Initialize the model
        model_init = exponential_vert_mass(bounds={'amp1':Amp_bounds, 'z1':hz_bounds})
        fit = LevMarLSQFitter()
        #
        # Fit the model to the data and print it out
        model_disk_vert = fit(model_init, distances[1:], np.cumsum(masses), maxiter=iters)
        print(model_disk_vert)
        #
        return model_disk_vert

    def disk_rad_mass_model(self, distances, masses, A_in, r_in, A_out, r_out, hz, A_in_bounds, r_in_bounds, A_out_bounds, r_out_bounds, iters=100000):
        """
        DESCRIPTION:
            Model a double exponential profile for the radial component of the
            disk. To get this, you need to integrate the double exponential
            density profile to some enclosed radius, R.

        VARIABLES:
            distances    : 1D array
            masses       : 1D array
            A_in         : float
            r_in         : float or int
            A_out        : float
            r_out        : float or int
            h_z          : float or int
            A_in_bounds  : tuple
            r_in_bounds  : tuple
            A_out_bounds : tuple
            r_out_bounds : tuple
            iters        : int

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to already have a good estimate for h_z (disk scale height)
            - Need to provide good estimates for the amplitudes and scale radii
            - Also, need to provide reasonable bounds for these parameters
            - Returns the enclosed mass model for the inner and outer radial
              components of the disk.
                - Returns amplitudes and scale radii for both components.
        """
        # Define the mass profile
        @custom_model
        def double_exponential_mass(r, amp1=A_in, r1=r_in, amp2=A_out, r2=r_out):
            mass_in = (4*np.pi*amp1*hz*r1)*(r1-np.exp(-r/r1)*(r1+r))
            mass_out = (4*np.pi*amp2*hz*r2)*(r2-np.exp(-r/r2)*(r2+r))
            return mass_in+mass_out
        #
        # Initialize the model
        model_init = double_exponential_mass(bounds={'amp1':A_in_bounds, 'r1':r_in_bounds, 'amp2':A_out_bounds, 'r2':r_out_bounds})
        fit = LevMarLSQFitter()
        #
        # Fit the model to the data and print it out
        model_disk_rad = fit(model_init, distances[1:], np.cumsum(masses), maxiter=iters)
        print(model_disk_rad)
        #
        return model_disk_rad

    def halo_nfw_mass_model(self, distances, masses, A_halo, a_halo, A_halo_bounds, a_halo_bounds, r_min=10, r_max=None, iters=100000):
        """
        DESCRIPTION:
            Model the halo of the galaxy with a regular NFW profile.

        VARIABLES:
            distances     : 1D array
            masses        : 1D array
            A_halo        : float or int
            a_halo        : float or int
            A_halo_bounds : tuple
            a_halo_bounds : tuple
            r_min         : float or int
            r_max         : float or int or None
            iters         : int

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to provide good estimates for the parameters and their bounds
            - Returns a model of the halo component of the galaxy
                - Returns an amplitude and scale radius
        """
        # Find the minimum index of the distance array depending on r_min; default to 10 kpc
        r_cut_min = np.where(distances > r_min)[0][0]
        #
        # If no maximum distance to fit to, fit over entire distance array
        if r_max == None:
            #
            # Define the model
            @custom_model
            def nfw_mass_model(r, amp=A_halo, a=a_halo):
                return amp*(np.log((a+r)/a)+a/(a+r)-1)
            #
            # Initialize the model
            model_init = nfw_mass_model(bounds={'amp':A_halo_bounds, 'a':a_halo_bounds})
            fit = LevMarLSQFitter()
            #
            # Fit the model to the data and print it out
            model_halo = fit(model_init, distances[r_cut_min:], np.cumsum(masses)[r_cut_min-1:], maxiter=iters)
            print(model_halo)
            #
            return model_halo
        else:
            # If there is a maximum distance to fit to, find the index in the distances array
            r_cut_max = np.where(distances > r_max)[0][0]
            #
            # Define the model
            @custom_model
            def nfw_mass_model(r, amp=A_halo, a=a_halo):
                return amp*(np.log((a+r)/a)+a/(a+r)-1)
            #
            # Initialize the model
            model_init = nfw_mass_model(bounds={'amp':A_halo_bounds, 'a':a_halo_bounds})
            fit = LevMarLSQFitter()
            #
            # Fit the model to the data and print it out
            model_halo = fit(model_init, distances[r_cut_min:r_cut_max], np.cumsum(masses)[r_cut_min-1:r_cut_max-1], maxiter=iters)
            print(model_halo)
            #
            return model_halo

    def halo_2p_mass_model(self, distances, masses, A_halo, a_halo, slope_in, slope_out, A_halo_bounds, a_halo_bounds, slope_in_bounds, slope_out_bounds, r_min=10, r_max=None, iters=100000):
        """
        DESCRIPTION:
            Model the halo of the galaxy with a generalized NFW profile.

        VARIABLES:
            distances        : 1D array
            masses           : 1D array
            A_halo           : float or int
            a_halo           : float or int
            slope_in         : float or int
            slope_out        : float or int
            A_halo_bounds    : tuple
            a_halo_bounds    : tuple
            slope_in_bounds  : tuple
            slope_out_bounds : tuple
            r_min            : float or int
            r_max            : float or int or None
            iters            : int

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to provide good estimates for the parameters and their bounds
            - Returns a model of the halo component of the galaxy
                - Returns an amplitude, scale radius, and an inner and outer slope
        """
        # Find the minimum index of the distance array depending on r_min; default to 10 kpc
        r_cut_min = np.where(distances > r_min)[0][0]
        #
        # If there is no maximum radius to fit to, fit over entire distance array
        if r_max == None:
            #
            # Define the model
            @custom_model
            def two_power_beta_fixed(r, amp=A_halo, a=a_halo, alpha=slope_in, beta=slope_out):
                return (amp/(3-alpha))*((r/a)**(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a)
            #
            # Initialize the model
            model_init = two_power_beta_fixed(bounds={'amp':A_halo_bounds, 'a':a_halo_bounds, 'alpha':slope_in_bounds, 'beta':slope_out_bounds})
            fit = LevMarLSQFitter()
            #
            # Fit the model to the data and print it out
            model_halo = fit(model_init, distances[r_cut_min:], np.cumsum(masses)[r_cut_min-1:], maxiter=iters)
            print(model_halo)
            #
            return model_halo
        else:
            # If there is a maximum radius to fit to, find the corresponding index in the distance array
            r_cut_max = np.where(distances > r_max)[0][0]
            #
            # Define the model
            @custom_model
            def two_power_beta_fixed(r, amp=A_halo, a=a_halo, alpha=slope_in, beta=slope_out):
                return (amp/(3-alpha))*((r/a)**(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-r/a)
            #
            # Initialize the model
            model_init = two_power_beta_fixed(bounds={'amp':A_halo_bounds, 'a':a_halo_bounds, 'alpha':slope_in_bounds, 'beta':slope_out_bounds})
            fit = LevMarLSQFitter()
            #
            # Fit the model to the data and print it out
            model_halo = fit(model_init, distances[r_cut_min:r_cut_max], np.cumsum(masses)[r_cut_min-1:r_cut_max-1], maxiter=iters)
            print(model_halo)
            #
            return model_halo


class DensityModelFit:

    def __init__(self):
        """
        Don't really have anything to put in here yet...
        Leaving blank for now.
        """
        pass

    def disk_vert_dens_model(self, distances, densities, Amp, hz, Amp_bounds, hz_bounds, iters=100000):
        """
        DESCRIPTION:
            Model a single exponential profile for the vertical component of the
            disk.

        VARIABLES:
            distances  : 1D array
            densities  : 1D array
            Amp        : float
            hz         : float
            Amp_bounds : tuple
            hz_bounds  : tuple
            iters      : int

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to provide good estimates for the amplitude and scale height
            - Also, need to provide reasonable bounds for these parameters
            - Returns a density model of the vertical component of the disk
                - Returns an amplitude and scale height
        """
        # Define the model
        @custom_model
        def exponential_vert_density(z, amp1=Amp, z1=hz):
            return amp1*np.exp(-np.abs(z)/z1)
        #
        # Initialize the model
        model_init = exponential_vert_density(bounds={'amp1':Amp_bounds, 'z1':hz_bounds})
        fit = LevMarLSQFitter()
        #
        # Fit the model to the data and print it out
        model_disk_vert = fit(model_init, distances[1:], densities, maxiter=iters)
        print(model_disk_vert)
        #
        return model_disk_vert

    def disk_rad_dens_model(self, distances, densities, A_in, r_in, A_out, r_out, A_in_bounds, r_in_bounds, A_out_bounds, r_out_bounds, r_cut=None, iters=100000):
        """
        DESCRIPTION:
            Model a double exponential profile for the radial component of the
            disk.

        VARIABLES:
            distances    : 1D array
            densiti      : 1D array
            A_in         : float
            r_in         : float or int
            A_out        : float
            r_out        : float or int
            A_in_bounds  : tuple
            r_in_bounds  : tuple
            A_out_bounds : tuple
            r_out_bounds : tuple
            r_cut        : float or int or None
            iters        : int

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to provide good estimates for the amplitudes and scale radii
            - Also, need to provide reasonable bounds for these parameters
            - Returns the enclosed mass model for the inner and outer radial
              components of the disk.
                - Returns amplitudes and scale radii for both components.
            - Can model the disk with two exponentials over the entire distance
              array, or as two single exponentials over specific distances.
        """
        # If there is no cut along the distance array, model disk as the sum of two exponentials
        if r_cut == None:
            # Define the model
            @custom_model
            def double_exponential_density(r, amp1=A_in, r1=r_in, amp2=A_out, r2=r_out):
                return amp1*np.exp(-r/r1) + amp2*np.exp(-r/r2)
            #
            # Initialize the model
            model_init = double_exponential_density(bounds={'amp1':A_in_bounds, 'r1':r_in_bounds, 'amp2':A_out_bounds, 'r2':r_out_bounds})
            fit = LevMarLSQFitter()
            #
            # Fit the model to the data and print it out
            model_disk_rad = fit(model_init, distances[1:], densities, maxiter=iters)
            print(model_disk_rad)
            #
            return model_disk_rad
        else:
            # Define the model as a single exponential
            @custom_model
            def double_exponential_density(r, amp=A_in, r_s=r_in):
                return amp*np.exp(-r/r_s)
            #
            # If there is an r_cut, find the corresponding index in the distance array
            cut = np.where(distances > r_cut)[0][0]
            #
            # Initialize a model for the inner and outer region seperately
            model_init_1 = double_exponential_density(bounds={'amp':A_in_bounds, 'r_s':r_in_bounds})
            model_init_2 = double_exponential_density(bounds={'amp':A_out_bounds, 'r_s':r_out_bounds})
            fit = LevMarLSQFitter()
            #
            # Fit the models to the data, combine them, and print them out
            model_disk_rad_1 = fit(model_init_1, distances[1:cut], densities[:cut-1], maxiter=iters)
            model_disk_rad_2 = fit(model_init_2, distances[cut:], densities[cut-1:], maxiter=iters)
            model_disk_rad = model_disk_rad_1+model_disk_rad_2
            print(model_disk_rad)
            #
            return model_disk_rad

    def halo_nfw_dens_model(self):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        pass

    def halo_2p_dens_model(self):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        pass


class Profiles:

    def __init__(self, directory):
        """
        Read in the fitting parameters and save them as attributes
        """
        self.fitting_data = pd.read_csv(directory+'/orbit_data/fitting_param.csv', index_col=0)
        pass

    def disk_radial_density(self, distances, gal):
        """
        DESCRIPTION:
            Model the disk as the sum of two exponentials, one for the inner
            region, and one for the outer. Here, the vertical component is
            integrated out already, resulting in a factor of h_z (the scale
            height) out front.

        VARIABLES:
            distances : 1D array
            gal       : str

        NOTES:
            - Returns the density at all points in the 'distances' array.
            - Uses values for the amplitudes, scale lengths, and scale height
              from data that was already compiled.
        """
        # Set all of the parameters
        A_disk_in = self.fitting_data['A_disk_in'][gal]
        r_in = self.fitting_data['r_in'][gal]
        A_disk_out = self.fitting_data['A_disk_out'][gal]
        r_out = self.fitting_data['r_out'][gal]
        h_z = self.fitting_data['h_z'][gal]
        #
        # Define the inner and outer density
        density_inner = A_disk_in*h_z*np.exp(-distances/r_in)
        density_outer = A_disk_out*h_z*np.exp(-distances/r_out)
        #
        return density_inner + density_outer

    def disk_radial_mass(self, distances, gal):
        """
        DESCRIPTION:
            Model the disk as the sum of two exponentials, one for the inner
            region, and one for the outer. Here, the vertical component is
            integrated out already, resulting in a factor of 2*h_z (the scale
            height) out front.

        VARIABLES:
            distances : 1D array
            gal       : str

        NOTES:
            - Returns the enclosed mass at all points in the 'distances' array.
            - Uses values for the amplitudes, scale lengths, and scale height
              from data that was already compiled.
        """
        # Set all of the parameters
        A_disk_in = self.fitting_data['A_disk_in'][gal]
        r_in = self.fitting_data['r_in'][gal]
        A_disk_out = self.fitting_data['A_disk_out'][gal]
        r_out = self.fitting_data['r_out'][gal]
        h_z = self.fitting_data['h_z'][gal]
        #
        # Define the inner and outer enclosed mass profiles
        mass_inner = (4*np.pi*A_disk_in*h_z*r_in)*(r_in-np.exp(-distances/r_in)*(r_in+distances))
        mass_outer = (4*np.pi*A_disk_out*h_z*r_out)*(r_out-np.exp(-distances/r_out)*(r_out+distances))
        #
        return mass_inner + mass_outer

    def halo_nfw_density(self):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        pass

    def halo_nfw_mass(self, distances, gal):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        A_halo = self.fitting_data['A_halo'][gal]
        a_halo = self.fitting_data['a_halo'][gal]
        #
        return A_halo*(np.log((a_halo+distances)/a_halo)+a_halo/(a_halo+distances)-1)


    def halo_2p_nfw_density(self):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        pass

    def halo_2p_nfw_mass(self, distances, gal):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        A_halo = self.fitting_data['A_halo'][gal]
        a_halo = self.fitting_data['a_halo'][gal]
        alpha = self.fitting_data['alpha'][gal]
        beta = self.fitting_data['beta'][gal]
        #
        return (A_halo/(3-alpha))*((distances/a_halo)**(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-distances/a_halo)
