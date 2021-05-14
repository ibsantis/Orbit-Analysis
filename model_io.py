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
import sys


#
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
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        @custom_model
        def exponential_vert_mass(z, amp1=Amp, z1=hz):
            mass = amp1*(2*z1)*(1-np.exp(-z/z1))
            return mass
        #
        # Fit the model to the data for various cutoff radii
        model_init = exponential_vert_mass(bounds={'amp1':Amp_bounds, 'z1':hz_bounds})
        fit = LevMarLSQFitter()
        model_disk_vert = fit(model_init, distances[1:], np.cumsum(masses), maxiter=iters)
        print(model_disk_vert)
        #
        return model_disk_vert

    def disk_rad_mass_model(self, distances, masses, A_in, r_in, A_out, r_out, hz, A_in_bounds, r_in_bounds, A_out_bounds, r_out_bounds, iters=100000):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        @custom_model
        def double_exponential_mass(r, amp1=A_in, r1=r_in, amp2=A_out, r2=r_out):
            mass_in = (4*np.pi*amp1*hz*r1)*(r1-np.exp(-r/r1)*(r1+r))
            mass_out = (4*np.pi*amp2*hz*r2)*(r2-np.exp(-r/r2)*(r2+r))
            return mass_in+mass_out
        #
        model_init = double_exponential_mass(bounds={'amp1':A_in_bounds, 'r1':r_in_bounds, 'amp2':A_out_bounds, 'r2':r_out_bounds})
        fit = LevMarLSQFitter()
        model_disk_rad = fit(model_init, distances[1:], np.cumsum(masses), maxiter=iters)
        print(model_disk_rad)
        #
        return model_disk_rad

    def halo_nfw_mass_model(self):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        pass

    def halo_2p_mass_model(self):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        pass


class DensityModelFit:

    def __init__(self):
        """
        Don't really have anything to put in here yet...
        Leaving blank for now.
        """
        pass

    def disk_rad_dens_model(self, distances, densities, A_in, r_in, A_out, r_out, A_in_bounds, r_in_bounds, A_out_bounds, r_out_bounds, r_cut=None, iters=100000):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        ############### TEST THISSSSS
        if r_cut == None:
            @custom_model
            def double_exponential_density(r, amp1=A_in, r1=r_in, amp2=A_out, r2=r_out):
                return amp1*np.exp(-r/r1) + amp2*np.exp(-r/r2)
            #
            # Fit the model to the data for various cutoff radii
            model_init = double_exponential_density(bounds={'amp1':A_in_bounds, 'r1':r_in_bounds, 'amp2':A_out_bounds, 'r2':r_out_bounds})
            fit = LevMarLSQFitter()
            model_disk_rad = fit(model_init, distances[1:], densities, maxiter=iters)
            print(model_disk_rad)
            #
            return model_disk_rad
        else:
            @custom_model
            def double_exponential_density(r, amp=A_in, r_s=r_in):
                return amp*np.exp(-r/r_s)
            #
            cut = np.where(distances > r_cut)[0][0]
            # Fit the model to the data for various cutoff radii
            model_init_1 = double_exponential_density(bounds={'amp':A_in_bounds, 'r_s':r_in_bounds})
            model_init_2 = double_exponential_density(bounds={'amp':A_out_bounds, 'r_s':r_out_bounds})
            fit = LevMarLSQFitter()
            model_disk_rad_1 = fit(model_init_1, distances[1:cut], densities[:cut-1], maxiter=iters)
            model_disk_rad_2 = fit(model_init_2, distances[cut:], densities[cut-1:], maxiter=iters)
            model_disk_rad = model_disk_rad_1+model_disk_rad_2
            print(model_disk_rad)
            #
            return model_disk_rad

    def disk_vert_dens_model(self, distances, densities, Amp, hz, Amp_bounds, hz_bounds, iters=100000):
        """
        DESCRIPTION:
            Blah blah blah

        VARIABLES:
            HMMM

        NOTES:
            Yes.
        """
        @custom_model
        def exponential_vert_density(z, amp1=Amp, z1=hz):
            return amp1*np.exp(-np.abs(z)/z1)
        #
        # Fit the model to the data for various cutoff radii
        model_init = exponential_vert_density(bounds={'amp1':Amp_bounds, 'z1':hz_bounds})
        fit = LevMarLSQFitter()
        model_disk_vert = fit(model_init, distances[1:], densities, maxiter=iters)
        print(model_disk_vert)
        #
        return model_disk_vert


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
