#!/usr/bin/python3

"""

    Intended for use with astropy's modeling packages and the FIRE-2 simulations.

    @author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

        This was written to help compute parameters for different density / mass
        models for:

            - Double exponential disk
                - This accounts for an "inner" and "outer" disk region

            - NFW halo

            - Generalized NFW halo with an inner and outer slope

"""


# Import the necessary tools
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from scipy import special
import pandas as pd


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
                         Units not important, but need to be the same as hz

            masses     : 1D array
                         Units generally Msun

            Amp        : float
                         Amplitude of the mass profile with units of M_sun (or
                         at least whatever units 'masses' array is in)

            hz         : float
                         Disk scale height.
                         Units usually in kpc

            Amp_bounds : tuple
                         Initial guess for the amplitude parameter.
                         Include a lower and upper bound guess in a tuple.

            hz_bounds  : tuple
                         Initial guess for the scale height parameter.
                         Include a lower and upper bound guess in a tuple.

            iters      : int
                         Maximum number of iterations astropy will do to converge
                         on a best-fit guess.

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
                           Units not important, but need to be the same as hz,
                           r_in, and r_out.

            masses       : 1D array
                           Units generally in Msun.

            A_in         : float
                           Amplitude of the inner stellar disk mass profile.
                           Units in Msun (or whatever units the 'masses' array
                           is in).

            r_in         : float or int
                           Inner disk scale length.
                           Generally in units of kpc.

            A_out        : float
                           Amplitude of the inner stellar disk mass profile.
                           Units in Msun (or whatever units the 'masses' array
                           is in).

            r_out        : float or int
                           Inner disk scale length.
                           Generally in units of kpc.

            h_z          : float or int
                           Disk scale height.

            A_in_bounds  : tuple
                           Initial guess for the inner disk amplitude parameter.
                           Include a lower and upper bound guess in a tuple.

            r_in_bounds  : tuple
                           Initial guess for the inner scale length.
                           Include a lower and upper bound guess in a tuple.

            A_out_bounds : tuple
                           Initial guess for the outer disk amplitude parameter.
                           Include a lower and upper bound guess in a tuple.

            r_out_bounds : tuple
                           Initial guess for the outer scale length.
                           Include a lower and upper bound guess in a tuple.

            iters        : int
                           Maximum number of iterations astropy will do to converge
                           on a best-fit guess.

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to already have a good estimate for h_z (disk scale height)
            - Need to provide good estimates for the amplitudes and scale radii
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
            Model the dark matter halo of the galaxy with a regular NFW profile.

        VARIABLES:
            distances     : 1D array
                            Units not important, but need to be the same as a_halo,
                            r_min, and r_max.

            masses        : 1D array
                            Units usually in Msun

            A_halo        : float or int
                            Amplitude of the halo mass profile.
                            Units usually in Msun (or whatever units the 'masses'
                            array is in).

            a_halo        : float or int
                            Halo scale length.
                            Units usually in kpc.

            A_halo_bounds : tuple
                            Initial guess for the amplitude parameter.
                            Include a lower and upper bound guess in a tuple.

            a_halo_bounds : tuple
                            Initial guess for the scale length parameter.
                            Include a lower and upper bound guess in a tuple.

            r_min         : float or int
                            Minimum radius to fit the data to.
                            Need to be in the same units as 'distances' array.

            r_max         : float or int or None
                            Maximum radius to fit the data to.
                            Need to be in the same units as 'distances' array.

            iters         : int
                            Maximum number of iterations astropy will do to converge
                            on a best-fit guess.

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to provide good estimates for the parameters and their bounds
            - Returns a model of the halo component of the galaxy
                - Returns an amplitude and scale radius
        """
        # Find the minimum index of the distance array depending on r_min
        r_cut_min = np.where(distances > r_min)[0][0]
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
        # If no maximum distance to fit to, fit over entire distance array
        if r_max == None:
            # Fit the model to the data
            model_halo = fit(model_init, distances[r_cut_min:], np.cumsum(masses)[r_cut_min-1:], maxiter=iters)
        #
        else:
            # If there is a maximum distance to fit to, find the index in the distances array
            r_cut_max = np.where(distances > r_max)[0][0]
            # Fit the model to the data
            model_halo = fit(model_init, distances[r_cut_min:r_cut_max], np.cumsum(masses)[r_cut_min-1:r_cut_max-1], maxiter=iters)
        # Print out and return the model
        print(model_halo)
        #
        return model_halo

    def halo_2p_mass_model(self, distances, masses, A_halo, a_halo, slope_in, slope_out, A_halo_bounds, a_halo_bounds, slope_in_bounds, slope_out_bounds, r_min=10, r_max=None, iters=100000):
        """
        DESCRIPTION:
            Model the halo of the galaxy with a generalized NFW profile.

        VARIABLES:
            distances        : 1D array
                               Units not important, but need to be the same as a_halo,
                               r_min, and r_max.

            masses           : 1D array
                               Units usually in Msun

            A_halo           : float or int
                               Amplitude of the halo mass profile.
                               Units usually in Msun (or whatever units the 'masses'
                               array is in).

            a_halo           : float or int
                               Halo scale length.
                               Units usually in kpc.

            slope_in         : float or int
                               Slope of the inner halo profile.

            slope_out        : float or int
                               Slope of the outer halo profile

            A_halo_bounds    : tuple
                               Initial guess for the amplitude parameter.
                               Include a lower and upper bound guess in a tuple.

            a_halo_bounds    : tuple
                               Initial guess for the scale length parameter.
                               Include a lower and upper bound guess in a tuple.

            slope_in_bounds  : tuple
                               Initial guess for the slope of them inner halo profile
                               paramter.
                               Include a lower and upper bound guess in a tuple.

            slope_out_bounds : tuple
                               Initial guess for the slope of them inner halo profile
                               paramter.
                               Include a lower and upper bound guess in a tuple.

            r_min            : float or int
                               Minimum radius to fit the data to.
                               Need to be in the same units as 'distances' array.

            r_max            : float or int or None
                               Maximum radius to fit the data to.
                               Need to be in the same units as 'distances' array.

            iters            : int
                               Maximum number of iterations astropy will do to converge
                               on a best-fit guess.

        NOTES:
            - Makes use of astropy's modeling functions
            - Need to provide good estimates for the parameters and their bounds
            - Returns a model of the halo component of the galaxy
                - Returns an amplitude, scale radius, and an inner and outer slope
        """
        # Find the minimum index of the distance array depending on r_min; default to 10 kpc
        r_cut_min = np.where(distances > r_min)[0][0]
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
        # If there is no maximum radius to fit to, fit over entire distance array
        if r_max == None:
            # Fit the model to the data and print it out
            model_halo = fit(model_init, distances[r_cut_min:], np.cumsum(masses)[r_cut_min-1:], maxiter=iters)
        #
        else:
            # If there is a maximum radius to fit to, find the corresponding index in the distance array
            r_cut_max = np.where(distances > r_max)[0][0]
            # Fit the model to the data and print it out
            model_halo = fit(model_init, distances[r_cut_min:r_cut_max], np.cumsum(masses)[r_cut_min-1:r_cut_max-1], maxiter=iters)
        #
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
                         Units not important, but need to be the same as hz

            densities  : 1D array
                         Units generally Msun / kpc^3

            Amp        : float
                         Amplitude of the mass profile with units of M_sun / kpc^3
                         (or at least whatever units 'densities' array is in)

            hz         : float
                         Disk scale height.
                         Units usually in kpc

            Amp_bounds : tuple
                         Initial guess for the amplitude parameter.
                         Include a lower and upper bound guess in a tuple.

            hz_bounds  : tuple
                         Initial guess for the scale height parameter.
                         Include a lower and upper bound guess in a tuple.

            iters      : int
                         Maximum number of iterations astropy will do to converge
                         on a best-fit guess.

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
                           Units not important, but need to be the same as hz,
                           r_in, and r_out.

            densities    : 1D array
                           Units generally in Msun / kpc^3.

            A_in         : float
                           Amplitude of the inner stellar disk mass profile.
                           Units in Msun / kpc^3 (or whatever units the 'densities'
                           array is in).

            r_in         : float or int
                           Inner disk scale length.
                           Generally in units of kpc.

            A_out        : float
                           Amplitude of the inner stellar disk mass profile.
                           Units in Msun / kpc^3 (or whatever units the 'densities'
                           array is in).

            r_out        : float or int
                           Inner disk scale length.
                           Generally in units of kpc.

            h_z          : float or int
                           Disk scale height.

            A_in_bounds  : tuple
                           Initial guess for the inner disk amplitude parameter.
                           Include a lower and upper bound guess in a tuple.

            r_in_bounds  : tuple
                           Initial guess for the inner scale length.
                           Include a lower and upper bound guess in a tuple.

            A_out_bounds : tuple
                           Initial guess for the outer disk amplitude parameter.
                           Include a lower and upper bound guess in a tuple.

            r_out_bounds : tuple
                           Initial guess for the outer scale length.
                           Include a lower and upper bound guess in a tuple.

            iters        : int
                           Maximum number of iterations astropy will do to converge
                           on a best-fit guess.

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
            TBD

        VARIABLES:
            TBD

        NOTES:
            There is currently no method for this in my pipeline, but I wanted
            to include a blank one for completeness-sake. Eventually...
        """
        pass

    def halo_2p_dens_model(self):
        """
        DESCRIPTION:
            TBD

        VARIABLES:
            TBD

        NOTES:
            There is currently no method for this in my pipeline, but I wanted
            to include a blank one for completeness-sake. Eventually...
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
                        Units need to be in kpc to be consistent with the
                        fitting data used.

            gal       : str
                        Name of the MW-mass galaxy you are interested in.

        NOTES:
            - Returns the density at all points in the 'distances' array.
            - Units in Msun / kpc^3
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

    def disk_radial_mass(self, distances, gal, custom_param=False):
        """
        DESCRIPTION:
            Model the disk as the sum of two exponentials, one for the inner
            region, and one for the outer. Here, the vertical component is
            integrated out already, resulting in a factor of 2*h_z (the scale
            height) out front.

        VARIABLES:
            distances : 1D array
                        Units need to be in kpc to be consistent with the
                        fitting data used.

            gal       : str
                        Name of the MW-mass galaxy you are interested in.

        NOTES:
            - Returns the enclosed mass at all points in the 'distances' array.
            - Units in Msun.
            - Uses values for the amplitudes, scale lengths, and scale height
              from data that was already compiled.
        """
        if custom_param:
            # Set all of the parameters
            A_disk_in = custom_param[0]
            r_in = custom_param[1]
            A_disk_out = custom_param[2]
            r_out = custom_param[3]
            h_z = custom_param[4]
        else:
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
            TBD

        VARIABLES:
            distances : 1D array
                        Units need to be in kpc to be consistent with the
                        fitting data used.

            gal       : str
                        Name of the MW-mass galaxy you are interested in.

        NOTES:
            There is currently no method for this in my pipeline, but I wanted
            to include a blank one for completeness-sake. Eventually...
        """
        pass

    def halo_nfw_mass(self, distances, gal):
        """
        DESCRIPTION:
            Model of the dark matter NFW halo mass profile.

        VARIABLES:
            distances : 1D array
                        Units need to be in kpc to be consistent with the
                        fitting data used.

            gal       : str
                        Name of the MW-mass galaxy you are interested in.

        NOTES:
            - Returns the enclosed mass at all points in the 'distances' array.
            - Units in Msun.
            - Uses values for the amplitudes, scale lengths, and scale height
              from data that was already compiled.
        """
        A_halo = self.fitting_data['A_halo'][gal]
        a_halo = self.fitting_data['a_halo'][gal]
        #
        return A_halo*(np.log((a_halo+distances)/a_halo)+a_halo/(a_halo+distances)-1)

    def halo_2p_nfw_density(self):
        """
        DESCRIPTION:
            TBD

        VARIABLES:
            distances : 1D array
                        Units need to be in kpc to be consistent with the
                        fitting data used.

            gal       : str
                        Name of the MW-mass galaxy you are interested in.

        NOTES:
            There is currently no method for this in my pipeline, but I wanted
            to include a blank one for completeness-sake. Eventually...
        """
        pass

    def halo_2p_nfw_mass(self, distances, gal, custom_param=False):
        """
        DESCRIPTION:
            Model of the generalized dark matter NFW halo mass profile.

        VARIABLES:
            distances : 1D array
                        Units need to be in kpc to be consistent with the
                        fitting data used.

            gal       : str
                        Name of the MW-mass galaxy you are interested in.

        NOTES:
            - Returns the enclosed mass at all points in the 'distances' array.
            - Units in Msun.
            - Uses values for the amplitudes, scale lengths, and scale height
              from data that was already compiled.
        """
        if custom_param:
            A_halo = custom_param[0]
            a_halo = custom_param[1]
            alpha = custom_param[2]
            beta = custom_param[3]
        else:
            A_halo = self.fitting_data['A_halo'][gal]
            a_halo = self.fitting_data['a_halo'][gal]
            alpha = self.fitting_data['alpha'][gal]
            beta = self.fitting_data['beta'][gal]
        #
        return (A_halo/(3-alpha))*((distances/a_halo)**(3-alpha))*special.hyp2f1(3.-alpha,-alpha+beta,4.-alpha,-distances/a_halo)
