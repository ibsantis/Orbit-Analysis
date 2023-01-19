#!/usr/bin/python3

"""

    Intended for use with the FIRE-2 simulations

    @author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

        This was written to create summary statistic plots with data output from
        orbit_io.py and model_io.py. Data that I import was previously compiled
        using summary_data.py and summary_data_dmo.py

"""

import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker
from scipy.interpolate import splrep, splev
import pandas as pd
import galpy


class SummaryDataSort:

    def __init__(self):
        """
        Initialize the sorting class.

        Want to save:
            - Host names to loop over in the following methods.
            - Oversampling factors.
                - baryon     : Calculated for luminous halos (Mstar > 1e4 Msun) in the
                               baryonic simulations (res 7100 Msun).
                - baryon_all : Calculated for all subhalos (Mhalo,peak > 1e8 Msun)
                               in the baryonic simulations.
                - dmo        : Calculated for all subhalos (Mhalo,peak > 1e8 Msun)
                               in the dark matter-only simulations.
                               NOTE: m12z and LG-pairs don't have factors for the DMO
                               runs because there are not DMO halo trees.
        """
        # Create a dictionary of host name arrays depending on if you want different samples
        self.host_names = {'all': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', 'm12z', \
                                   'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus'],
                           #
                           'all_no_z': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', \
                                        'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus'],
                           #
                           'all_no_r': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', 'm12z', \
                                        'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus'],
                           #
                           'all_energy': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', \
                                          'Romeo', 'Juliet', 'Thelma', 'Louise'],
                           #
                           'all_energy_new': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w', \
                                              'Romeo', 'Juliet', 'Thelma', 'Louise'],
                           #
                           'iso': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', 'm12z'], \
                           #
                           'iso_no_z': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w'], \
                           #
                           'iso_dmo': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w'], \
                           #
                           'lg': ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus'], \
                           #
                           'high-mass': ['Romulus', 'm12f', 'm12m', 'm12b', 'Thelma', 'Romeo'], \
                           #
                           'low-mass': ['m12c', 'm12i', 'Remus', 'Louise', 'm12w', 'Juliet', 'm12r'], \
                           #
                           'lg_no_RR': ['Romeo', 'Juliet', 'Thelma', 'Louise']}
        #
        # Oversampling factors
        self.oversample = {'baryon': {'m12b': 16, 'm12c': 13, 'm12f': 12, 'm12i': 20, 'm12m': 12,\
                                      'm12r': 19, 'm12w': 14, 'm12z': 21, 'Romeo': 16, 'Juliet': 14,\
                                      'Thelma': 16, 'Louise': 15, 'Romulus': 10, 'Remus': 15},\
                           'baryon_all': {'m12b': 17, 'm12c': 14, 'm12f': 14, 'm12i': 18, 'm12m': 14,\
                                          'm12r': 18, 'm12w': 19, 'm12z': 18, 'Romeo': 19, 'Juliet': 18,\
                                          'Thelma': 15, 'Louise': 15, 'Romulus': 10, 'Remus': 20},\
                           'dmo': {'m12b': 13, 'm12c': 12, 'm12f': 10, 'm12i': 14, 'm12m': 11,\
                                   'm12r': 16, 'm12w': 16, 'm12z': 0, 'Romeo': 0, 'Juliet': 0,\
                                   'Thelma': 0, 'Louise': 0, 'Romulus': 0, 'Remus': 0}}

    def data_read(self, directory, sim_type='baryon', hosts='all', point_mass=False):
        """
        DESCRIPTION:
            Reads in the summary data and stores it in a dictionary with each
            key being the host name.

        VARIABLES:
            directory : string
                        Home directory.

            sim_type  : string
                        Choose which subhalos you want to read in; same keys
                        as in the oversampling factors:
                        baryon     - luminous subhalos in baryonic simulations
                        baryon_all - luminous + dark subhalos in baryonic simulations
                        dmo        - dark subhalos in dark matter-only simulations

            hosts     : string
                        Choose any of the options listed in self.host_names within
                        the __init__ function above.

        NOTES:
            - The returned dictionary contains other dictionaries.
            - Each key in the dictionary is a host name
                - Each key in the sub-dictionaries is given in summary_data.py
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        data_dict = dict()
        #
        # Given the type of data you want, read in from the appropriate directory
        if sim_type == 'baryon':
            #
            if point_mass:
                for name in self.host_names[hosts]:
                    data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/summary_data/point_mass/data_'+name+'_point_mass', verbose=True)
                    data_dict[name] = data
            else:
                for name in self.host_names[hosts]:
                    data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/summary_data/data_'+name, verbose=True)
                    data_dict[name] = data
        #
        elif sim_type == 'all_baryon':
            for name in self.host_names[hosts]:
                data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/summary_data/data_'+name+'_dmo_selection', verbose=True)
                data_dict[name] = data
        #
        elif sim_type == 'dmo':
            for name in self.host_names[hosts]:
                data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/summary_data/data_'+name+'_dmo', verbose=True)
                data_dict[name] = data
        #
        return data_dict

    def data_read_potential(self, directory, sim_type='baryon', hosts='all'):
        """
        DESCRIPTION:
            Reads in the potential data and stores it in a dictionary with each
            key being the host name.

        VARIABLES:
            directory : string
                        Home directory.

            sim_type  : string
                        Choose which subhalos you want to read in; same keys
                        as in the oversampling factors:
                        baryon     - luminous subhalos in baryonic simulations
                        baryon_all - luminous + dark subhalos in baryonic simulations
                        dmo        - dark subhalos in dark matter-only simulations

            hosts     : string
                        Choose any of the options listed in self.host_names within
                        the __init__ function above.

        NOTES:
            - The returned dictionary contains other dictionaries.
            - Each key in the dictionary is a host name
                - Each key in the sub-dictionaries is given in potential_tree.py
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              potential_tree.py

            **THIS METHOD WAS USED PRIMARILY FOR THE PAPER I DATA**
        """
        data_dict = dict()
        #
        # Given the type of data, read in from the appropriate directory
        if sim_type == 'baryon':
            for name in self.host_names[hosts]:
                data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/potentials/'+name+'_potentials', verbose=True)
                data_dict[name] = data
        #
        elif sim_type == 'all_baryon':
            for name in self.host_names[hosts]:
                data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/potentials/'+name+'_potentials_all_subs', verbose=True)
                data_dict[name] = data
        #
        elif sim_type == 'dmo':
            for name in self.host_names[hosts]:
                data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/potentials/'+name+'_potentials_dmo', verbose=True)
                data_dict[name] = data
        #
        return data_dict

    def data_read_potential_full(self, directory, selection='sim', hosts='all'):
        """
        DESCRIPTION:
            Reads in the potential data and stores it in a dictionary with each
            key being the host name.

        VARIABLES:
            directory : string
                        Home directory.

            selection : string
                        Can either be 'sim' or 'model' depending on what you want

            hosts     : string
                        Choose any of the options listed in self.host_names within
                        the __init__ function above.

        NOTES:
            - The returned dictionary contains other dictionaries.
            - Each key in the dictionary is a host name
                - Each key in the sub-dictionaries is given in combine_potential_files.py
                  or in sat_potentials_model.py
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              sat_potentials_all_snap.py and combined with combine_potential_files.py,
              or generated from sat_potentials_model.py

            **THIS METHOD WAS USED PRIMARILY FOR THE PAPER II DATA**
        """
        # Set up empty dictionary to save data to
        data_dict = dict()
        #
        # Given the type of data, read in from the appropriate directory
        if selection == 'sim':
            for name in self.host_names[hosts]:
                data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/potentials/all_snapshots/'+name+'_potentials_all', verbose=True)
                data_dict[name] = data
        #
        elif selection == 'model':
            for name in self.host_names[hosts]:
                data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/potentials/all_snapshots/'+name+'_potentials_model', verbose=True)
                data_dict[name] = data
        #
        return data_dict

    def data_read_mass_profile(self, directory, hosts='all'):
        """
        DESCRIPTION:
            Reads in the mass profile data for each host and stores it in a
            dictionary, along with time and distance arrays used in generating
            the mass profile data.

        VARIABLES:
            directory : string
                        Home directory.

            hosts     : string
                        Choose any of the options listed in self.host_names within
                        the __init__ function above.

        NOTES:
            - Returns a dictionary.
            - Data was generated from mass_profile_parallel.py, and combined with
              mass_profile_comobine_files.py
            - This is used for Figure 3 of Paper II
        """
        # Set up an empty dictionary to save to
        data_dict = dict()
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/mass_profiles/'+name+'_mass_profile_all', verbose=True)
            data_dict[name] = data['mass.profile']
        #
        # Save extra useful information
        data_dict['snapshot'] = data['snapshot']
        data_dict['time'] = data['time']
        data_dict['rs'] = np.logspace(np.log10(5), np.log10(500), 25)
        data_dict['rs.interp'] = np.logspace(np.log10(data_dict['rs'][0]), np.log10(data_dict['rs'][-1]), 1000)
        #
        return data_dict

    def data_read_dadr(self, mass_profile, hosts='all'):
        """
        DESCRIPTION:
            Using the host halos enclosed mass profile, calculates the change in
            the tidal acceleration for each host at each snapshot from 5 to 500
            kpc.

        VARIABLES:
            mass_profile : dictionary
                           Data containing each host's mass profile over time

            hosts        : string
                           Choose any of the options listed in self.host_names within
                           the __init__ function above.

        NOTES:
            - Returns a dictionary.
            - Each element in the dictionary is data for a given host
            - Each element in a host's dictionary is the change in the tidal
              acceleration, given in cm / s^2
            - This method first reads in the mass profile and saves the tidal
              field at each distance and snapshot for a host.
              - Then it loops through the tidal field array, and numerically
                calculates the derivative at each distance in an interpolated
                distance array to calculate the change in the tidal acceleration.
        """
        # Set up an empty dictionary to save to, along with useful arrays
        da_dr_all = dict()
        rs = mass_profile['rs']
        rs_new = mass_profile['rs.interp']
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            tidal_field_ind = np.zeros(mass_profile[name].shape)
            da_dr_ind = np.zeros((mass_profile[name].shape[0], len(rs_new)))
            #
            # Loop through each snapshot
            for i in range(0, mass_profile[name].shape[0]):
                # Loop through each distance, i.e. each value of the enclosed mass
                for j in range(0, mass_profile[name].shape[1]):
                    # Calculate the tidal acceleration
                    tidal_field_ind[i,j] = (mass_profile[name][i][j]/rs[j+1]**2)*(6.67*10**(-11))*(2*10**(30)*100)/((1000**2)*(3.086*10**(16))**2) # in cm/s^2
            #
            # Loop through the data again and interpolate between the distance array
            for i in range(0, mass_profile[name].shape[0]):
                spline_fit = splrep(x=rs[1:], y=tidal_field_ind[i])
                da_dr_ind[i] = splev(rs_new, spline_fit, der=1)*(1/1000)*(1/(3.086*10**(18))) # Needed to convert the denominator to cm
            da_dr_all[name] = da_dr_ind
        #
        return da_dr_all

    def data_mask(self, dictionary, outliers=False, peri_sim=True, peri_model=False, current_sat=False, either=False, hosts='all'):
        """
        DESCRIPTION:
            Create a dictionary of masks for the satellites that depends on whether they
            have fallen into the host, whether they are currently in the host (at z = 0),
            whether they have experienced a pericenter in the simulation or model, and
            the outliers that have experienced pericenter in sim, but not model.

        VARIABLES:
            dictionary  : dictionary
                          Dictionary of data for all hosts. This is read in by
                          data_read()

            outliers    : boolean
                          Outliers are defined as the subhalos that have pericenters
                          in the simulations, but not in the model.

            peri_sim    : boolean
                          Set to True if you want subhalos that have expericenced
                          pericenter in the simulations.

            peri_model  : boolean
                          Set to True if you want subhalos that have expericenced
                          pericenter in the model.

            current_sat : boolean
                          Set to True if you want subhalos that are within
                          R_vir of the host galaxy at z = 0.

            either      : boolean
                          Set to True if you want subhalos that had a pericenter
                          in either the simulations or model.

            hosts       : string
                          Choose any of the options listed in self.host_names within
                          the __init__ function above.

        NOTES:
            - Returns a dictionary of masks where each key corresponds to each
              host galaxy.
              - This masking dictionary is used by ALL other methods in this class.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up a dictionary to save the masks to
        mask_dict = dict()
        #
        # If not interested in outliers, do the following masks
        if outliers == False:
            #
            if either == True:
                #
                # If not interested in whether they are currently satellites, do this
                if current_sat == False:
                    for name in self.host_names[hosts]:
                        mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['pericenter.check.sim'] | dictionary[name]['pericenter.check.galpy'])
                #
                # If interested in current sats only, do this
                elif current_sat == True:
                    for name in self.host_names[hosts]:
                        mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['pericenter.check.sim'] | dictionary[name]['pericenter.check.galpy'])*(dictionary[name]['d.tot.sim'][:,0] < dictionary[name]['host.radius'][0])
            #
            elif either == False:
                #
                # Pericenter in simulation but not required in the model
                if (peri_sim == True) & (peri_model == False):
                    #
                    # If not interested in whether they are currently satellites, do this
                    if current_sat == False:
                        for name in self.host_names[hosts]:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']
                    #
                    # If interested in current sats only, do this
                    elif current_sat == True:
                        for name in self.host_names[hosts]:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*(dictionary[name]['d.tot.sim'][:,0] < dictionary[name]['host.radius'][0])
                #
                # Pericenter required in simulation and in model
                elif (peri_sim == True) & (peri_model == True):
                    #
                    # If not interested in whether they are currently satellites, do this
                    if current_sat == False:
                        for name in self.host_names[hosts]:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*dictionary[name]['pericenter.check.galpy']
                    #
                    # If interested in current sats only, do this
                    elif current_sat == True:
                        for name in self.host_names[hosts]:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*dictionary[name]['pericenter.check.galpy']*(dictionary[name]['d.tot.sim'][:,0] < dictionary[name]['host.radius'][0])
                #
                # Pericenter not required in simulation or model
                elif (peri_sim == False) & (peri_model == False):
                        #
                        # If not interested in whether they are currently satellites, do this
                        if current_sat == False:
                            for name in self.host_names[hosts]:
                                mask_dict[name] = dictionary[name]['infall.check']
                        #
                        # If interested in current sats only, do this
                        elif current_sat == True:
                            for name in self.host_names[hosts]:
                                mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['d.tot.sim'][:,0] < dictionary[name]['host.radius'][0])
        #
        # If interested in outliers do this.
        elif outliers == True:
            for name in self.host_names[hosts]:
                mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*(~dictionary[name]['pericenter.check.galpy'])
        #
        return mask_dict

    def data_mask_apo(self, dictionary, hosts='all'):
        """
        DESCRIPTION:
            Create a dictionary of masks for the satellites that checks whether
            or not a satellite experienced an apocenter in the simulations
            ** Probably want to add another mask to check if a satellite has
               experienced an apocenter in the model as well **

        VARIABLES:
            dictionary  : dictionary
                          Dictionary of data for all hosts. This is read in by
                          data_read()

            hosts       : string
                          Choose any of the options listed in self.host_names within
                          the __init__ function above.

        NOTES:
            - Returns a dictionary of masks where each key corresponds to each
              host galaxy.
              - This masking dictionary is used by ALL other methods in this class.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up a dictionary to save the masks to
        mask_dict = dict()
        #
        # Loop through each host and save the masks in a dictionary
        for name in self.host_names[hosts]:
            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['apocenter.check.sim']
        #
        return mask_dict

    ### Everything above this can probably go in it's own class for reading and masking the data.

    def halo_id(self, data_dict, mask_dict, hosts='all'):
        """
        DESCRIPTION:
            Given the data and mask dictionaries generated by the methods above,
            return another dictionary to list various identifiers for the
            satellites to better distinguish them. This is purely a diagnostic
            tool.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

        NOTES:
            - Returns a dictionary.
                - Within the dictionary, there are three arrays that give:
                    - Host galaxy a satellite is in
                    - The "id" I have given it
                    - The merger tree indices of the satellite from the simulations
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up an empty list to save values to
        data = dict()
        names = []
        ids = []
        indices = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Loop through each subhalo and save the properties to the arrays
            for i in range(0, len(data_dict[name]['id'][mask_dict[name]])):
                names.append(name)
                ids.append(data_dict[name]['id'][mask_dict[name]][i])
                indices.append(data_dict[name]['indices.z0'][mask_dict[name]][:,0][i])
        #
        # Save the arrays to a dictionary
        data['host'] = np.asarray(names)
        data['ids'] = np.asarray(ids)
        data['indices'] = np.asarray(indices)
        #
        return data

    def print_keys(self, data_dict, host='m12b'):
        """
        TBD
        """
        for i in data_dict[host].keys():
            print(i)

    def host_vir_properties(self, data_dict, hosts='all'):
        """
        TBD
        """
        d = dict()
        R200m = []
        M200m = []
        Vcirc = []
        Evir = []
        Lvir = []
        G = 6.67*10**(-11)
        #
        for name in self.host_names[hosts]:
            radius = data_dict[name]['host.radius'][0]
            mass = data_dict[name]['host.mass'][0]
            energy = (G*2*10**(30))/(1000**3*3.086*10**(16))*mass/radius # in kmsis
            vel = np.sqrt(energy)
            ell = radius*vel
            #
            R200m.append(radius)
            M200m.append(mass)
            Vcirc.append(vel)
            Evir.append(energy)
            Lvir.append(ell)
        #
        d['mass'] = np.hstack(M200m)
        d['radius'] = np.hstack(R200m)
        d['vcirc'] = np.hstack(Vcirc)
        d['Evir'] = np.hstack(Evir)
        d['Lvir'] = np.hstack(Lvir)
        #
        return d

    def nperi(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Retrieve the number of pericenters a subhalo experiences from the
            data and save these values in an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up an empty list to save values to
        data = []
        #
        # Loop over the hosts
        for name in self.host_names[hosts]:
            # Determine whether or not oversampling, then save the data to the list
            if oversample:
                data.append(np.repeat(data_dict[name]['N.peri.'+selection][mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['N.peri.'+selection][mask_dict[name]])
        #
        return np.hstack(data)

    def nperi_model(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Retrieve the number of pericenters a subhalo experienced since a given
            definition of infall into the host galaxy.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose the key associated with the infall definition
                         you're interested in.
                         - sim         : infall time defined from simulation data with evolving R200m
                         - model       : infall time defined from model data with evolving R200m
                         - model.R200m : infall time defined from model data of when sat first crossed R200m

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up an empty list to save values to
        data = []
        #
        # Set the infall type
        if selection == 'sim':
            infall_type = 'first.infall.time.lb'
        elif selection == 'model':
            infall_type = 'first.infall.time.lb.model'
        elif selection == 'model.R200m':
            infall_type = 'infall.time.lb.model.R200m'
        else:
            print('Choose a correct infall type')
            raise KeyError('non-alphanumeric character in input')
            pass
        #
        # Loop over the hosts
        for name in self.host_names[hosts]:
            # Loop over each satellite
            for i in range(0, len(data_dict[name]['pericenter.time.lb.model'][mask_dict[name]])):
                # Mask out all of the null values
                m = (data_dict[name]['pericenter.time.lb.model'][mask_dict[name]][i] != -1)
                # Determine whether or not oversampling, then save the data to the list
                if oversample:
                    data.append(np.repeat(np.sum((data_dict[name]['pericenter.time.lb.model'][mask_dict[name]][i][m] < data_dict[name][infall_type][mask_dict[name]][i])), self.oversample[sim_type][name]))
                else:
                    data.append(np.sum((data_dict[name]['pericenter.time.lb.model'][mask_dict[name]][i][m] < data_dict[name][infall_type][mask_dict[name]][i])))
        #
        return np.hstack(data)

    def dperi_recent(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the recent pericenter distances a subhalo experiences, either
            in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced a pericenter, sets the most recent
              pericenter distance equal to the present-day distance.
        """
        # Set up an empty list to save values to
        data = []
        #
        # Loops through each host
        for name in self.host_names[hosts]:
            # Get all of the most recent pericenters
            temp_array = data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][:,0]
            # Mask out the cases where there are no pericenters
            mask_temp = (temp_array == -1)
            # For the null values, replace with present-day distances
            temp_array[mask_temp] = data_dict[name]['d.tot.sim'][mask_dict[name]][:,0][mask_temp]
            # Determine oversampling and save to list
            if oversample:
                data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
            else:
                data.append(temp_array)
        #
        return np.hstack(data)

    def dperi_min(self, data_dict, mask_dict, oversample=False, selection='sim', hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the minimum pericenter distances a subhalo experiences, either
            in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced a pericenter, sets the minimum
              pericenter distance equal to the present-day distance.
            - In the model, for a subhalo that experiences multiple pericenters,
              the pericenter distances are always going to be the same, so the
              minimum will be the same in the model.
        """
        # Set up an empty list to save values to
        data = []
        #
        # Loop through each host
        if selection == 'sim':
            for name in self.host_names[hosts]:
                # Loop through each subhalo in the host
                for i in range(0, len(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]])):
                    # Mask out the null pericenter values
                    mask_temp = (data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i] != -1)
                    # If there are no pericenters, append the present-day distance as the minimum
                    if np.sum(mask_temp) == 0:
                        if oversample:
                            data.append(np.repeat(data_dict[name]['d.tot.sim'][mask_dict[name]][i][0], self.oversample[sim_type][name]))
                        else:
                            data.append(data_dict[name]['d.tot.sim'][mask_dict[name]][i][0])
                    # If there are pericenters, append the minimum value to the list
                    else:
                        if oversample:
                            data.append(np.repeat(np.min(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i][mask_temp]), self.oversample[sim_type][name]))
                        else:
                            data.append(np.min(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i][mask_temp]))
        #
        elif selection == 'model':
            for name in self.host_names[hosts]:
                # Get all of the most recent pericenters
                temp_array = data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][:,0]
                # Mask out the cases where there are no pericenters
                mask_temp = (temp_array == -1)
                # For the null values, replace with present-day distances
                temp_array[mask_temp] = data_dict[name]['d.tot.sim'][mask_dict[name]][:,0][mask_temp]
                # Determine oversampling and save to list
                if oversample:
                    data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
                else:
                    data.append(temp_array)
        #
        return np.hstack(data)

    # Check up on this and whether or not I want to keep the max distance rule
    def dapo_recent(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the recent apocenter distances a subhalo experiences, either
            in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced an apocenter, sets the most recent
              apocenter distance equal to the maximum distance a subhalo has
              ever experienced.
              ** Need to check whether this is a problem. I don't want to count
                 an apocenter as the max distance outside of the virial radius **
        """
        # Set up an empty list to save values to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Select all recent apocenter values
            temp_array = data_dict[name]['apocenter.dist.'+selection][mask_dict[name]][:,0]
            # Mask out cases with null values
            mask_temp = (temp_array == -1)
            # For null cases, replace them with maximum distances the satellites experienced
            temp_array[mask_temp] = np.max(data_dict[name]['d.tot.sim'][mask_dict[name]][mask_temp], axis=1)
            # Oversample if needed and append to the list
            if oversample:
                data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
            else:
                data.append(temp_array)
        #
        return np.hstack(data)

    # Is this even relevant anymore?
    def dmax(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Finds the largest distance a satellite ever experienced.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - Does not require the satellite to have fell into the host
              galaxy/halo.
        """
        # Set up an empty list to save values to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            #
            # If selecting simulation data, get the max distance from the data file
            if selection == 'sim':
                if oversample:
                    data.append(np.repeat(data_dict[name]['max.dist.sim'][mask_dict[name]], self.oversample[sim_type][name]))
                else:
                    data.append(data_dict[name]['max.dist.sim'][mask_dict[name]])
            #
            # If selecting model data, just get the most recent apocenter distance since they are all the same
            elif selection == 'model':
                if oversample:
                    data.append(np.repeat(data_dict[name]['apocenter.dist.model'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
                else:
                    data.append(data_dict[name]['apocenter.dist.model'][mask_dict[name]][:,0])
        #
        return np.hstack(data)

    def tperi_recent(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the recent pericenter lookback times a subhalo experienced, either
            in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced a pericenter, sets the most recent
              pericenter time equal to 0 Gyr, i.e. present-day.
        """
        # Set up an empty list to save the data to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Get the most recent pericenter lookback time
            temp_array = data_dict[name]['pericenter.time.lb.'+selection][mask_dict[name]][:,0]
            # Mask out the null values
            mask_temp = (temp_array == -1)
            # If no pericenter, set the lookback time to present-day
            temp_array[mask_temp] = 0.0
            # Oversample if needed and append to the list
            if oversample:
                data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
            else:
                data.append(temp_array)
        #
        return np.hstack(data)

    def tperi_min(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the lookback times of the minimum pericenter a subhalo experienced
            in the simulation into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced a pericenter, sets the most recent
              pericenter time equal to 0 Gyr, i.e. present-day.
        """
        if selection == 'sim':
            # Set up an empty list to save the data to
            data = []
            #
            # Loop through each host galaxy
            for name in self.host_names[hosts]:
                # Loop through each subhalo
                for i in range(0, len(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]])):
                    # Mask out the null values
                    mask_temp = (data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i] != -1)
                    # If there are no pericenters, set the lookback time to present-day, oversample if needed and append to the list
                    if np.sum(mask_temp) == 0:
                        if oversample:
                            data.append(np.repeat(0.0, self.oversample[sim_type][name]))
                        else:
                            data.append(0.0)
                    #
                    # If there pericenters, find the minimum, select the time, oversample if needed and append to the list
                    else:
                        index = np.where(np.min(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i][mask_temp]) \
                                         == data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i][mask_temp])[0][0]
                        if oversample:
                            data.append(np.repeat(data_dict[name]['pericenter.time.lb.'+selection][mask_dict[name]][i][mask_temp][index], self.oversample[sim_type][name]))
                        else:
                            data.append(data_dict[name]['pericenter.time.lb.'+selection][mask_dict[name]][i][mask_temp][index])
            #
            return np.hstack(data)
        #
        else:
            data = self.tperi_recent(data_dict, mask_dict, selection=selection, oversample=oversample, hosts=hosts, sim_type=sim_type)
            return data

    # Do I want to keep this?
    def tapo_recent(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the recent pericenter lookback times a subhalo experienced, either
            in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced a pericenter, sets the most recent
              pericenter time equal to 0 Gyr, i.e. present-day.
        """
        # Set up an empty list to save the data to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Get the most recent pericenter lookback time
            temp_array = data_dict[name]['apocenter.time.lb.'+selection][mask_dict[name]][:,0]
            # Mask out the null values
            mask_temp = (temp_array == -1)
            # If no pericenter, set the lookback time to present-day
            temp_array[mask_temp] = 0.0
            # Oversample if needed and append to the list
            if oversample:
                data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
            else:
                data.append(temp_array)
        #
        return np.hstack(data)

    def first_infall(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the lookback times of satellite first infall into their MW-mass
            host together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up empty array to save to
        data = []
        #
        if selection == 'model':
            infall_str = 'first.infall.time.lb.'+selection
        else:
            infall_str = 'first.infall.time.lb'
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and save to the list
            if oversample:
                data.append(np.repeat(data_dict[name][infall_str][mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name][infall_str][mask_dict[name]])
        #
        return np.hstack(data)

    # optimized ; could probably absorb this into the method above eventually
    def first_infall_any(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the lookback times of satellite first infall into ANY more
            massive halo together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up empty array to save to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and save to list
            if oversample:
                data.append(np.repeat(data_dict[name]['first.infall.time.lb.any'][mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['first.infall.time.lb.any'][mask_dict[name]])
        #
        return np.hstack(data)

    # needs dox and optimization
    def recent_infall(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        # Set up empty array to save to
        data = []
        #
        if selection == 'model':
            time_name = 'all.infall.time.lb.'+selection
        else:
            time_name = 'all.infall.time.lb'
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            for i in range(0, len(data_dict[name][time_name][mask_dict[name]])):
                mask = (data_dict[name][time_name][mask_dict[name]][i] != -1)
                if np.sum(mask != 0):
                    if oversample:
                        data.append(np.repeat(data_dict[name][time_name][mask_dict[name]][i][mask][0], self.oversample[sim_type][name]))
                    else:
                        data.append(data_dict[name][time_name][mask_dict[name]][i][mask][0])
                else:
                    if oversample:
                        data.append(np.repeat(np.nan, self.oversample[sim_type][name]))
                    else:
                        data.append(np.nan)
                #
        return np.hstack(data)

    def infall_fixed(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
            TBD
        """
        # Set up empty array to save to
        data = []
        #
        time_name = 'infall.time.lb.model.R200m'
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            for i in range(0, len(data_dict[name][time_name][mask_dict[name]])):
                mask = (data_dict[name][time_name][mask_dict[name]][i] != -1)
                if np.sum(mask != 0):
                    if oversample:
                        data.append(np.repeat(data_dict[name][time_name][mask_dict[name]][i][mask], self.oversample[sim_type][name]))
                    else:
                        data.append(data_dict[name][time_name][mask_dict[name]][i][mask])
                else:
                    if oversample:
                        data.append(np.repeat(np.nan, self.oversample[sim_type][name]))
                    else:
                        data.append(np.nan)
                #
        return np.hstack(data)

    def mstar(self, data_dict, mask_dict, selection='z0', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the peak or present-day satellite stellar masses together
            into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose either 'peak' for the peak stellar mass, or 'z0'
                         for the present-day stellar mass.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up empty array to save data to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and append to list
            if oversample:
                data.append(np.repeat(data_dict[name]['M.star.'+selection][mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['M.star.'+selection][mask_dict[name]])
        #
        return np.hstack(data)

    def mhalo(self, data_dict, mask_dict, selection='z0', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the peak or present-day satellite halo masses together
            into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose either 'peak' for the peak stellar mass, or 'z0'
                         for the present-day stellar mass.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up empty list to save to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and append to list
            if oversample:
                data.append(np.repeat(data_dict[name]['M.halo.'+selection][mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['M.halo.'+selection][mask_dict[name]])
        #
        return np.hstack(data)

    def d_z0(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Get the present-day 1D satellite distances and save them in an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up empty list to save to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and append to list
            if oversample:
                data.append(np.repeat(data_dict[name]['d.tot.sim'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['d.tot.sim'][mask_dict[name]][:,0])
        #
        return np.hstack(data)

    # optimized, update once data is re-run and absorb velocities() method here
    def v_z0(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Get the present-day total/scalar velocities and save them in an array.
            ** I don't really use this metric in Paper I, worth deciding on if
               I should keep it or not **

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
              ** Should just add a "selection = tot/tan/rad" option and get rid
                 of velocities() function below. Requires me to re-name the
                 vtot property key name
        """
        # Set up empty list to save to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and append to list
            if oversample:
                data.append(np.repeat(data_dict[name]['v.tot.'+selection][mask_dict[name]][:,0], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['v.tot.'+selection][mask_dict[name]][:,0])
        #
        return np.hstack(data)

    def vperi_recent(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the recent pericenter velocities a subhalo experiences, either
            in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced a pericenter, sets the most recent
              pericenter velocity equal to the present-day velocity.
        """
        # Set up an empty list to save values to
        data = []
        #
        # Loops through each host
        for name in self.host_names[hosts]:
            # Get all of the most recent pericenters
            temp_array = data_dict[name]['pericenter.vel.'+selection][mask_dict[name]][:,0]
            # Mask out the cases where there are no pericenters
            mask_temp = (temp_array == -1)
            # For the null values, replace with present-day velocities
            temp_array[mask_temp] = data_dict[name]['v.tot.sim'][mask_dict[name]][:,0][mask_temp]
            # Determine oversampling and save to list
            if oversample:
                data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
            else:
                data.append(temp_array)
        #
        return np.hstack(data)

    def vperi_min(self, data_dict, mask_dict, oversample=False, selection='sim', hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the velocities at minimum pericenter, from either
            in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced a pericenter, sets the minimum
              pericenter velocity equal to the present-day velocity.
            - In the model, for a subhalo that experiences multiple pericenters,
              the pericenter velocities are always going to be the same, so the
              minimum will be the same in the model.
        """
        # Set up an empty list to save values to
        data = []
        #
        if selection == 'sim':
            # Loop through each host
            for name in self.host_names[hosts]:
                # Loop through each subhalo in the host
                for i in range(0, len(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]])):
                    # Mask out the null pericenter values
                    mask_temp = (data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i] != -1)
                    # If there are no pericenters, append the present-day velocity as the minimum
                    if np.sum(mask_temp) == 0:
                        if oversample:
                            data.append(np.repeat(data_dict[name]['v.tot.sim'][mask_dict[name]][i][0], self.oversample[sim_type][name]))
                        else:
                            data.append(data_dict[name]['v.tot.sim'][mask_dict[name]][i][0])
                    # If there are pericenters, append the value of the velocity at minimum to the list
                    else:
                        index = np.where(np.min(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i][mask_temp]) \
                                         == data_dict[name]['pericenter.dist.'+selection][mask_dict[name]][i][mask_temp])[0][0]
                        if oversample:
                            data.append(np.repeat(data_dict[name]['pericenter.vel.'+selection][mask_dict[name]][i][mask_temp][index], self.oversample[sim_type][name]))
                        else:
                            data.append(data_dict[name]['pericenter.vel.'+selection][mask_dict[name]][i][mask_temp][index])
        #
        elif selection == 'model':
            # Loop through each host
            for name in self.host_names[hosts]:
                # Get all of the most recent pericenters
                temp_array = data_dict[name]['pericenter.vel.'+selection][mask_dict[name]][:,0]
                # Mask out the cases where there are no pericenters
                mask_temp = (temp_array == -1)
                # For the null values, replace with present-day velocities
                temp_array[mask_temp] = data_dict[name]['v.tot.sim'][mask_dict[name]][:,0][mask_temp]
                # Determine oversampling and save to list
                if oversample:
                    data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
                else:
                    data.append(temp_array)
        return np.hstack(data)

    def potential(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon', norm='potential'):
        """
        DESCRIPTION:
            Groups the satellite potentials into an array and normalizes them.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read_potential()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

            norm       : string
                         Choose which type of normalization you want to apply
                         to the data. "potential" will normalize the subhalos
                         by simply subtracting the host potential at R200m from
                         the subhalo values. "kinetic" will normalize them such
                         that their total energy is 0 at the host R200m

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - Note that there are two different types of normalization; I only
              used the "kinetic" version in Paper I
            - Not all host galaxies have potential data. I usually use the
              hosts='all_energy' selection of hosts, which excludes m12z and R&R
            ** Should maybe point to screenshot I have of how I normalize in the
               kinetic scheme; or just explain it more here **
        """
        # Set up an empty list to save to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Determine which normalization, oversample if needed, and append data to list
            if norm == 'potential':
                if oversample:
                    data.append(np.repeat(data_dict[name]['subhalo.potential'][mask_dict[name]]-data_dict[name]['host.potential.R200m'], self.oversample[sim_type][name]))
                else:
                    data.append(data_dict[name]['subhalo.potential'][mask_dict[name]]-data_dict[name]['host.potential.R200m'])
            elif norm == 'kinetic':
                k = (-1)*data_dict[name]['host.potential.R200m']-data_dict[name]['KE.at.Rvir']
                if oversample:
                    data.append(np.repeat(data_dict[name]['subhalo.potential'][mask_dict[name]]+k, self.oversample[sim_type][name]))
                else:
                    data.append(data_dict[name]['subhalo.potential'][mask_dict[name]]+k)
        #
        return np.hstack(data)

    def kinetic_energy(self, data_dict, mask_dict, ke_type, oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Calculates the kinetic energies of the satellites based on a few
            different selections, and groups them together into an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read_potential()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            ke_type    : string
                         Type of kinetic energy you want. 'z0' is the present-day
                         kinetic energy, 'max' is the maximum kinetic energy a
                         satellite experienced, and 'peri' is the kinetic energy
                         at the most recent pericenter.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - Note that there are three different types of kinetic energy; I only
              used the "z0" version in Paper I
            - In Paper I, I only selected the 'all_energy' hosts because I used
              kinetic energy to find the total energy, and not all hosts had
              values for the satellite potential energy.
        """
        # Set up empty list to save to
        data = []
        #
        # Loop through all hosts
        for name in self.host_names[hosts]:
            # Determine type of kinetic energy
            if ke_type == 'max':
                # Loop through each satellite
                for i in range(0, len(data_dict[name]['v.tot.sim'][mask_dict[name]])):
                    if oversample:
                        data.append(np.repeat(np.nanmax(0.5*data_dict[name]['v.tot.sim'][mask_dict[name]][i]**2), self.oversample[sim_type][name]))
                    else:
                        data.append(np.nanmax(0.5*data_dict[name]['v.tot.sim'][mask_dict[name]][i]**2))
            #
            # If you want KE at pericenter, get data at most recent pericenter
            elif ke_type == 'peri':
                for i in range(0, len(data_dict[name]['pericenter.vel.sim'][mask_dict[name]])):
                    if (data_dict[name]['pericenter.vel.sim'][mask_dict[name]][i][0] == -1):
                        if oversample:
                            data.append(np.repeat(0.5*data_dict[name]['v.tot.sim'][mask_dict[name]][i][0]**2, self.oversample[sim_type][name]))
                        else:
                            data.append(0.5*data_dict[name]['v.tot.sim'][mask_dict[name]][i][0]**2)
                    else:
                        if oversample:
                            data.append(np.repeat(0.5*data_dict[name]['pericenter.vel.sim'][mask_dict[name]][i][0]**2, self.oversample[sim_type][name]))
                        else:
                            data.append(0.5*data_dict[name]['pericenter.vel.sim'][mask_dict[name]][i][0]**2)
            #
            # If present-day KE, select present-day velocity, oversample if needed, and append to list
            elif ke_type == 'z0':
                if oversample:
                    data.append(np.repeat(0.5*data_dict[name]['v.tot.sim'][mask_dict[name]][:,0]**2, self.oversample[sim_type][name]))
                else:
                    data.append(0.5*data_dict[name]['v.tot.sim'][mask_dict[name]][:,0]**2)
        #
        return np.hstack(data)

    # optimized, but still needs a lot of work.
    def mass_masking_property(self, data_dict, mask_dict, prop, mass_array, mass_type='M.star.z0', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Takes a mass array and saves a property of interest in a dictionary
            where each entry in the dictionary are the properties for galaxies
            that fall in the mass bins supplied by the mass array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read().

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            prop       : string
                         Property that you want to save to a dictionary.
                         Only takes one property.
                         The properties that you can select are the keys in
                         data_dict. See orbit_io.py for the output properties
                         that are saved in data_dict().

            mass_array : 1D array
                         Numpy array of mass values that are the bin edges for which
                         you want to select galaxies.

            mass_type  : string
                         The mass type you are selecting the galaxies by. Can be
                         stellar or halo mass, and either present-day or peak
                         values.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a dictionary.
            - Each key in the dictionary corresponds to one of the mass bins.
              There are len(mass_array)-1 bins, and dictionary keys, total.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py. However, only galaxies that
              fall in the mass bins are saved.
              ** This is still very much so a work in progress! I have not written
                 this to account for all properties. Especially properties in
                 data_dict, which is an array for each satellite. I have not
                 yet thought about how to best account for those.
        """
        # Set up an empty dictionary to save to
        props = dict()
        #
        # Loop through the mass bins
        for i in range(0, len(mass_array)-1):
            # Set up empty list to save data to
            prop_list = []
            # Loop through each of the hosts
            for name in self.host_names[hosts]:
                # Select satellites within the mass bin
                prop_mask = ((data_dict[name][mass_type][mask_dict[name]] > mass_array[i])*(data_dict[name][mass_type][mask_dict[name]] < mass_array[i+1]))
                # Oversample if needed and save data to list
                if oversample:
                    prop_list.append(np.repeat(data_dict[name][prop][mask_dict[name]][prop_mask], self.oversample[sim_type][name]))
                else:
                    prop_list.append(data_dict[name][prop][mask_dict[name]][prop_mask])
            #
            # Save the galaxies in the mass bin to the dictionary
            props['bin.'+str(i+1)] = np.hstack(prop_list)
        #
        return props

    # optimized, but will get rid of and absorb into v_z0 function...
    def velocities(self, data_dict, mask_dict, selection='tan', oversample=False, hosts='all', sim_type='baryon'):
        """
        Only works with the simulation data right now, not the model data...

        selection = rad or tan

        Only saves the z = 0 value
        """
        # Set up empty array to save data to
        data = []
        #
        # Loop over each host
        for name in self.host_names[hosts]:
            # oversample if needed and append data to list
            if oversample:
                data.append(np.repeat(data_dict[name]['v.'+selection+'.sim'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['v.'+selection+'.sim'][mask_dict[name]][:,0])
        #
        return np.hstack(data)

    def L_z0(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Get the present-day specific angular momentum for satellite galaxies
            and save them in an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Select whether you want simulation or model data.
                          ** Right now, only works with simulation data... **

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
              ** Only works with the simulation data right now, but in time, I
                 will adapt this to work with the model data **
        """
        # Set up empty list to save data to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and append data to list
            if oversample:
                data.append(np.repeat(data_dict[name]['L.tot.'+selection][mask_dict[name]][:,0], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['L.tot.'+selection][mask_dict[name]][:,0])
        #
        return np.hstack(data)

    def L_rec(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the specific angular momenta of satellites at their most recent
            pericenters, either in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.
                         ** Right now, only works with simulation data **

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - Only works if there are actual pericenters, will not work on all
              satellites.
              ** Only works with the simulation data right now, not the model
                 data... **
        """
        # Set up empty list to save to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and append data to list
            if oversample:
                data.append(np.repeat(data_dict[name]['L.at.peri'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['L.at.peri'][mask_dict[name]][:,0])
        #
        return np.hstack(data)

    def L_min(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the specific angular momenta of satellites at their minimum
            pericenters, either in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.
                         ** Right now, only works with simulation data **

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - Only works if there are actual pericenters, will not work on all
              satellites.
              ** Only works with the simulation data right now, not the model
                 data... **
        """
        # Set up empty list to save data to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Loop through each satellite
            for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                # Find the index in the array of pericenters for each satellite
                # that corresponds to the minimum pericenter
                index = np.where(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]) == data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp])[0]
                # Oversample if needed and save the data to the list
                if oversample:
                    data.append(np.repeat(data_dict[name]['L.at.peri'][mask_dict[name]][i][mask_temp][index], self.oversample[sim_type][name]))
                else:
                    data.append(data_dict[name]['L.at.peri'][mask_dict[name]][i][mask_temp][index])
        #
        return np.hstack(data)

    def L_infall(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Groups the specific angular momenta of satellites at their infall,
            either in the simulation or model, together into one array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.
                         ** Right now, only works with simulation data **

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - Only works if there are actual pericenters, will not work on all
              satellites.
              ** Only works with the simulation data right now, not the model
                 data... **
        """
        # Set up empty list to save to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Set up lookback time array
            lb = np.flip(data_dict[name]['time.sim'][-1]-data_dict[name]['time.sim'])
            # Loop through each satellite
            for i in range(0, len(data_dict[name]['infall.check'][mask_dict[name]])):
                # Check if the satellite fell into the host
                if (data_dict[name]['first.infall.time.lb'][mask_dict[name]][i] != -1):
                    # Find the angular momentum of the satellite at infall, oversample if needed, and append save to the list
                    if oversample:
                        data.append(np.repeat(data_dict[name]['L.tot.sim'][mask_dict[name]][i][np.argmin(np.abs(data_dict[name]['first.infall.time.lb'][mask_dict[name]][i]-lb))], self.oversample[sim_type][name]))
                    else:
                        data.append(data_dict[name]['L.tot.sim'][mask_dict[name]][i][np.argmin(np.abs(data_dict[name]['first.infall.time.lb'][mask_dict[name]][i]-lb))])
        #
        return np.hstack(data)

    # Needs more dox
    # Should probably re-run the data and change the keys to include "sim"
    def eccentricity(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Calculates the average eccentricity for a subhalo and stores them
            all in an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced at least one pericenter and one
              apocenter, then the eccentricity is set to -1.
        """
        # Set up an empty list to save values to
        data = []
        #
        for name in self.host_names[hosts]:
            for i in range(0, len(data_dict[name]['eccentricity.'+selection][mask_dict[name]])):
                mask = (data_dict[name]['eccentricity.'+selection][mask_dict[name]][i] != -1)
                if (np.sum(mask) > 0):
                    if oversample:
                        data.append(np.repeat(np.average(data_dict[name]['eccentricity.'+selection][mask_dict[name]][i][mask]), self.oversample[sim_type][name]))
                    else:
                        data.append(np.average(data_dict[name]['eccentricity.'+selection][mask_dict[name]][i][mask]))
        #
        return np.hstack(data)

    # This is currently a diagnostic method... need dox
    def eccentricity_recent(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Calculates the average eccentricity for a subhalo and stores them
            all in an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
            - If a subhalo has not experienced at least one pericenter and one
              apocenter, then the eccentricity is set to -1.
        """
        # Set up an empty list to save values to
        data = []
        #
        for name in self.host_names[hosts]:
            for i in range(0, len(data_dict[name]['eccentricity.'+selection][mask_dict[name]])):
                mask = (data_dict[name]['eccentricity.'+selection][mask_dict[name]][i] != -1)
                if (np.sum(mask) != 0):
                    if oversample:
                        data.append(np.repeat(data_dict[name]['eccentricity.'+selection][mask_dict[name]][i][mask][0], self.oversample[sim_type][name]))
                    else:
                        data.append(data_dict[name]['eccentricity.'+selection][mask_dict[name]][i][mask][0])
        #
        return np.hstack(data)

    # Needs more dox
    # Should probably re-run the data and change the keys to include "sim"
    def period_recent(self, data_dict, mask_dict, selection='sim', choice='peri', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Calculates the recent period for a subhalo and stores them
            all in an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            TBD
        """
        # Set up an empty list to save values to
        data = []
        #
        period_type = 'orbit.period.'+choice+'.'+selection
        #
        for name in self.host_names[hosts]:
            for i in range(0, len(data_dict[name][period_type][mask_dict[name]])):
                if (data_dict[name][period_type][mask_dict[name]][i][0] != -1):
                    if oversample:
                        data.append(np.repeat(data_dict[name][period_type][mask_dict[name]][i][0], self.oversample[sim_type][name]))
                    else:
                        data.append(data_dict[name][period_type][mask_dict[name]][i][0])
        #
        return np.hstack(data)

    # Is this necessary?
    def period_average(self, data_dict, mask_dict, selection='sim', choice='peri', oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Calculates the average period for a subhalo and stores them
            all in an array.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            selection  : string
                         Choose whether you want values from simulation data or
                         data from the model.

            oversample : boolean
                         Choose whether you want to oversample the subhalos or not.

            hosts      : string
                         Choose which host galaxies you want to analyze; choices
                         listed in __init__().

            sim_type   : string
                         Choose which type of data you are analyzing. This is
                         only used for oversampling factors and does not matter
                         if you are not oversampling.

        NOTES:
            TBD
        """
        # Set up an empty list to save values to
        data = []
        #
        per_type_peri = 'orbit.period.peri.'+selection
        per_type_apo = 'orbit.period.apo.'+selection
        #
        if choice == 'both':
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name][per_type_peri][mask_dict[name]])):
                    mask_peri = (data_dict[name][per_type_peri][mask_dict[name]][i] != -1)
                    mask_apo = (data_dict[name][per_type_apo][mask_dict[name]][i] != -1)
                    if (np.sum(mask_peri) > 0) & (np.sum(mask_apo) > 0):
                        if oversample:
                            data.append(np.repeat(np.average(np.hstack((data_dict[name][per_type_peri][mask_dict[name]][i][mask_peri], data_dict[name][per_type_apo][mask_dict[name]][i][mask_apo]))), self.oversample[sim_type][name]))
                        else:
                            data.append(np.average(np.hstack((data_dict[name][per_type_peri][mask_dict[name]][i][mask_peri], data_dict[name][per_type_apo][mask_dict[name]][i][mask_apo]))))
                    #
                    elif (np.sum(mask_peri) > 0) & (np.sum(mask_apo) == 0):
                        if oversample:
                            data.append(np.repeat(np.average(data_dict[name][per_type_peri][mask_dict[name]][i][mask_peri]), self.oversample[sim_type][name]))
                        else:
                            data.append(np.average(data_dict[name][per_type_peri][mask_dict[name]][i][mask_peri]))
        #
        else:
            if choice == 'peri':
                period_type = per_type_peri
            elif choice == 'apo':
                period_type = per_type_apo
                #
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name][period_type][mask_dict[name]])):
                    mask = (data_dict[name][period_type][mask_dict[name]][i] != -1)
                    if (np.sum(mask) > 0):
                        if oversample:
                            data.append(np.repeat(np.average(data_dict[name][period_type][mask_dict[name]][i][mask]), self.oversample[sim_type][name]))
                        else:
                            data.append(np.average(data_dict[name][period_type][mask_dict[name]][i][mask]))

        #
        return np.hstack(data)

    # Needs dox
    def da_dr(self, data_dict, mask_dict, mass_profile_dict, da_dr_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        Finding da_dr at d_peri,min
        """
        data = []
        data_dist = []
        data_time = []
        #
        # Loop through all of the hosts
        for name in self.host_names[hosts]:
            #
            if selection == 'sim':
                # Loop through each of the satellites in a given host
                for i in range(0, len(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]])):
                    mask = (data_dict[name]['d.tot.'+selection][mask_dict[name]][i] != -1)
                    initial = (-1)*1e6
                    d_initial = -1
                    t_initial = -1
                    #
                    # Loop through the number of snapshots
                    for j in range(0, len(data_dict[name]['d.tot.'+selection][mask_dict[name]][i][mask])):
                        d_ind = np.where(np.min(np.abs(data_dict[name]['d.tot.'+selection][mask_dict[name]][i][mask][j] - mass_profile_dict['rs.interp'])) == np.abs(data_dict[name]['d.tot.'+selection][mask_dict[name]][i][mask][j] - mass_profile_dict['rs.interp']))[0][0]
                        if (np.flip(np.abs(da_dr_dict[name]), axis=0)[j][d_ind] > initial):
                            initial = np.flip(np.abs(da_dr_dict[name]), axis=0)[j][d_ind]
                            d_initial = mass_profile_dict['rs.interp'][d_ind]
                            t_initial = np.flip(mass_profile_dict['time'], axis=0)[j]
                    #
                    if oversample:
                        data.append(np.repeat(initial, self.oversample[sim_type][name]))
                        data_dist.append(np.repeat(d_initial, self.oversample[sim_type][name]))
                        data_time.append(np.repeat(t_initial, self.oversample[sim_type][name]))
                    else:
                        data.append(initial)
                        data_dist.append(d_initial)
                        data_time.append(t_initial)
            #
            elif selection == 'model':
                # Loop through each of the satellites in a given host
                for i in range(0, len(data_dict[name]['pericenter.dist.'+selection][mask_dict[name]])):
                    mask = (data_dict[name]['d.tot.'+selection][mask_dict[name]][i] != -1)
                    initial = (-1)*1e6
                    d_initial = -1
                    t_initial = -1
                    #
                    # Loop through the number of snapshots
                    for j in range(0, len(data_dict[name]['d.tot.'+selection][mask_dict[name]][i][mask])-2): # Need to avoid probing snapshots that don't have da/dr data
                        d_ind = np.where(np.min(np.abs(data_dict[name]['d.tot.'+selection][mask_dict[name]][i][mask][j] - mass_profile_dict['rs.interp'])) == np.abs(data_dict[name]['d.tot.'+selection][mask_dict[name]][i][mask][j] - mass_profile_dict['rs.interp']))[0][0]
                        if (np.flip(np.abs(da_dr_dict[name]), axis=0)[0][d_ind] > initial):
                            initial = np.flip(np.abs(da_dr_dict[name]), axis=0)[0][d_ind]
                            d_initial = mass_profile_dict['rs.interp'][d_ind]
                            t_initial = np.flip(mass_profile_dict['time'], axis=0)[j]
                    #
                    if oversample:
                        data.append(np.repeat(initial, self.oversample[sim_type][name]))
                        data_dist.append(np.repeat(d_initial, self.oversample[sim_type][name]))
                        data_time.append(np.repeat(t_initial, self.oversample[sim_type][name]))
                    else:
                        data.append(initial)
                        data_dist.append(d_initial)
                        data_time.append(t_initial)
        #
        d = dict()
        d['dadr'] = np.hstack(data)
        d['dadr.dist.interp'] =  np.hstack(data_dist)
        d['dadr.time.interp'] =  np.hstack(data_time)
        d['dadr.time.lb.interp'] =  np.hstack(mass_profile_dict['time'][-1] - data_time)
        #s
        return d

    # Needs a lot of documenting
    def energies(self, data_dict, mask_dict, potential_dict, mass_profile_dict, time_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data_z0 = []
        data_infall = []
        data_all = dict()
        data_all_time = dict()
        tlb = time_dict['time'][-1] - np.flip(time_dict['time'])
        data_all_host = []
        #
        for name in self.host_names[hosts]:
            # Set the host potential at 100 kpc at z = 0
            phi_host_z0 = potential_dict[name]['host.pot.100kpc'][-1] - potential_dict[name]['host.pot.R200m'] - potential_dict[name]['KE.at.Rvir']
            #
            # Set up the enclosed mass ratio within 100-ish kpc
            M_enc_100kpc_z0 = mass_profile_dict[name][-1][16]
            M_enc_100kpc_z = mass_profile_dict[name][:,16]
            M_enc_ratio = M_enc_100kpc_z/M_enc_100kpc_z0
            #
            # Define the host potential at 100 kpc across all snapshots
            phi_host_z = np.flip(phi_host_z0*M_enc_ratio) # Goes from z = 0 to some other z
            #
            # Set up null array to save the normalized subhalo potentials to
            sub_pot = (-1)*np.ones(potential_dict[name]['subhalo.pot'].shape)
            sub_energy = (-1)*np.ones(potential_dict[name]['subhalo.pot'].shape)
            sub_pot_snaps = (-1)*np.ones(potential_dict[name]['subhalo.pot'].shape, int)
            sub_pot_tlb = (-1)*np.ones(potential_dict[name]['subhalo.pot'].shape)
            sub_host_energy = (-1)*np.ones(potential_dict[name]['subhalo.pot'].shape[0])
            #
            host_energies = ((6.67*10**(-11)*2*10**(30))/(10**3*3.086*10**(16)*1000**2))*(data_dict[name]['host.mass'][0]/data_dict[name]['host.radius'][0])
            #
            # Loop through all of the satellites
            for i in range(0, sub_pot.shape[0]):
                # Create the right masks
                mask_pot_exist = (np.flip(potential_dict[name]['subhalo.pot'][i]) != -1)
                mask_pot_finite = np.isfinite(np.flip(potential_dict[name]['subhalo.pot'][i]))
                mask_vel_exist = (data_dict[name]['v.tot.sim'][i][:len(potential_dict[name]['subhalo.pot'][i])] != -1)
                mask_vel_finite = np.isfinite(data_dict[name]['v.tot.sim'][i][:len(potential_dict[name]['subhalo.pot'][i])])
                mask_host_finite = np.isfinite(np.flip(potential_dict[name]['host.pot.100kpc']))
                mask_host_exist = (phi_host_z[:len(potential_dict[name]['subhalo.pot'][i])] != 0)
                mask_tot = mask_pot_exist*mask_pot_finite*mask_vel_exist*mask_vel_finite*mask_host_exist*mask_host_finite
                #
                # Calculate the normalized subhalo potential energy
                sub_pot[i][mask_tot] = np.flip(potential_dict[name]['subhalo.pot'][i])[mask_tot] - np.flip(potential_dict[name]['host.pot.100kpc'])[mask_tot] + phi_host_z[:len(potential_dict[name]['subhalo.pot'][i])][mask_tot]
                #
                # Keep which snapshots it has data for
                sub_pot_snaps[i][mask_tot] = np.flip(time_dict['index'])[:len(potential_dict[name]['subhalo.pot'][i])][mask_tot]
                sub_pot_tlb[i][mask_tot] = tlb[:len(potential_dict[name]['subhalo.pot'][i])][mask_tot]
                #
                # Calculate the total orbital energy
                sub_energy[i][mask_tot] = sub_pot[i][mask_tot] + 0.5*(data_dict[name]['v.tot.sim'][i][:len(potential_dict[name]['subhalo.pot'][i])][mask_tot])**2
                sub_host_energy[i] = host_energies
            #
            # Save all energy data for the satellites.
            data_all[name] = sub_energy
            data_all_time[name] = sub_pot_tlb
            #
            # Save data for the z = 0 stuff
            if oversample:
                data_z0.append(np.repeat(sub_energy[:,0][mask_dict[name]], self.oversample[sim_type][name]))
                data_all_host.append(np.repeat(sub_host_energy[mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data_z0.append(sub_energy[:,0][mask_dict[name]])
                data_all_host.append(sub_host_energy[mask_dict[name]])
            #
            for i in range(0, np.sum(mask_dict[name])):
                infall_mask = (tlb[:len(sub_energy[mask_dict[name]][i])] <= data_dict[name]['first.infall.time.lb'][mask_dict[name]][i])
                if oversample:
                    data_infall.append(np.repeat(sub_energy[mask_dict[name]][i][infall_mask][-1], self.oversample[sim_type][name]))
                else:
                    data_infall.append(sub_energy[mask_dict[name]][i][infall_mask][-1])
        #
        d = dict()
        d['energy.z0'] = np.hstack(data_z0)
        d['energy.infall'] = np.hstack(data_infall)
        d['energy.all'] = data_all
        d['energy.all.tlb'] = data_all_time
        d['E.vir'] = np.hstack(data_all_host)
        #
        return d

    # Needs a lot of documenting
    def energies_model(self, data_dict, mask_dict, potential_dict_sim, potential_dict_model, time_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        # Import the relevant potentials
        from galpy.potential import DoubleExponentialDiskPotential # For disks
        from galpy.potential import TwoPowerSphericalPotential # For DM halos
        from galpy.potential import evaluatePotentials
        from astropy import units as u
        #
        # Set up the potential model
        fitting_data = pd.read_csv('/Users/isaiahsantistevan/simulation/orbit_data/fitting_param.csv', index_col=0)
        #
        # Set up empty dictionary/lists to save data to
        d = dict()
        data_z0 = []
        data_infall = []
        data_all = dict()
        data_all_time = dict()
        #
        tlb = time_dict['time'][-1] - np.flip(time_dict['time'])
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Set up the model host potential
            disk_outer = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_out'][name]*u.solMass/u.kpc**3, hr=fitting_data['r_out'][name]*u.kpc, hz=fitting_data['h_z'][name]*u.kpc)
            disk_inner = DoubleExponentialDiskPotential(amp=fitting_data['A_disk_in'][name]*u.solMass/u.kpc**3, hr=fitting_data['r_in'][name]*u.kpc, hz=fitting_data['h_z'][name]*u.kpc)
            halo_2p = TwoPowerSphericalPotential(amp=fitting_data['A_halo'][name]*u.solMass, a=fitting_data['a_halo'][name]*u.kpc, alpha=fitting_data['alpha'][name], beta=fitting_data['beta'][name])
            potential_two_power = disk_inner+disk_outer+halo_2p
            #
            # Set the global potential at 100 kpc
            d100 = 100/np.sqrt(2)
            d200m = data_dict[name]['host.radius'][0]/np.sqrt(2)
            phi_model_100kpc = evaluatePotentials(potential_two_power, d100*u.kpc, d100*u.kpc)
            #
            # Set the potential at z = 0
            phi_model_z0 = evaluatePotentials(potential_two_power, d100*u.kpc, d100*u.kpc) - evaluatePotentials(potential_two_power, d200m*u.kpc, d200m*u.kpc) - potential_dict_sim[name]['KE.at.Rvir']
            # Set the potential across all time as just the potential at z = 0
            phi_model_z = phi_model_z0*np.ones(potential_dict_model[name]['model.potential'].shape[1])
            #
            # Set up null array to save the normalized subhalo potentials to
            sub_pot_model = (-1)*np.ones(potential_dict_model[name]['model.potential'].shape)
            sub_energy_model = (-1)*np.ones(potential_dict_model[name]['model.potential'].shape)
            sub_pot_snaps_model = (-1)*np.ones(potential_dict_model[name]['model.potential'].shape, int)
            sub_pot_tlb_model = (-1)*np.ones(potential_dict_model[name]['model.potential'].shape)
            sub_kin_model = (-1)*np.ones(potential_dict_model[name]['model.potential'].shape)
            #
            # Loop through all of the satellites and calculate the potential the same way that I do in the simulations
            for j in range(0, sub_pot_model.shape[0]):
                # Create a mask for the subhalo data
                # Calculate the normalized subhalo energy
                sub_pot_model[j] = potential_dict_model[name]['model.potential'][j] - phi_model_100kpc + phi_model_z
                #
                # Keep which snapshots it has data for
                sub_pot_snaps_model[j] = np.flip(time_dict['index'])
                sub_pot_tlb_model[j] = tlb
                #
                # Calculate the total orbital energy
                sub_energy_model[j] = sub_pot_model[j] + 0.5*(data_dict[name]['v.tot.model'][j])**2
                sub_kin_model[j] =  0.5*(data_dict[name]['v.tot.model'][j])**2
            #
            # Save all energy data for the satellites.
            data_all[name] = sub_energy_model
            data_all_time[name] = sub_pot_tlb_model
            #
            # Save data for the z = 0 stuff
            if oversample:
                data_z0.append(np.repeat(sub_energy_model[:,0][mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data_z0.append(sub_energy_model[:,0][mask_dict[name]])
            #
            for i in range(0, np.sum(mask_dict[name])):
                infall_mask = (tlb[:len(sub_energy_model[mask_dict[name]][i])] <= data_dict[name]['first.infall.time.lb'][mask_dict[name]][i])
                if oversample:
                    data_infall.append(np.repeat(sub_energy_model[mask_dict[name]][i][infall_mask][-1], self.oversample[sim_type][name]))
                else:
                    data_infall.append(sub_energy_model[mask_dict[name]][i][infall_mask][-1])
            #
            # Save the data to the dictionary
            d['energy.z0'] = np.hstack(data_z0)
            d['energy.infall'] = np.hstack(data_infall)
            d['energy.all'] = data_all
            d['energy.all.tlb'] = data_all_time
        #
        return d

class SummaryDataPlot(SummaryDataSort):

    def __init__(self):
        """
        DESCRIPTION:
            Initializes the plotting class and sets important attributes.

        VARIABLES:


        NOTES:
            - Saves three attributes
                - self.colors is an array of hex color codes. All 14 are different
                  enough from one another to be distinct from one another to color-
                  blind people.
                - self.labels is a dictionary of plotting labels.
                - self.titles is a dictionary of plotting titles.
        """
        SummaryDataSort.__init__(self)
        #
        self.colors = ['#2f4f4f', '#006400', '#8b0000', '#000080', '#00ced1',\
                       '#ff8c00', '#c71585', '#7fff00', '#00fa9a', '#0000ff',\
                       '#ff00ff', '#1e90ff', '#f0e68c', '#ffc0cb']
        #
        self.labels = {'d.sim': 'd$_{\\rm peri,sim}$ [kpc]',\
                       'd.sim.min': 'd$_{\\rm peri,min,sim}$ [kpc]',\
                       'delta_d': '(d$_{\\rm peri,min}$ - d$_{\\rm peri,recent}$) [kpc]',\
                       'delta_d_frac': '($d_{\\rm peri,min}$ - $d_{\\rm peri,recent}$)/$d_{\\rm peri,recent}$',\
                       'delta_ell_frac': '($\\ell_{\\rm peri,min}$ - $\\ell_{\\rm peri,recent}$)/$\\ell_{\\rm peri,recent}$',\
                       'd.peri': 'd$_{\\rm peri}$ [kpc]',\
                       'd.peri.recent': 'd$_{\\rm peri, recent}$ [kpc]',\
                       'd.peri.min': 'd$_{\\rm peri, min}$ [kpc]',\
                       'd.peri.text': 'Pericenter distance [kpc]',\
                       'd.apo.text': 'Apocenter distance [kpc]',\
                       'd.model': 'd$_{\\rm peri,model}$ [kpc]',\
                       'd.z0': 'Host distance $d$ [kpc]',\
                       'delta.d.frac': '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$)/d$_{\\rm peri,sim}$',\
                       'delta.d': '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$) [kpc]',\
                       'delta.dapo.frac': '(d$_{\\rm apo,model}$ - d$_{\\rm apo,sim}$)/d$_{\\rm apo,sim}$',\
                       'delta.dapo': '(d$_{\\rm apo,model}$ - d$_{\\rm apo,sim}$) [kpc]',\
                       'delta.tapo': '(t$_{\\rm apo,model}$ - t$_{\\rm apo,sim}$) [kpc]',\
                       'v.tan': 'Tangential velocity [km s$^{-1}$]',\
                       'v.rad': 'v$_{\\rm rad}(z = 0)$ [km s$^{-1}$]',\
                       'v.tot': 'Total velocity [km s$^{-1}$]',\
                       'delta.v': '($v_{\\rm peri,model}$ - $v_{\\rm peri,sim}$) [kpc]',\
                       't.sim': 't$_{\\rm peri,lb,sim}$ [Gyr]',\
                       't.sim.min': 't$_{\\rm peri,min,lb,sim}$ [Gyr]',\
                       'delta_t': '$t^{\\rm lb}_{\\rm peri,lb,min}$ - $t^{\\rm lb}_{\\rm peri,lb,recent}$ [Gyr]',\
                       'delta_t_frac': '(t$^{\\rm lb}_{\\rm peri,min}$ - t$^{\\rm lb}_{\\rm peri,lb,recent}$)/t$^{\\rm lb}_{\\rm peri,lb,recent}$',\
                       't.model': 't$_{\\rm peri,lb,model}$ [Gyr]',\
                       't.peri': 't$_{\\rm peri,lb}$ [Gyr]',\
                       't.peri.recent': 't$_{\\rm peri,recent,lb}$ [Gyr]',\
                       't.peri.min': 't$_{\\rm peri,min,lb}$ [Gyr]',\
                       't.peri.text': 'Pericenter lookback time [Gyr]',\
                       't.infall': 't$_{\\rm infall,lb}$ [Gyr]',\
                       't.infall.any': 't$_{\\rm infall,any,lb}$ [Gyr]',\
                       't.infall.text': 'Infall lookback time [Gyr]',\
                       't.infall.diff': '(t$_{\\rm infall,any,lb}$ - t$_{\\rm infall,lb}$) [Gyr]',\
                       't.lb': 'Lookback time [Gyr]',\
                       'delta_t_infall': '(t$_{\\rm infall,model,lb}$ - t$_{\\rm infall,sim,lb}$) [Gyr]',\
                       'delta.t.frac': '(t$_{\\rm peri,model}$ - t$_{\\rm peri,sim}$)/t$_{\\rm peri,sim}$',\
                       'delta.t': '(t$_{\\rm peri,model}$ - t$_{\\rm peri,sim}$) [Gyr]',\
                       'N.sim': 'N$_{\\rm peri,sim}$',\
                       'N.model': 'N$_{\\rm peri,model}$',\
                       'N.peri': 'N$_{\\rm peri}$',\
                       'N.peri.text': 'Pericenter Number',\
                       'N.delta': 'N$_{\\rm model}$ - N$_{\\rm sim}$',\
                       'M.star.z0': '$M_{\\rm star}$ [$M_{\\odot}$]',\
                       'M.star.peak': 'log$_{\\rm 10}$[$M_{\\rm star, peak}$/$M_{\\odot}$]',\
                       'M.halo.z0': 'log$_{\\rm 10}$[$M_{\\rm halo}(z = 0)$/$M_{\\odot}$]',\
                       'M.halo.peak': '$M_{\\rm halo, peak}$ [$M_{\\odot}$]',\
                       'KE.max.sim': 'KE$_{\\rm max,sim}$ [10$^4$ km$^2$/s$^2$]',\
                       'KE.peri.sim': 'KE$_{\\rm peri,sim}$ [10$^4$ km$^2$/s$^2$]',\
                       'KE.z0.sim': 'KE$_{\\rm sim}(z = 0)$ [10$^4$ km$^2$/s$^2$]',\
                       'E.tot': '$E$ [10$^4$ km$^2$/s$^2$]',\
                       'E.tot.sim': 'E$_{\\rm sim}(z = 0)$ [10$^4$ km$^2$/s$^2$]',\
                       'E.tot.model': 'E$_{\\rm model}(z = 0)$ [10$^4$ km$^2$/s$^2$]',\
                       'L.tot': '$\\ell$ [10$^4$ kpc km s$^{-1}$]',\
                       'L.tot.sim': 'L$_{\\rm sim}(z = 0)$ [10$^4$ kpc km s$^{-1}$]',\
                       'L.tot.model': 'L$_{\\rm model}(z = 0)$ [10$^4$ kpc km s$^{-1}$]',\
                       'L.diff': '$L(z=0) - L_{\\rm infall,MW}$ [10$^4$ kpc km s$^{-1}$]',\
                       'L.frac': '($L(z=0) - L_{\\rm infall,MW}$)/$L_{\\rm infall,MW}$',\
                       'ecc': 'Eccentricity',\
                       'ecc.model': 'Model Ecccentricity',\
                       'ecc.delta': '$e_{\\rm model} - e_{\\rm sim}$',\
                       'ecc.frac': '$(e_{\\rm model} - e_{\\rm sim})/e_{\\rm sim}$',\
                       'period': 'Orbital Period [Gyr]',\
                       'period.model': 'Model Orbital Period [Gyr]',\
                       'period.delta': '$T_{\\rm model} - T_{\\rm sim}$ [Gyr]',\
                       'dadr': '$|da/dr|$ [$s^{-2}$]',\
                       'dadr.log': 'log$_{\\rm 10}$($|da/dr|$)',\
                       'dadr.min': '$|da/dr|_{dperi,min}$ [$s^{-2}$]',\
                       'dadr.max': '$|da/dr|_{max}$ [$s^{-2}$]',\
                       'dadr.min.t': '$t_{\\rm da/dr, dperi, min}$ [Gyr]',\
                       'dadr.max.t': '$t_{\\rm da/dr, max}$ [Gyr]',\
                       'dadr.diff': '$|da/dr|_{dperi,min}-|da/dr|_{max}$ [$s^{-2}$]',\
                       'dadr.diff.log': 'log $|da/dr|_{dperi,min}$ - log $|da/dr|_{max}$ [$s^{-2}$]',\
                       'dadr.diff.t': '$t_{\\rm da/dr,dperi,min} - t_{\\rm da/dr,max}$ [Gyr]',\
                       'dadr.frac': '$(|da/dr|_{dperi,min}-|da/dr|_{max})/|da/dr|_{dperi,min}$',\
                       #
                       # I don't really use the ones below here...
                       'delta_dmin_dm_v_star': '(d$_{\\rm min,dm}$ - d$_{\\rm min,star}$) [kpc]',\
                       'delta_drec_dm_v_star': '(d$_{\\rm recent,dm}$ - d$_{\\rm recent,star}$) [kpc]',\
                       'd_min_dm': 'd$_{\\rm min,dm}$ [kpc]',\
                       'd_recent_dm': 'd$_{\\rm recent,dm}$ [kpc]'}
        #
        # I don't really make use of a lot of these titles, but I like the idea of using them someday.
        self.titles = {'d.sim': 'Recent Minimum Distances',\
                       'd.model': 'Recent Minimum Distances',\
                       'd.z0': 'Present-day Distances',\
                       't.sim': 'Recent Minimum Distance Lookback Times',\
                       't.model': 'Recent Minimum Distance Lookback Times',\
                       't.infall': 'Host Infall Lookback Times',\
                       'N.sim': 'Pericenters',\
                       'N.model': 'Pericenters',\
                       'N.delta': 'Pericenters'}

    def binning_scheme(self, x, xtype, binsize, binedges=None):
        """
        DESCRIPTION:
            Function that makes bins... finish later.
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
                if minn < 0:
                    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
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
            Function that finds median and scatter... finish later.
        """
        if 'M.' in xtype:
            x = np.log10(x)
        if 'M.' in ytype:
            y = np.log10(y)
        #
        if 'N.' not in ytype:
            #
            onesigp = 84.13
            onesigm = 15.87
            twosigp = 100 # 99.86
            twosigm = 0   # 0.14
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

    def scatter_plot(self, x, y, xtype, ytype, file_path_and_name, x_out=None, y_out=None, limits=None, title=None):
        """
        DESCRIPTION:
            Plots two quantities in a scatter plot.

        VARIABLES:
            - x                  : 1D array
            - y                  : 1D array
            - x_out              : 1D array
            - y_out              : 1D array
            - xtype              : string
            - ytype              : string
            - limits             : tuple of two tuples
            - title              : string
            - file_path_and_name : string

        NOTES:
            - If plotting mass quantities, this takes the log first.
            - If comparing pericenter distances or times, this plots a 1-to-1
              line from the bottom left corner to the top right.
        """
        if 'M.' in xtype:
            x = np.log10(x)
            if x_out:
                x_out = np.log10(x_out)
        if 'M.' in ytype:
            y = np.log10(y)
            if y_out:
                y_out = np.log10(y_out)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x, y, color='k', s=50, marker='x', alpha=0.5)
        if x_out:
            ax.scatter(x_out, y_out, color='r', s=50, marker='x', alpha=0.5)
        if ('.sim' in xtype) & ('.sim' in ytype or '.model' in ytype):
            ax.set_xlim(left=limits[0], right=limits[1])
            ax.set_ylim(bottom=limits[0], top=limits[1])
            ax.plot([0, 1], [0, 1], linestyle=':', color='k', transform=ax.transAxes)
        elif limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(self.labels[xtype], fontsize=28)
        plt.ylabel(self.labels[ytype], fontsize=28)
        if title:
            plt.title(self.titles[title], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()
        pass

    def median_plot(self, x, y, xtype, ytype, binsize, file_path_and_name, binedges=None, limits=None, title=None, hl=False, w_scatter=False, axis_labels=None):
        """
        DESCRIPTION:
            Bins the x-axis quantity and plots either the mean or median, along
            with the standard deviation or 68% scatter, of the y-axis quantity.

        VARIABLES:
            - x                  : 1D array
            - y                  : 1D array
            - xtype              : string
            - ytype              : string
            - binsize            : float or int
            - binedges           : tuple
            - limits             : tuple of two tuples
            - title              : string
            - file_path_and_name : string

        NOTES:
            - If plotting mass quantities, this takes the log first.
            - If the x-axis quantity is an integer quantity (pericenter number
              in particular), script bins differently than if not an integer
              quantity.
            - If the y-axis quantity is an integer quantity, then the method
              calculates the mean and standard deviation.
            - If the y-axis quantity is not an integer quantity, then the method
              calculates the median and 68% scatter.
        """
        #
        # Create the bins to use in finding the median + scatter and for plotting
        binss, half_binss = self.binning_scheme(x=x, xtype=xtype, binedges=binedges, binsize=binsize)
        #
        # Calculate the median or mean + scatters
        med, upper, lower, highest, lowest = self.median_and_scatter(x=x, y=y, xtype=xtype, ytype=ytype, bins=binss)
        #
        if w_scatter:
            ls = '-s'
        else:
            ls = '-'
        f, ax = plt.subplots(figsize=(11, 8))
        #
        if axis_labels:
            ax.set_xlabel(axis_labels[0], fontsize=28)
            ax.set_ylabel(axis_labels[1], fontsize=28)
        else:
            ax.set_xlabel(self.labels[xtype], fontsize=28)
            ax.set_ylabel(self.labels[ytype], fontsize=28)
        if title:
            ax.set_title(title, fontsize=24)
        if 'M.' in xtype and 'M.' not in ytype:
            plt.plot(10**(binss[:-1]+half_binss), med, ls, color=self.colors[1], markersize=10, alpha=0.5)
            plt.fill_between(10**(binss[:-1]+half_binss), upper, lower, color=self.colors[1], alpha=0.3)
            plt.fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=self.colors[1], alpha=0.15)
            ax.set_xscale('log')
            ax.set_yscale('linear')
            if limits:
                plt.xlim(10**(limits[0][0]), 10**(limits[0][1]))
                plt.ylim(limits[1])
            if hl:
                plt.hlines(y=0, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), color='k', linestyles='dotted', alpha=0.5)
            #
            if 't.' in ytype:
                # Instantiate the cosmology class and run this method first to set up scalefactors
                cc = ut.cosmology.CosmologyClass()
                red = np.array([0, 1])
                cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
                #
                axis_2_label = 'redshift'
                axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
                axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
                axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
                ax2 = ax.twinx()
                ax2.set_xscale('log')
                ax2.set_yscale('linear')
                ax2.set_yticks(axis_2_tick_locations)
                ax2.set_yticklabels(axis_2_tick_labels, fontsize=28)
                ax2.set_ylim(limits[1])
                ax2.set_ylabel(axis_2_label, labelpad=9)
                ax2.tick_params(pad=3)
        #
        elif 'M.' in xtype and 'M.' in ytype:
            plt.plot(10**(binss[:-1]+half_binss), 10**med, ls, color=self.colors[1], markersize=10, alpha=0.5)
            plt.fill_between(10**(binss[:-1]+half_binss), 10**upper, 10**lower, color=self.colors[1], alpha=0.3)
            plt.fill_between(10**(binss[:-1]+half_binss), 10**highest, 10**lowest, color=self.colors[1], alpha=0.15)
            plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
            if limits:
                plt.xlim(10**(limits[0][0]), 10**(limits[0][1]))
                plt.ylim(10**(limits[1][0]), 10**(limits[1][1]))
            ax.set_xscale('log')
            ax.set_yscale('log')
        #
        else:
            plt.plot(binss[:-1]+half_binss, med, ls, color=self.colors[1], markersize=10, alpha=0.5)
            plt.fill_between(binss[:-1]+half_binss, upper, lower, color=self.colors[1], alpha=0.3)
            plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=self.colors[1], alpha=0.15)
            ax.set_xscale('linear')
            ax.set_yscale('linear')
            if limits:
                plt.xlim(limits[0])
                plt.ylim(limits[1])
            if hl:
                plt.hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], color='k', linestyles='dotted', alpha=0.5)
            #
            if 't.' in xtype:
                # Instantiate the cosmology class and run this method first to set up scalefactors
                cc = ut.cosmology.CosmologyClass()
                red = np.array([0, 1])
                cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
                #
                axis_2_label = 'redshift'
                axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
                axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
                axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
                ax2 = ax.twiny()
                ax2.set_xscale('linear')
                ax2.set_yscale('linear')
                ax2.set_xticks(axis_2_tick_locations)
                ax2.set_xticklabels(axis_2_tick_labels, fontsize=28)
                ax2.set_xlim(limits[0])
                ax2.set_xlabel(axis_2_label, labelpad=9)
                ax2.tick_params(pad=3)
            #
            if 't.' in ytype:
                # Instantiate the cosmology class and run this method first to set up scalefactors
                cc = ut.cosmology.CosmologyClass()
                red = np.array([0, 1])
                cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
                #
                axis_2_label = 'redshift'
                axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
                axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
                axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
                ax2 = ax.twinx()
                ax2.set_xscale('linear')
                ax2.set_yscale('linear')
                ax2.set_yticks(axis_2_tick_locations)
                ax2.set_yticklabels(axis_2_tick_labels, fontsize=28)
                ax2.set_ylim(limits[1])
                ax2.set_ylabel(axis_2_label, labelpad=9)
                ax2.tick_params(pad=3)
        #
        ax.tick_params(axis='both', which='major', labelsize=28)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def median_plot_mult(self, x, y, xtype, ytype, labels, binsize, file_path_and_name, binedges=None, limits=None, title=None, legend_on=True, hl=False, w_scatter=False):
        """
        DESCRIPTION:
            Bins the x-axis quantity and plots either the mean or median, along
            with the standard deviation or 68% scatter, of the y-axis quantity.

        VARIABLES:
            - x                  : list
            - y                  : list
            - xtype              : list
            - ytype              : list
            - labels             : list
            - binsize            : float or int
            - limits             : tuple of two tuples
            - title              : string
            - file_path_and_name : string

        NOTES:
            - If plotting mass quantities, this takes the log first.
            - If the x-axis quantity is an integer quantity (pericenter number
              in particular), script bins differently than if not an integer
              quantity.
            - If the y-axis quantity is an integer quantity, then the method
              calculates the mean and standard deviation.
            - If the y-axis quantity is not an integer quantity, then the method
              calculates the median and 68% scatter.
        """
        if len(x) == 2:
            colorss = ['#006400', '#000080']
        else:
            colorss = self.colors
        #
        f, ax = plt.subplots(figsize=(11, 8))
        #
        for j in range(0, len(x)):
            #
            # Create the bins to use in finding the median + scatter and for plotting
            binss, half_binss = self.binning_scheme(x=x[j], xtype=xtype[j], binedges=binedges, binsize=binsize)
            #
            # Calculate the median or mean + scatters
            med, upper, lower, highest, lowest = self.median_and_scatter(x=x[j], y=y[j], xtype=xtype[j], ytype=ytype[j], bins=binss)
            #
            if w_scatter:
                ls = '-s'
            else:
                ls = '-'
            # PLOTTING
            if 'M.' in xtype[j] and 'M.' not in ytype[j]:
                plt.plot(10**(binss[:-1]+half_binss), med, ls, color=colorss[j], markersize=10, alpha=0.5, label=labels[j])
                plt.fill_between(10**(binss[:-1]+half_binss), upper, lower, color=colorss[j], alpha=0.3)
                plt.fill_between(10**(binss[:-1]+half_binss), highest, lowest, color=colorss[j], alpha=0.15)
                if hl:
                    plt.hlines(y=0, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), color='k', linestyles='dotted', alpha=0.5)
            elif 'M.' in xtype[j] and 'M.' in ytype[j]:
                plt.plot(10**(binss[:-1]+half_binss), 10**med, ls, color=colorss[j], markersize=10, alpha=0.5, label=labels[j])
                plt.fill_between(10**(binss[:-1]+half_binss), 10**upper, 10**lower, color=colorss[j], alpha=0.3)
                plt.fill_between(10**(binss[:-1]+half_binss), 10**highest, 10**lowest, color=colorss[j], alpha=0.15)
                ax.axhline(y=3*10**7, xmin=10**(8), xmax=10**(9), color='k', linestyle=':')
                #
            else:
                plt.plot(binss[:-1]+half_binss, med, ls, color=colorss[j], markersize=10, alpha=0.5, label=labels[j])
                plt.fill_between(binss[:-1]+half_binss, upper, lower, color=colorss[j], alpha=0.3)
                plt.fill_between(binss[:-1]+half_binss, highest, lowest, color=colorss[j], alpha=0.15)
                if hl:
                    plt.hlines(y=0, xmin=limits[0][0], xmax=limits[0][1], color='k', linestyles='dotted', alpha=0.5)
        if 'M.' in xtype[0]:
            plt.xscale('log')
        if 'M.' in ytype[0]:
            plt.yscale('log')
        if limits:
            if 'M.' in xtype[0] and 'M.' not in ytype[0]:
                plt.xlim(10**(limits[0][0]), 10**(limits[0][1]))
                plt.ylim(limits[1])
            elif 'M.' in xtype[0] and 'M.' in ytype[0]:
                plt.xlim(10**(limits[0][0]), 10**(limits[0][1]))
                plt.ylim(10**(limits[1][0]), 10**(limits[1][1]))
            else:
                plt.xlim(limits[0])
                plt.ylim(limits[1])
        if 't.' in ytype[0]:
            # Instantiate the cosmology class and run this method first to set up scalefactors
            cc = ut.cosmology.CosmologyClass()
            red = np.array([0, 1])
            cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
            #
            axis_2_label = 'redshift'
            axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
            axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
            axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
            ax2 = ax.twinx()
            if 'M.' in xtype[0]:
                ax2.set_xscale('log')
            else:
                ax2.set_xscale('linear')
            ax2.set_yscale('linear')
            ax2.set_yticks(axis_2_tick_locations)
            ax2.set_yticklabels(axis_2_tick_labels, fontsize=28)
            ax2.set_ylim(limits[1])
            ax2.set_ylabel(axis_2_label, labelpad=9)
            ax2.tick_params(pad=3)
        if 't.' in xtype[0]:
            # Instantiate the cosmology class and run this method first to set up scalefactors
            cc = ut.cosmology.CosmologyClass()
            red = np.array([0, 1])
            cc.convert_time(time_name_get='time.lookback', time_name_input='redshift', values=red)
            #
            axis_2_label = 'redshift'
            axis_2_tick_labels = ['6', '3', '2', '1', '0.7', '0.5', '0.3', '0.2', '0.1', '0']
            axis_2_tick_values = [float(v) for v in axis_2_tick_labels]
            axis_2_tick_locations = cc.convert_time('time.lookback', 'redshift', axis_2_tick_values)
            ax2 = ax.twiny()
            if 'M.' in ytype[0]:
                ax2.set_yscale('log')
            else:
                ax2.set_yscale('linear')
            ax2.set_xscale('linear')
            ax2.set_xticks(axis_2_tick_locations)
            ax2.set_xticklabels(axis_2_tick_labels, fontsize=28)
            ax2.set_xlim(limits[0])
            ax2.set_xlabel(axis_2_label, labelpad=9)
            ax2.tick_params(pad=3)
        ax.set_xlabel(self.labels[xtype[0]], fontsize=28)
        ax.set_ylabel(self.labels[ytype[0]], fontsize=28)
        if legend_on:
            ax.legend(prop={'size': 24}, loc='best')
        if title:
            plt.title(title, fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=28)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def median_plot_mult_one_scatter(self, x, y, xtype, ytype, labels, binsize, file_path_and_name, limits=None, title=None):
        """
        DESCRIPTION:
            Bins the x-axis quantity and plots either the mean or median, along
            with the standard deviation or 68% scatter, of the y-axis quantity.

        VARIABLES:
            - x                  : list
            - y                  : list
            - xtype              : list
            - ytype              : list
            - labels             : list
            - binsize            : float or int
            - limits             : tuple of two tuples
            - title              : string
            - file_path_and_name : string

        NOTES:
            - If plotting mass quantities, this takes the log first.
            - If the x-axis quantity is an integer quantity (pericenter number
              in particular), script bins differently than if not an integer
              quantity.
            - If the y-axis quantity is an integer quantity, then the method
              calculates the mean and standard deviation.
            - If the y-axis quantity is not an integer quantity, then the method
              calculates the median and 68% scatter.
        """
        colorss = ['#006400', '#000080']
        #
        x_all = np.hstack(x)
        y_all = np.hstack(y)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        #
        if 'M.' in xtype[0]:
            x_all = np.log10(x_all)
            x[0] = np.log10(x[0])
            x[1] = np.log10(x[1])
        if 'M.' in ytype[0]:
            y_all = np.log10(y_all)
            y[0] = np.log10(y[0])
            y[1] = np.log10(y[1])
        #
        if 'N.' not in xtype[0] and 'N.' not in ytype[0]:
            minn = binsize*np.floor(np.min(x_all)/binsize)
            maxx = binsize*np.ceil(np.max(x_all)/binsize)
            if minn < 0:
                bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
            else:
                bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
            bins = np.linspace(minn, maxx, bin_num)
            half_bin = (bins[1]-bins[0])/2
            #
            onesigp = 84.13
            onesigm = 15.87
            twosigp = 100
            twosigm = 0
            #
            med_all = np.zeros(len(bins)-1)
            med_1 = np.zeros(len(bins)-1)
            med_2 = np.zeros(len(bins)-1)
            #
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            lowest = np.zeros(len(bins)-1)
            highest = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask_all = (x_all >= bins[i]) & (x_all <= bins[i+1])
                med_all[i] = np.nanmedian(y_all[mask_all])
                upper[i] = np.nanpercentile(y_all[mask_all], onesigp)
                lower[i] = np.nanpercentile(y_all[mask_all], onesigm)
                highest[i] = np.nanpercentile(y_all[mask_all], twosigp)
                lowest[i] = np.nanpercentile(y_all[mask_all], twosigm)
                #
                mask_1 = (x[0] >= bins[i]) & (x[0] <= bins[i+1])
                med_1[i] = np.nanmedian(y[0][mask_1])
                #
                mask_2 = (x[1] >= bins[i]) & (x[1] <= bins[i+1])
                med_2[i] = np.nanmedian(y[1][mask_2])
        #
        if 'N.' in xtype[0] and 'N.' not in ytype[0]:
            minn = int(binsize*np.floor(np.min(x_all)/binsize))-0.5
            maxx = int(binsize*np.ceil(np.max(x_all)/binsize))+0.5
            bin_num = int((np.abs(maxx)+np.abs(minn))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            onesigp = 84.13
            onesigm = 15.87
            twosigp = 100
            twosigm = 0
            #
            med_all = np.zeros(len(bins)-1)
            med_1 = np.zeros(len(bins)-1)
            med_2 = np.zeros(len(bins)-1)
            #
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            lowest = np.zeros(len(bins)-1)
            highest = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask_all = (x_all >= bins[i]) & (x_all <= bins[i+1])
                med_all[i] = np.nanmedian(y_all[mask_all])
                upper[i] = np.nanpercentile(y_all[mask_all], onesigp)
                lower[i] = np.nanpercentile(y_all[mask_all], onesigm)
                highest[i] = np.nanpercentile(y_all[mask_all], twosigp)
                lowest[i] = np.nanpercentile(y_all[mask_all], twosigm)
                #
                mask_1 = (x[0] >= bins[i]) & (x[0] <= bins[i+1])
                med_1[i] = np.nanmedian(y[0][mask_1])
                #
                mask_2 = (x[1] >= bins[i]) & (x[1] <= bins[i+1])
                med_2[i] = np.nanmedian(y[1][mask_2])
        #
        if 'N.' not in xtype[0] and 'N.' in ytype[0]:
            minn = binsize*np.floor(np.min(x_all)/binsize)
            maxx = binsize*np.ceil(np.max(x_all)/binsize)
            if minn < 0:
                bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
            else:
                bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
            bins = np.linspace(minn, maxx, bin_num)
            half_bin = (bins[1]-bins[0])/2
            #
            twosigp = 100
            twosigm = 0
            #
            means_all = np.zeros(len(bins)-1)
            means_1 = np.zeros(len(bins)-1)
            means_2 = np.zeros(len(bins)-1)
            #
            scatter = np.zeros(len(bins)-1)
            lowest = np.zeros(len(bins)-1)
            highest = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask_all = (x_all >= bins[i]) & (x_all <= bins[i+1])
                means_all[i] = np.nanmean(y_all[mask_all])
                scatter[i] = np.nanstd(y_all[mask_all])
                highest[i] = np.nanpercentile(y_all[mask_all], twosigp)
                lowest[i] = np.nanpercentile(y_all[mask_all], twosigm)
                #
                mask_1 = (x[0] >= bins[i]) & (x[0] <= bins[i+1])
                means_1 = np.nanmean(y[0][mask_1])
                #
                mask_2 = (x[1] >= bins[i]) & (x[1] <= bins[i+1])
                means_2 = np.nanmean(y[0][mask_2])
            #
            upper = means_all+scatter
            lower = means_all-scatter
            med_all = means_all
            med_1 = means_1
            med_2 = means_2
        #
        if 'N.' in xtype[0] and 'N.' in ytype[0]:
            minn = int(binsize*np.floor(np.min(x_all)/binsize))-0.5
            maxx = int(binsize*np.ceil(np.max(x_all)/binsize))+0.5
            bin_num = int((np.abs(maxx)+np.abs(minn))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            half_bin = (bins[1]-bins[0])/2
            #
            twosigp = 100
            twosigm = 0
            #
            means_all = np.zeros(len(bins)-1)
            means_1 = np.zeros(len(bins)-1)
            means_2 = np.zeros(len(bins)-1)
            #
            scatter = np.zeros(len(bins)-1)
            lowest = np.zeros(len(bins)-1)
            highest = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask_all = (x_all >= bins[i]) & (x_all <= bins[i+1])
                means_all[i] = np.nanmean(y_all[mask_all])
                scatter[i] = np.nanstd(y_all[mask_all])
                highest[i] = np.nanpercentile(y_all[mask_all], twosigp)
                lowest[i] = np.nanpercentile(y_all[mask_all], twosigm)
                #
                mask_1 = (x[0] >= bins[i]) & (x[0] <= bins[i+1])
                means_1 = np.nanmean(y[0][mask_1])
                #
                mask_2 = (x[1] >= bins[i]) & (x[1] <= bins[i+1])
                means_2 = np.nanmean(y[0][mask_2])
            #
            med_all = means_all
            med_1 = means_1
            med_2 = means_2
        #
        plt.plot(bins[:-1]+half_bin, med_all, color='k', markersize=10, alpha=0.8)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color='k', alpha=0.2)
        plt.fill_between(bins[:-1]+half_bin, highest, lowest, color='k', alpha=0.1)
        plt.plot(bins[:-1]+half_bin, med_1, color=colorss[0], markersize=10, alpha=0.5, label=labels[0])
        plt.plot(bins[:-1]+half_bin, med_2, color=colorss[1], markersize=10, alpha=0.5, label=labels[1])
        #
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(self.labels[xtype[0]], fontsize=28)
        plt.ylabel(self.labels[ytype[0]], fontsize=28)
        plt.legend(prop={'size': 18}, loc='best')
        if title:
            plt.title(self.titles[title], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def plot_hist(self, x, xtype, binsize, file_path_and_name, pdf=False, xlimits=None, title=None, binedges=None, x_labels=None):
        """
        DESCRIPTION:
            Plots a histogram of a given property.

        VARIABLES:
            - x                  : 1D array
            - xtype              : string
            - binsize            : float or int
            - xlimits            : tuple
            - pdf                : boolean
            - title              : string
            - file_path_and_name : string

        NOTES:
            - If plotting mass quantities, this takes the log first.
            - Bins things slightly differently if the x-axis quantity is an
              integer quantity.
            - Plots either a PDF or regular histogram depending on what 'pdf'
              is set to.
        """
        if pdf:
            y_label = 'Probability'
        else:
            y_label = 'N'
        #
        if 'M.' in xtype:
            x = np.log10(x)
        #
        # Create the bins to use in finding the median + scatter and for plotting
        binss, half_binss = self.binning_scheme(x=x, xtype=xtype, binedges=binedges, binsize=binsize)
        #
        # Plot the data
        plt.figure(figsize=(10, 8))
        #
        if 'N.' not in xtype:
            #
            # Calculate the scatter
            onesigp = 84.13
            onesigm = 15.87
            sigma_one_op = np.nanpercentile(x, onesigp)
            sigma_one_om = np.nanpercentile(x, onesigm)
            #
            y_med = np.max(np.histogram(x, binss, density=pdf)[0])*1.1
            #
            plt.hist(x, binss, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            plt.errorbar(np.median(x), y_med, xerr=np.array([[np.median(x)-sigma_one_om],[sigma_one_op-np.median(x)]]), color='k', lw=5, capsize=0)
            plt.scatter(np.median(x), y_med, s=250, marker='s', c='k')
        #
        elif 'N.' in xtype:
            #
            y_mean = np.max(np.histogram(x, binss, density=pdf)[0])*1.1
            #
            plt.hist(x, binss, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            #plt.errorbar(np.mean(x), y_mean, xerr=np.array([[2*np.std(x)],[2*np.std(x)]]), color='k', lw=5, capsize=0, alpha=0.3)
            plt.errorbar(np.mean(x), y_mean, xerr=np.array([[np.std(x)],[np.std(x)]]), color='k', lw=5, capsize=0)
            plt.scatter(np.mean(x), y_mean, s=250, marker='s', c='k')
        #
        plt.xlim(xlimits)
        if x_labels:
            plt.xlabel(x_labels, fontsize=28)
        else:
            plt.xlabel(self.labels[xtype], fontsize=28)
        plt.ylabel(y_label, fontsize=34)
        if title:
            plt.title(title, fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=28)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def plot_hist_mult(self, x, xtype, labels, binsize, file_path_and_name, med_location=None, pdf=False, xlimits=None, title=None, legend_on=True, leg_loc=None, binedges=None):
        """
        DESCRIPTION:
            Plots a histogram of a given property.

        VARIABLES:
            - x                  : list
            - xtype              : list
            - binsize            : float or int
            - xlimits            : tuple
            - pdf                : boolean
            - title              : string
            - labels             : list
            - file_path_and_name : string

        NOTES:
            - If plotting mass quantities, this takes the log first.
            - Bins things slightly differently if the x-axis quantity is an
              integer quantity.
            - Plots either a PDF or regular histogram depending on what 'pdf'
              is set to.
        """
        if len(x) == 2:
            colorss = ['#006400', '#000080']
        else:
            colorss = self.colors
        #
        if pdf:
            y_label = 'Probability'
        else:
            y_label = 'N'
        #
        # Plot the data
        plt.figure(figsize=(10, 8))
        #
        for i in range(0, len(x)):
            #
            if 'M.' in xtype[i]:
                x[i] = np.log10(x[i])
            #
            # Create the bins to use in finding the median + scatter and for plotting
            binss, half_binss = self.binning_scheme(x=x[i], xtype=xtype[i], binedges=binedges, binsize=binsize)
            #
            if 'N.' not in xtype[i]:
                #
                # Calculate the scatter
                onesigp = 84.13
                onesigm = 15.87
                sigma_one_op = np.nanpercentile(x[i], onesigp)
                sigma_one_om = np.nanpercentile(x[i], onesigm)
                #
                if med_location:
                    y_med = med_location[i]
                else:
                    y_med = np.max(np.histogram(x[i], binss, density=pdf)[0])*1.1
                #
                plt.hist(x[i], binss, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
                plt.errorbar(np.median(x[i]), y_med, xerr=np.array([[np.median(x[i])-sigma_one_om],[sigma_one_op-np.median(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.8)
                plt.scatter(np.median(x[i]), y_med, s=250, marker='s', c=colorss[i], alpha=0.8)
            #
            elif 'N.' in xtype[i]:
                #
                if med_location:
                    y_mean = med_location[i]
                else:
                    y_mean = np.max(np.histogram(x[i], binss, density=pdf)[0])*1.1
                #
                plt.hist(x[i], binss, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
                #plt.errorbar(np.mean(x[i]), y_mean, xerr=np.array([[2*np.std(x[i])],[2*np.std(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.3)
                plt.errorbar(np.mean(x[i]), y_mean, xerr=np.array([[np.std(x[i])],[np.std(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.8)
                plt.scatter(np.mean(x[i]), y_mean, s=250, marker='s', c=colorss[i])
        #
        plt.xlim(xlimits)
        plt.xlabel(self.labels[xtype[0]], fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        if legend_on:
            if leg_loc:
                plt.legend(prop={'size': 18}, loc=leg_loc)
            else:
                plt.legend(prop={'size': 18}, loc='best')
        if title:
            plt.title(title, fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()
