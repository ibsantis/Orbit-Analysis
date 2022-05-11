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
                           'all_energy': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', \
                                          'Romeo', 'Juliet', 'Thelma', 'Louise'],
                            #
                           'iso': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', 'm12z'], \
                           #
                           'iso_no_z': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w'], \
                           #
                           # Don't really need this selection anymore
                           'iso_dmo': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12w'], \
                           #
                           'lg': ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus'], \
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

    def data_read(self, directory, sim_type='baryon', hosts='all'):
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
                - Each key in the sub-dictionaries is given in summary_data.py
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
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

    def data_mask_nperi(self, dictionary, nperi, hosts='all'):
        """
        OLD METHOD:
        ===========
        This was primarily used to create masks for the satellites that had
        experienced 0, 1, and > 1 pericenters. I used this in previous plots
        that are no longer included in Paper I, so it would be okay to eventually
        delete this method.
        """
        # Set up a dictionary to save the masks to
        mask_dict = dict()
        #
        # Depending on the pericenter number you are interested in, create a
        # dictionary of masks, similar to data_mask and data_mask_apo
        if nperi == 0:
            for name in self.host_names[hosts]:
                mask_dict[name] = dictionary[name]['infall.check']*(~dictionary[name]['pericenter.check.sim'])
        #
        if nperi == 1:
            for name in self.host_names[hosts]:
                mask_dict[name] = np.zeros(len(dictionary[name]['infall.check']), bool)
                for i in range(0, len(mask_dict[name])):
                    peri_mask = (dictionary[name]['pericenter.dist.sim'][i] != -1)
                    if (dictionary[name]['infall.check'][i]) and (np.sum(peri_mask) == 1):
                        mask_dict[name][i] = True
        #
        if nperi > 1:
            for name in self.host_names[hosts]:
                mask_dict[name] = np.zeros(len(dictionary[name]['infall.check']), bool)
                for i in range(0, len(mask_dict[name])):
                    peri_mask = (dictionary[name]['pericenter.dist.sim'][i] != -1)
                    if (dictionary[name]['infall.check'][i]) and (np.sum(peri_mask) > 1):
                        mask_dict[name][i] = True
        #
        return mask_dict

    def halo_id(self, data_dict, mask_dict, hosts='all'):
        """
        DESCRIPTION:
            Given the data and mask dictionaries generated by the methods above,
            return another dictionary to list various identifiers for the
            satellites to better distinguish them. This is purely a diagnostic
            tool.
            ** Might want to think about incorporating this into the data_read()
               function, but this also requires the masking dictionary. The
               masking dictionary is not entirely necessary though... **

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

    def delta_nperi(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Calculates the difference in the number of pericenters a subhalo
            experiences between the model and the simulation.

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
        # Set up an empty list to append values to.
        data = []
        #
        if oversample == True:
            # Loop through hosts and append data to the list
            for name in self.host_names[hosts]:
                data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]],self.oversample[sim_type][name]) - \
                             np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]],self.oversample[sim_type][name]))
        #
        elif oversample == False:
            # Loop through hosts and append data to the list
            for name in self.host_names[hosts]:
                data.append(data_dict[name]['N.peri.galpy'][mask_dict[name]] - \
                            data_dict[name]['N.peri.sim'][mask_dict[name]])
        #
        return np.hstack(data)

    # optimized
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
        # Quick label fix if interested in the model property
        if selection == 'model':
            selection = 'galpy' # Eventually, I'm going to re-run the data with the "galpy" label replaced with "model"
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

    # optimized
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
        # Quick fix on labels
        if selection == 'model':
            selection = 'galpy'
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

    # optimized
    def dperi_min(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
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
        for name in self.host_names[hosts]:
            # Loop through each subhalo in the host
            for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                # Mask out the null pericenter values
                mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                # If there are no pericenters, append the present-day distance as the minimum
                if np.sum(mask_temp) == 0:
                    if oversample:
                        data.append(np.repeat(data_dict[name]['d.tot.sim'][mask_dict[name]][i][0], self.oversample[sim_type][name]))
                    else:
                        data.append(data_dict[name]['d.tot.sim'][mask_dict[name]][i][0])
                # If there are pericenters, append the minimum value to the list
                else:
                    if oversample:
                        data.append(np.repeat(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]), self.oversample[sim_type][name]))
                    else:
                        data.append(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]))
        #
        return np.hstack(data)

    # Old and ready to delete after Paper I submission
    def dperi_first(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        OLD METHOD:
        ===========
        I am not using first pericenters as a metric anymore in Paper I, therefore,
        it is probably worth getting rid of this method eventually.
        """
        # Set up an empty list to save values to
        data = []
        #
        # Determines whether oversampling or not, then
        if oversample == False:
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    #
                    if np.sum(mask_temp) == 0:
                        data.append(data_dict[name]['d.tot.sim'][mask_dict[name]][i][0])
                    #
                    else:
                        data.append(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp][-1])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    #
                    if np.sum(mask_temp) == 0:
                        data.append(np.repeat(data_dict[name]['d.tot.sim'][mask_dict[name]][i][0], self.oversample[sim_type][name]))
                    #
                    else:
                        data.append(np.repeat(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp][-1], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    # optimized
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
        # Quick label fix
        if selection == 'model':
            selection = 'galpy'
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

    # optimized
    def delta_dperi(self, data_dict, mask_dict, fraction=False, oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Similar to delta_nperi(), except calculates the difference in
            pericenter distances between the model and the simulation.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            fraction   : boolean
                         Set to True if interested in fractional difference in
                         pericenter distances.

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
            - This only calculates the difference between the MOST RECENT
              pericenter distances.
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Initialize an empty list to save data to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Save the most recent pericenters
            temp_array_model = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
            # If no pericenters, replace with present-day distances
            mask_temp = (temp_array_model == -1)
            temp_array_model[mask_temp] = data_dict[name]['d.tot.sim'][mask_dict[name]][:,0][mask_temp]
            #
            temp_array_sim = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
            mask_temp = (temp_array_sim == -1)
            temp_array_sim[mask_temp] = data_dict[name]['d.tot.sim'][mask_dict[name]][:,0][mask_temp]
            #
            # Determine what kind of difference, oversample if needed, and append to list
            if fraction:
                if oversample:
                    data.append((np.repeat(temp_array_model, self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim, self.oversample[sim_type][name]))\
                                 /np.repeat(temp_array_sim, self.oversample[sim_type][name]))
                else:
                    data.append((temp_array_model - temp_array_sim)/temp_array_sim)
            else:
                if oversample:
                    data.append(np.repeat(temp_array_model,self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim,self.oversample[sim_type][name]))
                else:
                    data.append(temp_array_model - temp_array_sim)
        #
        return np.hstack(data)

    # optimized
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
        # Quick label fix
        if selection == 'model':
            selection = 'galpy'
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

    # optimized
    def tperi_min(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
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
        # Set up an empty list to save the data to
        data = []
        #
        # Loop through each host galaxy
        for name in self.host_names[hosts]:
            # Loop through each subhalo
            for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                # Mask out the null values
                mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                # If there are no pericenters, set the lookback time to present-day, oversample if needed and append to the list
                if np.sum(mask_temp) == 0:
                    if oversample:
                        data.append(np.repeat(0.0, self.oversample[sim_type][name]))
                    else:
                        data.append(0.0)
                #
                # If there pericenters, find the minimum, select the time, oversample if needed and append to the list
                else:
                    index = np.where(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]) \
                                     == data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp])[0][0]
                    if oversample:
                        data.append(np.repeat(data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][i][mask_temp][index], self.oversample[sim_type][name]))
                    else:
                        data.append(data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][i][mask_temp][index])
        #
        return np.hstack(data)

    # optimized
    def delta_tperi_new(self, data_dict, mask_dict, fraction=False, oversample=False, hosts='all', sim_type='baryon'):
        """
        DESCRIPTION:
            Similar to delta_nperi(), except calculates the difference in
            pericenter lookback times between the model and the simulation.

        VARIABLES:
            data_dict  : dictionary
                         Dictionary of data created by data_read()

            mask_dict  : dictionary
                         Dictionary of masks created by data_mask(). This is used
                         on data_dict to mask out the subhalos you want.

            fraction   : boolean
                         Set to True if interested in fractional difference in
                         pericenter lookback times.

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
            - This only calculates the time difference between the MOST RECENT
              pericenter events.
            - Returns a 1D array.
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        # Set up an empty list to save data to
        data = []
        #
        # Loop through each host
        for name in self.host_names[hosts]:
            # Save data from the model to a temporary array
            temp_array_model = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
            # Mask out null values
            mask_temp = (temp_array_model == -1)
            # Set null values to present-day
            temp_array_model[mask_temp] = 0.0
            #
            # Repeat for the simulation
            temp_array_sim = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
            mask_temp = (temp_array_sim == -1)
            temp_array_sim[mask_temp] = 0.0
            #
            # If fractional difference, oversample if needed
            if fraction:
                if oversample:
                    ratio = (np.repeat(temp_array_model,self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim,self.oversample[sim_type][name]))\
                                 /np.repeat(temp_array_sim,self.oversample[sim_type][name])
                    # For cases where the lookback time is present-day, set the fraction to present-day
                    # i.e., do not save infinite values
                    ratio[~np.isfinite(ratio)] = 0
                    data.append(ratio)
                else:
                    ratio = (temp_array_model - temp_array_sim)/temp_array_sim
                    ratio[~np.isfinite(ratio)] = 0
                    data.append(ratio)
            # If not interested in fractional difference, oversample if needed and save data to array
            else:
                if oversample:
                    data.append(np.repeat(temp_array_model,self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim,self.oversample[sim_type][name]))
                else:
                    data.append(temp_array_model - temp_array_sim)
        #
        return np.hstack(data)

    # optimized
    def first_infall(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
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
        # Loop through each host
        for name in self.host_names[hosts]:
            # Oversample if needed and save to the list
            if oversample:
                data.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]])
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

    # optimized
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

    # optimized
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

    # optimized
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
    def v_z0(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
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
                data.append(np.repeat(data_dict[name]['v.tot.sim'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
            else:
                data.append(data_dict[name]['v.tot.sim'][mask_dict[name]][:,0])
        #
        return np.hstack(data)

    # optimized
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

    # optimized
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

    # optimized
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

    # optimized
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

    # optimized
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
                       'v.tan': 'Tangential velocity [km s$^{-1}$]',\
                       'v.rad': 'v$_{\\rm rad}(z = 0)$ [km s$^{-1}$]',\
                       'v.tot': 'Total velocity [km s$^{-1}$]',\
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

    def binning_scheme(self):
        """
        DESCRIPTION:
            Function that makes bins... finish later.
        """
        pass

    def median_and_scatter(self):
        """
        DESCRIPTION:
            Function that finds median and scatter... finish later.
        """
        pass

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

    def median_plot(self, x, y, xtype, ytype, binsize, file_path_and_name, binedges=None, limits=None, title=None):
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
        if 'M.' in xtype:
            x = np.log10(x)
        if 'M.' in ytype:
            y = np.log10(y)
        #
        if 'N.' not in xtype and 'N.' not in ytype:
            #
            if binedges:
                bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                bins = np.linspace(binedges[0], binedges[1], bin_num)
                half_bin = (bins[1]-bins[0])/2
            else:
                minn = binsize*np.floor(np.min(x)/binsize)
                maxx = binsize*np.ceil(np.max(x)/binsize)
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
        if 'N.' in xtype and 'N.' not in ytype:
            if binedges:
                bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                bins = np.linspace(binedges[0], binedges[1], bin_num)
                half_bin = (bins[1]-bins[0])/2
            else:
                minn = int(binsize*np.floor(np.min(x)/binsize))-0.5
                maxx = int(binsize*np.ceil(np.max(x)/binsize))+0.5
                bin_num = int((np.abs(maxx)+np.abs(minn))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
            onesigp = 84.13
            onesigm = 15.87
            twosigp = 100
            twosigm = 0
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
        if 'N.' not in xtype and 'N.' in ytype:
            if binedges:
                bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                bins = np.linspace(binedges[0], binedges[1], bin_num)
                half_bin = (bins[1]-bins[0])/2
            else:
                minn = binsize*np.floor(np.min(x)/binsize)
                maxx = binsize*np.ceil(np.max(x)/binsize)
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
        if 'N.' in xtype and 'N.' in ytype:
            if binedges:
                bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                bins = np.linspace(binedges[0], binedges[1], bin_num)
                half_bin = (bins[1]-bins[0])/2
            else:
                minn = int(binsize*np.floor(np.min(x)/binsize))-0.5
                maxx = int(binsize*np.ceil(np.max(x)/binsize))+0.5
                bin_num = int((np.abs(maxx)+np.abs(minn))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                half_bin = (bins[1]-bins[0])/2
            #
            twosigp = 100
            twosigm = 0
            #
            means = np.zeros(len(bins)-1)
            scatter = np.zeros(len(bins)-1)
            highest = np.zeros(len(bins)-1)
            lowest = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                means[i] = np.nanmean(y[mask])
                scatter[i] = np.nanstd(y[mask])
                highest[i] = np.nanpercentile(y[mask], twosigp)
                lowest[i] = np.nanpercentile(y[mask], twosigm)
            #
            upper = means+scatter
            lower = means-scatter
            med = means
        #
        f, ax = plt.subplots(figsize=(11, 8))
        #ax.minorticks_on()
        ax.set_xlabel(self.labels[xtype], fontsize=28)
        ax.set_ylabel(self.labels[ytype], fontsize=28)
        if title:
            ax.set_title(self.titles[title], fontsize=24)
        if 'M.' in xtype and 'M.' not in ytype:
            plt.plot(10**(bins[:-1]+half_bin), med, color=self.colors[1], markersize=10, alpha=0.5)
            plt.fill_between(10**(bins[:-1]+half_bin), upper, lower, color=self.colors[1], alpha=0.3)
            plt.fill_between(10**(bins[:-1]+half_bin), highest, lowest, color=self.colors[1], alpha=0.15)
            ax.set_xscale('log')
            ax.set_yscale('linear')
            if limits:
                plt.xlim(10**(limits[0][0]), 10**(limits[0][1]))
                plt.ylim(limits[1])
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
            plt.plot(10**(bins[:-1]+half_bin), 10**med, color=self.colors[1], markersize=10, alpha=0.5)
            plt.fill_between(10**(bins[:-1]+half_bin), 10**upper, 10**lower, color=self.colors[1], alpha=0.3)
            plt.fill_between(10**(bins[:-1]+half_bin), 10**highest, 10**lowest, color=self.colors[1], alpha=0.15)
            plt.hlines(y=3*10**4, xmin=10**(limits[0][0]), xmax=10**(limits[0][1]), colors='k', linestyles='dotted', alpha=0.5)
            if limits:
                plt.xlim(10**(limits[0][0]), 10**(limits[0][1]))
                plt.ylim(10**(limits[1][0]), 10**(limits[1][1]))
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], markersize=10, alpha=0.5)
            plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
            plt.fill_between(bins[:-1]+half_bin, highest, lowest, color=self.colors[1], alpha=0.15)
            ax.set_xscale('linear')
            ax.set_yscale('linear')
            if limits:
                plt.xlim(limits[0])
                plt.ylim(limits[1])
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

    def median_plot_mult(self, x, y, xtype, ytype, labels, binsize, file_path_and_name, binedges=None, limits=None, title=None, legend_on=True):
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
            if 'M.' in xtype[j]:
                x[j] = np.log10(x[j])
            if 'M.' in ytype[j]:
                y[j] = np.log10(y[j])
            #
            if 'N.' not in xtype[j] and 'N.' not in ytype[j]:
                if binedges:
                    bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                    bins = np.linspace(binedges[0], binedges[1], bin_num)
                    half_bin = (bins[1]-bins[0])/2
                else:
                    minn = binsize*np.floor(np.min(x[j])/binsize)
                    maxx = binsize*np.ceil(np.max(x[j])/binsize)
                    if minn < 0:
                        bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
                    else:
                        bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
                    bins = np.linspace(minn, maxx, bin_num)
                    half_bin = (bins[1]-bins[0])/2
                #
                onesigp = 84.13
                onesigm = 15.87
                #twosigp = 97.72
                #twosigm = 2.28 # play around with this more...
                twosigp = 100
                twosigm = 0
                #
                med = np.zeros(len(bins)-1)
                lower = np.zeros(len(bins)-1)
                upper = np.zeros(len(bins)-1)
                lowest = np.zeros(len(bins)-1)
                highest = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
                    med[i] = np.nanmedian(y[j][mask])
                    upper[i] = np.nanpercentile(y[j][mask], onesigp)
                    lower[i] = np.nanpercentile(y[j][mask], onesigm)
                    highest[i] = np.nanpercentile(y[j][mask], twosigp)
                    lowest[i] = np.nanpercentile(y[j][mask], twosigm)
            #
            if 'N.' in xtype[j] and 'N.' not in ytype[j]:
                if binedges:
                    bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                    bins = np.linspace(binedges[0], binedges[1], bin_num)
                    half_bin = (bins[1]-bins[0])/2
                else:
                    minn = int(binsize*np.floor(np.min(x[j])/binsize))-0.5
                    maxx = int(binsize*np.ceil(np.max(x[j])/binsize))+0.5
                    bin_num = int((np.abs(maxx)+np.abs(minn))/binsize+1)
                    bins = np.linspace(minn, maxx, bin_num)
                    #
                    half_bin = (bins[1]-bins[0])/2
                onesigp = 84.13
                onesigm = 15.87
                twosigp = 100
                twosigm = 0
                #
                med = np.zeros(len(bins)-1)
                lower = np.zeros(len(bins)-1)
                upper = np.zeros(len(bins)-1)
                lowest = np.zeros(len(bins)-1)
                highest = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
                    med[i] = np.nanmedian(y[j][mask])
                    upper[i] = np.nanpercentile(y[j][mask], onesigp)
                    lower[i] = np.nanpercentile(y[j][mask], onesigm)
                    highest[i] = np.nanpercentile(y[j][mask], twosigp)
                    lowest[i] = np.nanpercentile(y[j][mask], twosigm)
            #
            if 'N.' not in xtype[j] and 'N.' in ytype[j]:
                if binedges:
                    bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                    bins = np.linspace(binedges[0], binedges[1], bin_num)
                    half_bin = (bins[1]-bins[0])/2
                else:
                    minn = binsize*np.floor(np.min(x[j])/binsize)
                    maxx = binsize*np.ceil(np.max(x[j])/binsize)
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
                means = np.zeros(len(bins)-1)
                scatter = np.zeros(len(bins)-1)
                lowest = np.zeros(len(bins)-1)
                highest = np.zeros(len(bins)-1)
                upper = np.zeros(len(bins)-1)
                lower = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
                    means[i] = np.nanmean(y[j][mask])
                    scatter[i] = np.nanstd(y[j][mask])
                    highest[i] = np.nanpercentile(y[j][mask], twosigp)
                    lowest[i] = np.nanpercentile(y[j][mask], twosigm)
                    upper[i] = means[i]+scatter[i]
                    lower[i] = means[i]-scatter[i]
                    if (upper[i] > highest[i]):
                        upper[i] = highest[i]
                    if (lower[i] < lowest[i]):
                        lower[i] = lowest[i]
                #
                med = means
            #
            if 'N.' in xtype[j] and 'N.' in ytype[j]:
                if binedges:
                    bin_num = int((binedges[1]-binedges[0])/binsize + 1)
                    bins = np.linspace(binedges[0], binedges[1], bin_num)
                    half_bin = (bins[1]-bins[0])/2
                else:
                    minn = int(binsize*np.floor(np.min(x[j])/binsize))-0.5
                    maxx = int(binsize*np.ceil(np.max(x[j])/binsize))+0.5
                    bin_num = int((np.abs(maxx)+np.abs(minn))/binsize+1)
                    bins = np.linspace(minn, maxx, bin_num)
                    half_bin = (bins[1]-bins[0])/2
                #
                twosigp = 100
                twosigm = 0
                #
                means = np.zeros(len(bins)-1)
                scatter = np.zeros(len(bins)-1)
                lowest = np.zeros(len(bins)-1)
                highest = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (x[j] >= bins[i]) & (x[j] <= bins[i+1])
                    means[i] = np.nanmean(y[j][mask])
                    scatter[i] = np.nanstd(y[j][mask])
                    highest[i] = np.nanpercentile(y[j][mask], twosigp)
                    lowest[i] = np.nanpercentile(y[j][mask], twosigm)
                #
                upper = means+scatter
                lower = means-scatter
                med = means
            #
            # PLOTTING
            if 'M.' in xtype[j] and 'M.' not in ytype[j]:
                plt.plot(10**(bins[:-1]+half_bin), med, color=colorss[j], markersize=10, alpha=0.5, label=labels[j])
                plt.fill_between(10**(bins[:-1]+half_bin), upper, lower, color=colorss[j], alpha=0.3)
                plt.fill_between(10**(bins[:-1]+half_bin), highest, lowest, color=colorss[j], alpha=0.15)
            elif 'M.' in xtype[j] and 'M.' in ytype[j]:
                plt.plot(10**(bins[:-1]+half_bin), 10**med, color=colorss[j], markersize=10, alpha=0.5, label=labels[j])
                plt.fill_between(10**(bins[:-1]+half_bin), 10**upper, 10**lower, color=colorss[j], alpha=0.3)
                plt.fill_between(10**(bins[:-1]+half_bin), 10**highest, 10**lowest, color=colorss[j], alpha=0.15)
                ax.axhline(y=3*10**7, xmin=10**(8), xmax=10**(9), color='k', linestyle=':')
                #
            else:
                plt.plot(bins[:-1]+half_bin, med, color=colorss[j], markersize=10, alpha=0.5, label=labels[j])
                plt.fill_between(bins[:-1]+half_bin, upper, lower, color=colorss[j], alpha=0.3)
                plt.fill_between(bins[:-1]+half_bin, highest, lowest, color=colorss[j], alpha=0.15)
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
        ax.set_xlabel(self.labels[xtype[0]], fontsize=28)
        ax.set_ylabel(self.labels[ytype[0]], fontsize=28)
        if legend_on:
            ax.legend(prop={'size': 24}, loc='best')
        if title:
            plt.title(self.titles[title], fontsize=24)
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

    def plot_hist(self, x, xtype, binsize, file_path_and_name, pdf=False, xlimits=None, title=None):
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
        # Plot the data
        plt.figure(figsize=(10, 8))
        #
        if 'N.' not in xtype:
            minn = binsize*np.floor(np.min(x)/binsize)
            maxx = binsize*np.ceil(np.max(x)/binsize)
            if minn < 0:
                bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
            else:
                bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize)+1)
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            # Calculate the scatter
            onesigp = 84.13
            onesigm = 15.87
            sigma_one_op = np.nanpercentile(x, onesigp)
            sigma_one_om = np.nanpercentile(x, onesigm)
            #
            y_med = np.max(np.histogram(x, bin_array, density=pdf)[0])*1.1
            #
            plt.hist(x, bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            plt.errorbar(np.median(x), y_med, xerr=np.array([[np.median(x)-sigma_one_om],[sigma_one_op-np.median(x)]]), color='k', lw=5, capsize=0)
            plt.scatter(np.median(x), y_med, s=250, marker='s', c='k')
        #
        elif 'N.' in xtype:
            minn = int(binsize*np.floor(np.min(x)/binsize))-0.5
            maxx = int(binsize*np.ceil(np.max(x)/binsize))+0.5
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            y_mean = np.max(np.histogram(x, bin_array, density=pdf)[0])*1.1
            #
            plt.hist(x, bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            plt.errorbar(np.mean(x), y_mean, xerr=np.array([[2*np.std(x)],[2*np.std(x)]]), color='k', lw=5, capsize=0, alpha=0.3)
            plt.errorbar(np.mean(x), y_mean, xerr=np.array([[np.std(x)],[np.std(x)]]), color='k', lw=5, capsize=0)
            plt.scatter(np.mean(x), y_mean, s=250, marker='s', c='k')
        #
        plt.xlim(xlimits)
        plt.xlabel(self.labels[xtype], fontsize=36)
        plt.ylabel(y_label, fontsize=34)
        if title:
            plt.title(self.titles[title], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def plot_hist_mult(self, x, xtype, labels, binsize, file_path_and_name, med_location=None, pdf=False, xlimits=None, title=None, legend_on=True):
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
            y_label = 'PDF'
        else:
            y_label = 'N'
        #
        # Plot the data
        plt.figure(figsize=(10, 8))
        #
        for i in range(0, len(x)):
            if 'M.' in xtype[i]:
                x[i] = np.log10(x[i])
            if 'N.' not in xtype[i]:
                minn = binsize*np.floor(np.min(x[i])/binsize)
                maxx = binsize*np.ceil(np.max(x[i])/binsize)
                if minn < 0:
                    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize)-1)
                else:
                    bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize)+1)
                bin_array = np.linspace(minn, maxx, bin_num)
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
                    y_med = np.max(np.histogram(x[i], bin_array, density=pdf)[0])*1.1
                #
                plt.hist(x[i], bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
                plt.errorbar(np.median(x[i]), y_med, xerr=np.array([[np.median(x[i])-sigma_one_om],[sigma_one_op-np.median(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.8)
                plt.scatter(np.median(x[i]), y_med, s=250, marker='s', c=colorss[i], alpha=0.8)
            #
            elif 'N.' in xtype[i]:
                minn = int(binsize*np.floor(np.min(x[i])/binsize))-0.5
                maxx = int(binsize*np.ceil(np.max(x[i])/binsize))+0.5
                bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
                bin_array = np.linspace(minn, maxx, bin_num)
                #
                if med_location:
                    y_mean = med_location[i]
                else:
                    y_mean = np.max(np.histogram(x[i], bin_array, density=pdf)[0])*1.1
                #
                plt.hist(x[i], bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
                plt.errorbar(np.mean(x[i]), y_mean, xerr=np.array([[2*np.std(x[i])],[2*np.std(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.3)
                plt.errorbar(np.mean(x[i]), y_mean, xerr=np.array([[np.std(x[i])],[np.std(x[i])]]), c=colorss[i], lw=5, capsize=0, alpha=0.8)
                plt.scatter(np.mean(x[i]), y_mean, s=250, marker='s', c=colorss[i])
        #
        plt.xlim(xlimits)
        plt.xlabel(self.labels[xtype[0]], fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        if legend_on:
            plt.legend(prop={'size': 18}, loc='center right')
        if title:
            plt.title(self.titles[title], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()
