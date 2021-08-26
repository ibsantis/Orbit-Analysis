#!/usr/bin/python3

"""
@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

    This was written to create summary statistic plots with data output from
    orbit_io.py and model_io.py. Data that I import was previously compiled
    using summary_data.py and summary_data_dmo.py

"""

import utilities as ut
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


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
                           'iso': ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w'],
                           'lg': ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']}
        #
        # Oversampling factors
        self.oversample = {'baryon': {'m12b': 16, 'm12c': 14, 'm12f': 13, 'm12i': 22, 'm12m': 12,\
                                      'm12r': 20, 'm12w': 14, 'm12z': 21, 'Romeo': 16, 'Juliet': 14,\
                                      'Thelma': 17, 'Louise': 16, 'Romulus': 10, 'Remus': 17},\
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
                        Choose either 'all', 'iso', or 'lg' to select different
                        samples.

        NOTES:
            - The dictionary that gets returned is a dictionary of dictionaries.
            - Each key in the dictionary is a host name
                - Each key in the sub-dictionaries is given in summary_data.py
            - Data is arranged in order of self.host_names[hosts]. Subhalos in
              each host are arranged in the same way they were generated from
              summary_data.py or summary_data_dmo.py.
        """
        data_dict = dict()
        #
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
                          in the simulations, but not in the data.

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

            hosts       : Choose either 'all', 'iso', or 'lg' to select different
                          samples.

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
                        mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['pericenter.check.sim'] | dictionary[name]['pericenter.check.galpy'])*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
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
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
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
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*dictionary[name]['pericenter.check.galpy']*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
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
                                mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
        #
        # If interested in outliers do this.
        elif outliers == True:
            for name in self.host_names[hosts]:
                mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*(~dictionary[name]['pericenter.check.galpy'])
        #
        return mask_dict

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
        # Determines if working with sim or model data, and whether oversampling or not.
        # Then appends values to list.
        if selection == 'sim':
            #
            if oversample == False:
                for name in self.host_names[hosts]:
                    data.append(data_dict[name]['N.peri.sim'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    data.append(np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        elif selection == 'model':
            if oversample == False:
                for name in self.host_names[hosts]:
                    data.append(data_dict[name]['N.peri.galpy'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]], self.oversample[sim_type][name]))
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
        # Determines if working with sim or model data, then whether oversampling or not.
        # Also masks values with no pericenter and sets them equal to d(z = 0)
        if (selection == 'sim'):
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
        #
        elif (selection == 'model'):
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
        #
        return np.hstack(data)

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
        # Determines whether oversampling or not, then
        if oversample == False:
            count = 0
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    #
                    if np.sum(mask_temp) == 0:
                        data.append(data_dict[name]['dtot.sim'][mask_dict[name]][i][0])
                    #
                    else:
                        data.append(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]))
                    #
                    # ...?
                    if np.sum(mask_temp) == 1:
                        count += 1
            print(count)
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    #
                    if np.sum(mask_temp) == 0:
                        data.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][i][0], self.oversample[sim_type][name]))
                    #
                    else:
                        data.append(np.repeat(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]), self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def delta_dperi(self, data_dict, mask_dict, fraction=False, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if fraction == False:
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    temp_array_sim = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    data.append(temp_array_model - temp_array_sim)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    temp_array_sim = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    data.append(np.repeat(temp_array_model,self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim,self.oversample[sim_type][name]))
        #
        elif fraction == True:
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    temp_array_sim = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    data.append((temp_array_model - temp_array_sim)/temp_array_sim)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    temp_array_sim = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    data.append((np.repeat(temp_array_model,self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim,self.oversample[sim_type][name]))\
                                 /np.repeat(temp_array_sim,self.oversample[sim_type][name]))
        return np.hstack(data)

    def tperi_recent(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if (selection == 'sim'):
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
        #
        elif (selection == 'model'):
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(np.repeat(temp_array, self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def tperi_min(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        data = []
        if oversample == False:
            count = 0
            #
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    #
                    if np.sum(mask_temp) == 0:
                        data.append(0.0)
                    #
                    else:
                        index = np.where(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]) \
                                         == data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp])[0][0]
                        data.append(data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][i][mask_temp][index])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    #
                    if np.sum(mask_temp) == 0:
                        data.append(np.repeat(0.0, self.oversample[sim_type][name]))
                    #
                    else:
                        index = np.where(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]) \
                                         == data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp])[0][0]
                        data.append(np.repeat(data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][i][mask_temp][index], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def delta_tperi(self, data_dict, mask_dict, fraction=False, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if fraction == False:
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = 0.0
                    #
                    temp_array_sim = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = 0.0
                    #
                    data.append(temp_array_model - temp_array_sim)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = 0.0
                    #
                    temp_array_sim = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = 0.0
                    #
                    data.append(np.repeat(temp_array_model,self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim,self.oversample[sim_type][name]))
        #
        elif fraction == True:
            if oversample == False:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = 0.0
                    #
                    temp_array_sim = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = 0.0
                    #
                    ratio = (temp_array_model - temp_array_sim)/temp_array_sim
                    ratio[~np.isfinite(ratio)] = 0
                    data.append(ratio)
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    temp_array_model = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = 0.0
                    #
                    temp_array_sim = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = 0.0
                    #
                    ratio = (np.repeat(temp_array_model,self.oversample[sim_type][name]) - \
                                 np.repeat(temp_array_sim,self.oversample[sim_type][name]))\
                                 /np.repeat(temp_array_sim,self.oversample[sim_type][name])
                    ratio[~np.isfinite(ratio)] = 0
                    data.append(ratio)
        return np.hstack(data)

    def first_infall(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names[hosts]:
                data.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                data.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def first_infall_any(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names[hosts]:
                data.append(data_dict[name]['first.infall.time.lb.any'][mask_dict[name]])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                data.append(np.repeat(data_dict[name]['first.infall.time.lb.any'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def mstar(self, data_dict, mask_dict, selection='z0', oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if selection == 'z0':
            if oversample == False:
                for name in self.host_names[hosts]:
                    data.append(data_dict[name]['Mstar.z0'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    data.append(np.repeat(data_dict[name]['Mstar.z0'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        elif selection == 'peak':
            if oversample == False:
                for name in self.host_names[hosts]:
                    data.append(data_dict[name]['Mstar.peak'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    data.append(np.repeat(data_dict[name]['Mstar.peak'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def mhalo(self, data_dict, mask_dict, selection='z0', oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if selection == 'z0':
            if oversample == False:
                for name in self.host_names[hosts]:
                    data.append(data_dict[name]['Mhalo.z0'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    data.append(np.repeat(data_dict[name]['Mhalo.z0'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        elif selection == 'peak':
            if oversample == False:
                for name in self.host_names[hosts]:
                    data.append(data_dict[name]['Mhalo.peak'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names[hosts]:
                    data.append(np.repeat(data_dict[name]['Mhalo.peak'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def d_z0(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names[hosts]:
                data.append(data_dict[name]['dtot.sim'][mask_dict[name]][:,0])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                data.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def v_z0(self, data_dict, mask_dict, oversample=False, hosts='all', sim_type='baryon'):
        """
        TBD
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names[hosts]:
                data.append(data_dict[name]['vtot.sim'][mask_dict[name]][:,0])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                data.append(np.repeat(data_dict[name]['vtot.sim'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
        #
        return np.hstack(data)


    def kinetic_energy(self, data_dict, mask_dict, ke_type, oversample=False, hosts='all', sim_type='baryon'):
        data = []
        #
        if ke_type == 'max':
            if oversample == False:
                for name in self.host_names[hosts]:
                    for i in range(0, len(data_dict[name]['vtot.sim'][mask_dict[name]])):
                        data.append(np.nanmax(0.5*data_dict[name]['vtot.sim'][mask_dict[name]][i]**2))
            else:
                for name in self.host_names[hosts]:
                    for i in range(0, len(data_dict[name]['vtot.sim'][mask_dict[name]])):
                        data.append(np.repeat(np.nanmax(0.5*data_dict[name]['vtot.sim'][mask_dict[name]][i]**2), self.oversample[sim_type][name]))
        #
        elif ke_type == 'peri':
            if oversample == False:
                for name in self.host_names[hosts]:
                    for i in range(0, len(data_dict[name]['pericenter.vel.sim'][mask_dict[name]])):
                        if (data_dict[name]['pericenter.vel.sim'][mask_dict[name]][i][0] == -1):
                            data.append(0.5*data_dict[name]['vtot.sim'][mask_dict[name]][i][0]**2)
                        else:
                            data.append(0.5*data_dict[name]['pericenter.vel.sim'][mask_dict[name]][i][0]**2)
            else:
                for name in self.host_names[hosts]:
                    for i in range(0, len(data_dict[name]['pericenter.vel.sim'][mask_dict[name]])):
                        if (data_dict[name]['pericenter.vel.sim'][mask_dict[name]][i][0] == -1):
                            data.append(np.repeat(0.5*data_dict[name]['vtot.sim'][mask_dict[name]][i][0]**2, self.oversample[sim_type][name]))
                        else:
                            data.append(np.repeat(0.5*data_dict[name]['pericenter.vel.sim'][mask_dict[name]][i][0]**2, self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def mass_masking_property(self, data_dict, mask_dict, prop, mass_array, mass_type='Mstar.z0', oversample=False, hosts='all', sim_type='baryon'):
        """
        STILL NEEDS A LOT OF WORK AND CHECKING...
        """
        props = dict()
        prop_low = []
        prop_mid = []
        #
        if oversample == False:
            for name in self.host_names[hosts]:
                mask_low = ((data_dict[name][mass_type][mask_dict[name]] > mass_array[0])*(data_dict[name][mass_type][mask_dict[name]] < mass_array[1]))
                mask_mid = (data_dict[name][mass_type][mask_dict[name]] > mass_array[1])
                if prop == 't.infall':
                    prop_low.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_low])
                    prop_mid.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_mid])
                elif prop == 'dz0':
                    prop_low.append(data_dict[name]['dtot.sim'][mask_dict[name]][mask_low][:,0])
                    prop_mid.append(data_dict[name]['dtot.sim'][mask_dict[name]][mask_mid][:,0])
            #
            props['low'] = np.hstack(prop_low)
            props['mid'] = np.hstack(prop_mid)
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                mask_low = ((data_dict[name][mass_type][mask_dict[name]] > mass_array[0])*(data_dict[name][mass_type][mask_dict[name]] < mass_array[1]))
                mask_mid = (data_dict[name][mass_type][mask_dict[name]] > mass_array[1])
                if prop == 't.infall':
                    prop_low.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_low], self.oversample[sim_type][name]))
                    prop_mid.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_mid], self.oversample[sim_type][name]))
                elif prop == 'dz0':
                    prop_low.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][mask_low][:,0], self.oversample[sim_type][name]))
                    prop_mid.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][mask_mid][:,0], self.oversample[sim_type][name]))
            #
            props['low'] = np.hstack(prop_low)
            props['mid'] = np.hstack(prop_mid)
        #
        return props

    def velocities(self, data_dict, mask_dict, selection='tan', oversample=False, hosts='all', sim_type='baryon'):
        """
        Only works with the simulation data right now, not the model data...

        selection = rad or tan
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names[hosts]:
                data.append(data_dict[name]['v.'+selection+'.z0'][mask_dict[name]])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                data.append(np.repeat(data_dict[name]['v.'+selection+'.z0'][mask_dict[name]], self.oversample[sim_type][name]))
        #
        return np.hstack(data)

    def L_z0(self, data_dict, mask_dict, selection='sim', oversample=False, hosts='all', sim_type='baryon'):
        """
        Only works with the simulation data right now, not the model data...
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names[hosts]:
                data.append(data_dict[name]['Ltot.sim'][mask_dict[name]][:,0])
        #
        elif oversample == True:
            for name in self.host_names[hosts]:
                data.append(np.repeat(data_dict[name]['Ltot.sim'][mask_dict[name]][:,0], self.oversample[sim_type][name]))
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
                       'd.sim.min.recent': '(d$_{\\rm peri,min,sim}$ - d$_{\\rm peri,sim}$) [kpc]',\
                       'd.sim.min.recent.frac': '(d$_{\\rm peri,min,sim}$ - d$_{\\rm peri,sim}$)/d$_{\\rm peri,sim}$',\
                       'd.peri': 'd$_{\\rm peri}$ [kpc]',\
                       'd.model': 'd$_{\\rm peri,model}$ [kpc]',\
                       'd.z0': 'd(z = 0) [kpc]',\
                       'delta.d.frac': '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$)/d$_{\\rm peri,sim}$',\
                       'delta.d': '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$) [kpc]',\
                       'v.tan': 'v$_{\\rm tan}(z = 0)$ [km s$^{-1}$]',\
                       'v.rad': 'v$_{\\rm rad}(z = 0)$ [km s$^{-1}$]',\
                       'v.tot': 'v$_{\\rm tot}(z = 0)$ [km s$^{-1}$]',\
                       't.sim': 't$_{\\rm peri,lb,sim}$ [Gyr]',\
                       't.sim.min': 't$_{\\rm peri,min,lb,sim}$ [Gyr]',\
                       't.sim.min.recent': '(t$_{\\rm peri,min,lb,sim}$ - t$_{\\rm peri,lb,sim}$) [Gyr]',\
                       't.sim.min.recent.frac': '(t$_{\\rm peri,min,lb,sim}$ - t$_{\\rm peri,lb,sim}$)/t$_{\\rm peri,lb,sim}$',\
                       't.model': 't$_{\\rm peri,lb,model}$ [Gyr]',\
                       't.peri': 't$_{\\rm peri}$ [Gyr]',\
                       't.infall': 't$_{\\rm infall,lb}$ [Gyr]',\
                       't.infall.any': 't$_{\\rm infall,any,lb}$ [Gyr]',\
                       't.infall.diff': '(t$_{\\rm infall,any,lb}$ - t$_{\\rm infall,lb}$) [Gyr]',\
                       'delta.t.frac': '(t$_{\\rm peri,model}$ - t$_{\\rm peri,sim}$)/t$_{\\rm peri,sim}$',\
                       'delta.t': '(t$_{\\rm peri,model}$ - t$_{\\rm peri,sim}$) [Gyr]',\
                       'N.sim': 'N$_{\\rm peri,sim}$',\
                       'N.model': 'N$_{\\rm peri,model}$',\
                       'N.delta': 'N$_{\\rm model}$ - N$_{\\rm sim}$',\
                       'M.star.z0': 'log$_{\\rm 10}$[M$_{\\rm star}(z = 0)$/M$_{\\odot}$]',\
                       'M.star.peak': 'log$_{\\rm 10}$[M$_{\\rm star, peak}$/M$_{\\odot}$]',\
                       'M.halo.z0': 'log$_{\\rm 10}$[M$_{\\rm halo}(z = 0)$/M$_{\\odot}$]',\
                       'M.halo.peak': 'log$_{\\rm 10}$[M$_{\\rm halo, peak}$/M$_{\\odot}$]',\
                       'KE.max.sim': 'KE$_{\\rm max,sim}$ [10$^4$ km$^2$/s$^2$]',\
                       'KE.peri.sim': 'KE$_{\\rm peri,sim}$ [10$^4$ km$^2$/s$^2$]',\
                       'L.tot': 'L$_{\\rm tot}(z = 0)$ [10$^4$ kpc km s$^{-1}$]'}
        #
        self.titles = {'d.sim': 'Recent Minimum Distances',\
                       'd.model': 'Recent Minimum Distances',\
                       'd.z0': 'Present-day Distances',\
                       't.sim': 'Recent Minimum Distance Lookback Times',\
                       't.model': 'Recent Minimum Distance Lookback Times',\
                       't.infall': 'Host Infall Lookback Times',\
                       'N.sim': 'Pericenters',\
                       'N.model': 'Pericenters',\
                       'N.delta': 'Pericenters'}

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
            x_out = np.log10(x_out)
        if 'M.' in ytype:
            y = np.log10(y)
            y_out = np.log10(y_out)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x, y, color='k', s=50, marker='x', alpha=0.5)
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

    def median_plot(self, x, y, xtype, ytype, binsize, file_path_and_name, limits=None, title=None):
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
        if 'N.' in xtype and 'N.' in ytype:
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
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        plt.fill_between(bins[:-1]+half_bin, highest, lowest, color=self.colors[1], alpha=0.15)
        if limits:
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

    def median_plot_mult(self, x, y, xtype, ytype, labels, binsize, file_path_and_name, limits=None, title=None):
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
        f, ax = plt.subplots(figsize=(10, 8))
        #
        for j in range(0, len(x)):
            if 'M.' in xtype[j]:
                x[j] = np.log10(x[j])
            if 'M.' in ytype[j]:
                y[j] = np.log10(y[j])
            #
            if 'N.' not in xtype[j] and 'N.' not in ytype[j]:
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
            if 'N.' in xtype[j] and 'N.' in ytype[j]:
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
            plt.plot(bins[:-1]+half_bin, med, color=colorss[j], marker='s', markersize=10, alpha=0.5, label=labels[j])
            plt.fill_between(bins[:-1]+half_bin, upper, lower, color=colorss[j], alpha=0.3)
            plt.fill_between(bins[:-1]+half_bin, highest, lowest, color=colorss[j], alpha=0.15)
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
            y_label = 'PDF'
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
                bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            # Calculate the scatter
            onesigp = 84.13
            onesigm = 15.87
            sigma_one_op = np.nanpercentile(x, onesigp)
            sigma_one_om = np.nanpercentile(x, onesigm)
            #
            y_med = np.max(np.histogram(x, bin_array, normed=pdf)[0])*1.1
            #
            plt.hist(x, bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            plt.errorbar(np.median(x), y_med, xerr=np.array([[np.median(x)-sigma_one_om],[sigma_one_op-np.median(x)]]), color='k', lw=5, capsize=8)
            plt.scatter(np.median(x), y_med, s=250, marker='s', c='k')
        #
        elif 'N.' in xtype:
            minn = int(binsize*np.floor(np.min(x)/binsize))-0.5
            maxx = int(binsize*np.ceil(np.max(x)/binsize))+0.5
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            y_mean = np.max(np.histogram(x, bin_array, normed=pdf)[0])*1.1
            #
            plt.hist(x, bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            plt.errorbar(np.mean(x), y_mean, xerr=np.array([[2*np.std(x)],[2*np.std(x)]]), color='k', lw=5, capsize=8, alpha=0.3)
            plt.errorbar(np.mean(x), y_mean, xerr=np.array([[np.std(x)],[np.std(x)]]), color='k', lw=5, capsize=8)
            plt.scatter(np.mean(x), y_mean, s=250, marker='s', c='k')
        #
        plt.xlim(xlimits)
        plt.xlabel(self.labels[xtype], fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        if title:
            plt.title(self.titles[title], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def plot_hist_mult(self, x, xtype, labels, binsize, file_path_and_name, pdf=False, xlimits=None, title=None):
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
        colorss = ['#006400', '#000080']
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
                    bin_num = int(np.around((np.abs(minn)+np.abs(maxx))/binsize+1))
                else:
                    bin_num = int(np.around((np.abs(maxx)-np.abs(minn))/binsize+1))
                bin_array = np.linspace(minn, maxx, bin_num)
                #
                # Calculate the scatter
                onesigp = 84.13
                onesigm = 15.87
                sigma_one_op = np.nanpercentile(x[i], onesigp)
                sigma_one_om = np.nanpercentile(x[i], onesigm)
                #
                y_med = np.max(np.histogram(x[i], bin_array, normed=pdf)[0])*1.1
                #
                plt.hist(x[i], bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
                plt.errorbar(np.median(x[i]), y_med, xerr=np.array([[np.median(x[i])-sigma_one_om],[sigma_one_op-np.median(x[i])]]), c=colorss[i], lw=5, capsize=8, alpha=0.8)
                plt.scatter(np.median(x[i]), y_med, s=250, marker='s', c=colorss[i], alpha=0.8)
            #
            elif 'N.' in xtype[i]:
                minn = int(binsize*np.floor(np.min(x[i])/binsize))-0.5
                maxx = int(binsize*np.ceil(np.max(x[i])/binsize))+0.5
                bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
                bin_array = np.linspace(minn, maxx, bin_num)
                #
                y_mean = np.max(np.histogram(x[i], bin_array, normed=pdf)[0])*1.1
                #
                plt.hist(x[i], bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=colorss[i], alpha=0.4, label=labels[i])
                plt.errorbar(np.mean(x[i]), y_mean, xerr=np.array([[2*np.std(x[i])],[2*np.std(x[i])]]), c=colorss[i], lw=5, capsize=8, alpha=0.3)
                plt.errorbar(np.mean(x[i]), y_mean, xerr=np.array([[np.std(x[i])],[np.std(x[i])]]), c=colorss[i], lw=5, capsize=8, alpha=0.8)
                plt.scatter(np.mean(x[i]), y_mean, s=250, marker='s', c=colorss[i])
        #
        plt.xlim(xlimits)
        plt.xlabel(self.labels[xtype[0]], fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        plt.legend(prop={'size': 18}, loc='best')
        if title:
            plt.title(self.titles[title], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()
