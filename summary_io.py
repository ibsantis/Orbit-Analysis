#!/usr/bin/python3

"""
@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

    This was written to create summary statistic plots with data output from
    orbit_io.py and model_io.py. Data that I import was previously compiled
    using summary_data.py

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
            - Host names to loop over in the following methods
            - Oversampling factors
                NOTE: It would be nice to not hard-code these factors...
        """
        # Create a list of host names
        self.host_names = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', 'm12z', \
                           'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']
        #
        # Oversampling factors
        self.oversample = {'m12b': 16, 'm12c': 14, 'm12f': 13, 'm12i': 22, 'm12m': 12,\
                           'm12r': 20, 'm12w': 14, 'm12z': 21, 'Romeo': 16, 'Juliet': 14,\
                           'Thelma': 17, 'Louise': 16, 'Romulus': 10, 'Remus': 17}

    def data_read(self, directory):
        """
        TBD
        """
        data_dict = dict()
        for name in self.host_names:
            data = ut.io.file_hdf5(directory+'/orbit_data/hdf5_files/summary_data/data_'+name, verbose=True)
            data_dict[name] = data
        #
        return data_dict

    def data_mask(self, dictionary, outliers=False, peri_sim=True, peri_model=False, current_sat=False, either=False):
        """
        DESCRIPTION:
            Create a dictionary of masks for the satellites that depends on whether they
            have fallen into the host, whether they are currently in the host (at z = 0),
            whether they have experienced a pericenter in the simulation or model, and
            the outliers that have experienced pericenter in sim, but not model.

        VARIABLES:
            dictionary  : dictionary
            outliers    : bool
            peri_sim    : bool
            peri_model  : bool
            current_sat : bool

        NOTES:
            - TBD
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
                    for name in self.host_names:
                        mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['pericenter.check.sim'] | dictionary[name]['pericenter.check.galpy'])
                #
                # If interested in current sats only, do this
                elif current_sat == True:
                    for name in self.host_names:
                        mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['pericenter.check.sim'] | dictionary[name]['pericenter.check.galpy'])*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
            #
            elif either == False:
                #
                # Pericenter in simulation but not required in the model
                if (peri_sim == True) & (peri_model == False):
                    #
                    # If not interested in whether they are currently satellites, do this
                    if current_sat == False:
                        for name in self.host_names:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']
                    #
                    # If interested in current sats only, do this
                    elif current_sat == True:
                        for name in self.host_names:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
                #
                # Pericenter required in simulation and in model
                elif (peri_sim == True) & (peri_model == True):
                    #
                    # If not interested in whether they are currently satellites, do this
                    if current_sat == False:
                        for name in self.host_names:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*dictionary[name]['pericenter.check.galpy']
                    #
                    # If interested in current sats only, do this
                    elif current_sat == True:
                        for name in self.host_names:
                            mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*dictionary[name]['pericenter.check.galpy']*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
                #
                # Pericenter not required in simulation or model
                elif (peri_sim == False) & (peri_model == False):
                        #
                        # If not interested in whether they are currently satellites, do this
                        if current_sat == False:
                            for name in self.host_names:
                                mask_dict[name] = dictionary[name]['infall.check']
                        #
                        # If interested in current sats only, do this
                        elif current_sat == True:
                            for name in self.host_names:
                                mask_dict[name] = dictionary[name]['infall.check']*(dictionary[name]['dtot.sim'][:,0] < dictionary[name]['host.radius'][0])
        #
        # If interested in outliers do this.
        elif outliers == True:
            for name in self.host_names:
                mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*(~dictionary[name]['pericenter.check.galpy'])
        return mask_dict

    def mass_masking_property(self, data_dict, mask_dict, prop, mass_type='Mstar.z0', oversample=False):
        props = dict()
        prop_low = []
        prop_mid = []
        prop_high = []
        #
        if oversample == False:
            for name in self.host_names:
                mask_low = (data_dict[name][mass_type][mask_dict[name]] < 1e5)
                mask_mid = ((data_dict[name][mass_type][mask_dict[name]] > 1e5)*(data_dict[name][mass_type][mask_dict[name]] < 1e7))
                mask_high = (data_dict[name][mass_type][mask_dict[name]] > 1e7)
                if prop == 't.infall':
                    prop_low.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_low])
                    prop_mid.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_mid])
                    prop_high.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_high])
                elif prop == 'dz0':
                    prop_low.append(data_dict[name]['dtot.sim'][mask_dict[name]][mask_low][:,0])
                    prop_mid.append(data_dict[name]['dtot.sim'][mask_dict[name]][mask_mid][:,0])
                    prop_high.append(data_dict[name]['dtot.sim'][mask_dict[name]][mask_high][:,0])
            #
            props['low'] = np.hstack(prop_low)
            props['mid'] = np.hstack(prop_mid)
            props['high'] = np.hstack(prop_high)
        #
        elif oversample == True:
            for name in self.host_names:
                mask_low = (data_dict[name][mass_type][mask_dict[name]] < 1e5)
                mask_mid = ((data_dict[name][mass_type][mask_dict[name]] > 1e5)*(data_dict[name][mass_type][mask_dict[name]] < 1e7))
                mask_high = (data_dict[name][mass_type][mask_dict[name]] > 1e7)
                if prop == 't.infall':
                    prop_low.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_low], self.oversample[name]))
                    prop_mid.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_mid], self.oversample[name]))
                    prop_high.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]][mask_high], self.oversample[name]))
                elif prop == 'dz0':
                    prop_low.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][mask_low][:,0], self.oversample[name]))
                    prop_mid.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][mask_mid][:,0], self.oversample[name]))
                    prop_high.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][mask_high][:,0], self.oversample[name]))
            #
            props['low'] = np.hstack(prop_low)
            props['mid'] = np.hstack(prop_mid)
            props['high'] = np.hstack(prop_high)
        #
        return props

    def delta_nperi(self, data_dict, mask_dict, oversample=True):
        """
        DESCRIPTION:
            TBD

        VARIABLES:
            TBD

        NOTES:
            - TBD
        """
        data = []
        if oversample == True:
            for name in self.host_names:
                data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]],self.oversample[name]) - \
                             np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]],self.oversample[name]))
        #
        elif oversample == False:
            for name in self.host_names:
                data.append(data_dict[name]['N.peri.galpy'][mask_dict[name]] - \
                            data_dict[name]['N.peri.sim'][mask_dict[name]])
        #
        return np.hstack(data)

    def nperi(self, data_dict, mask_dict, selection='sim', oversample=False):
        """
        TBD
        """
        data = []
        #
        if selection == 'sim':
            if oversample == False:
                for name in self.host_names:
                    data.append(data_dict[name]['N.peri.sim'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]], self.oversample[name]))
        #
        elif selection == 'model':
            if oversample == False:
                for name in self.host_names:
                    data.append(data_dict[name]['N.peri.galpy'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]], self.oversample[name]))
        return np.hstack(data)

    def dperi_recent(self, data_dict, mask_dict, selection='sim', oversample=False):
        """
        TBD
        """
        data = []
        #
        if (selection == 'sim'):
            if oversample == False:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(np.repeat(temp_array, self.oversample[name]))
        #
        elif (selection == 'model'):
            if oversample == False:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    data.append(np.repeat(temp_array, self.oversample[name]))
        #
        return np.hstack(data)

    def dperi_min(self, data_dict, mask_dict, oversample=False):
        data = []
        if oversample == False:
            count = 0
            for name in self.host_names:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    if np.sum(mask_temp) == 1:
                        count += 1
                    data.append(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp]))
            print(count)
        elif oversample == True:
            for name in self.host_names:
                for i in range(0, len(data_dict[name]['pericenter.dist.sim'][mask_dict[name]])):
                    mask_temp = (data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i] != -1)
                    data.append(np.repeat(np.min(data_dict[name]['pericenter.dist.sim'][mask_dict[name]][i][mask_temp])), self.oversample[name])
        return np.hstack(data)

    def delta_dperi(self, data_dict, mask_dict, fraction=False, oversample=False):
        """
        TBD
        """
        data = []
        #
        if fraction == False:
            if oversample == False:
                for name in self.host_names:
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
                for name in self.host_names:
                    temp_array_model = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    temp_array_sim = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    data.append(np.repeat(temp_array_model,self.oversample[name]) - \
                                 np.repeat(temp_array_sim,self.oversample[name]))
        #
        elif fraction == True:
            if oversample == False:
                for name in self.host_names:
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
                for name in self.host_names:
                    temp_array_model = data_dict[name]['pericenter.dist.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    temp_array_sim = data_dict[name]['pericenter.dist.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = data_dict[name]['dtot.sim'][mask_dict[name]][:,0][mask_temp]
                    #
                    data.append((np.repeat(temp_array_model,self.oversample[name]) - \
                                 np.repeat(temp_array_sim,self.oversample[name]))\
                                 /np.repeat(temp_array_sim,self.oversample[name]))
        return np.hstack(data)

    def tperi_recent(self, data_dict, mask_dict, selection='sim', oversample=False):
        """
        TBD
        """
        data = []
        #
        if (selection == 'sim'):
            if oversample == False:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(np.repeat(temp_array, self.oversample[name]))
        #
        elif (selection == 'model'):
            if oversample == False:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(temp_array)
            #
            elif oversample == True:
                for name in self.host_names:
                    temp_array = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array == -1)
                    temp_array[mask_temp] = 0.0
                    data.append(np.repeat(temp_array, self.oversample[name]))
        #
        return np.hstack(data)

    def delta_tperi(self, data_dict, mask_dict, fraction=False, oversample=False):
        """
        TBD
        """
        data = []
        #
        if fraction == False:
            if oversample == False:
                for name in self.host_names:
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
                for name in self.host_names:
                    temp_array_model = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = 0.0
                    #
                    temp_array_sim = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = 0.0
                    #
                    data.append(np.repeat(temp_array_model,self.oversample[name]) - \
                                 np.repeat(temp_array_sim,self.oversample[name]))
        #
        elif fraction == True:
            if oversample == False:
                for name in self.host_names:
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
                for name in self.host_names:
                    temp_array_model = data_dict[name]['pericenter.time.lb.galpy'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_model == -1)
                    temp_array_model[mask_temp] = 0.0
                    #
                    temp_array_sim = data_dict[name]['pericenter.time.lb.sim'][mask_dict[name]][:,0]
                    mask_temp = (temp_array_sim == -1)
                    temp_array_sim[mask_temp] = 0.0
                    #
                    ratio = (np.repeat(temp_array_model,self.oversample[name]) - \
                                 np.repeat(temp_array_sim,self.oversample[name]))\
                                 /np.repeat(temp_array_sim,self.oversample[name])
                    ratio[~np.isfinite(ratio)] = 0
                    data.append(ratio)
        return np.hstack(data)

    def first_infall(self, data_dict, mask_dict, oversample=False):
        """
        TBD
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names:
                data.append(data_dict[name]['first.infall.time.lb'][mask_dict[name]])
        #
        elif oversample == True:
            for name in self.host_names:
                data.append(np.repeat(data_dict[name]['first.infall.time.lb'][mask_dict[name]], self.oversample[name]))
        #
        return np.hstack(data)

    def mstar(self, data_dict, mask_dict, selection='z0', oversample=False):
        """
        TBD
        """
        data = []
        #
        if selection == 'z0':
            if oversample == False:
                for name in self.host_names:
                    data.append(data_dict[name]['Mstar.z0'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['Mstar.z0'][mask_dict[name]], self.oversample[name]))
        #
        elif selection == 'peak':
            if oversample == False:
                for name in self.host_names:
                    data.append(data_dict[name]['Mstar.peak'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['Mstar.peak'][mask_dict[name]], self.oversample[name]))
        #
        return np.hstack(data)

    def mhalo(self, data_dict, mask_dict, selection='z0', oversample=False):
        """
        TBD
        """
        data = []
        #
        if selection == 'z0':
            if oversample == False:
                for name in self.host_names:
                    data.append(data_dict[name]['Mhalo.z0'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['Mhalo.z0'][mask_dict[name]], self.oversample[name]))
        #
        elif selection == 'peak':
            if oversample == False:
                for name in self.host_names:
                    data.append(data_dict[name]['Mhalo.peak'][mask_dict[name]])
            #
            elif oversample == True:
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['Mhalo.peak'][mask_dict[name]], self.oversample[name]))
        #
        return np.hstack(data)

    def d_z0(self, data_dict, mask_dict, oversample=False):
        """
        TBD
        """
        data = []
        #
        if oversample == False:
            for name in self.host_names:
                data.append(data_dict[name]['dtot.sim'][mask_dict[name]][:,0])
        #
        elif oversample == True:
            for name in self.host_names:
                data.append(np.repeat(data_dict[name]['dtot.sim'][mask_dict[name]][:,0], self.oversample[name]))
        #
        return np.hstack(data)


class SummaryDataPlot(SummaryDataSort):

    def __init__(self):
        """
        TBD
        """
        SummaryDataSort.__init__(self)
        #
        self.colors = ['#2f4f4f', '#006400', '#8b0000', '#000080', '#00ced1',\
                       '#ff8c00', '#c71585', '#7fff00', '#00fa9a', '#0000ff',\
                       '#ff00ff', '#1e90ff', '#f0e68c', '#ffc0cb']
        #
        self.labels = {'d.sim': 'd$_{\\rm peri,sim}$ [kpc]',\
                       'd.model': 'd$_{\\rm peri,model}$ [kpc]',\
                       'd.z0': 'd(z = 0) [kpc]',\
                       'delta.d.frac': '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$)/d$_{\\rm peri,sim}$',\
                       'delta.d': '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$) [kpc]',\
                       't.sim': 't$_{\\rm peri,lb,sim}$ [Gyr]',\
                       't.model': 't$_{\\rm peri,lb,model}$ [Gyr]',\
                       't.infall': 't$_{\\rm infall,lb}$ [Gyr]',\
                       'delta.t.frac': '(t$_{\\rm peri,model}$ - t$_{\\rm peri,sim}$)/t$_{\\rm peri,sim}$',\
                       'delta.t': '(t$_{\\rm peri,model}$ - t$_{\\rm peri,sim}$) [Gyr]',\
                       'N.sim': 'N$_{\\rm peri,sim}$',\
                       'N.model': 'N$_{\\rm peri,model}$',\
                       'delta.N': 'N$_{\\rm model}$ - N$_{\\rm sim}$',\
                       'M.star.z0': 'log$_{\\rm 10}$[M$_{\\rm star}(z = 0)$/M$_{\\odot}$]',\
                       'M.star.peak': 'log$_{\\rm 10}$[M$_{\\rm star, peak}$/M$_{\\odot}$]',\
                       'M.halo.z0': 'log$_{\\rm 10}$[M$_{\\rm halo}(z = 0)$/M$_{\\odot}$]',\
                       'M.halo.peak': 'log$_{\\rm 10}$[M$_{\\rm halo, peak}$/M$_{\\odot}$]'}
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
        TBD
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
        if (xtype == 'd.sim' or xtype == 't.sim') & (ytype == 'd.model' or ytype == 't.model'):
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

    def dperi_comparison_median(self, x, y, binsize, file_path_and_name, limits=None):
        """
        TBD
        """
        minn = int(binsize*np.floor(np.min(x)/binsize))
        maxx = int(binsize*np.ceil(np.max(x)/binsize))
        bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
        bins = np.linspace(minn, maxx, bin_num)
        #
        half_bin = (bins[1]-bins[0])/2
        onesigp = 84.13
        onesigm = 15.87
        #
        med = np.zeros(len(bins)-1)
        lower = np.zeros(len(bins)-1)
        upper = np.zeros(len(bins)-1)
        #
        for i in range(0, len(bins)-1):
            mask = (x >= bins[i]) & (x <= bins[i+1])
            med[i] = np.nanmedian(y[mask])
            upper[i] = np.nanpercentile(y[mask], onesigp)
            lower[i] = np.nanpercentile(y[mask], onesigm)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel('d$_{\\rm peri,sim}$ [kpc]', fontsize=28)
        plt.ylabel('d$_{\\rm peri,model}$ [kpc]', fontsize=28)
        plt.title('Most Recent Pericenter', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def plot_hist(self, x, binsize, file_path_and_name, pdf=False, xtype='d.sim', xlimits=None):
        """
        TBD
        """
        if pdf:
            den = True
            y_label = 'PDF'
        else:
            den = False
            y_label = 'N'
        #
        # Plot the data
        plt.figure(figsize=(10, 8))
        #
        if 'N.' not in xtype:
            if binsize*np.floor(np.min(x)/binsize) % 1 != 0:
                minn = int(np.floor(binsize*np.floor(np.min(x)/binsize)))
            else:
                minn = int(binsize*np.floor(np.min(x)/binsize))
            if binsize*np.ceil(np.max(x)/binsize) % 1 != 0:
                maxx = int(np.ceil(binsize*np.ceil(np.max(x)/binsize)))
            else:
                maxx = int(binsize*np.ceil(np.max(x)/binsize))
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            # Calculate the scatter
            onesigp = 84.13
            onesigm = 15.87
            sigma_one_op = np.nanpercentile(x, onesigp)
            sigma_one_om = np.nanpercentile(x, onesigm)
            #
            y_med = np.max(np.histogram(x, bin_array, normed=den)[0])*1.1
            #
            plt.hist(x, bin_array, density=den, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            plt.errorbar(np.median(x), y_med, xerr=np.array([[np.median(x)-sigma_one_om],[sigma_one_op-np.median(x)]]), color='k', lw=5, capsize=8)
            plt.scatter(np.median(x), y_med, s=250, marker='s', c='k')
        #
        elif 'N.' in xtype:
            minn = int(binsize*np.floor(np.min(x)/binsize))-0.5
            maxx = int(binsize*np.ceil(np.max(x)/binsize))+0.5
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            y_mean = np.max(np.histogram(x, bin_array, normed=den)[0])*1.1
            #
            plt.hist(x, bin_array, density=den, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            plt.errorbar(np.mean(x), y_mean, xerr=np.array([[2*np.std(x)],[2*np.std(x)]]), color='k', lw=5, capsize=8, alpha=0.3)
            plt.errorbar(np.mean(x), y_mean, xerr=np.array([[np.std(x)],[np.std(x)]]), color='k', lw=5, capsize=8)
            plt.scatter(np.mean(x), y_mean, s=250, marker='s', c='k')
        #
        plt.xlim(xlimits)
        plt.xlabel(self.labels[xtype], fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        plt.title(self.titles[xtype], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def delta_dperi_hist(self, x, binsize, file_path_and_name, xlimits=None, fraction=True, pdf=True):
        """
        TBD
        """
        #
        if fraction:
            x_label = '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$)/d$_{\\rm peri,sim}$'
        else:
            x_label = '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$) [kpc]'
        #
        minn = int(binsize*np.floor(np.min(x)/binsize))
        maxx = int(binsize*np.ceil(np.max(x)/binsize))
        bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
        bin_array = np.linspace(minn, maxx, bin_num)
        #
        y_med = np.max(np.histogram(x, bin_array, normed=pdf)[0])*1.1
        #
        # Calculate the scatter
        onesigp = 84.13
        onesigm = 15.87
        sigma_one_op = np.nanpercentile(x, onesigp)
        sigma_one_om = np.nanpercentile(x, onesigm)
        #
        # Plot the data
        plt.figure(figsize=(10, 8))
        plt.hist(x, bin_array, density=True, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
        #
        plt.errorbar(np.median(x), y_med, xerr=np.array([[np.median(x)-sigma_one_om],[sigma_one_op-np.median(x)]]), color='k', lw=5, capsize=8)
        plt.scatter(np.median(x), y_med, s=250, marker='s', c='k')
        #
        plt.xlim(xlimits)
        plt.xlabel(x_label, fontsize=28)
        plt.ylabel('PDF', fontsize=28)
        plt.title('Recent Pericenter Distances', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def delta_dperi_vs_prop_median(self, x, y, file_path_and_name, binsize, versus='d.sim', limits=None, fraction=True):
        """
        TBD
        """
        if fraction:
            y_label = '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$)/d$_{\\rm peri,sim}$'
        else:
            y_label = '(d$_{\\rm peri,model}$ - d$_{\\rm peri,sim}$) [kpc]'
        #
        if 'N.' not in versus:
            if 'M.' not in versus:
                minn = int(binsize*np.floor(np.min(x)/binsize))
                maxx = int(binsize*np.ceil(np.max(x)/binsize))
                bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
                onesigp = 84.13
                onesigm = 15.87
                #
                med = np.zeros(len(bins)-1)
                lower = np.zeros(len(bins)-1)
                upper = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (x >= bins[i]) & (x <= bins[i+1])
                    med[i] = np.nanmedian(y[mask])
                    upper[i] = np.nanpercentile(y[mask], onesigp)
                    lower[i] = np.nanpercentile(y[mask], onesigm)
            elif 'M.' in versus:
                if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                    minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
                else:
                    minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
                if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                    maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
                else:
                    maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
                bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
                #
                onesigp = 84.13
                onesigm = 15.87
                #
                med = np.zeros(len(bins)-1)
                lower = np.zeros(len(bins)-1)
                upper = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (np.log10(x) >= bins[i]) & (np.log10(x) <= bins[i+1])
                    med[i] = np.nanmedian(y[mask])
                    upper[i] = np.nanpercentile(y[mask], onesigp)
                    lower[i] = np.nanpercentile(y[mask], onesigm)
            #
        if 'N.' in versus:
            minn = int(binsize*np.floor(np.min(x)/binsize))-0.5
            maxx = int(binsize*np.ceil(np.max(x)/binsize))+0.5
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            onesigp = 84.13
            onesigm = 15.87
            #
            med = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                med[i] = np.nanmedian(y[mask])
                upper[i] = np.nanpercentile(y[mask], onesigp)
                lower[i] = np.nanpercentile(y[mask], onesigm)
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(self.labels[versus], fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        plt.title('Most Recent Pericenter', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def tperi_comparison_median(self, x, y, binsize, file_path_and_name, limits=None):
        """
        TBD
        """
        if binsize*np.floor(np.min(x)/binsize) % 1 != 0:
            minn = int(np.floor(binsize*np.floor(np.min(x)/binsize)))
        else:
            minn = int(binsize*np.floor(np.min(x)/binsize))
        if binsize*np.ceil(np.max(x)/binsize) % 1 != 0:
            maxx = int(np.ceil(binsize*np.ceil(np.max(x)/binsize)))
        else:
            maxx = int(binsize*np.ceil(np.max(x)/binsize))
        bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
        bins = np.linspace(minn, maxx, bin_num)
        half_bin = (bins[1]-bins[0])/2
        #
        onesigp = 84.13
        onesigm = 15.87
        #
        med = np.zeros(len(bins)-1)
        lower = np.zeros(len(bins)-1)
        upper = np.zeros(len(bins)-1)
        #
        for i in range(0, len(bins)-1):
            mask = (x >= bins[i]) & (x <= bins[i+1])
            med[i] = np.nanmedian(y[mask])
            upper[i] = np.nanpercentile(y[mask], onesigp)
            lower[i] = np.nanpercentile(y[mask], onesigm)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel('t$_{\\rm peri,sim}$ [Gyr]', fontsize=28)
        plt.ylabel('t$_{\\rm peri,model}$ [Gyr]', fontsize=28)
        plt.title('Most Recent Pericenter', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def delta_tperi_hist(self, x, binsize, file_path_and_name, xlimits=None, fraction=True, pdf=True):
        """
        TBD
        """
        #
        if fraction:
            x_label = '(t$_{\\rm peri,lb,model}$ - t$_{\\rm peri,lb,sim}$)/t$_{\\rm peri,lb,sim}$'
        else:
            x_label = '(t$_{\\rm peri,lb,model}$ - t$_{\\rm peri,lb,sim}$) [Gyr]'
        #
        if binsize*np.floor(np.min(x)/binsize) % 1 != 0:
            minn = int(np.floor(binsize*np.floor(np.min(x)/binsize)))
        else:
            minn = int(binsize*np.floor(np.min(x)/binsize))
        if binsize*np.ceil(np.max(x)/binsize) % 1 != 0:
            maxx = int(np.ceil(binsize*np.ceil(np.max(x)/binsize)))
        else:
            maxx = int(binsize*np.ceil(np.max(x)/binsize))
        bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
        bin_array = np.linspace(minn, maxx, bin_num)
        #
        y_med = np.max(np.histogram(x, bin_array, normed=pdf)[0])*1.1
        #
        # Calculate the scatter
        onesigp = 84.13
        onesigm = 15.87
        sigma_one_op = np.nanpercentile(x, onesigp)
        sigma_one_om = np.nanpercentile(x, onesigm)
        #
        # Plot the data
        plt.figure(figsize=(10, 8))
        plt.hist(x, bin_array, density=True, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
        #
        plt.errorbar(np.median(x), y_med, xerr=np.array([[np.median(x)-sigma_one_om],[sigma_one_op-np.median(x)]]), color='k', lw=5, capsize=8)
        plt.scatter(np.median(x), y_med, s=250, marker='s', c='k')
        #
        if xlimits:
            plt.xlim(xlimits)
        plt.xlabel(x_label, fontsize=28)
        plt.ylabel('PDF', fontsize=28)
        plt.title('Recent Pericenter Lookback Times', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def delta_tperi_vs_prop_median(self, x, y, file_path_and_name, binsize, versus='t.sim', limits=None, fraction=True):
        """
        TBD
        """
        #
        if fraction:
            y_label = '(t$_{\\rm peri,lb,model}$ - t$_{\\rm peri,lb,sim}$)/t$_{\\rm peri,lb,sim}$'
        else:
            y_label = '(t$_{\\rm peri,lb,model}$ - t$_{\\rm peri,lb,sim}$) [Gyr]'
        #
        if 'N.' not in versus:
            if 'M.' not in versus:
                minn = int(binsize*np.floor(np.min(x)/binsize))
                maxx = int(binsize*np.ceil(np.max(x)/binsize))
                bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
                onesigp = 84.13
                onesigm = 15.87
                #
                med = np.zeros(len(bins)-1)
                lower = np.zeros(len(bins)-1)
                upper = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (x >= bins[i]) & (x <= bins[i+1])
                    med[i] = np.nanmedian(y[mask])
                    upper[i] = np.nanpercentile(y[mask], onesigp)
                    lower[i] = np.nanpercentile(y[mask], onesigm)
            elif 'M.' in versus:
                if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                    minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
                else:
                    minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
                if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                    maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
                else:
                    maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
                bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
                #
                onesigp = 84.13
                onesigm = 15.87
                #
                med = np.zeros(len(bins)-1)
                lower = np.zeros(len(bins)-1)
                upper = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (np.log10(x) >= bins[i]) & (np.log10(x) <= bins[i+1])
                    med[i] = np.nanmedian(y[mask])
                    upper[i] = np.nanpercentile(y[mask], onesigp)
                    lower[i] = np.nanpercentile(y[mask], onesigm)
            #
        if 'N.' in versus:
            minn = int(binsize*np.floor(np.min(x)/binsize))-0.5
            maxx = int(binsize*np.ceil(np.max(x)/binsize))+0.5
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            onesigp = 84.13
            onesigm = 15.87
            #
            med = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                med[i] = np.nanmedian(y[mask])
                upper[i] = np.nanpercentile(y[mask], onesigp)
                lower[i] = np.nanpercentile(y[mask], onesigm)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(self.labels[versus], fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        plt.title('Most Recent Pericenter', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def delta_nperi_vs_prop_median(self, x, y, binsize, file_path_and_name, versus='d.sim', limits=None):
        #
        if 'N.' not in versus:
            if 'M.' not in versus:
                minn = int(binsize*np.floor(np.min(x)/binsize))
                maxx = int(binsize*np.ceil(np.max(x)/binsize))
                bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
                #
                means = np.zeros(len(bins)-1)
                scatter = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (x >= bins[i]) & (x <= bins[i+1])
                    means[i] = np.nanmean(y[mask])
                    scatter[i] = np.nanstd(y[mask])
            elif 'M.' in versus:
                if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                    minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
                else:
                    minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
                if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                    maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
                else:
                    maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
                bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
                bins = np.linspace(minn, maxx, bin_num)
                #
                half_bin = (bins[1]-bins[0])/2
                #
                means = np.zeros(len(bins)-1)
                scatter = np.zeros(len(bins)-1)
                #
                for i in range(0, len(bins)-1):
                    mask = (np.log10(x) >= bins[i]) & (np.log10(x) <= bins[i+1])
                    means[i] = np.nanmean(y[mask])
                    scatter[i] = np.nanstd(y[mask])
        if 'N.' in versus:
            means = np.zeros(np.max(x)+1)
            scatter = np.zeros(np.max(x)+1)
            bins = np.arange(np.max(x)+1)
            #
            for i in range(0, np.max(x)+1):
                mask = (x == i)
                means[i] = np.nanmean(y[mask])
                scatter[i] = np.std(y[mask])
        #
        f, ax = plt.subplots(figsize=(10, 8))
        if 'N.' not in versus:
            plt.plot(bins[:-1]+half_bin, means, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
            plt.errorbar(bins[:-1]+half_bin, means, yerr=np.array([scatter, scatter]), color=self.colors[1], alpha=0.5, lw=5, capsize=8)
        elif 'N.' in versus:
            plt.plot(np.arange(np.max(x)+1), means, color=self.colors[1], alpha=0.5)
            plt.errorbar(np.arange(np.max(x)+1), means, yerr=np.array([scatter, scatter]), color=self.colors[1], alpha=0.5, lw=5, capsize=8)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(self.labels[versus], fontsize=28)
        plt.ylabel('N$_{\\rm model}$ - N$_{\\rm sim}$', fontsize=28)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def nperi_vs_prop_median(self, x, y, binsize, file_path_and_name, xtype='d.sim', ytype='N.sim', limits=None):
        if 'M.' not in xtype:
            minn = int(binsize*np.floor(np.min(x)/binsize))
            maxx = int(binsize*np.ceil(np.max(x)/binsize))
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            #
            means = np.zeros(len(bins)-1)
            scatter = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                means[i] = np.nanmean(y[mask])
                scatter[i] = np.nanstd(y[mask])
        elif 'M.' in xtype:
            if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
            else:
                minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
            if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
            else:
                maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
            bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            #
            means = np.zeros(len(bins)-1)
            scatter = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (np.log10(x) >= bins[i]) & (np.log10(x) <= bins[i+1])
                means[i] = np.nanmean(y[mask])
                scatter[i] = np.nanstd(y[mask])
        #
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, means, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.errorbar(bins[:-1]+half_bin, means, yerr=np.array([scatter, scatter]), color=self.colors[1], alpha=0.5, lw=5, capsize=8)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(self.labels[xtype], fontsize=28)
        plt.ylabel(self.labels[ytype], fontsize=28)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def mstar_hist(self, x, binsize, file_path_and_name, log=False, selection='z0', xlimits=None, pdf=True):
        """
        TBD
        """
        #
        if log == True:
            if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
            else:
                minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
            if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
            else:
                maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
            bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            if selection == 'z0':
                x_label = 'log$_{\\rm 10}$($M_{\\rm star}(z = 0)/M_{\\odot}$)'
            elif selection == 'peak':
                x_label = 'log$_{\\rm 10}$($M_{\\rm star,peak}/M_{\\odot}$)'
            #
            if pdf:
                y_label = 'PDF'
            else:
                y_label = 'N'
            #
            # Calculate the scatter
            onesigp = 84.13
            onesigm = 15.87
            sigma_one_op = np.nanpercentile(np.log10(x), onesigp)
            sigma_one_om = np.nanpercentile(np.log10(x), onesigm)
            #
            y_med = np.max(np.histogram(np.log10(x), bin_array, normed=pdf)[0])*1.1
            #
            # Plot the data
            plt.figure(figsize=(10, 8))
            plt.hist(np.log10(x), bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            #
            plt.errorbar(np.median(np.log10(x)), y_med, xerr=np.array([[np.median(np.log10(x))-sigma_one_om],[sigma_one_op-np.median(np.log10(x))]]), color='k', lw=5, capsize=8)
            plt.scatter(np.median(np.log10(x)), y_med, s=250, marker='s', c='k')
            #
            if xlimits:
                plt.xlim(xlimits)
            plt.xlabel(x_label, fontsize=28)
            plt.ylabel(y_label, fontsize=28)
            plt.title('Satellite M$_{\\rm star}$ Histogram', fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=24)
            plt.tight_layout()
            plt.savefig(file_path_and_name)
            plt.close()
        elif log == False:
            print('Haven\'t thought far enough ahead to bin outside of log-space :(')

    def infall_vs_prop_median(self, x, y, binsize, file_path_and_name, xtype='d.z0', limits=None):
        #
        if 'M.' not in xtype:
            minn = int(binsize*np.floor(np.min(x)/binsize))
            maxx = int(binsize*np.ceil(np.max(x)/binsize))
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            onesigp = 84.13
            onesigm = 15.87
            #
            med = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                med[i] = np.nanmedian(y[mask])
                upper[i] = np.nanpercentile(y[mask], onesigp)
                lower[i] = np.nanpercentile(y[mask], onesigm)
        elif 'M.' in xtype:
            if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
            else:
                minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
            if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
            else:
                maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
            bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            #
            onesigp = 84.13
            onesigm = 15.87
            #
            med = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (np.log10(x) >= bins[i]) & (np.log10(x) <= bins[i+1])
                med[i] = np.nanmedian(y[mask])
                upper[i] = np.nanpercentile(y[mask], onesigp)
                lower[i] = np.nanpercentile(y[mask], onesigm)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(self.labels[xtype], fontsize=28)
        plt.ylabel('t$_{\\rm infall,lb}$ [Gyr]', fontsize=28)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def mstar_vs_prop_median(self, x, y, binsize, file_path_and_name, xtype='M.z0', ytype='d.z0', limits=None):
        if xtype == 'M.z0':
            x_label = 'log$_{\\rm 10}$($M_{\\rm star}(z = 0)/M_{\\odot}$)'
        elif xtype == 'M.peak':
            x_label = 'log$_{\\rm 10}$($M_{\\rm star, peak}/M_{\\odot}$)'
        #
        if ytype == 'd.z0':
            y_label = 'd(z = 0) [kpc]'
        #
        if 'M.' not in xtype:
            minn = int(binsize*np.floor(np.min(x)/binsize))
            maxx = int(binsize*np.ceil(np.max(x)/binsize))
            bin_num = int((np.abs(minn)+np.abs(maxx))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            onesigp = 84.13
            onesigm = 15.87
            #
            med = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (x >= bins[i]) & (x <= bins[i+1])
                med[i] = np.nanmedian(y[mask])
                upper[i] = np.nanpercentile(y[mask], onesigp)
                lower[i] = np.nanpercentile(y[mask], onesigm)
        elif 'M.' in xtype:
            if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
            else:
                minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
            if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
            else:
                maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
            bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
            bins = np.linspace(minn, maxx, bin_num)
            #
            half_bin = (bins[1]-bins[0])/2
            #
            onesigp = 84.13
            onesigm = 15.87
            #
            med = np.zeros(len(bins)-1)
            lower = np.zeros(len(bins)-1)
            upper = np.zeros(len(bins)-1)
            #
            for i in range(0, len(bins)-1):
                mask = (np.log10(x) >= bins[i]) & (np.log10(x) <= bins[i+1])
                med[i] = np.nanmedian(y[mask])
                upper[i] = np.nanpercentile(y[mask], onesigp)
                lower[i] = np.nanpercentile(y[mask], onesigm)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(x_label, fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def mhalo_hist(self, x, binsize, file_path_and_name, log=False, selection='z0', xlimits=None, pdf=True):
        """
        TBD
        """
        #
        if log == True:
            if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
                minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
            else:
                minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
            if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
                maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
            else:
                maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
            bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
            bin_array = np.linspace(minn, maxx, bin_num)
            #
            if pdf:
                y_label = 'PDF'
            else:
                y_label = 'N'
            #
            y_med = np.nanmax(np.histogram(np.log10(x), bin_array, normed=pdf)[0])*1.1
            #
            if selection == 'z0':
                x_label = 'log$_{\\rm 10}$($M_{\\rm halo}(z = 0)/M_{\\odot}$)'
            elif selection == 'peak':
                x_label = 'log$_{\\rm 10}$($M_{\\rm halo,peak}/M_{\\odot}$)'
            #
            # Calculate the scatter
            onesigp = 84.13
            onesigm = 15.87
            sigma_one_op = np.nanpercentile(np.log10(x), onesigp)
            sigma_one_om = np.nanpercentile(np.log10(x), onesigm)
            #
            # Plot the data
            plt.figure(figsize=(10, 8))
            plt.hist(np.log10(x), bin_array, density=pdf, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
            #
            plt.errorbar(np.median(np.log10(x)), y_med, xerr=np.array([[np.median(np.log10(x))-sigma_one_om],[sigma_one_op-np.median(np.log10(x))]]), color='k', lw=5, capsize=8)
            plt.scatter(np.median(np.log10(x)), y_med, s=250, marker='s', c='k')
            #
            if xlimits:
                plt.xlim(xlimits)
            plt.xlabel(x_label, fontsize=28)
            plt.ylabel(y_label, fontsize=28)
            plt.title('Satellite M$_{\\rm halo}$ Histogram', fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=24)
            plt.tight_layout()
            plt.savefig(file_path_and_name)
            plt.close()
        elif log == False:
            print('Haven\'t thought far enough ahead to bin outside of log-space :(')

    def mstar_mhalo_median(self, x, y, binsize, file_path_and_name, masstype='z0',limits=None):
        if masstype == 'z0':
            x_label = self.labels['M.halo.z0']
            y_label = self.labels['M.star.z0']
        elif masstype == 'peak':
            x_label = self.labels['M.halo.peak']
            y_label = self.labels['M.star.peak']
        #
        if binsize*np.floor(np.min(np.log10(x))/binsize) % 1 != 0:
            minn = int(np.floor(binsize*np.floor(np.min(np.log10(x))/binsize)))
        else:
            minn = int(binsize*np.floor(np.min(np.log10(x))/binsize))
        if binsize*np.ceil(np.max(np.log10(x))/binsize) % 1 != 0:
            maxx = int(np.ceil(binsize*np.ceil(np.max(np.log10(x))/binsize)))
        else:
            maxx = int(binsize*np.ceil(np.max(np.log10(x))/binsize))
        bin_num = int((np.abs(maxx)-np.abs(minn))/binsize+1)
        bins = np.linspace(minn, maxx, bin_num)
        #
        half_bin = (bins[1]-bins[0])/2
        #
        onesigp = 84.13
        onesigm = 15.87
        #
        med = np.zeros(len(bins)-1)
        lower = np.zeros(len(bins)-1)
        upper = np.zeros(len(bins)-1)
        #
        for i in range(0, len(bins)-1):
            mask = (np.log10(x) >= bins[i]) & (np.log10(x) <= bins[i+1])
            med[i] = np.nanmedian(np.log10(y)[mask])
            upper[i] = np.nanpercentile(np.log10(y)[mask], onesigp)
            lower[i] = np.nanpercentile(np.log10(y)[mask], onesigm)
        #
        f, ax = plt.subplots(figsize=(10, 8))
        plt.plot(bins[:-1]+half_bin, med, color=self.colors[1], marker='s', markersize=10, alpha=0.5)
        plt.fill_between(bins[:-1]+half_bin, upper, lower, color=self.colors[1], alpha=0.3)
        if limits:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        plt.xlabel(x_label, fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()
