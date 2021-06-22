#!/usr/bin/python3

"""
@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>

    This was written to create summary statistic plots with data output from
    orbit_io.py and model_io.py. Data that I import was previously compiled
    using summary_data.py

"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class SummaryDataSort:

    def __init__(self):
        """
        Initialize the sorting class.

        Want to save:
            - Host names to loop over in the following methods
            - Oversampling factors for either:
                - CASE 1: Cases where there are pericenters in the simulations,
                          but not required in the model
                - CASE 2: Cases where there are pericenters in both the
                          simulations and the model.

                NOTE: It would be nice to not hard-code these factors...
        """
        # Create a list of host names
        self.host_names = ['m12b', 'm12c', 'm12f', 'm12i', 'm12m', 'm12r', 'm12w', 'm12z', \
                           'Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']
        #
        # Oversampling factors for CASE 1
        self.oversample1 = {'m12b': 16, 'm12c': 14, 'm12f': 13, 'm12i': 22, 'm12m': 12,\
                           'm12r': 20, 'm12w': 14, 'm12z': 21, 'Romeo': 16, 'Juliet': 14,\
                           'Thelma': 17, 'Louise': 16, 'Romulus': 10, 'Remus': 17}
        #
        # Oversampling factors for CASE 2
        self.oversample2 = {'m12b': 15, 'm12c': 14, 'm12f': 12, 'm12i': 21, 'm12m': 12,\
                           'm12r': 20, 'm12w': 13, 'm12z': 22, 'Romeo': 16, 'Juliet': 13,\
                           'Thelma': 16, 'Louise': 15, 'Romulus': 10, 'Remus': 17}

    def data_mask(self, dictionary, outliers=False, peri_sim=True, peri_model=False, current_sat=False):
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
        # If interested in outliers do this.
        elif outliers == True:
            for name in self.host_names:
                mask_dict[name] = dictionary[name]['infall.check']*dictionary[name]['pericenter.check.sim']*(~dictionary[name]['pericenter.check.galpy'])
        return mask_dict

    def delta_nperi(self, data_dict, mask_dict, oversample=True, peri_selection='sim'):
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
            if peri_selection == 'sim':
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]],self.oversample1[name]) - \
                                 np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]],self.oversample1[name]))
            elif peri_selection == 'both':
                for name in self.host_names:
                    data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]],self.oversample2[name]) - \
                                 np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]],self.oversample2[name]))
        #
        elif oversample == False:
            for name in self.host_names:
                data.append(data_dict[name]['N.peri.galpy'][mask_dict[name]] - \
                            data_dict[name]['N.peri.sim'][mask_dict[name]])
        #
        return np.hstack(data)

    def nperi(self, data_dict, mask_dict, oversample=False, selection='sim', peri_selection='sim'):
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
                if peri_selection == 'sim':
                    for name in self.host_names:
                        data.append(np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]], self.oversample1[name]))
                elif peri_selection == 'both':
                    for name in self.host_names:
                        data.append(np.repeat(data_dict[name]['N.peri.sim'][mask_dict[name]], self.oversample2[name]))
        #
        elif selection == 'model':
            if oversample == False:
                for name in self.host_names:
                    data.append(data_dict[name]['N.peri.galpy'][mask_dict[name]])
            #
            elif oversample == True:
                if peri_selection == 'sim':
                    for name in self.host_names:
                        data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]], self.oversample1[name]))
                elif peri_selection == 'both':
                    for name in self.host_names:
                        data.append(np.repeat(data_dict[name]['N.peri.galpy'][mask_dict[name]], self.oversample2[name]))
        return np.hstack(data)

class SummaryDataPlot(SummaryDataSort):

    def __init__(self):
        """
        TBD
        """
        SummaryDataSort.__init__(self)
        self.colors = ['#2f4f4f', '#006400', '#8b0000', '#000080', '#00ced1',\
                       '#ff8c00', '#c71585', '#7fff00', '#00fa9a', '#0000ff',\
                       '#ff00ff', '#1e90ff', '#f0e68c', '#ffc0cb']

    def delta_nperi_hist(self, x, bin_array, file_path_and_name):
        """
        TBD
        """
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.hist(x, bins=bin_array, density=True, linestyle='solid', linewidth=2, histtype='stepfilled', color=self.colors[3], alpha=0.4)
        #plt.errorbar(np.mean(delta_No_tot), 0.36, xerr=np.array([[np.mean(delta_No_tot)-sigma_one_om],[sigma_one_op-np.mean(delta_No_tot)]]), color='k', lw=5, capsize=8)
        plt.errorbar(np.mean(x), 0.40, xerr=np.array([[2*np.std(x)],[2*np.std(x)]]), color='k', lw=5, capsize=8, alpha=0.3)
        plt.errorbar(np.mean(x), 0.40, xerr=np.array([[np.std(x)],[np.std(x)]]), color='k', lw=5, capsize=8)
        plt.scatter(np.mean(x), 0.40, s=250, marker='s', c='k')
        #plt.text(6.5, 0.45,'Mean: '+str(np.around(np.mean(delta_No_tot), 2)), fontsize=18)
        plt.xlabel('N$_{\\rm model}$ - N$_{\\rm sim}$', fontsize=28)
        plt.ylabel('PDF', fontsize=28)
        plt.ylim(ymax=0.43)
        plt.title('Pericenters', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()

    def delta_nperi_scatter(self, x, y, file_path_and_name, versus='sim'):
        """
        TBD
        """
        if versus == 'sim':
            y_label = 'N$_{\\rm sim}$'
        elif versus == 'model':
            y_label = 'N$_{\\rm model}$'
        #
        f, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x, y, color='k', s=50, marker='x', alpha=0.5)
        plt.ylim(-0.5, 13.5)
        plt.xlabel('N$_{\\rm model}$ - N$_{\\rm sim}$', fontsize=28)
        plt.ylabel(y_label, fontsize=28)
        plt.title('Pericenters', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(file_path_and_name)
        plt.close()
