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
    pass

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