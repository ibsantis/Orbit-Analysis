


"""
    Try using the get_subhalos_match() function in elvis_plot.py
"""


import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np

hal = halo.io.IO.read_catalogs(snapshot_value=600, snapshot_value_kind='index', simulation_directory='/share/wetzellab/m12i/m12i_r7100', rockstar_directory='asdf')
