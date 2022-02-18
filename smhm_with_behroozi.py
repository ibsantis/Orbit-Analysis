import os
import subprocess
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

"""
    This part of the script is what I used to generate the data in 'smhm_behroozi_values.txt'
    stored in '/simulation/orbit_data/'.

    Tried making this fancy, but really just needed to type:
    python <path/to/his/script> <redshift> </path/to/parameters>
    i.e.
    python ~/Desktop/umachine-dr1/data/smhm/params/gen_smhm.py 0 ~/Desktop/umachine-dr1/data/smhm/params/smhm_med_sat_params.txt
"""

#dir = '~/Desktop/umachine-dr1/data/smhm/params'
#script = '~/Desktop/umachine-dr1/data/smhm/params/gen_smhm.py'
#params = '~/Desktop/umachine-dr1/data/smhm/params/smhm_med_sat_params.txt'

#os.system('python '+dir+'/gen_smhm.py 0 '+dir+'/smhm_med_sat_params.txt')


################################################################################

"""
    This part involves opening the data and making a quick plot to look at it, but
    I don't use this plot for anything, just to check it.
"""

df = pd.read_csv('~/simulation/orbit_data/smhm_behroozi_values.txt', sep=' ', header=0, skiprows=[1,2])

x_data = df['Log10(Mpeak/Msun)'][:15]
y_data = df['Log10(Median_SM/Msun)'][:15]

ff = interpolate.interp1d(x=x_data, y=y_data, bounds_error=False, fill_value='extrapolate')
x_new = np.linspace(8,11.5,300)


f = plt.subplots(1, 1, figsize=(10,8))
plt.scatter(x_new, ff(x_new), color='k', s=1)
plt.plot(df['Log10(Mpeak/Msun)'][:15], df['Log10(Median_SM/Msun)'][:15])
plt.xlabel('$M_{\\rm halo, peak}$ [$M_{\\odot}$]')
plt.ylabel('$M_{\\rm star}$ [$M_{\\odot}$]')
plt.tight_layout()
#plt.show()
plt.savefig('/Users/isaiahsantistevan/Desktop/smhm_test.pdf')
