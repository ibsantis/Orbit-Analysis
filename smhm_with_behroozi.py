import os
import subprocess
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

#dir = '~/Desktop/umachine-dr1/data/smhm/params'
#script = '~/Desktop/umachine-dr1/data/smhm/params/gen_smhm.py'
#params = '~/Desktop/umachine-dr1/data/smhm/params/smhm_med_sat_params.txt'

#os.system('python '+dir+'/gen_smhm.py 0 '+dir+'/smhm_med_sat_params.txt')

#x = subprocess.Popen(['python '+script+' 0 '+params], stdout=subprocess.PIPE, shell=True)

#x = subprocess.getoutput('python '+dir+'/gen_smhm.py 0 '+dir+'/smhm_med_sat_params.txt')

#df = pd.DataFrame(eval(x))

#print(df)

#print(x)


################################################################################


#d = dict()
#f = open('~/simulation/orbit_data/smhm_behroozi_values.txt')

df = pd.read_csv('~/simulation/orbit_data/smhm_behroozi_values.txt', sep=' ', header=0, skiprows=[1,2])

x_data = df['Log10(Mpeak/Msun)'][:15]
y_data = df['Log10(Median_SM/Msun)'][:15]

ff = interpolate.interp1d(x=x_data, y=y_data, bounds_error=False, fill_value='extrapolate')
x_new = np.linspace(8,11.5,300)


#print(df['Log10(Mpeak/Msun)'])

#halo_masses = df['Log10(Mpeak/Msun)']
#print(halo_masses)
#print(halo_masses[:5])
f = plt.subplots(1, 1, figsize=(10,8))
plt.scatter(x_new, ff(x_new), color='k', s=1)
plt.plot(df['Log10(Mpeak/Msun)'][:15], df['Log10(Median_SM/Msun)'][:15])
plt.xlabel('$M_{\\rm halo, peak}$ [$M_{\\odot}$]')
plt.ylabel('$M_{\\rm star}$ [$M_{\\odot}$]')
plt.tight_layout()
#plt.show()
plt.savefig('/Users/isaiahsantistevan/Desktop/smhm_test.pdf')
