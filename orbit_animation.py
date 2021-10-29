from galpy.orbit import Orbit
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
from astropy import units as u
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import animation
print('Read in the tools')

### Set path and initial parameters
sim_data = orbit_io.OrbitRead(gal1='m12i', location='mac')
print('Set paths')

data = ut.io.file_hdf5(file_name_base=sim_data.home_dir+'/orbit_data/hdf5_files/summary_data/data_Romulus')

colors = ['#2f4f4f', '#006400', '#8b0000', '#000080', '#00ced1',\
          '#ff8c00', '#c71585', '#7fff00', '#00fa9a', '#0000ff',\
          '#ff00ff', '#1e90ff', '#f0e68c', '#ffc0cb']

fig = plt.figure()
ax = plt.axes(projection='3d')
#
mask = (data['d.sim'][2][:,0] != -1)
ax.scatter3D(data['d.sim'][2][:,0][0], data['d.sim'][2][:,1][0], data['d.sim'][2][:,2][0], c=colors[0])
ax.plot3D(data['d.sim'][2][:,0][mask], data['d.sim'][2][:,1][mask], data['d.sim'][2][:,2][mask], colors[0], alpha=0.5)
mask = (data['d.sim'][4][:,0] != -1)
ax.scatter3D(data['d.sim'][4][:,0][0], data['d.sim'][4][:,1][0], data['d.sim'][4][:,2][0], c=colors[1])
ax.plot3D(data['d.sim'][4][:,0][mask], data['d.sim'][4][:,1][mask], data['d.sim'][4][:,2][mask], colors[1], alpha=0.5)
mask = (data['d.sim'][7][:,0] != -1)
ax.scatter3D(data['d.sim'][7][:,0][0], data['d.sim'][7][:,1][0], data['d.sim'][7][:,2][0], c=colors[2])
ax.plot3D(data['d.sim'][7][:,0][mask], data['d.sim'][7][:,1][mask], data['d.sim'][7][:,2][mask], colors[2], alpha=0.5)
mask = (data['d.sim'][31][:,0] != -1)
ax.scatter3D(data['d.sim'][31][:,0][0], data['d.sim'][31][:,1][0], data['d.sim'][31][:,2][0], c=colors[3])
ax.plot3D(data['d.sim'][31][:,0][mask], data['d.sim'][31][:,1][mask], data['d.sim'][31][:,2][mask], colors[3], alpha=0.5)
mask = (data['d.sim'][9][:,0] != -1)
ax.scatter3D(data['d.sim'][9][:,0][0], data['d.sim'][9][:,1][0], data['d.sim'][9][:,2][0], c=colors[4])
ax.plot3D(data['d.sim'][9][:,0][mask], data['d.sim'][9][:,1][mask], data['d.sim'][9][:,2][mask], colors[4], alpha=0.5)
#
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')


# EXAMPLE
# Single halo
fig = plt.figure()
ax = plt.axes(projection='3d')
mask = (data['d.sim'][2][:,0] != -1)
ax.set_xlim(np.min(data['d.sim'][2][:,0][mask]), np.max(data['d.sim'][2][:,0][mask]))
ax.set_ylim(np.min(data['d.sim'][2][:,1][mask]), np.max(data['d.sim'][2][:,1][mask]))
ax.set_zlim(np.min(data['d.sim'][2][:,2][mask]), np.max(data['d.sim'][2][:,2][mask]))
ax.set_xlabel('X [kpc]', labelpad=15.)
ax.set_ylabel('Y [kpc]', labelpad=15.)
ax.set_zlabel('Z [kpc]', labelpad=15.)
#ax.set_xlim(-400, 400)
#ax.set_ylim(-400, 400)
#ax.set_zlim(-400, 400)
line, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[0])
ax.scatter(0,0,0,color='black')

def init():
    line.set_data(np.array([]),np.array([]))
    line.set_3d_properties(np.array([]))
    return line,

def animate(i):
    x = data['d.sim'][2][:,0][mask][:i]
    y = data['d.sim'][2][:,1][mask][:i]
    z = data['d.sim'][2][:,2][mask][:i]
    #line.set_data(x,y,z)
    line.set_data(x,y)
    line.set_3d_properties(z)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data['d.sim'][2][:,0][mask])+1, interval=1, blit=False)
plt.show()




# EXAMPLE 2
# Multiple halos
fig = plt.figure()
ax = plt.axes(projection='3d')
mask2 = (data['d.sim'][2][:,0] != -1)
mask4 = (data['d.sim'][4][:,0] != -1)
mask29 = (data['d.sim'][29][:,0] != -1)
mask31 = (data['d.sim'][31][:,0] != -1)
#
ax.set_xlim(np.min(np.concatenate((data['d.sim'][2][:,0][mask2],data['d.sim'][4][:,0][mask4],data['d.sim'][29][:,0][mask29],data['d.sim'][31][:,0][mask31]))), np.max(np.concatenate((data['d.sim'][2][:,0][mask2],data['d.sim'][4][:,0][mask4],data['d.sim'][29][:,0][mask29],data['d.sim'][31][:,0][mask31]))))
ax.set_ylim(np.min(np.concatenate((data['d.sim'][2][:,1][mask2],data['d.sim'][4][:,1][mask4],data['d.sim'][29][:,1][mask29],data['d.sim'][31][:,1][mask31]))), np.max(np.concatenate((data['d.sim'][2][:,1][mask2],data['d.sim'][4][:,1][mask4],data['d.sim'][29][:,1][mask29],data['d.sim'][31][:,1][mask31]))))
ax.set_zlim(np.min(np.concatenate((data['d.sim'][2][:,2][mask2],data['d.sim'][4][:,2][mask4],data['d.sim'][29][:,2][mask29],data['d.sim'][31][:,2][mask31]))), np.max(np.concatenate((data['d.sim'][2][:,2][mask2],data['d.sim'][4][:,2][mask4],data['d.sim'][29][:,2][mask29],data['d.sim'][31][:,2][mask31]))))
#
ax.set_xlabel('X [kpc]', labelpad=15.)
ax.set_ylabel('Y [kpc]', labelpad=15.)
ax.set_zlabel('Z [kpc]', labelpad=15.)
#ax.set_xlim(-400, 400)
#ax.set_ylim(-400, 400)
#ax.set_zlim(-400, 400)
line2, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[0])
line4, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[1])
line29, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[2])
line8, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[3])
ax.scatter(0,0,0,color='black')

def init():
    line2.set_data(np.array([]),np.array([]))
    line2.set_3d_properties(np.array([]))
    line4.set_data(np.array([]),np.array([]))
    line4.set_3d_properties(np.array([]))
    line29.set_data(np.array([]),np.array([]))
    line29.set_3d_properties(np.array([]))
    line8.set_data(np.array([]),np.array([]))
    line8.set_3d_properties(np.array([]))
    return line2, line4, line29, line8,

def animate(i):
    x2 = data['d.sim'][2][:,0][mask2][:i]
    y2 = data['d.sim'][2][:,1][mask2][:i]
    z2 = data['d.sim'][2][:,2][mask2][:i]
    #
    x4 = data['d.sim'][4][:,0][mask4][:i]
    y4 = data['d.sim'][4][:,1][mask4][:i]
    z4 = data['d.sim'][4][:,2][mask4][:i]
    #
    x29 = data['d.sim'][29][:,0][mask29][:i]
    y29 = data['d.sim'][29][:,1][mask29][:i]
    z29 = data['d.sim'][29][:,2][mask29][:i]
    #
    x8 = data['d.sim'][31][:,0][mask31][:i]
    y8 = data['d.sim'][31][:,1][mask31][:i]
    z8 = data['d.sim'][31][:,2][mask31][:i]
    #line.set_data(x,y,z)
    line2.set_data(x2,y2)
    line2.set_3d_properties(z2)
    #
    line4.set_data(x4,y4)
    line4.set_3d_properties(z4)
    #
    line29.set_data(x29,y29)
    line29.set_3d_properties(z29)
    #
    line8.set_data(x8,y8)
    line8.set_3d_properties(z8)
    #
    return line2, line4, line29, line8,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data['d.sim'][2][:,0][mask2])-100, interval=0.1, blit=False)
plt.tight_layout()

#writergif = animation.PillowWriter(fps=30)
#anim.save('/Users/isaiahsantistevan/Desktop/test.gif',writer=writergif)

#anim.save('/Users/isaiahsantistevan/Desktop/test.gif')
plt.show()





# EXAMPLE 3 Plotting two subhalos orbiting each other in Louise
# Multiple halos
fig = plt.figure()
ax = plt.axes(projection='3d')
mask6 = (data['d.sim'][6][:,0] != -1)
mask116 = (data['d.sim'][116][:,0] != -1)
#
ax.set_xlim(np.min(np.concatenate((data['d.sim'][6][:,0][mask6],data['d.sim'][116][:,0][mask116]))), np.max(np.concatenate((data['d.sim'][6][:,0][mask6],data['d.sim'][116][:,0][mask116]))))
ax.set_ylim(np.min(np.concatenate((data['d.sim'][6][:,1][mask6],data['d.sim'][116][:,1][mask116]))), np.max(np.concatenate((data['d.sim'][6][:,1][mask6],data['d.sim'][116][:,1][mask116]))))
ax.set_zlim(np.min(np.concatenate((data['d.sim'][6][:,2][mask6],data['d.sim'][116][:,2][mask116]))), np.max(np.concatenate((data['d.sim'][6][:,2][mask6],data['d.sim'][116][:,2][mask116]))))
#
ax.set_xlabel('X [kpc]', labelpad=15.)
ax.set_ylabel('Y [kpc]', labelpad=15.)
ax.set_zlabel('Z [kpc]', labelpad=15.)
#
line6, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[0])
line116, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[1])
ax.scatter(0,0,0,color='black')

def init():
    line6.set_data(np.array([]),np.array([]))
    line6.set_3d_properties(np.array([]))
    line116.set_data(np.array([]),np.array([]))
    line116.set_3d_properties(np.array([]))
    return line6, line116,

def animate(i):
    x6 = data['d.sim'][6][:,0][mask6][:i]
    y6 = data['d.sim'][6][:,1][mask6][:i]
    z6 = data['d.sim'][6][:,2][mask6][:i]
    #
    x116 = data['d.sim'][116][:,0][mask116][:i]
    y116 = data['d.sim'][116][:,1][mask116][:i]
    z116 = data['d.sim'][116][:,2][mask116][:i]
    #
    line6.set_data(x6,y6)
    line6.set_3d_properties(z6)
    #
    line116.set_data(x116,y116)
    line116.set_3d_properties(z116)
    #
    ax.view_init(elev=30, azim=0.6*i)
    #
    return line6, line116,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data['d.sim'][6][:,0][mask6])-10, interval=0.1, blit=False)
plt.tight_layout()

writergif = animation.PillowWriter(fps=30)
anim.save('/Users/isaiahsantistevan/Desktop/test.gif',writer=writergif)

anim.save('/Users/isaiahsantistevan/Desktop/duo.gif')
#plt.show()






# EXAMPLE 3 Plotting two subhalos orbiting each other in Juliet
# Mstar[8] = 2e8, Mstar[143] = 3e4
# Multiple halos
fig = plt.figure()
ax = plt.axes(projection='3d')
mask8 = (data['d.sim'][8][:,0] != -1)
mask143 = (data['d.sim'][143][:,0] != -1)
#
ax.set_xlim(np.min(np.concatenate((data['d.sim'][8][:,0][mask8],data['d.sim'][143][:,0][mask143]))), np.max(np.concatenate((data['d.sim'][8][:,0][mask8],data['d.sim'][143][:,0][mask143]))))
ax.set_ylim(np.min(np.concatenate((data['d.sim'][8][:,1][mask8],data['d.sim'][143][:,1][mask143]))), np.max(np.concatenate((data['d.sim'][8][:,1][mask8],data['d.sim'][143][:,1][mask143]))))
ax.set_zlim(np.min(np.concatenate((data['d.sim'][8][:,2][mask8],data['d.sim'][143][:,2][mask143]))), np.max(np.concatenate((data['d.sim'][8][:,2][mask8],data['d.sim'][143][:,2][mask143]))))
#
ax.set_xlabel('X [kpc]', labelpad=15.)
ax.set_ylabel('Y [kpc]', labelpad=15.)
ax.set_zlabel('Z [kpc]', labelpad=15.)
#
line8, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[0])
line143, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[1])
ax.scatter(0,0,0,color='black')

def init():
    line8.set_data(np.array([]),np.array([]))
    line8.set_3d_properties(np.array([]))
    line143.set_data(np.array([]),np.array([]))
    line143.set_3d_properties(np.array([]))
    return line8, line143,

def animate(i):
    x8 = data['d.sim'][8][:,0][mask8][:i]
    y8 = data['d.sim'][8][:,1][mask8][:i]
    z8 = data['d.sim'][8][:,2][mask8][:i]
    #
    x143 = data['d.sim'][143][:,0][mask143][:i]
    y143 = data['d.sim'][143][:,1][mask143][:i]
    z143 = data['d.sim'][143][:,2][mask143][:i]
    #
    line8.set_data(x8,y8)
    line8.set_3d_properties(z8)
    #
    line143.set_data(x143,y143)
    line143.set_3d_properties(z143)
    #
    ax.view_init(elev=30, azim=0.6*i)
    #
    return line8, line143,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data['d.sim'][8][:,0][mask8])-10, interval=0.1, blit=False)
plt.tight_layout()
writergif = animation.PillowWriter(fps=30)
anim.save('/Users/isaiahsantistevan/Desktop/duo_juliet.gif',writer=writergif)
#plt.show()



# EXAMPLE 4 Plotting two subhalos orbiting each other in m12z
# Mstar[1] = 6.2e8, Mstar[60] = 4.3e4
# Multiple halos
fig = plt.figure()
ax = plt.axes(projection='3d')
mask1 = (data['d.sim'][1][:,0] != -1)
mask60 = (data['d.sim'][60][:,0] != -1)
#
ax.set_xlim(np.min(np.concatenate((data['d.sim'][1][:,0][mask1],data['d.sim'][60][:,0][mask60]))), np.max(np.concatenate((data['d.sim'][1][:,0][mask1],data['d.sim'][60][:,0][mask60]))))
ax.set_ylim(np.min(np.concatenate((data['d.sim'][1][:,1][mask1],data['d.sim'][60][:,1][mask60]))), np.max(np.concatenate((data['d.sim'][1][:,1][mask1],data['d.sim'][60][:,1][mask60]))))
ax.set_zlim(np.min(np.concatenate((data['d.sim'][1][:,2][mask1],data['d.sim'][60][:,2][mask60]))), np.max(np.concatenate((data['d.sim'][1][:,2][mask1],data['d.sim'][60][:,2][mask60]))))
#
ax.set_xlabel('X [kpc]', labelpad=15.)
ax.set_ylabel('Y [kpc]', labelpad=15.)
ax.set_zlabel('Z [kpc]', labelpad=15.)
#
line1, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[0])
line60, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[1])
ax.scatter(0,0,0,color='black')

def init():
    line1.set_data(np.array([]),np.array([]))
    line1.set_3d_properties(np.array([]))
    line60.set_data(np.array([]),np.array([]))
    line60.set_3d_properties(np.array([]))
    return line1, line60,

def animate(i):
    x1 = data['d.sim'][1][:,0][mask1][:i]
    y1 = data['d.sim'][1][:,1][mask1][:i]
    z1 = data['d.sim'][1][:,2][mask1][:i]
    #
    x60 = data['d.sim'][60][:,0][mask60][:i]
    y60 = data['d.sim'][60][:,1][mask60][:i]
    z60 = data['d.sim'][60][:,2][mask60][:i]
    #
    line1.set_data(x1,y1)
    line1.set_3d_properties(z1)
    #
    line60.set_data(x60,y60)
    line60.set_3d_properties(z60)
    #
    ax.view_init(elev=30, azim=-0.9*i)
    #
    return line1, line60,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data['d.sim'][1][:,0][mask1]), interval=0.1, blit=False)
plt.tight_layout()
writergif = animation.PillowWriter(fps=30)
anim.save('/Users/isaiahsantistevan/Desktop/duo_m12z.gif',writer=writergif)
#plt.show()






# EXAMPLE 5 Plotting two subhalos orbiting each other in Romulus
# Mstar[13] = 6.2e8, Mstar[111] = 4.3e4
# Multiple halos
fig = plt.figure()
ax = plt.axes(projection='3d')
mask13 = (data['d.sim'][13][:,0] != -1)
mask111 = (data['d.sim'][111][:,0] != -1)
#
ax.set_xlim(np.min(np.concatenate((data['d.sim'][13][:,0][mask13],data['d.sim'][111][:,0][mask111]))), np.max(np.concatenate((data['d.sim'][13][:,0][mask13],data['d.sim'][111][:,0][mask111]))))
ax.set_ylim(np.min(np.concatenate((data['d.sim'][13][:,1][mask13],data['d.sim'][111][:,1][mask111]))), np.max(np.concatenate((data['d.sim'][13][:,1][mask13],data['d.sim'][111][:,1][mask111]))))
ax.set_zlim(np.min(np.concatenate((data['d.sim'][13][:,2][mask13],data['d.sim'][111][:,2][mask111]))), np.max(np.concatenate((data['d.sim'][13][:,2][mask13],data['d.sim'][111][:,2][mask111]))))
#
ax.set_xlabel('X [kpc]', labelpad=15.)
ax.set_ylabel('Y [kpc]', labelpad=15.)
ax.set_zlabel('Z [kpc]', labelpad=15.)
#
line13, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[0])
line111, = ax.plot3D(np.array([]),np.array([]),np.array([]), colors[1])
ax.scatter(0,0,0,color='black')

def init():
    line13.set_data(np.array([]),np.array([]))
    line13.set_3d_properties(np.array([]))
    line111.set_data(np.array([]),np.array([]))
    line111.set_3d_properties(np.array([]))
    return line13, line111,

def animate(i):
    x13 = data['d.sim'][13][:,0][mask13][:i]
    y13 = data['d.sim'][13][:,1][mask13][:i]
    z13 = data['d.sim'][13][:,2][mask13][:i]
    #
    x111 = data['d.sim'][111][:,0][mask111][:i]
    y111 = data['d.sim'][111][:,1][mask111][:i]
    z111 = data['d.sim'][111][:,2][mask111][:i]
    #
    line13.set_data(x13,y13)
    line13.set_3d_properties(z13)
    #
    line111.set_data(x111,y111)
    line111.set_3d_properties(z111)
    #
    ax.view_init(elev=30, azim=0.9*i)
    #
    return line13, line111,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(data['d.sim'][13][:,0][mask13]), interval=0.1, blit=False)
plt.tight_layout()
writergif = animation.PillowWriter(fps=30)
anim.save('/Users/isaiahsantistevan/Desktop/duo_Romulus.gif',writer=writergif)
#plt.show()
