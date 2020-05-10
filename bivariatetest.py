# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:43:50 2020

@author: goblo
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import geomdl
from geomdl import NURBS
from geomdl.visualization import VisMPL
from matplotlib import cm
from geomdl import knotvector
from geomdl import utilities
from geomdl import operations
from matplotlib.colors import Normalize
from matplotlib import cm

file = open('./prgm/AV/brute_25x25', 'rb')
surf = pickle.load(file)
file.close()
surf.enr_points_finaux(surf.points)

# Regularly-spaced, coarse grid
dx, dy = 0.1,0.1
xmax, ymax = 4, 4
x = np.arange(-xmax, xmax, dx)
y = np.arange(-ymax, ymax, dy)
X, Y = np.meshgrid(x, y)

R = 3;
B = (X)**2 + Y**2 <= R**2; # B est un filtre pour simuler le contour d'une lentille
B.astype(np.int)
B = np.multiply(B, 1)

def f(X,Y):
    return  np.exp(-1/10*((X)**2 + (Y)**2))

Z = f(X,Y)
Zcut = Z
Zcut[B == 0] = False
np.random.seed(1)
noise = (np.random.randn(*Z.shape) * 0.05)

#on agit comme si on avait le filtre qui détecte les contours et qui supprime les points qui ne concernent pas la lentille
noisy = (Z + noise)*B

noisy[B == 0] = -3 #float("NaN")  #False


ctrlpts = [[[X[j,i],Y[j,i],noisy[j,i],1] for j in range(len(y))]for i in range(len(x))]



fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.view_init(elev=5, azim=45)
ax.set_axis_off()
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_zlim(-2,6)

ax.scatter(X, Y, noisy, s=5, c='r')
#ax.plot_surface(X, Y, Z*B, color='b' , alpha = 0.9)

fig.tight_layout()
plt.show()

# Generate surface
surf = NURBS.Surface()
surf.degree_u = 3
surf.degree_v = 3
surf.ctrlpts2d = ctrlpts
surf.knotvector_u =  knotvector.generate(surf.degree_u, len(ctrlpts))
#surf.knotvector_u[4:len(ctrlpts)-1] = 0.6*np.ones(len(ctrlpts)-1-4)

surf.knotvector_v = knotvector.generate(surf.degree_v, len(ctrlpts[0]))
#surf.knotvector_v[4:len(ctrlpts[0])-1] = 0.6*np.ones(len(ctrlpts)-1-4)

operations.refine_knotvector(surf, [0,0])

# nombre de points dans le fichier .csv en sortie (une seule direction)
surf.sample_size = len(ctrlpts)

# Visualize surface
surf.vis = VisMPL.VisSurfTriangle(ctrlpts=True, axes=False, legend=False)
surf.render(colormap=cm.summer)




# points de la lentille après sampling - à exporter en format .csv ou .txt
surface_points = surf.evalpts

ZRES = np.zeros([len(ctrlpts),len(ctrlpts[0])])


Xres = [surface_points[len(ctrlpts[0])*i][0] for i in range(len(ctrlpts))]
Yres = [surface_points[i][1] for i in range(len(ctrlpts[0]))]
for i in range(len(ctrlpts)):
    for j in range(len(ctrlpts[0])):
        ZRES[i,j] = surface_points[len(ctrlpts[0])*i + j][2]


XRES,YRES = np.meshgrid(np.asarray(Xres),np.asarray(Yres))

ZINIT = f(XRES,YRES)*B




## comparaison surface sous-jacente et résultat
#xres =  np.arange(-xmax, xmax, (2*xmax)/surf.sample_size[0])
#yres =  np.arange(-ymax, ymax, (2*ymax)/surf.sample_size[0])
#XRES, YRES = np.meshgrid(xres, yres)
#ZRES = 4*np.exp(- XRES**2 - YRES**2) -  1.5*np.exp(-1/0.05 * ((XRES-1)**2 + (YRES-1)**2))

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter(XRES, YRES, ZRES, s=5, c='r')
#ax.scatter(XRES, YRES, ZINIT, s=5, c='b')
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(XRES, YRES, 100*(np.abs(ZRES - ZINIT))/(1+np.abs(ZINIT)),alpha=0.7,cmap=cm.summer)
#
#ax.set_xlim(-R*0.4,R*0.4)
#ax.set_ylim(-R*0.4,R*0.4)
