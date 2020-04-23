import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import scipy.signal as sci
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
plt.close('all')
from Camera import Camera
from Ecran import Ecran
from nc import *
from util import surface_refEcran, montage_refEcran, allo_refEcran,  show_sgmf

# choisir l'échantillon à caractériser parmi la panoplie collector best of 2018-2020 "ah les années poly, c'était beau" :
#lentille_anto, miroir_plan_anto, lentille_biconvexe, lentille_plano_convexe, miroir_plan
echantillon = "lentille_biconvexe"
# NOUS -------------------------------------
import dataPG as pg
import dataAV as av
nb_pix = 48 # Largeur d'un carré en pixel
l_pix = 0.277e-3 # Largeur d'un pixel
carre=nb_pix*l_pix
c = (888,576)
K1 = np.genfromtxt('./calibration/data_PG/camera.txt')
K2 = np.genfromtxt('./calibration/data_AV/camera.txt')
d1 = np.genfromtxt('./calibration/data_PG/dist.txt')
d2 = np.genfromtxt('./calibration/data_AV/dist.txt')



#################################################################
# Écran -------------------------------------
w = np.array( [1600, 900] ) #pixels
W = w * 0.277e-3 #m
ecran = Ecran( W, w ,c )
# Camera Point Grey -------------------------------------
sgmf1 = "./data/" + echantillon + "/cam_match_PG.npy"
R1 = np.array(pg.R)
T1 = np.array(pg.T)
w1 = np.array( [3376, 2704] )
W1 = w1* 1.69e-6
cam1 = Camera(ecran, K1, R1, T1, W1, sgmf1)
cam1.dist=d1
cam1.mask = cv2.imread("./data/" + echantillon + "/conf_PG.png", 0).astype('bool') #quadrupleconf est meilleure que conf pour PG mais pas pour AV

# Allied vision -------------------------------------
sgmf2 = "./data/" + echantillon + "/cam_match_AV.npy"
R2 = np.array(av.R)
T2 = np.array(av.T)
w2 = np.array( [780, 580] )
W2 = w2 * 8.3e-6
cam2 = Camera(ecran, K2, R2, T2, W2, sgmf2)
cam2.dist=d2
cam2.mask = cv2.imread("./data/" + echantillon + '/conf_AV.png', 0).astype('bool')

#################################################################
# # Notre miroir :
# y1=-0.0; y2=-0.15
y1=0.04; y2=-0.14
# x1=0.07; x2=-0.07
x1=0.06; x2=-0.06

#direction de recherche normale a lecran
d = np.array([0,0,-1])
t = np.array([(x1+x2)/2, (y1+y2)/2, -16e-2])
# t = np.array([0,0,0e-2])
#direction de recherche normale a une camera
# d = cam1.camToEcran(np.array([0,0,1,0]))[:3]
# t = cam1.camToEcran(np.array([0,0,0,1]))[:3] + d*20e-2

h=0.01e-2
l=3e-2

grid = []
o = t
dk=2e-3
Lx=(x1-x2)/2;
Ly=(y1-y2)/2;
kx=int(np.floor(Lx/dk)); ky=int(np.floor(Ly/dk))


# searchVolumeBasis = graham( d, [1,0,0], [0,1,0] )
# v1 = searchVolumeBasis[0]; v2 = searchVolumeBasis[1]; v3 = searchVolumeBasis[2]
# for j in np.arange(-kx, kx):
#    for i in np.arange(-ky, ky):
#        a = o + i*dk*v3 + j*dk*v2
#        grid.append(a)
# surf=Surface(grid)
# search(surf, d, h, l, cam1, cam2, ecran)
#
# surface_refEcran(surf)
#
# import pickle
# f = open("miroir", "wb") # remember to open the file in binary mode
# pickle.dump(surf,f)
# f.close()


# TRAITEMENT DES DONNÉES ------------------------------------------------


import pickle
surf= pickle.load( open( "miroir", "rb" ) )
surf.good_points.clear()
for point in surf.points:
    if point.valmin < 0.01:
        if point.valmin > 0.001:
            # if point.pmin[0] > -0.04:
            surf.good_points.append(point)
g = surf.good_points
surf.enr_points_finaux(surf.good_points)
surface_refEcran(surf)
L=10e-2
montage_refEcran(surf, ecran, cam1, cam2, L, t, d)

# xs=surf.x_f; ys=surf.y_f;  zs=surf.z_f;
#
# # plot raw data
# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(xs, ys, zs, color='b')
#
# # do fit
# tmp_A = []
# tmp_b = []
# for i in range(len(xs)):
#     tmp_A.append([xs[i], ys[i], 1])
#     tmp_b.append(zs[i])
# b = np.matrix(tmp_b).T
# A = np.matrix(tmp_A)
# fit = (A.T * A).I * A.T * b
# errors = b - A * fit
# residual = np.linalg.norm(errors)
# fit.item(0)
# fit
#
# X,Y = np.meshgrid(xs, ys)
# # plot plane
# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(xs, ys, zs, color='b')
# Z = fit.item(0) * X + fit.item(1) * Y + fit.item(2)
# ax.plot_wireframe(X,Y,Z, color='k')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
#
# n=np.array([-fit.item(0), fit.item(1), 1])
# n=n/np.linalg.norm(n)
# a=np.array([0,0,1])
# v=np.cross(a, n)
# s=np.linalg.norm(v)
# c=a@n
# matv = np.array([ [0,-v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0] ])
# R = np.eye(3) + matv + matv@matv*(1-c)/(s**2)
#
# a,b,c,d,e,f,g,h,i = R.ravel()
# xcac = a*xs+ b*ys + c*zs
# ycac = d*xs + e*ys + f*zs
# zcac = g*xs + h*ys + i*zs
#
# # plot plane
# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(xcac, ycac, zcac, color='b')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()
#
#
# # Option pour la surface
# fig = plt.figure()
# ax = Axes3D(fig)
# surf1= ax.plot_trisurf(xcac[xcac>-0.04],ycac[xcac>-0.04],zcac[xcac>-0.04], linewidth=0.1)
# fig.colorbar(surf1, shrink=0.5, aspect=5)
# plt.savefig('miroir.png')
# plt.show()
