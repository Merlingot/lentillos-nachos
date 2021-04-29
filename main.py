import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import scipy.signal as sci
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from numpy.linalg import lstsq
plt.close('all')
from Camera import Camera
from Ecran import Ecran
from nc import *
from util import surface_refEcran, montage_refEcran, allo_refEcran,  show_sgmf, show_point
import prgm

# INITIALISATION       ------------------------------------------------
# choisir l'échantillon à caractériser parmi la panoplie collector best of 2018-2020 "ah les années poly, c'était beau" :;)
#lentille_anto, miroir_plan_anto, lentille_biconvexe, lentille_plano_convexe, miroir_plan
echantillon = "miroir_plan"
#NOUS -------------------------------------
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
cam1.mask = cv2.imread("./data/" + echantillon + "/confcrop_PG.png", 0).astype('bool')
# Allied vision -------------------------------------
sgmf2 = "./data/" + echantillon + "/cam_match_AV.npy"
R2 = np.array(av.R)
T2 = np.array(av.T)
w2 = np.array( [780, 580] )
W2 = w2 * 8.3e-6
cam2 = Camera(ecran, K2, R2, T2, W2, sgmf2)
cam2.dist=d2
cam2.mask = cv2.imread("./data/" + echantillon + '/confcrop_AV.png', 0).astype('bool')
#----------------------------------------------------------------------------

#################################################################
# # Notre miroir :
# y1=-0.0; y2=-0.15
y1=-0.0; y2=-0.10
# x1=0.07; x2=-0.07
x1=0.04; x2=-0.04

#direction de recherche normale a lecran
d = np.array([0,0,-1])
t = np.array([0.00, -0.05, -15e-2])
# t = np.array([0,0,0e-2])
#direction de recherche normale a une camera
# d = cam1.camToEcran(np.array([0,0,1,0]))[:3]
# t = cam1.camToEcran(np.array([0,0,0,1]))[:3] + d*20e-2

h=0.1e-2
l=5e-2
eps = h/5 # tolérance sur l'encadrement de la distance parcourue selon d pour trouver p_min

grid = []
o = t
dk=0.0025
Lx=0.15
Ly=0.15
kx=int(np.floor(Lx/dk/2)); ky=int(np.floor(Ly/dk/2))

# print(str(int(4*kx*ky)) + " points dans la grille de départ")

searchVolumeBasis = graham( d, [1,0,0], [0,1,0] )
v1 = searchVolumeBasis[0]; v2 = searchVolumeBasis[1]; v3 = searchVolumeBasis[2]
for j in np.arange(-kx, kx):
    for i in np.arange(-ky, ky):
        a = o + i*dk*v3 + j*dk*v2
        grid.append(a)
surf=Surface(grid)

## RECONSTRUCTION ------------------------------------------------

search(surf, d, h, l, eps, cam1, cam2, ecran)

import pickle
f=open( "./allo", "wb")
pickle.dump(surf, f)
