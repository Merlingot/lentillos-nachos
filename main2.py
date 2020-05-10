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
from util import *
import prgm
import pickle
import dataPG as pg
import dataAV as av



# INITIALISATION       ------------------------------------------------
#lentille_biconvexe, lentille_plano_convexe, miroir_plan
echantillon = "lentille_plano_convexe"

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
sgmf1 = "./data/" + echantillon + "/cam_match_PG.png"

R1 = np.array(pg.R)
T1 = np.array(pg.T)
w1 = np.array( [3376, 2704] )
W1 = w1* 1.69e-6
cam1 = Camera(ecran, K1, R1, T1, W1, sgmf1)
cam1.dist=d1
cam1.mask = cv2.imread("./data/" + echantillon + "/confcrop_PG.png", 0).astype('bool')
# Allied vision -------------------------------------
sgmf2 = "./data/" + echantillon + "/cam_match_AV.png"
R2 = np.array(av.R)
T2 = np.array(av.T)
w2 = np.array( [780, 580] )
W2 = w2 * 8.3e-6
cam2 = Camera(ecran, K2, R2, T2, W2, sgmf2)
cam2.dist=d2
cam2.mask = cv2.imread("./data/" + echantillon + '/confcrop_AV.png', 0).astype('bool')

#----------------------------------------------------------------------------
# Choisir la camera de reference
cam=cam1; camR=cam2

grid = []
Nx=50;Ny=50
(ky,kx) = np.nonzero(camR.mask)
maxX=np.max(kx);minX=np.min(kx)
maxY=np.max(ky);minY=np.min(ky)
for i in np.arange(int(minX), int(maxX), int(((maxX-minX)/Nx)) ):
    for j in np.arange(int(minY), int(maxY), int(((maxY-minY)/Ny)) ):
        a = i*np.array([1,0,0]) + j*np.array([0,1,0])+np.array([0,0,1])
        if echantillon=='lentille_biconvexe':
            if (camR.mask[a[1],a[0]] > 0 and a[0]<520):
                grid.append(a)
        else:
            if (camR.mask[a[1],a[0]] > 0):
                grid.append(a)
surf=Surface(grid)

plt.imshow(camR.mask)
plt.scatter(surf.x_i, surf.y_i)

h=5e-2;L=40e-2

prgm.search(surf, h, L, camR, cam, ecran)
f=open( "./prgm/AV/{}_brute_50x50".format(echantillon), "wb")
pickle.dump(surf, f)
# # # -----------------------------------------------------
# file = open('./prgm/AV/{}_brute_50x50'.format(echantillon), 'rb')
# surf = pickle.load(file)
# file.close()
surf.enr_points_finaux(surf.points)
montage_refEcran(surf, ecran, cam1, cam2, 10e-2, np.array([0,0,0]), np.array([0,0,1]))



# for p in surf.points:
#     # show_sgmf_prgm(camR, cam, p)
#     if p.pmin[2]>-0.09:
#         show_sgmf_prgm(camR, cam, p)
