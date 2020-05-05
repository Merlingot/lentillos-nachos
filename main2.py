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



# INITIALISATION       ------------------------------------------------
# choisir l'échantillon à caractériser parmi la panoplie collector best of 2018-2020 "ah les années poly, c'était beau" :;
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
camR = Camera(ecran, K1, R1, T1, W1, sgmf1)
camR.dist=d1
camR.mask = cv2.imread("./data/" + echantillon + "/confcrop_PG.png", 0).astype('bool')
# Allied vision -------------------------------------
sgmf2 = "./data/" + echantillon + "/cam_match_AV.npy"
R2 = np.array(av.R)
T2 = np.array(av.T)
w2 = np.array( [780, 580] )
W2 = w2 * 8.3e-6
cam = Camera(ecran, K2, R2, T2, W2, sgmf2)
cam.dist=d2
cam.mask = cv2.imread("./data/" + echantillon + '/confcrop_AV.png', 0).astype('bool')

#----------------------------------------------------------------------------
grid = []
Nx=20;Ny=20
(ky,kx) = np.nonzero(camR.mask)
maxX=np.max(kx);minX=np.min(kx)
maxY=np.max(ky);minY=np.min(ky)
for i in np.arange(int(minX), int(maxX), int(((maxX-minX)/Nx)) ):
    for j in np.arange(int(minY), int(maxY), int(((maxY-minY)/Ny)) ):
        a = i*np.array([1,0,0]) + j*np.array([0,1,0])+np.array([0,0,1])
        grid.append(a)
surf=Surface(grid)

h=5e-2
L=100e-2

prgm.search(surf, h, L, camR, cam, ecran)


# f=open( "./allocacaewew", "wb")
# pickle.dump(surf, f)
# -----------------------------------------------------


# file = open('allocacaewew', 'rb')
# dump information to that file
# surf = pickle.load(file)
# close the file
# file.close()


surf.enr_points_finaux(surf.points)
# montage_refEcran(surf, ecran, cam1, cam2, 10e-2, np.array([0,0,0]), np.array([0,0,1]))

# fig = plt.figure()
# ax = Axes3D(fig)
# allo= ax.plot3D(surf.x_f, surf.y_f, surf.z_f, 'o')
# plt.show()

for p in surf.points:
    show_caca(cam2, cam1, p)
