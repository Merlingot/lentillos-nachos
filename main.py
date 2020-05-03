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
from util import surface_refEcran, montage_refEcran, allo_refEcran,  show_sgmf



# INITIALISATION       ------------------------------------------------

# choisir l'échantillon à caractériser parmi la panoplie collector best of 2018-2020 "ah les années poly, c'était beau" :
#lentille_anto, miroir_plan_anto, lentille_biconvexe, lentille_plano_convexe, miroir_plan
echantillon = "miroir_plan"
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
# sgmf1 = "./data/" + echantillon + "/cam_match_median7_PG.png"
sgmf1 = "./data/" + echantillon + "/cam_match_PG.npy"
R1 = np.array(pg.R)
T1 = np.array(pg.T)
w1 = np.array( [3376, 2704] )
W1 = w1* 1.69e-6
cam1 = Camera(ecran, K1, R1, T1, W1, sgmf1)
cam1.dist=d1
cam1.mask = cv2.imread("./data/" + echantillon + "/conf_PG.png", 0).astype('bool')

# Allied vision -------------------------------------
# sgmf2 = "./data/" + echantillon + "/cam_match_median7_AV.png"
sgmf2 = "./data/" + echantillon + "/cam_match_AV.npy"
R2 = np.array(av.R)
T2 = np.array(av.T)
w2 = np.array( [780, 580] )
W2 = w2 * 8.3e-6
cam2 = Camera(ecran, K2, R2, T2, W2, sgmf2)
cam2.dist=d2
cam2.mask = cv2.imread("./data/" + echantillon + '/conf_AV.png', 0).astype('bool')


# # SHOW DIFFERENT SGMF APPROXIMATIONS
#
# sgmf17 = cv2.imread("./data/" + echantillon + "/cam_match_median7_PG.png",-1)/900
# sgmf15 = cv2.imread("./data/" + echantillon + "/cam_match_median5_PG.png",-1)/900
# sgmf10 = cv2.imread("./data/" + echantillon + "/cam_match_PG.png",-1)/900
#
# spl = UnivariateSpline(np.arange(0,len(sgmf10[:,1000,1])),sgmf10[:,1000,1])
# sgmfspl = spl(np.arange(0,len(sgmf10[:,1000,1])))
#
#
# plt.plot(sgmf10[:,1000,1],'-o')
# plt.plot(sgmfspl)
# plt.plot(sgmf15[:,1000,1],'-o')
# plt.plot(sgmf17[:,1000,1],'-o')
# plt.legend(["raw","UnivariateSpline","5 pixel window","7 pixel window"])
# plt.show()




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




#
## RECONSTRUCTION         ------------------------------------------------
#
search(surf, d, h, l, eps, cam1, cam2, ecran)
# print("recherche terminée")
#surf.get_good_points(1)
#surf.enr_points_finaux(surf.good_points)
#g = surf.good_points




# TRAITEMENT DES DONNÉES ------------------------------------------------

#


def func(x,a,b,x0,y0):
    return y0**2 + np.sqrt( b**2 * ((x-x0)/a)**2 + 1) -1
#    return np.piecewise(x, [x < x0, x > x0], [lambda x:a1*(x-x01), lambda x:a2*(x-x02)])

#ramp = lambda u: np.maximum( u, 0 )
#step = lambda u: ( u > 0 ).astype(float)
#
#def SegmentedLinearReg( X, Y, breakpoints ):
#    nIterationMax = 10
#
#    breakpoints = np.sort( np.array(breakpoints) )
#
#    dt = np.min( np.diff(X) )
#    ones = np.ones_like(X)
#
#    for i in range( nIterationMax ):
#        # Linear regression:  solve A*p = Y
#        Rk = [ramp( X - xk ) for xk in breakpoints ]
#        Sk = [step( X - xk ) for xk in breakpoints ]
#        A = np.array([ ones, X ] + Rk + Sk )
#        p =  lstsq(A.transpose(), Y, rcond=None)[0] 
#
#        # Parameters identification:
#        a, b = p[0:2]
#        ck = p[ 2:2+len(breakpoints) ]
#        dk = p[ 2+len(breakpoints): ]
#
#        # Estimation of the next break-points:
#        newBreakpoints = breakpoints - dk/ck 
#
#        # Stop condition
#        if np.max(np.abs(newBreakpoints - breakpoints)) < dt/5:
#            break
#
#        breakpoints = newBreakpoints
#    else:
#        print( 'maximum iteration reached' )
#
#    # Compute the final segmented fit:
#    Xsolution = np.insert( np.append( breakpoints, max(X) ), 0, min(X) )
#    ones =  np.ones_like(Xsolution) 
#    Rk = [ c*ramp( Xsolution - x0 ) for x0, c in zip(breakpoints, ck) ]
#
#    Ysolution = a*ones + b*Xsolution + np.sum( Rk, axis=0 )
#
#    return Xsolution, Ysolution



#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

for point in surf.points:
    # surf.good_points.append(point)
    if point.valmin < 0.01:
        if len(point.vecP[:,2]) >= 13:
            spl = UnivariateSpline(-point.vecP[:,2], point.vecV, k = 4, s = 0)
            roots = spl.derivative(1).roots()
            dev1 = spl.derivative(1)
            dev2 = spl.derivative(2)
            if len(roots) == 1:
                if dev2(roots) > 0:
                    # on ne garde que les points dont la proportion de montée est d'un certain pourcentage de la plage
                    if np.max(np.ediff1d( point.vecV)) < 0.005:
                        if sum(dev1(-point.vecP[:,2]) <= 0)/len(-point.vecP[:,2]) > 0.25:
                            if sum(dev1(-point.vecP[:,2]) >= 0)/len(-point.vecP[:,2]) > 0.25 :
#                                p_min, val_min, n1, n2 = parabolic_search(point,t,d,cam1, cam2, ecran)
#                                # print(p_min[0],p_min[1],p_min[2],val_min,n1)
#                                point.pmin=p_min; point.valmin=val_min
                                
                                show_sgmf(cam1, cam2, point)
                                
                                xdata = point.vecP[:,2]; ydata = point.vecV
                                popt, pcov = curve_fit(func, xdata, ydata)
                                
                                plt.figure()
                                plt.plot(xdata, func(xdata,*popt), '-r')
#                                initialBreakpoints = [-18e-2]
#                                plt.plot( *SegmentedLinearReg( xdata, ydata, initialBreakpoints ), '-r' );
                                plt.plot(xdata,ydata,'o')
                                plt.show()
                                ax.scatter(point.pmin[0], point.pmin[1], point.pmin[2],color="r")
                                surf.good_points.append(point)
plt.show()

g = surf.good_points
surf.enr_points_finaux(surf.good_points)
#
# surface_refEcran(surf)

#Montrer tout le montage
#L=10e-2 #longueur des flêches
#montage_refEcran(surf, ecran, cam1, cam2, L, t, d)
#surface_refEcran(surf)
# allo_refEcran(g[10], ecran, cam1, cam2, L, t, d)


# VISUALISATION DES DONNÉES ------------------------------------------------
# print(int(len(g))," points reconstruits")





for p in g[:]:
    show_sgmf(cam1, cam2, p)
