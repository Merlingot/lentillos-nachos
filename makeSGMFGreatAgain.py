import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import scipy.signal as sci
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline, RectBivariateSpline
plt.close('all')
from Camera import Camera
from Ecran import Ecran


#################################################################


#ne fonctionne pas avec .._anto, (trop grosses dimensions pour la univariate smoothing)

# choisir l'échantillon à caractériser parmi la panoplie collector best of 2018-2020 "ah les années poly, c'était beau" :
#miroir_plan_anto, lentille_biconvexe, lentille_plano_convexe, miroir_plan
echantillon = "lentille_plano_convexe"


#################################################################
ecran='allo'
# Camera Point Grey -------------------------------------
sgmf1 = "./data/" + echantillon + "/cam_match_PG.png"
sgmf1=cv2.imread(sgmf1)
mask1= np.transpose(cv2.imread("./data/" + echantillon + "/quadrupleconf_PG.png", 0).astype('bool')) #quadrupleconf est meilleure que conf pour PG mais pas pour AV
# Allied vision -------------------------------------
sgmf2 = "./data/" + echantillon + "/cam_match_AV.png"
sgmf2=cv2.imread(sgmf2)

mask2= np.transpose(cv2.imread("./data/" + echantillon + '/conf_AV.png', 0).astype('bool'))


mat1= np.load("./data/" + echantillon + "/cam_match_PG.npy")
mat2= np.load("./data/" + echantillon + "/cam_match_AV.npy")

plt.imshow(sgmf1[:,:,2])

plt.imshow(mat1[:,:,1])









# #################################################################
#
# ###### Caméra POINT GREY
#
# #là je suis perdu avec les 0 ou end-1 ... je sais pas si quand je montre shape ça me dit le dernier indice ou le nombre de cases. donc j'efface pas
#
# plage_x= np.arange(0,int(w1[1]/2)-1)
# plage_y= np.arange(0,int(w1[0]/2)-1)
#
#
# sgmf = cv2.imread(sgmf1)
# sgmf_median = cv2.medianBlur(sgmf,5)
# data_v = sgmf_median[:,:,1]
# data_u = sgmf_median[:,:,2]
#
#
#
# FULL_SGMF = np.zeros([len(plage_x),len(plage_y),2])
#
# for tranche_x in plage_y :
#     sgmf_continuous_1D = UnivariateSpline(plage_x, data_v[plage_x,tranche_x])
#     FULL_SGMF[plage_x,tranche_x,1] = sgmf_continuous_1D(plage_x)
#
# for tranche_y in plage_x :
#     sgmf_continuous_1D = UnivariateSpline(plage_y, data_u[tranche_y,plage_y])
#     FULL_SGMF[tranche_y,plage_y,0] = sgmf_continuous_1D(plage_y)
#
#
# np.save("./data/" + echantillon + "/cam_match_PG",FULL_SGMF)
#
#
#
#
#
# ###### Caméra ALLIED VISION
#
# #là je suis perdu avec les 0 ou end-1 ... je sais pas si quand je montre shape ça me dit le dernier indice ou le nombre de cases. donc j'efface pas
#
# plage_x= np.arange(0,int(w2[1])-1)
# plage_y= np.arange(0,int(w2[0])-1)
#
#
# sgmf = cv2.imread(sgmf1)
# sgmf_median = cv2.medianBlur(sgmf,5)
# data_v = sgmf_median[:,:,1]
# data_u = sgmf_median[:,:,2]
#
#
#
# FULL_SGMF = np.zeros([len(plage_x),len(plage_y),2])
#
# for tranche_x in plage_y :
#     sgmf_continuous_1D = UnivariateSpline(plage_x, data_v[plage_x,tranche_x])
#     FULL_SGMF[plage_x,tranche_x,1] = sgmf_continuous_1D(plage_x)
#
# for tranche_y in plage_x :
#     sgmf_continuous_1D = UnivariateSpline(plage_y, data_u[tranche_y,plage_y])
#     FULL_SGMF[tranche_y,plage_y,0] = sgmf_continuous_1D(plage_y)
#
#
# np.save("./data/" + echantillon + "/cam_match_AV",FULL_SGMF)
