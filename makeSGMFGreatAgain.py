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
mask1= cv2.imread("./data/" + echantillon + '/extremeconf_PG.png', 0).astype('bool')
# Allied vision -------------------------------------
sgmf2 = "./data/" + echantillon + "/cam_match_AV.png"
sgmf2=cv2.imread(sgmf2)

mask2= cv2.imread("./data/" + echantillon + '/extremeconf_AV.png', 0).astype('bool')








#################################################################

###### Caméra POINT GREY

#là je suis perdu avec les 0 ou end-1 ... je sais pas si quand je montre shape ça me dit le dernier indice ou le nombre de cases. donc j'efface pas

colonnes = np.arange(0,np.shape(sgmf1)[1])
lignes = np.arange(0,np.shape(sgmf1)[0])

# print("les colonnes doivent être",colonnes)
# print("les lignes doivent être",lignes)


sgmf = sgmf1
sgmf_median = cv2.medianBlur(sgmf,7)
data_v = sgmf_median[:,:,1]#*mask1
data_u = sgmf_median[:,:,2]#*mask1



FULL_SGMF = np.zeros([len(lignes),len(colonnes),2])
print(np.shape(FULL_SGMF))

for colonne in colonnes :
    sgmf_continuous_1D = UnivariateSpline(lignes, data_v[lignes,colonne],s=None)
    FULL_SGMF[lignes,colonne,1] = sgmf_continuous_1D(lignes)

for ligne in lignes :
    sgmf_continuous_1D = UnivariateSpline(colonnes, data_u[ligne,colonnes],s=None)
    FULL_SGMF[ligne,colonnes,0] = sgmf_continuous_1D(colonnes)

plt.imshow(sgmf1[:,:,2])
plt.show()

plt.imshow(FULL_SGMF[:,:,0])
plt.show()

plt.plot(lignes,sgmf1[lignes,1000,1],'o')
plt.plot(lignes,sgmf_median[lignes,1000,1],'-o')
plt.plot(lignes,FULL_SGMF[lignes,1000,1])
plt.show()

np.save("./data/" + echantillon + "/cam_match_PG",FULL_SGMF/255)




#
# ###### Caméra ALLIED VISION
#
# #là je suis perdu avec les 0 ou end-1 ... je sais pas si quand je montre shape ça me dit le dernier indice ou le nombre de cases. donc j'efface pas
#
#
colonnes = np.arange(0,np.shape(sgmf2)[1])
lignes = np.arange(0,np.shape(sgmf2)[0])

# print("les colonnes doivent être",colonnes)
# print("les lignes doivent être",lignes)


sgmf = sgmf2
sgmf_median = cv2.medianBlur(sgmf,7)
data_v = sgmf_median[:,:,1]#*mask2
data_u = sgmf_median[:,:,2]#*mask2



FULL_SGMF = np.zeros([len(lignes),len(colonnes),2])
print(np.shape(FULL_SGMF))

for colonne in colonnes :
    sgmf_continuous_1D = UnivariateSpline(lignes, data_v[lignes,colonne],s=None)
    FULL_SGMF[lignes,colonne,1] = sgmf_continuous_1D(lignes)

for ligne in lignes :
    sgmf_continuous_1D = UnivariateSpline(colonnes, data_u[ligne,colonnes],s=None)
    FULL_SGMF[ligne,colonnes,0] = sgmf_continuous_1D(colonnes)

plt.imshow(sgmf2[:,:,2])
plt.show()

plt.imshow(FULL_SGMF[:,:,0])
plt.show()

plt.plot(lignes,sgmf2[lignes,450,1],'o')
plt.plot(lignes,FULL_SGMF[lignes,450,1])
plt.show()
np.save("./data/" + echantillon + "/cam_match_AV",FULL_SGMF/255)
