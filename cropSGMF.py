# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:48:47 2020

@author: goblo
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

def cropSGMF(sgmf,name):
    mat = cv2.imread(sgmf,0)
    cadre = np.array([int(0.2 * np.shape(mat)[1]) , int(0.1 * np.shape(mat)[1]) , int(0.20 * np.shape(mat)[0]),int(0.3* np.shape(mat)[0]) ])
    print(cadre)
    mat[:,:cadre[0]] = 0
    mat[:, -cadre[1]:] = 0
    mat[:cadre[2],:] = 0
    mat[-cadre[3]:,:] = 0
    cv2.imwrite(name,mat)

sgmf1 = "./data/" + "lentille_plano_convexe" + "/extremeconf_PG.png"
hihi = cropSGMF(sgmf1,"./data/" + "lentille_plano_convexe" + "/confcrop_PG.png")
plt.figure()
plt.imshow(cv2.imread("./data/" + "lentille_plano_convexe" + "/confcrop_PG.png"))
plt.show()
print(cv2.imread("./data/" + "lentille_plano_convexe" + "/conf_PG.png",0).astype('bool').shape)

# sgmf1 = "./data/" + "lentille_plano_convexe" + "/extremeconf_AV.png"
# hihi = cropSGMF(sgmf1,"./data/" + "lentille_plano_convexe" + "/confcrop_AV.png")
# plt.figure()
# plt.imshow(cv2.imread("./data/" + "lentille_plano_convexe" + "/confcrop_AV.png"))
# plt.show()
# print(cv2.imread("./data/" + "lentille_plano_convexe" + "/conf_AV.png",0).astype('bool').shape)
