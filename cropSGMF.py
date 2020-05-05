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
    cadre = int(0.15 * np.shape(mat)[1])
    print(cadre)
    mat[:,:cadre] = 0
    mat[:, -cadre:] = 0
    mat[:cadre,:] = 0
    mat[-cadre:,:] = 0
    cv2.imwrite(name,mat)

sgmf1 = "./data/" + "miroir_plan" + "/conf_AV.png"
hihi = cropSGMF(sgmf1,"./data/" + "miroir_plan" + "/confcrop_AV.png")

plt.figure()
plt.imshow(cv2.imread("./data/" + "miroir_plan" + "/confcrop_AV.png"))
plt.show()

print(cv2.imread("./data/" + "miroir_plan" + "/conf_AV.png",0).astype('bool').shape)
