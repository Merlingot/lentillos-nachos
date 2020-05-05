""" Functions needed fot the search algorithm """
from Camera import Camera
from Ecran import Ecran
from Surface import Surface, Point
# import seaborn as sns
from util import show_sgmf


import numpy as np
from numpy import abs
from numpy.linalg import norm
import matplotlib.pyplot as plt
#from PyAstronomy import pyasl



def search(surface, h, L, camR, cam, ecran):
    """
    Find the position (vector p_min) and value (min) of the minimum on each point of the grid
    Args:
        h (float) : incrément de recherche
        grid (list[np.array([x,y,z])]) :
        cam1, cam2 : caméra
        ecran : ecran
    Return:
        Objet surface
    """
    N=int(np.floor(L/h)) # nombre d'itérations (de descentes) pour un seul point à partir du stenope
    for pixR in surface.grid: # Loop sur les points
        # Trouver la direction de descente
        CR=camR.cacmouE(pixR) # Point sur la cam2 dans ref ecran
        d=((camR.S-CR)/norm(camR.S-CR))[:3] #direction de descente ref ecran
        # Point initial de recherche (stenope)
        p=np.array([camR.S[0], camR.S[1], camR.S[2]])
        # Recherche du minimum :
        succes, tup, _ = find_minimum(pixR, p, N, d, h, camR, cam, ecran)
        # Continuer avec le point étudié seulement si au moins un bon point:
        if succes:
            (p_min, val_min, n_min, vecV, vecP) = tup
            # Faire le refine search
            p_minus = p_min - h*d;
            hr=h/80; Nr = int(2*h/hr)
            succes_r, tup_r, tup_pix = find_minimum(pixR, p_minus, Nr ,d, hr, camR, cam, ecran)
            if succes_r:
                (p_min, val_min, n_min, vecV_r, vecP_r) = tup_r
                (vecUR, vecER, vecU, vecE, vecEth) = tup_pix
                # Enregistrer les resultats apres refine search
                point = Point();
                point.vecP=vecP; point.vecV=vecV #grossier
                point.vecPr=vecP_r; point.vecVr=vecV_r #fin
                point.pmin=p_min; point.valmin=val_min; point.nmin=n_min
                point.vecUr=vecUR; point.vecEr=vecER
                point.vecU=vecU; point.vecE=vecE; point.vecEth=vecEth

                surface.ajouter_point(point)


def find_minimum(pixR, p, N, d, h, camR, cam, ecran):
    val_min = 1e10 #infini
    succes=False; p_min=None; n_min=None
    vecP=np.zeros((N,3)); vecV=np.zeros(N); vecB=np.zeros(N, dtype=bool)
    vecUR=[]; vecER=[]; vecU=[]; vecE=[]; vecEth=[]
    n=0;
    while n<N: # Loop sur la descente du point
        b, val, nR, tup_r, tup_c= evaluatePoint(pixR, p, camR, cam, ecran)
        if b:
            if val < val_min:
                succes = True
                val_min = val
                p_min = np.array([p[0],p[1],p[2]])
                n_min = nR

                (ur, er)=tup_r; (u,e,eth)=tup_c
                vecUR.append(ur);vecER.append(er)
                vecU.append(u);vecE.append(e);vecEth.append(eth)
        vecV[n]=val; vecP[n]=p; vecB[n]=b;
        p += h*d
        n+=1
    # Arranger les vecteurs pour enlever les NaN:
    vecV=vecV[vecB]; vecP=vecP[vecB]
    return succes, (p_min, val_min, n_min, vecV, vecP), (vecUR, vecER, vecU, vecE, vecEth)

def evaluatePoint(pixR, p, camR, cam, ecran):
    """
    Evaluate the inconsistensy m of two measurements from cam1 and cam2 at a point p
    Args:
        p = np.array([x,y,z])
        cam1, cam2 : measurements nb. 1 and 2
    Returns:
        Inconsistensy, two normals
    """
    P = homogene(p)

    # Normale de la cam de reference : ------------------------
    # Point sur l'ecran pour camR
    eR=camR.SGMF(pixR);
    # print(pixR, eR)
    ER=ecran.pixelToSpace(eR)
    # Point sur la cam pour camR
    CR=camR.cacmouE(pixR)
    # Normale
    nR=normale(P, ER, CR)

    #Autre camera :
    b, val, tup = pipicaca(P, nR, cam, ecran)

    return b, val, nR, (pixR, eR), tup

def pipicaca(P, nR, cam, ecran):
    """
    *homogene
    """
    # Avoir le pixel de camera qui voit P
    pixCam = cam.projectPoint( P ) #[u,v,1]
    if isinstance(pixCam, np.ndarray): # Sur CCD ?
        # Appliquer la SGMF : pixel theorique !
        pixEcranSGMF = cam.SGMF(pixCam) #[u,v,1]
        if isinstance(pixEcranSGMF, np.ndarray): # Dans masque ?
            g = (cam.S-P)/norm(cam.S-P) #Direction P->cam.S
            G = 2*nR-g #Direction P->Ecran
            alpha = -P[2]/G[2]
            vecEcran = P + alpha*G
            pixEcran = ecran.spaceToPixel(vecEcran)
            return True, taxi(pixEcranSGMF, pixEcran), (pixCam, pixEcran, pixEcranSGMF)
        else :
            return False, None, None
    else:
        return False, None, None



# Vector functions

def homogene(vec):
    """ np.array([x,y,z]) -> np.array([x,y,z,1])   """
    if vec.size == 3:
        return np.array([vec[0], vec[1], vec[2], 1])
    else:
        return vec

def cartesienne(vec):
    """ np.array([x,y,z,-]) - > np.array([x,y,z]) """
    if vec.size == 4:
        return np.array([vec[0], vec[1], vec[2]])
    else:
        return vec

def normale(P,E,C):
    """
    *homogene
    Calculer une normale avec 3 points dans le même référentiel
    P:point E:écran C:caméra np.array([x,y,z,1])
    """
    r = P-E; p = P-C # r = vec(EP), p = vec(CP)
    r = r/np.linalg.norm(r); p = p/np.linalg.norm(p)
    n = - r - p
    n = n/np.linalg.norm(n)
    return n

def taxi(u, w):
    return np.sum(np.absolute(u-w))
















#
