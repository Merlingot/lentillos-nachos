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



def search(surface, d, h, L, eps, cam1, cam2, ecran):
    """
    Find the position (vector p_min) and value (min) of the minimum on each point of the grid
    Args:
        d (np.array([x,y,z])) : direction de recherche (vecteur unitaire)
        h (float) : incrément de recherche
        grid (list[np.array([x,y,z])]) :
        cam1, cam2 : caméra
        ecran : ecran
    Return:
        Objet surface
    """
    N=int(np.floor(L/h)) # nombre d'itérations (de descentes) pour un seul point
    for point in surface.grid: # Loop sur les points
        p=np.array([ point[0], point[1], point[2] ])
        succes, tup = find_minimum(p, N, d, h, cam1, cam2, ecran)
        # Continuer avec le point étudié seulement si au moins un bon point:
        if succes:
            (p_min, val_min, n_min, vecV, vecP) = tup
            # Faire le refine search
            p_minus = p_min - 2*h*d;
            hr=h/40; Nr = int(4*h/hr)
            succes_r, tup_r = find_minimum(p_minus, Nr ,d, hr, cam1, cam2, ecran)
            if succes_r:
                (p_min, val_min, n_min, vecV_r, vecP_r) = tup_r
                # Enregistrer les resultats apres refine search
                point = Point();
                point.vecP=vecP; point.vecV=vecV #grossier
                point.vecPr=vecP_r; point.vecVr=vecV_r #fin
                point.pmin=p_min; point.valmin=val_min; point.nmin=n_min

                surface.ajouter_point(point)


def find_minimum(p, N, d, h, cam1, cam2, ecran):
    val_min = 1e10 #infini
    succes=False; p_min=None; n_min=None
    vecP=np.zeros((N,3)); vecV=np.zeros(N); vecB=np.zeros(N, dtype=bool)
    n=0;
    while n<N: # Loop sur la descente du point
        b, val, n1, _, _, n2, _, _ = evaluatePoint(p, cam1, cam2, ecran)
        if b:
            if val < val_min:
                succes = True
                val_min = val
                p_min = np.array([p[0],p[1],p[2]])
                n_min = (n1+n2)/2
        vecV[n]=val; vecP[n]=p; vecB[n]=b;
        p += h*d
        n+=1
    # Arranger les vecteurs pour enlever les NaN:
    vecV=vecV[vecB]; vecP=vecP[vecB]
    return succes, (p_min, val_min, n_min, vecV, vecP)

def evaluatePoint(p, cam1, cam2, ecran):
    """
    Evaluate the inconsistensy m of two measurements from cam1 and cam2 at a point p
    Args:
        p = np.array([x,y,z])
        cam1, cam2 : measurements nb. 1 and 2
    Returns:
        Inconsistensy, two normals
    """
    P = homogene(p)

    n1, u1, e1 = normal_at(P, cam1, ecran); n2, u2, e2 = normal_at(P, cam2, ecran)

    if isinstance(n1, np.ndarray) and isinstance(n2, np.ndarray) :
        return True, m1(n1, n2), cartesienne(n1), u1, e1, cartesienne(n2), u2, e2
    else:
        return False, None, None, None, None, None, None, None

def normal_at(P, cam, ecran):
    """
    *homogene
    Évaluer la normale en un point p
    Args:
        P : np.array([x,y,z,1])
            Point dans le référentiel de l'écran, homogène
        cam: Structure Camera
            Caméra qui regarde le point
        ecran : Structure écran
            Écran qui shoote des pattern
    returns
        n = np.array([x,y,z,0]) (unit vector)
    """
    # Avoir le pixel de camera qui voit P
    pixCam = cam.projectPoint( P ) #[u,v,1]
#    print("camera",cam.w,"pixCam",pixCam)
    if isinstance(pixCam, np.ndarray): # Sur CCD ?
        # Appliquer la SGMF
        pixEcran = cam.SGMF(pixCam) #[u,v,1]
        # print(pixEcran)
        if isinstance(pixEcran, np.ndarray): # Dans masque ?
            # Transformer de pixel a metres
            E = ecran.pixelToSpace(pixEcran) #[x,y,0,1]
            return normale(P, E, cam.S), pixCam, pixEcran
        else :
            return None, None, None
    else:
        return None, None, None


def ternary_search(point, p_minus, p_plus, d, eps, cam1, cam2, ecran):
    """
    Ternary search for the minimum of inconsistency m calculated between two points
    p_minus and p_plus until. Precision eps on the distance between
    Args:
        p_minus, p_plus = np.array([x,y,z])
        cam1, cam2 : measurements nb. 1 and 2
    Returns:
        New minimum inconsistency value, new point, new normal associated
    """
    vecP=[]; vecV=[]

    p1 = p_minus
    p4 = p_plus
    _, val1, _, _, _, _, _, _ = evaluatePoint(p1, cam1, cam2, ecran)
    _, val4, _, _, _, _, _, _ = evaluatePoint(p4, cam1, cam2, ecran)
    h = np.linalg.norm(p4 - p1)
    if isinstance(val1,np.float) and isinstance(val4,np.float) :
        while h > eps:
            p2 = p1 + 1/3*h*d
            p3 = p4 - 1/3*h*d
            _, val2, _, _, _, _, _, _ = evaluatePoint(p2, cam1, cam2, ecran)
            _, val3, _, _, _, _, _, _ = evaluatePoint(p3, cam1, cam2, ecran)
            if isinstance(val2,np.float) and isinstance(val3,np.float) :
                if val2 > val3:
                    p1 = p2
                    vecP.append(p2);vecV.append(val2)
                else:
                    p4 = p3
                    vecP.append(p3);vecV.append(val3)
                h = np.linalg.norm(p4 - p1)

    p_min = (p1 + p4)/2
    _, val_min, n1, _, _, n2, _, _ = evaluatePoint(p_min, cam1, cam2, ecran)
    vecP.append(p_min); vecV.append(val_min)
    point.vecP_ternary = np.array(vecP); point.vecV_ternary = np.array(vecV);
    return p_min, val_min, n1, n2

def parabolic_search(point,t,d,cam1, cam2, ecran):
    # Find the maximum
    x,y = point.vecP[:,2], point.vecV
    # print(point.pmin[0],point.pmin[1],point.pmin[2],point.valmin,point.nmin[0],point.nmin[1],point.nmin[2])
    epos, mi, xb, yb, p = pyasl.quadExtreme(x, y, mode="min", dp=(3,3),fullOutput=True)
    p_min = point.vecP[0,:]
    p_min[2] = epos
    # newx = np.linspace(min(xb), max(xb), 100)
    # model = np.polyval(p, newx)
    # # print(epos)
    #
    # # Plot the "data"
    # plt.plot(x, y, 'bp')
    # # Mark the points used in the fitting (shifted, because xb is shifted)
    # plt.plot(xb+x[mi], yb, 'rp')
    # # Overplot the model (shifted, because xb is shifted)
    # plt.plot(newx+x[mi], model, 'r--')
    # plt.show()
    _, val_min, n1, _, _, n2, _, _ = evaluatePoint(p_min, cam1, cam2, ecran)

    return p_min, val_min, n1, n2

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

def m2(n1, n2):
    """
    *homogene
    Inconsistensy of the current point p.
    Definition: m=1-absolute_value( n1<dot_product>n2 )
    Args:
        n1, n2 : np.array([x,y,z,0])
    """
    return 1 - np.abs(n1@n2)

def m1(n1, n2):
    """
    *homogene
    Inconsistensy of the current point p.
    Definition : m=n1<cross_product>n2
    Args:
        n1, n2 : np.array([x,y,z,0])"""

    return norm(np.cross(n1[:3], n2[:3]))


# Autre fonctions ------------------
def getApproxZDirection(cam1, cam2):

    """ Donne la direction approximative de la table dans le référentiel de l'écran"""

    zE_E = np.array([0,0,-1])
    zC_C = np.array([0,0,1])
    zC1_E = cam1.camToEcran(zC_C)
    zC1_E /= np.linalg.norm(zC1_E)
    zC2_E = cam2.camToEcran(zC_C)
    zC2_E /= np.linalg.norm(zC2_E)

    z1_E = zE_E + zC1_E; z1_E /=np.linalg.norm(z1_E)
    z2_E = zE_E + zC2_E; z2_E /=np.linalg.norm(z2_E)
    return (z1_E + z2_E)/2

def graham(v1, v2, v3):
    """
    Find the orthogonal basis with direction v1 as d
    Args:
    Return:
    """
    u1 = v1
    e1 = u1/np.linalg.norm(u1)
    u2 = v2 - ( np.dot(u1, v2) / np.dot(u1,u1) ) * u1
    e2 = u2/np.linalg.norm(u2)
    u3 = v3 - ( np.dot(u1, v3) / np.dot(u1,u1) ) * u1 - ( np.dot(u2, v3) / np.dot(u2,u2) ) * u2
    e3 = u3/np.linalg.norm(u3)

    return np.concatenate((e1,e2,e3), axis=0).reshape(3,3)













#
