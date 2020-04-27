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
    for p in surface.grid: # Loop sur les points
        # print('------------------ POINT ----------------')
        point = Point(N)
        n=0; index_min=None; n_min=None
        p_initial = np.array([ p[0], p[1], p[2] ])
        p_min = np.array([ p[0], p[1], p[2] ]); val_min = 1e10 #(infini)
        while n<N: # Loop sur la descente du point
            b, val, n1, u1, e1, n2, u2, e2 = evaluatePoint(p, cam1, cam2, ecran)
            if b:
                if val < val_min:
                    index_min = n
                    val_min = val
                    p_min = np.array([p[0],p[1],p[2]])
                    n_min=(n1+n2)/2
            point.vecV[n]=val; point.vecP[n]=p; point.vecB[n]=b; point.vecN1[n]=n1; point.vecN2[n]=n2
            point.vecU1[n]=u1; point.vecE1[n]=e1;
            point.vecU2[n]=u2; point.vecE2[n]=e2;
            p += h*d
            n+=1
            
        p_minus = p_min - h*d
        p_plus = p_min + h*d
#        print("p_min avant ternary",p_min)
        p_min, val_min, _= ternary_search(p_minus, p_plus, d, eps, cam1, cam2, ecran)
#        print("p_min après ternary",p_min)
       
        # Enregistrer les valeurs minimales du point
        point.pmin=p_min; point.valmin=val_min; point.indexmin=index_min
        point.nmin=n_min
        # Arranger les vecteurs pour enlever les NaN:
        point.vecV=point.vecV[point.vecB]
        point.vecP=point.vecP[point.vecB]
        point.vecN1 = point.vecN1[point.vecB];
        point.vecU1 = point.vecU1[point.vecB];
        point.vecE1 = point.vecE1[point.vecB];
        point.vecN2 = point.vecN2[point.vecB];
        point.vecU2 = point.vecU2[point.vecB];
        point.vecE2 = point.vecE2[point.vecB];
        # Enregistrer le point étudié seulement si au moins un bon point:
        if point.indexmin:
            surface.ajouter_point(point)
            # show_sgmf(cam1, cam2, point)

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

def ternary_search(p_minus, p_plus, d, eps, cam1, cam2, ecran):
    """
    Ternary search for the minimum of inconsistency m calculated between two points
    p_minus and p_plus until. Precision eps on the distance between
    Args:
        p_minus, p_plus = np.array([x,y,z])
        cam1, cam2 : measurements nb. 1 and 2
    Returns:
        New minimum inconsistency value, new point, new normal associated
    """
    
    p1 = p_minus
    p4 = p_plus
    _, val1, _, _, _, _, _, _ = evaluatePoint(p1, cam1, cam2, ecran)
    _, val4, _, _, _, _, _, _ = evaluatePoint(p4, cam1, cam2, ecran)
    h = np.linalg.norm(p4 - p1)
    if isinstance(val1,np.float) and isinstance(val4,np.float) :
        while h > eps:
#            print("h = ",h)
            p2 = p1 + 1/3*h*d
            p3 = p4 - 1/3*h*d
            _, val2, _, _, _, _, _, _ = evaluatePoint(p2, cam1, cam2, ecran)
            _, val3, _, _, _, _, _, _ = evaluatePoint(p3, cam1, cam2, ecran)
#            print("val2",val2)
    #        print("p2",p2)
    #        print("evaluatePoint(p2, cam1, cam2, ecran)",evaluatePoint(p4, cam1, cam2, ecran))
    #        print("normal_at(homogene(p2), cam1, ecran)",normal_at(homogene(p4), cam1, ecran))
    #        print("normal_at(homogene(p2), cam2, ecran)",normal_at(homogene(p4), cam2, ecran))
    #        print("cam1.projectPoint( homogene(pmin) )",cam1.projectPoint( (p_minus+p_plus)/2 ))
    #        print("cam2.projectPoint( homogene(p4) )",cam2.projectPoint( homogene(p4) ))
    #        
            if isinstance(val2,np.float) and isinstance(val3,np.float) :
                if val2 > val3:
                    p1 = p2
    #                print("augmente p1")
                else:
                    p4 = p3
    #                print("diminue p4")
                h = np.linalg.norm(p4 - p1)
#            print("ternary search h = ",h)
    p_min = (p1 + p4)/2
    _, val_min, _, _, _, _, _, _ = evaluatePoint(p_min, cam1, cam2, ecran)
    return p_min, val_min, evaluatePoint(p_min, cam1, cam2, ecran) 

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
