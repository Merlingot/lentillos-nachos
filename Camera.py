import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread, imsave
# import seaborn as sns


class Camera:

    """
    Input:
        -K      : Matrice intrinsèque de la camera contenant (fx,fy,s,cx,cy)             | np.narray()
        -R      : Matrice de rotation pour le passage entre ref_{ecran} -> ref_{cam}     | np.narray()
        -T      : Matrice de translation pour le passage entre ref_{ecran} -> ref{cam}   | np.narray()
        -W      : Vecteur de la taille du CCD de la camera en [mm]                       | np.array()
        -w      : Vecteur de la taille du CCD de la camera en [pix]                      | np.array()
        -SGMF   : String du nom du PNG de cartographie de pixel entre camera et ecran | str()
    """
    def __init__(self, ecran, K, R, T, W, sgmf):

        # Setup
        self.ecran = ecran

        ## SGMF
        #Importing cartography
        # sgmfXY = cv2.imread(sgmf,-1)
        sgmfXY = np.load(sgmf)
        self.sgmf = np.zeros( sgmfXY.shape ) #SHAPE Lignes,Colonnes,CHANNEL
        self.sgmf[:,:,0] = sgmfXY[:,:,0] * self.ecran.w[0] # channel X
        self.sgmf[:,:,1] = sgmfXY[:,:,1] * self.ecran.w[1] # channel Y
        self.w = np.array([self.sgmf.shape[1], self.sgmf.shape[0]])                             # Taille du CCD en [w]=pixels

        # Intrinsèque
        self.K = K                              # Tout information
        self.fx=K[0,0]; self.fy=K[1,1]
        self.cu=K[0,2]; self.cv=K[1,2]
        self.f = ( self.fx + self.fy ) / 2.     # Focale camera [f]=pixels
        self.c = np.array([self.cu, self.cv])   # Centre optique du CCD [c]=pix
        self.s = K[0,1]                         # Skew
        self.W = W                              # Taille du CCD en [W]=m


        # BINNING ?
        self.sx = self.W[0]/self.w[0]                     # Taille d'un pixel [m/pixel]
        self.sy = self.W[1]/self.w[1]                     # Taille d'un pixel [m/pixel]

        self.F = ( self.fx*self.sx + self.fy*self.sy ) / 2. #Focale utile [m]

        # Extrinsèque (Ecran -> Camera)
        self.R = R                              # Matrice de rotation [-]
        self.rvec, _ = cv2.Rodrigues(R)
        self.T = T                              # Matrice de translation [m]

        # Matrices de Passage
        self.eToC = np.block([ [R , T.reshape(3,1)] ,
                [ np.zeros((1,3)).reshape(1,3) , 1]
                ])
        self.cToE = np.block([ [np.transpose(R) , -(np.transpose(R)@T).reshape(3,1)] ,
                [ np.zeros((1,3)).reshape(1,3) , 1]
                ])

        self.Kinv = np.array([[1/self.fx,-self.s/(self.fx*self.fy),self.s*self.cv/(self.fx*self.fy) - self.cu/self.fx],
                         [0,1/self.fy,-self.cv/self.fy],
                         [0,0,1],
                         [0,0,-1/self.F]])

        # Position du sténopé de la caméra dans le référentiel de l'écran
        self.S = self.camToEcran( np.array( [0,0,0,1]) )
        # Normale de la caméra dans le référentiel de l'écran
        self.normale = self.camToEcran( np.array([0,0,1,0]))


    def ecranToCam(self, P):
        """
        * homogene
        [px, py, pz, 1]^E -> [px, py, pz, 1]^C """
        return self.eToC@P

    def camToEcran(self, P):
        """
        * homogene
        [px, py, pz, 1]^C -> [px, py, pz, 1]^E"""
        return self.cToE@P

    def projectPoint(self, P):
        """
        A scene view is formed by projecting 3D points into the image plane using a perspective transformation.
        *homogene
        [px,py,pz,1]^E -> [ex,ey,1]^C """
        imgpt, _ = cv2.projectPoints(np.array([P[:3]]), self.rvec, self.T, self.K, distCoeffs=self.dist) #[ex,ey]

        # condition dans la sgmf
        ex,ey = imgpt[0][0][0],imgpt[0][0][1]
        if 0<ex<self.w[0]-1 and 0<ey<self.w[1]-1:
            return np.array([ex,ey,1])
        else:
            return None

    def SGMF(self, vecPix):
        u,v = int(np.round(vecPix[0])), int(np.round(vecPix[1]))
        # INDEXATION LIGNE (v), COLONNE (u) !!!!!!
        if self.mask[v,u] :
            ex, ey = self.sgmf[v,u,0], self.sgmf[v,u,1] #les channels
            return np.array([ex,ey,1])
        else:
            return None





    # FONCTIONS D'AFFICHAGE -------------------------------------------------

    def pixelToSpace(self, vecPix):
        """
        * homogene
        [u,v,1] -> [U,V-F,1]

        Prend la coordonnée d'un pixel de la caméra (u,v,1) [pixel] et le transforme en coordonnées dans le référentiel de la caméra (U,V,-F,1) [m]
        Args:
            vecPix : np.array([u,v,1])
            Vecteur de position en pixel
        Returns:
            np.array([U,V,-F,1])
            Vecteur de position en m
        """
        return -self.F*self.Kinv@vecPix


    def cacmouE(self, vecPix):
        """
        * homogène (fonction d'affichage)
        [u,v,1] -> [X,Y,Z,1]

        Prend la coordonnée d'un pixel de la caméra (u,v,1) [pixel] et le transforme
        1) en coordonnées dans le référentiel de la caméra (U,V,-F,1) [m]
        2) en coordonnées dans le référentiel de l'écran (X,Y,Z,1) [m]
        Args:
            vecPix : np.array([u,v,1])
        Returns:
            np.array([X,Y,Z,1])
        """
        return self.camToEcran( self.pixelToSpace(vecPix) )

    # def cacmouC(self, vecPix):
    #     """
    #     * homogène (fonction d'affichage)
    #     [u,v,1] -> [U,V,-F,1]
    #
    #     Prend la coordonnée d'un pixel de la caméra (u,v,1) [pixel] et le transforme en coordonnées dans le référentiel de la caméra (U,V,-F,1) [m]
    #     Args:
    #         vecPix : np.array([u,v,1])
    #     Returns:
    #         np.array([U,V,-F,1])
    #     """
    #     return self.pixelToSpace(vecPix)
