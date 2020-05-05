import numpy as np
import scipy.signal as sci
from scipy.interpolate import UnivariateSpline

class Point:

    def __init__(self):
        self.pmin=None
        self.valmin=None
        self.nmin=None
        self.indexmin=None
        self.vecP=None
        self.vecV=None
        self.vecPr=None
        self.vecVr=None
        self.vecB=None
        # self.vecN1=np.zeros((N,3))
        # self.vecN2=np.zeros((N,3))
        # self.vecU1=np.zeros((N,3))
        # self.vecE1=np.zeros((N,3))
        # self.vecE2=np.zeros((N,3))
        # self.vecU2=np.zeros((N,3))

class Surface:

    def __init__(self, grid):
        self.grid=grid
        self.points=[]
        self.good_points=[]
        self.nb_points=len(grid) #Nombre de point dans la grille
        # Points initiaux de la grille
        self.x_i=None; self.y_i=None; self.z_i=None
        # Points finaux de la grille
        self.x_f=None; self.y_f=None; self.z_f=None

        # self.enr_points_initiaux()

    def ajouter_point(self, point):
        self.points.append(point)
        if len(self.points) > self.nb_points :
            print('Erreur nombre de points sur la surface')

    def enr_points_initiaux(self):
        n=len(self.grid)
        self.x_i,self.y_i,self.z_i=np.zeros(n),np.zeros(n),np.zeros(n)
        for i in range(len(self.grid)):
            p=self.grid[i]
            self.x_i[i]=p[0]; self.y_i[i]=p[1]; self.z_i[i]=p[2]

    def enr_points_finaux(self, points):
        n=len(points)
        self.x_f,self.y_f,self.z_f=np.zeros(n),np.zeros(n),np.zeros(n)
        self.u, self.v, self.w=np.zeros(n),np.zeros(n),np.zeros(n)
        for i in range(n):
            p=points[i]
            self.x_f[i]=p.pmin[0]; self.y_f[i]=p.pmin[1]; self.z_f[i]=p.pmin[2]
            self.u[i]=p.nmin[0];self.v[i]=p.nmin[1];self.w[i]=p.nmin[2];

    def get_good_points(self, critere):
        self.good_points.clear()
        for point in self.points:
            if point.valmin < critere:
                self.good_points.append(point)
