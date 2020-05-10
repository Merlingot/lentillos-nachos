import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import scipy.signal as sci
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from numpy.linalg import lstsq

from util import *
import pickle

echantillon='lentille_biconvexe'

# Trouver la matrice de rotation avec le miroir ------------------------------
file = open('./prgm/AV/miroir_plan_brute_25x25', 'rb')
surf = pickle.load(file)
file.close()
surf.enr_points_finaux(surf.points)
xs,ys,zs = surf.x_f, surf.y_f, surf.z_f
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

a=fit.item(0); b=fit.item(1); c=-1; d=fit.item(2)
n=np.array([a,b,c])
a=n/np.linalg.norm(n)
b=np.array([0,0,-1])
v=np.cross(a,b)
s=np.linalg.norm(v)
c=a@b
v1,v2,v3=v[0],v[1],v[2]
vx=np.array([[0,-v3,v2],[v3,0,-v1],[-v2,v1,0]])
vx2=vx@vx
R= np.eye(3) + vx + vx2*(1-c)/s**2

# Continuer avec la lentille

file = open('./prgm/AV/{}_brute_50x50'.format(echantillon), 'rb')
surf = pickle.load(file)
file.close()
surf.enr_points_finaux(surf.points)
xs,ys,zs = surf.x_f, surf.y_f, surf.z_f
# Rotation
x,y,z=np.zeros(len(xs)), np.zeros(len(xs)), np.zeros(len(xs))
for i in range(len(xs)):
    vec=np.array([xs[i],ys[i],zs[i]])
    rot=R@vec
    x[i],y[i],z[i]= rot[0],rot[1],rot[2]

# TRANSFORMATION DES DONNÉES !!
# Mettre a zero
z=z-np.min(z)
# METTRE EN CM
Z=z*1e2
X=(x)*1e2
Y=(y)*1e2
pmin=[np.min(X), np.min(Y),np.min(Z)]
pmax=[np.max(X), np.max(Y),np.max(Z)]

# Enlever les données de marde :
mean= np.mean(Z)
std= np.std(Z)
X=X[(np.absolute(Z-mean)<1*std)]
Y=Y[(np.absolute(Z-mean)<1*std)]
Z=Z[(np.absolute(Z-mean)<1*std)]

# Remettre à zéro !
Z=Z-np.min(Z)
pmin=[np.min(X), np.min(Y),np.min(Z)]
pmax=[np.max(X), np.max(Y),np.max(Z)]

#
# from scipy.optimize import curve_fit
# from scipy.interpolate import SmoothBivariateSpline
# def theo(pts,p00,p10,p01,p20,p11,p02):
#     x=pts[:,0]; y=pts[:,1];
#     return p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2
# def func(x,y,p00,p10,p01,p20,p11,p02):
#     return p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2
# pts = np.vstack([X, Y]).T
# popt, pcov = curve_fit( theo, pts, Z  )
#
# th=np.genfromtxt('caca.txt',usecols=(0,1,2))
# x_th,y_th,z_th=th[:,0]/10,th[:,1]/10,th[:,2]/10
# pts_th = np.vstack([x_th, y_th]).T
# popt_th, pcov_th = curve_fit( theo, pts_th, z_th )
# xlin=np.linspace(pmin[0],pmax[0],num=1000);
# ylin= np.linspace(pmin[1],pmax[1],num=1000)
# xgrid,ygrid=np.meshgrid( xlin ,ylin)
#
# spl=SmoothBivariateSpline(X,Y,theo(pts,*popt))
# splth=SmoothBivariateSpline(x_th,y_th,z_th)
# ix=np.argmin(np.absolute(spl.ev(xlin,ylin,dx=1)))
# iy=np.argmin(np.absolute(spl.ev(xlin,ylin,dy=1)))
# max_x=x_th[np.argmax(z_th)]
# max_y=y_th[np.argmax(z_th)]
# delta_x=xlin[ix]-max_x
# delta_y=ylin[iy]-max_y
# delta_z=spl.ev(xlin[ix],ylin[iy])-np.max(z_th)
# xgridth,ygridth=np.meshgrid(x_th,y_th)
#
#
# from matplotlib import cm
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(X-delta_x,Y-delta_y,Z-delta_z, c='tab:red')
# ax.plot_surface(xgridth,ygridth,func(xgridth,ygridth, *popt_th),alpha=0.1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.legend()
# plt.savefig('./fig/{}/party.png'.format(echantillon),format='png')
# plt.show()
#
# # ERREUR ---------------------------------------------
# col="#8e82fe"
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# # valeurs théorique à nos valeurs de points
# pts = np.vstack([X-delta_x, Y-delta_y]).T
# th = theo(pts,*popt_th)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(X-delta_x,Y-delta_y,Z-delta_z, c='tab:red')
# ax.scatter(X-delta_x,Y-delta_y, th )
# # ax.plot_surface(xgridth,ygridth,func(xgridth,ygridth, *popt_th),alpha=0.1)
# plt.show()
# err = ((Z-delta_z)-th)
#
# mean=np.mean(err)
# std=np.std(err)
#
#
# fig, ax = plt.subplots(figsize=(8, 3))
# # fig.subplots_adjust(hspace=0)
# ax.set_aspect(1.)
# # ax.fill_between(np.linspace(pmin[0],pmax[0]), mean-std, mean+std, color=col, alpha=0.4)
# ax.plot( X, Z-delta_z , 'k.', label='Surface reconstruite')
# ax.plot( X, th , 'b.', label='Surface étalon')
# ax.set_xlabel('Position en x (cm)')
# ax.legend()
# divider = make_axes_locatable(ax)
# ax2 = divider.append_axes("right", 1.2, pad=0.75)
# ax2.set_xlabel('Erreur (cm)')
# ax2.hist(err, bins=50, color=col, alpha=0.5)
# # make some labels invisible
# # ax2.xaxis.set_tick_params(labelbottom=False)
# # ax2.yaxis.set_tick_params(labelleft=False)
# ax.set_ylabel('Hauteur en z (cm)')
# ax2.set_ylabel('Occurence')
# plt.tight_layout()
# plt.savefig('./fig/{}/hist_x.eps'.format(echantillon), format='eps')
# plt.show()
#
#
# fig, ax = plt.subplots(figsize=(8, 3))
# fig.subplots_adjust(hspace=1)
# ax.set_aspect(1.)
# # ax.fill_between(np.linspace(pmin[1],pmax[1]), mean-std, mean+std, color=col, alpha=0.4)
# ax.plot( Y, Z-delta_z, 'k.', label='Surface reconstruite')
# ax.plot( Y, th , 'b.', label='Surface étalon')
# ax.legend()
# ax.set_xlabel('Position en y (cm)')
# divider = make_axes_locatable(ax)
# ax2 = divider.append_axes("right", 1.2, pad=0.75)
# ax2.set_xlabel('Erreur (cm)')
# ax2.hist(err, bins=50, color=col, alpha=0.5)
# # make some labels invisible
# # ax2.xaxis.set_tick_params(labelbottom=False)
# # ax2.yaxis.set_tick_params(labelleft=False)
# ax.set_ylabel('Hauteur en z (cm)')
# ax2.set_ylabel('Occurence')
# plt.tight_layout()
# plt.savefig('./fig/{}/hist_y.eps'.format(echantillon), format='eps')
# plt.show()
#
#
#
#
#
#
#
#
#
# # TRANCHES  ---------------------------------------------
# col="#8e82fe"
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# # fig, ax = plt.subplots()
# # ax.plot( X, Z, 'k.', markersize='2')
# # ax.set_ylabel('Hauteur en z \n (cm)')
# # ax.set_xlabel('Largeur en x (cm)')
# # ax.set_aspect('equal')
# # plt.savefig('./fig/{}/tranche_x.eps'.format(echantillon), format='eps')
# # plt.show()
# #
# # fig, ax = plt.subplots()
# # ax.plot( Y, Z, 'k.', markersize='2')
# # ax.set_ylabel('Hauteur en z \n (cm)')
# # ax.set_xlabel('Largeur en y (cm)')
# # ax.set_aspect('equal')
# # plt.savefig('./fig/{}/tranche_y.eps'.format(echantillon), format='eps')
# # plt.show()
#
# # fig, ax = plt.subplots()
# # ax.plot( X, Y, 'k.', markersize='2')
# # ax.set_ylabel('y(cm)')
# # ax.set_xlabel('x (cm)')
# # ax.set_aspect('equal')
# # plt.savefig('./fig/{}/rond.eps'.format(echantillon), format='eps')
# # plt.show()


# INTERPOLATION --------------------------------------------------------
from scipy.interpolate import griddata

pts = np.vstack([X, Y]).T
(grid_x, grid_y) = np.meshgrid(np.linspace(pmin[0],pmax[0],200),np.linspace(pmin[1],pmax[1],200) )
grid_z2 = griddata(pts, Z, (grid_x, grid_y), method='cubic')
fig,ax=plt.subplots(1,1)
mycmap=plt.get_cmap('plasma')
cp = ax.contourf(grid_x, grid_y, grid_z2, cmap=mycmap)
cbar = fig.colorbar(cp, label = 'Hauteur en z (cm)')
# Add a colorbar to a plot
plt.xlabel('x'); plt.ylabel('y')
plt.savefig('./fig/{}/heatmap.eps'.format(echantillon), format='eps')
plt.show()



# # # TRISURF -----------------------------------
import matplotlib.tri as mtri
import scipy.spatial
from matplotlib import cm
# plot final solution

pts = np.vstack([X, Y]).T
tess = scipy.spatial.Delaunay(pts) # tessilation
# Create the matplotlib Triangulation object
xx = tess.points[:, 0]
yy = tess.points[:, 1]
tri = tess.vertices # or tess.simplices depending on scipy version
triDat = mtri.Triangulation(x=pts[:, 0], y=pts[:, 1], triangles=tri)

# Trisurf :
fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')
set_aspect_3D(pmin, pmax,ax)
ax.plot_trisurf(triDat, Z, linewidth=0, edgecolor='none',
                antialiased=False, cmap=cm.jet)
ax.set_ylabel('y (cm)');ax.set_xlabel('x (cm)')
ax.set_zlabel('z (cm)')
# ax.scatter(x,y,z)
# ax.set_title(r'trisurf with delaunay triangulation',
          # fontsize=16, color='k')
ax.set_zlim3d(bottom=0)
plt.savefig('./fig/{}/trisurf.eps'.format(echantillon), format='eps')
plt.show()
#
#
# PLOTLYYYYY
import plotly.figure_factory as FF
import plotly.graph_objs as go
simplices = tess.simplices
fig = FF.create_trisurf(x=X, y=Y, z=Z,
                         colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', '#c8dcc8'],
                         # height=800,width=1000,
                         show_colorbar=False,
                         simplices=simplices,
                         aspectratio=dict(x=1, y=1, z=1))
fig.update_layout(showlegend=False)
set_aspect_plotly(pmin,pmax, fig)
fig.write_image("./fig/{}/surface.pdf".format(echantillon))
fig.show()
