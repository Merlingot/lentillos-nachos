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

echantillon='miroir_plan'

file = open('./prgm/AV/miroir_plan_brute_25x25', 'rb')
surf = pickle.load(file)
file.close()
surf.enr_points_finaux(surf.points)

# Premiere etape : Rapporter a zero -----------------------------------
xs,ys,zs = surf.x_f, surf.y_f, surf.z_f

# Deuxieme etape : redresser le plan -----------------------------------
# do fit
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

print ("solution:")
print ("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
print ("residual:")
print (residual)

# X,Y = np.meshgrid(xs, ys)
# plot plane
# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(xs, ys, zs, color='b')
# Z = fit.item(0) * X + fit.item(1) * Y + fit.item(2)
# ax.plot_wireframe(X,Y,Z, color='k')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

a=fit.item(0); b=fit.item(1); c=-1; d=fit.item(2)
n=np.array([a,b,c])
a=n/np.linalg.norm(n)
b=np.array([0,0,1])
v=np.cross(a,b)
s=np.linalg.norm(v)
c=a@b
v1,v2,v3=v[0],v[1],v[2]
vx=np.array([[0,-v3,v2],[v3,0,-v1],[-v2,v1,0]])
vx2=vx@vx
R= np.eye(3) + vx + vx2*(1-c)/s**2

# Rotation de 45 degree
t=0
Rz=np.array([[np.cos(t), -np.sin(t),0],[np.sin(t), np.cos(t),0], [0,0,1]])
x,y,z=np.zeros(len(xs)), np.zeros(len(xs)), np.zeros(len(xs))
for i in range(len(xs)):
    vec=np.array([xs[i],ys[i],zs[i]])
    rot=Rz@(R@vec)
    x[i],y[i],z[i]= rot[0],rot[1],rot[2]

# TRANSROMATION DES DONNÉES !!
# Mettre a zero
z=z-np.min(z)
# METTRE EN CM
Z=z*1e2
X=(x)*1e2
Y=(y)*1e2
pmin=[np.min(X), np.min(Y),np.min(Z)]
pmax=[np.max(X), np.max(Y),np.max(Z)]

# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(x, y, z, color='b')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()



# TRANCHES  ---------------------------------------------

mean= np.mean(Z)
std= np.std(Z)
col="#8e82fe"
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(8, 3))
fig.subplots_adjust(hspace=0)
# ax.set_aspect(1.)
ax.fill_between(np.linspace(pmin[0],pmax[0]), mean-std, mean+std, color=col, alpha=0.4)
ax.plot( X, Z, 'k.')
ax.set_xlabel('Largeur en x (cm)')
divider = make_axes_locatable(ax)
ax2 = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
ax2.hist(Z, bins=50, orientation="horizontal", color=col, alpha=0.5)
# make some labels invisible
# ax2.xaxis.set_tick_params(labelbottom=False)
ax2.yaxis.set_tick_params(labelleft=False)
ax.set_ylabel('Hauteur en z (cm)')
ax2.set_xlabel('Occurence')
plt.tight_layout()
plt.savefig('./fig/{}_hist_x.eps'.format(echantillon), format='eps')
plt.show()


fig, ax = plt.subplots(figsize=(8, 3))
fig.subplots_adjust(hspace=0)
# ax.set_aspect(1.)
ax.fill_between(Y, mean-std, mean+std, color=col, alpha=0.4)
ax.plot( Y, Z, 'k.')
ax.set_xlabel('Largeur en y (cm)')
divider = make_axes_locatable(ax)
ax2 = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
ax2.hist(Z, bins=50, orientation="horizontal", color=col, alpha=0.5)
# make some labels invisible
# ax2.xaxis.set_tick_params(labelbottom=False)
ax2.yaxis.set_tick_params(labelleft=False)
ax.set_ylabel('Hauteur en z (cm)')
ax2.set_xlabel('Occurence')
plt.tight_layout()
plt.savefig('./fig/{}_hist_y.eps'.format(echantillon), format='eps')
plt.show()


fig, ax = plt.subplots()
ax.plot( X, Z, 'k.', markersize='2')
ax.set_ylabel('Hauteur en z \n (cm)')
ax.set_xlabel('Largeur en x (cm)')
ax.set_aspect('equal')
plt.savefig('./fig/{}_tranche_x.eps'.format(echantillon), format='eps')
plt.show()

fig, ax = plt.subplots()
ax.plot( Y, Z, 'k.', markersize='2')
ax.set_ylabel('Hauteur en z \n (cm)')
ax.set_xlabel('Largeur en y (cm)')
ax.set_aspect('equal')
plt.savefig('./fig/{}_tranche_y.eps'.format(echantillon), format='eps')
plt.show()


# INTERPOLATION --------------------------------------------------------
from scipy.interpolate import griddata

pts = np.vstack([X, Y]).T
(grid_x, grid_y) = np.meshgrid(np.linspace(pmin[0],pmax[0],200),np.linspace(pmin[1],pmax[1],200) )
grid_z2 = griddata(pts, Z, (grid_x, grid_y), method='cubic')
fig,ax=plt.subplots(1,1)
mycmap=plt.get_cmap('plasma')
cp = ax.contourf(grid_x, grid_y, grid_z2-mean, cmap=mycmap)
cbar = fig.colorbar(cp, label = 'Deviation par rapport à la moyenne (cm)')
# Add a colorbar to a plot
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
plt.savefig('./fig/{}_heatmap.eps'.format(echantillon), format='eps')
plt.show()

# # TRISURF -----------------------------------
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

# # Trisurf :
# fig = plt.figure(figsize=(6,6))
# ax = fig.gca(projection='3d')
# set_aspect_3D(pmin, pmax,ax)
# ax.plot_trisurf(triDat, Z, linewidth=0, edgecolor='none',
#                 antialiased=False, cmap=cm.jet)
# # ax.scatter(x,y,z)
# ax.set_title(r'trisurf with delaunay triangulation',
#           fontsize=16, color='k')
# ax.set_zlim3d(bottom=0)
# plt.show()


#
# PLOTLYYYYY
import plotly.figure_factory as FF
import plotly.graph_objs as go
simplices = tess.simplices
fig = FF.create_trisurf(x=X, y=Y, z=Z,
                         colormap=['rgb(50, 0, 75)', 'rgb(200, 0, 200)', '#c8dcc8'],
                         height=800,width=1000,
                         show_colorbar=True,
                         simplices=simplices, title="Miroir plan",
                         aspectratio=dict(x=1, y=1, z=0.3))
fig.update_layout(showlegend=False)
set_aspect_plotly(pmin,pmax, fig)
fig.write_image("./fig/{}_surface.pdf".format(echantillon))
# fig.show()



#
