import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
import cycler


O_CAM_REF_CAM = np.array([0,0,0])
DIR_CAM_REF_CAM = np.array([0,0,1])

def cam_refEcran(cam, rgb_cam, L, legende, S=50):
    # Points de départ des flèches
    oCam = cam.S[:3]
    # Vecteurs unitaires
    dirCam = cam.normale[:3]
    # Plan ccd cam2
    coin1=cam.cacmouE(np.array([0,0,1]))
    coin2=cam.cacmouE(np.array([cam.w[0], 0,1]))
    coin3=cam.cacmouE(np.array([0, cam.w[1],1]))
    coin4=cam.cacmouE(np.array([cam.w[0],cam.w[1],1]))
    ccd = go.Mesh3d(
        x = [coin1[0], coin2[0], coin3[0], coin4[0] ],
        y = [coin1[1], coin2[1], coin3[1], coin4[1] ],
        z = [coin1[2], coin2[2], coin3[2], coin4[2] ],
        color='rgb({},{},{})'.format(rgb_cam[0],rgb_cam[1],rgb_cam[2])
        )
    data = [ccd]
    data += fleche(oCam, dirCam, rgb=rgb_cam, s=1/S, l=L, name=legende)
    return data


def ecran_refEcran(ecran, rgb_ecran, L, S=50):
    # Points de départ des flèches
    oEcran = np.array([0,0,0])
    # Vecteurs unitaires
    dirEcran = np.array([0,0,1])
    coin1 = ecran.pixelToSpace(np.array([0,0,1]))
    coin2 = ecran.pixelToSpace(np.array([ecran.w[0],0,1]))
    coin3 = ecran.pixelToSpace(np.array([0,ecran.w[1],1]))
    coin4 = ecran.pixelToSpace(np.array([ecran.w[0],ecran.w[1],1]))
    ecran = go.Mesh3d(
            x = [coin1[0], coin2[0], coin3[0], coin4[0] ],
            y = [coin1[1], coin2[1], coin3[1], coin4[1] ],
            z = [coin1[2], coin2[2], coin3[2], coin4[2] ],
            color='rgb({},{},{})'.format(rgb_ecran[0],rgb_ecran[1],rgb_ecran[2]),
            opacity=0.1
            )
    data =[ecran]
    data += fleche(oEcran, np.array([1,0,0]), rgb=(0,100,100), s=1/S, l=L, name='x')
    data += fleche(oEcran, np.array([0,1,0]), rgb=(0,200,200), s=1/S, l=L, name='y')
    data += fleche(oEcran, dirEcran, rgb=rgb_ecran, s=1/S, l=L, name='Ecran')
    return data


def grilles_refEcran(surf, rgb_grille_i, rgb_grille_f, t, d, L, S=50):
    # Points de départ des flèches
    oRecherche = t
    # Vecteurs unitaires
    dirRecherche = d

    # Grille initiale
    # data_grille_init = go.Mesh3d(
    #     x = surf.x_i,
    #     y = surf.y_i,
    #     z = surf.z_i,
    #     # mode = 'markers',
    #     # marker = dict(size=9)
    #     opacity=0.1
    #     )
    # Grille finale
    data_grille_finale= go.Scatter3d(
        x = surf.x_f,
        y = surf.y_f,
        z = surf.z_f,
        mode = 'markers',
        marker = dict(size=1)
        )
    # data = [data_grille_init, data_grille_finale]
    data =  [data_grille_finale]
    ## recherche
    # data += fleche(oRecherche, dirRecherche, rgb=rgb_grille_i, s=1/S, l=L, name='Direction recherche')

    # VECTEUR NORMAUX
    # for point in surf.good_points:
    #     data += fleche(point.pmin, point.nmin, rgb=rgb_grille_i, s=1/500, l=1e-2)

    return data

def montage_refEcran(surf, ecran, cam1, cam2, L, t, d):

    S=50
    # codes rgb
    rgb_ecran=(255,0,0)
    rgb_grille_i=(0,0,0)
    rgb_grille_f=(0,0,0)
    rgb_cam1=(0,255,0)
    rgb_cam2=(0,0,255)

    data=[]
    data += ecran_refEcran(ecran, rgb_ecran, L, S)
    data += grilles_refEcran(surf, rgb_grille_i, rgb_grille_f, t, d, L, S)
    data += cam_refEcran(cam1, rgb_cam1, L, 'cam1 PG', S)
    data += cam_refEcran(cam2, rgb_cam2, L, 'cam2 AV', S)

    fig = go.Figure(data)

    fig.update_layout(
    scene = dict(xaxis_title='X', yaxis_title='Y',zaxis_title='Z',     aspectratio=dict(x=1, y=1, z=1),
    aspectmode='manual',
    camera = dict(
    up=dict(x=0, y=0, z=-1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=-1.25)
    )))

    set_aspect_3D_plotly(cam1, fig)
    fig.update_layout(showlegend=True)
    # fig.write_image("fig_{}.eps".format(stra))
    fig.show()



def surface_refEcran(surf):

    S=50

    # Grille finale
    data_grille_finale= go.Scatter3d(
        x = surf.x_f,
        y = surf.y_f,
        z = surf.z_f,
        mode = 'markers',
        marker = dict(size=1)
        # opacity=0.2
        )
    data =  [data_grille_finale]

    fig = go.Figure(data)

    fig.update_layout(
    scene = dict(xaxis_title='X', yaxis_title='Y',zaxis_title='Z',     aspectratio=dict(x=1, y=1, z=1),
    aspectmode='manual',
    camera = dict(
    up=dict(x=0, y=0, z=-1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=-1.25)
    )))

    # set_aspect_3D_plotly(cam1, fig)
    # fig.update_layout(showlegend=True)
    # fig.write_image("fig_{}.eps".format(stra))
    fig.show()


def allo_refEcran(pp, ecran, cam1, cam2, L, t, d):

    S=50
    # codes rgb
    rgb_ecran=(255,0,0)
    rgb_grille_i=(0,0,0)
    rgb_grille_f=(0,0,0)
    rgb_cam1=(0,255,0)
    rgb_cam2=(0,0,255)

    data=[]
    data += ecran_refEcran(ecran, rgb_ecran, L, S)

    for i in range(len(pp.vecP)):
        data += point_refEcran(pp.vecP[i],  ecran, cam1, L, S, rgb_cam1)
        data += point_refEcran(pp.vecP[i], ecran, cam2, L, S, rgb_cam2)
        data += fleche(pp.vecP[i], pp.vecN1[i], rgb=rgb_cam1, name='n1', s=1/S, l=L)
        data += fleche(pp.vecP[i], pp.vecN2[i], rgb=rgb_cam2, name='n2',  s=1/S, l=L)

    data += cam_refEcran(cam1, rgb_cam1, L, 'cam1 PG', S)
    data += cam_refEcran(cam2, rgb_cam2, L, 'cam2 AV', S)

    fig = go.Figure(data)

    fig.update_layout(
    scene = dict(xaxis_title='X', yaxis_title='Y',zaxis_title='Z',     aspectratio=dict(x=1, y=1, z=1),
    aspectmode='manual',
    camera = dict(
    up=dict(x=0, y=0, z=-1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=-1.25)
    )))

    # set_aspect_3D_plotly(cam1, fig)
    fig.update_layout(showlegend=True)
    # fig.write_image("fig_{}.eps".format(stra))
    fig.show()


# def point_refEcran(p, ecran, cam, L, S, rgb_cam):
#
#     P = np.array([p[0], p[1], p[2], 1])
#     c = cam.camToCCD( cam.ecranToCam(P) )
#     u = cam.spaceToPixel(c)
#     vecPix = cam.pixCamToEcran(u)
#     E = ecran.pixelToSpace(vecPix)
#     UC = cam.camToEcran(c)
#
#     cacE = go.Scatter3d(
#         x = [P[0], E[0] ],
#         y = [P[1], E[1]],
#         z = [P[2], E[2]],
#         mode = 'lines',
#         marker=dict(color='rgb({},{},{})'.format(rgb_cam[0],rgb_cam[1],rgb_cam[2]) )
#
#         )
#     cacU = go.Scatter3d(
#         x = [P[0], UC[0] ],
#         y = [P[1], UC[1]],
#         z = [P[2], UC[2]],
#         mode = 'lines',
#         marker=dict(color='rgb({},{},{})'.format(rgb_cam[0],rgb_cam[1],rgb_cam[2]) )
#         )
#     data = [cacE, cacU]
#
#     return data



# # CAMERA ---------
# def cam_refCam(cam, rgb_cam, rgb_ccd, L, legende, S=50):
#     # Points de départ des flèches
#     oCam = O_CAM_REF_CAM
#     # Vecteurs unitaires
#     dirCam = DIR_CAM_REF_CAM
#     # Plan ccd cam2
#     coin1=cam.cacmouC(np.array([0,0,1]))
#     coin2=cam.cacmouC(np.array([cam.w[0], 0,1]))
#     coin3=cam.cacmouC(np.array([0, cam.w[1],1]))
#     coin4=cam.cacmouC(np.array([cam.w[0],cam.w[1],1]))
#     ccd = go.Mesh3d(
#         x = [coin1[0], coin2[0], coin3[0], coin4[0] ],
#         y = [coin1[1], coin2[1], coin3[1], coin4[1] ],
#         z = [coin1[2], coin2[2], coin3[2], coin4[2] ],
#         color='rgb({},{},{})'.format(rgb_ccd[0],rgb_ccd[1],rgb_ccd[2])
#         )
#     data = [ccd]
#     data += fleche(oCam, dirCam, rgb=rgb_cam, s=1/S, l=L, name=legende)
#     return data
#
#
# def montage_refCam(cam, leg, L):
#
#     S=50
#     # codes rgb
#     rgb_cam=(0,255,0)
#     rgb_ccd=(255,0,0)
#
#     data=[]
#     data += cam_refCam(cam, rgb_cam, rgb_ccd, L, leg, S)
#
#     fig = go.Figure(data)
#
#     fig.update_layout(
#     scene = dict(xaxis_title='X', yaxis_title='Y',zaxis_title='Z',     aspectratio=dict(x=1, y=1, z=1),
#     aspectmode='manual'
#     ))
#
#     # set_aspect_3D_plotly(cam, fig)
#     fig.update_layout(showlegend=True)
#     # fig.write_image("fig_{}.eps".format(stra))
#     fig.show()




# Tous refs --------------

def fleche(vecI, vecDir, rgb=(0,0,0), s=1/10, l=1, name='trace'):
    """
    vecI : np.array([x,y,z])
        coordonées du point de départ de la flèche
    vecDir : np.array([x,y,z])
        vecteur unitaire donnant la direction du vecteur
    scale : nombre entre 0 et 1
        grossit ou diminue la dimension du cone
    l : nombre > 0
        longueur de la tige de la flèche
    rgb : tuple (r,g,b)
        code rgb de la couleur de la flêche
    """
    xi,yi,zi = vecI
    xf,yf,zf =  vecI + vecDir*l
    u,v,w = vecDir
    color = 'rgb({},{},{})'.format(rgb[0],rgb[1],rgb[2])

    cone = go.Cone(
    x=[xf], y=[yf], z=[zf],
    u=[u], v=[v], w=[w],
    showscale=False,
    cauto=True,
    sizemode='absolute',
    sizeref=s,
    colorscale=[[0, color], [1, color]],
    anchor='cm',
    name=name
     )
    tige = go.Scatter3d(
    x =[xi,xf], y=[yi,yf], z=[zi,zf],
    name=name,
    mode='lines',
    line=dict(
        width=5, color=color
    ) )

    return [ tige, cone ]


def set_aspect_3D_plotly(cam, fig):
    """
    Fait en sorte que la visualisation soit pas dégueu
    """
    k=1.5
    X=np.array([0, cam.S[0]*k])
    Y=np.array([0, cam.S[1]*k])
    Z=np.array([0, cam.S[2]*k])
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    fig.add_trace(go.Scatter3d(x=Xb,
                   y=Yb,
                   z=-1*Zb,
                   mode='markers',
                    marker=dict(
                        color='rgba(255,255,255,0)',
                        opacity=0
                    )))










# Visualisation en matplotlib --------------------------------------------------

def set_aspect_3D(cam, ax):
    ## Set aspect -----------------------------------------
    X=np.array([0, cam.S[0]])
    Y=np.array([0, cam.S[1]])
    Z=np.array([0, cam.S[2]])
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')
    ## Set aspect -----------------------------------------


def show_sgmf(cam1, cam2, point):
    # Visualisation of SGMF points
    f, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2)

    ax1.imshow(cam1.sgmf[:,:,0]*cam1.mask, cmap="Greys")
    ax1.set_xlabel("u")
    ax1.set_ylabel("v")
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.set_title("SGMF PG")
    for pt in point.vecU1:
        ax1.scatter( pt[0], pt[1], color='r')

    ax2.imshow(cam2.sgmf[:,:,0]*cam2.mask, cmap="Greys")
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.set_title("SGMF AV")
    for pt in point.vecU2:
        ax2.scatter( pt[0], pt[1], color='g')

    ax3.plot(point.vecP[:,2], point.vecV, '-o', color='b')
    # ax3.plot(point.vecP_ternary[:,2], point.vecV_ternary, 'o', color='b',markersize = 1)
    ax3.plot(point.pmin[2],point.valmin,'x',color = 'r')
    ax3.set_xlabel("Distance parcourue selon d (m)")
    ax3.set_ylabel("Inconsistency value")

    ax4.set_aspect('equal', 'box')
    ax4.axhline(0); ax4.axhline(900)
    ax4.axvline(0); ax4.axvline(1600)
    ax4.set_xlabel("u'")
    ax4.set_ylabel("v'")
    ax4.xaxis.set_label_position('top')
    ax4.xaxis.tick_top()
    for pt in point.vecE1:
        ax4.scatter( pt[0], pt[1], color='g')
    for pt in point.vecE2:
        ax4.scatter( pt[0], pt[1], color='r')

    plt.show()


def show_point(point):
    # Visualisation of SGMF points
    f, ax3  = plt.subplots()

    ax3.plot(point.vecP[:,2], point.vecV, '-o', color='tab:blue')
    ax3.plot(point.vecPr[:,2], point.vecVr, '-o', color='tab:orange')
    # ax3.plot(point.vecP_ternary[:,2], point.vecV_ternary, 'o', color='b',markersize = 1)
    ax3.plot(point.pmin[2],point.valmin,'x',color = 'tab:pink')
    ax3.set_xlabel("Distance parcourue selon d (m)")
    ax3.set_ylabel("Inconsistency value")

    plt.show()

def show_caca(camR, cam, point):
    # Visualisation of SGMF points
    f, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, figsize=(10,7))

    ax1.imshow(camR.sgmf[:,:,0]*camR.mask, cmap="Greys")
    ax1.set_xlabel("u")
    ax1.set_ylabel("v")
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.set_title("SGMF CAM REF")
    for pt in point.vecUr:
        ax1.scatter( pt[0], pt[1], color='r')

    ax2.imshow(cam.sgmf[:,:,0]*cam.mask, cmap="Greys")
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.set_title("SGMF CAM 2")
    for pt in point.vecU:
        ax2.scatter( pt[0], pt[1], color='g')

    ax3.plot(point.vecP[:,2], point.vecV, '-o', color='b')
    # ax3.plot(point.vecP_ternary[:,2], point.vecV_ternary, 'o', color='b',markersize = 1)
    ax3.plot(point.pmin[2],point.valmin,'x',color = 'r')
    ax3.set_xlabel("Distance parcourue selon d (m)")
    ax3.set_ylabel("Inconsistency value")

    ax4.set_aspect('equal', 'box')
    ax4.axhline(0); ax4.axhline(900)
    ax4.axvline(0); ax4.axvline(1600)
    ax4.set_xlabel("u'")
    ax4.set_ylabel("v'")
    ax4.xaxis.set_label_position('top')
    ax4.xaxis.tick_top()
    for pt in point.vecEr:
        ax4.scatter( pt[0], pt[1], color='r')
    for pt in point.vecE:
        ax4.scatter( pt[0], pt[1], color='g')
    for pt in point.vecE:
        ax4.scatter( pt[0], pt[1], color='g', marker='x')

    plt.show()
