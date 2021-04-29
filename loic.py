# Curve fit :
from scipy.optimize import curve_fit, leastsq
import math
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]

r, x0, y0, z0 = sphereFit(X,Y,Z)

def cercle(x,y,r, x0,y0,z0):
    return z0 + np.sqrt( r**2 - (x-x0)**2 - (y-y0)**2 )

ZZ = cercle( X,Y,r, x0, y0, z0 )

fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')
ax.scatter3D(X,Y,ZZ,color='tab:orange')
ax.scatter3D(X,Y,Z)
plt.show()
# # TRAITEMENT DES DONNÉES ------------------------------------------------
#
# #
#
# #
# def func(x,a,b,x0,y0):
#     return y0**2 + np.sqrt( b**2 * ((x-x0)/a)**2 + 1) -1
#    # return np.piecewise(x, [x < x0, x > x0], [lambda x:a1*(x-x01), lambda x:a2*(x-x02)])
#
# ramp = lambda u: np.maximum( u, 0 )
# step = lambda u: ( u > 0 ).astype(float)
#
# def SegmentedLinearReg( X, Y, breakpoints ):
#    nIterationMax = 10
#
#    breakpoints = np.sort( np.array(breakpoints) )
#
#    dt = np.min( np.diff(X) )
#    ones = np.ones_like(X)
#
#    for i in range( nIterationMax ):
#        # Linear regression:  solve A*p = Y
#        Rk = [ramp( X - xk ) for xk in breakpoints ]
#        Sk = [step( X - xk ) for xk in breakpoints ]
#        A = np.array([ ones, X ] + Rk + Sk )
#        p =  lstsq(A.transpose(), Y, rcond=None)[0]
#
#        # Parameters identification:
#        a, b = p[0:2]
#        ck = p[ 2:2+len(breakpoints) ]
#        dk = p[ 2+len(breakpoints): ]
#
#        # Estimation of the next break-points:
#        newBreakpoints = breakpoints - dk/ck
#
#        # Stop condition
#        if np.max(np.abs(newBreakpoints - breakpoints)) < dt/5:
#            break
#
#        breakpoints = newBreakpoints
#    else:
#        print( 'maximum iteration reached' )
#
#    # Compute the final segmented fit:
#    Xsolution = np.insert( np.append( breakpoints, max(X) ), 0, min(X) )
#    ones =  np.ones_like(Xsolution)
#    Rk = [ c*ramp( Xsolution - x0 ) for x0, c in zip(breakpoints, ck) ]
#
#    Ysolution = a*ones + b*Xsolution + np.sum( Rk, axis=0 )
#
#    return Xsolution, Ysolution
#
#
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for point in surf.points:
#     # surf.good_points.append(point)
#     if point.valmin < 0.01:
#         if len(point.vecP[:,2]) >= 13:
#             spl = UnivariateSpline(-point.vecP[:,2], point.vecV, k = 4, s = 0)
#             roots = spl.derivative(1).roots()
#             dev1 = spl.derivative(1)
#             dev2 = spl.derivative(2)
#             if len(roots) == 1:
#                 if dev2(roots) > 0:
#                     # on ne garde que les points dont la proportion de montée est d'un certain pourcentage de la plage
#                     if np.max(np.absolute(np.ediff1d( point.vecV))) < 0.01:
#                         if sum(dev1(-point.vecP[:,2]) <= 0)/len(-point.vecP[:,2]) > 0.25:
#                             if sum(dev1(-point.vecP[:,2]) >= 0)/len(-point.vecP[:,2]) > 0.25 :
# #                                p_min, val_min, n1, n2 = parabolic_search(point,t,d,cam1, cam2, ecran)
# #                                # print(p_min[0],p_min[1],p_min[2],val_min,n1)
# #                                point.pmin=p_min; point.valmin=val_min
#
#                                 show_point(point)
#
#                                 xdata = point.vecP[:,2]; ydata = point.vecV
#                                 popt, pcov = curve_fit(func, xdata, ydata)
#
#                                 plt.figure()
#                                 plt.plot(xdata, func(xdata,*popt), '-r')
# #                                initialBreakpoints = [-18e-2]
# #                                plt.plot( *SegmentedLinearReg( xdata, ydata, initialBreakpoints ), '-r' );
#                                 plt.plot(xdata,ydata,'o')
#                                 plt.show()
#                                 ax.scatter(point.pmin[0], point.pmin[1], point.pmin[2],color="r")
#                                 surf.good_points.append(point)
# plt.show()
