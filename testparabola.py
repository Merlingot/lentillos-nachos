from __future__ import print_function, division
import numpy as np
from PyAstronomy import pyasl
import matplotlib.pylab as plt

# Create some data (a Gaussian)
x = np.arange(100.0)
y = np.exp(-(x-50.2714)**2/(2.*5.**2))

# Find the maximum
epos, mi = pyasl.quadExtreme(x, y, mode="max")
print("Maximum found at index: ", mi, ", value at maximum: ", y[mi])
print("Maximum found by parabolic fit: ", epos)
print()

# Find the maximum, use a wider range for the
# parabolic fit.
print("Using 5 points to each side of the maximum")
epos, mi = pyasl.quadExtreme(x, y, mode="max", dp=(5,5))
print("Maximum found at index: ", mi, ", value at maximum: ", y[mi])
print("Maximum found by parabolic fit: ", epos)
print()

# Do as above, but get the full output
print("Using 2 points to each side of the maximum")
epos, mi, xb, yb, p = pyasl.quadExtreme(x, y, mode="max", dp=(2,2), fullOutput=True)
# Evaluate polynomial at a number of points.
# Note that, internally, the x-value of the extreme point has
# been subtracted before the fit. Therefore, we need to re-shift
# it in the plot.
newx = np.linspace(min(xb), max(xb), 100)
model = np.polyval(p, newx)

# Plot the "data"
plt.plot(x, y, 'bp')
# Mark the points used in the fitting (shifted, because xb is shifted)
plt.plot(xb+x[mi], yb, 'rp')
# Overplot the model (shifted, because xb is shifted)
plt.plot(newx+x[mi], model, 'r--')
plt.show()





# SHOW DIFFERENT SGMF APPROXIMATIONS
sgmf17 = cv2.imread("./data/" + echantillon + "/cam_match_median7_PG.png",-1)/900
sgmf15 = cv2.imread("./data/" + echantillon + "/cam_match_median5_PG.png",-1)/900
sgmf10 = cv2.imread("./data/" + echantillon + "/cam_match_PG.png",-1)/900

spl = UnivariateSpline(np.arange(0,len(sgmf10[:,1000,1])),sgmf10[:,1000,1])
sgmfspl = spl(np.arange(0,len(sgmf10[:,1000,1])))


plt.plot(sgmf10[:,1000,1],'-o')
plt.plot(sgmfspl)
plt.plot(sgmf15[:,1000,1],'-o')
plt.plot(sgmf17[:,1000,1],'-o')
plt.legend(["raw","UnivariateSpline","5 pixel window","7 pixel window"])
plt.show()
