import matplotlib.pyplot as plt
import numpy as np
import cv2

def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255; # Some values seem to go above 255. However RGB channels has to be within 0-255
    return sobel

def confidenceMap(sgmf, name):
    ## (1) read and extract the blue channel
    img = cv2.imread(sgmf)
    # uncomment for pre-blurring
    # img = cv2.medianBlur(img,7)


    red = img[:,:,1]     # CHANNEL Y
    green = img[:,:,2]   # CHANNEL X



    # Adaptive Thresholding
    green2 = cv2.adaptiveThreshold(green, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 3, 2)

    red2 = cv2.adaptiveThreshold(red, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 3, 2)

    edgeImgG = np.max( np.array([ edgedetect(red2) ]), axis=0 )
    edgeImgR = np.max( np.array([ edgedetect(green2) ]), axis=0 )


    edgeImgR[edgeImgR <= np.mean(edgeImgR)] = 0;
    edgeImgG[edgeImgG <= np.mean(edgeImgG)] = 0;

    # Blur the image
    # changer le (21,21)PG en (7,7)AV
    maskimg = cv2.bilateralFilter(edgeImgR,5,75,75)*cv2.bilateralFilter(edgeImgG,5,75,75)



    maskimg = 255 - maskimg * 255
    plt.imshow(maskimg)
    plt.show()

    # maskimg = cv2.medianBlur(img,3)
    from scipy.ndimage import maximum_filter, minimum_filter
    def midpoint(img):
        maxf = maximum_filter(img, (4, 4))
        minf = minimum_filter(img, (4, 4))
        midpoint = (maxf + minf) / 2
        midpoint[midpoint < 0] = 0
        return midpoint
    plt.imshow(midpoint(maskimg))
    plt.show()

    cv2.imwrite(name, midpoint(maskimg))



echantillon = "lentille_plano_convexe"
camera = "PG"
confidenceMap('./data/'+ echantillon +'/cam_match_' + camera + '.png', './data/'+ echantillon +'/extremeconf_' + camera + '.png')
#

# apply medianBlur to SGMF


# sgmf1 = "./data/" + echantillon + "/cam_match_PG.png"
# sgmf2 = "./data/" + echantillon + "/cam_match_AV.png"
#
# window = 7
#
# blur1 = cv2.medianBlur(cv2.imread(sgmf1,-1),window)
# blur2 = cv2.medianBlur(cv2.imread(sgmf2,-1),window)
#
# cv2.imwrite("./data/" + echantillon + "/cam_match_median" + str(window) +"_PG.png", blur1)
# cv2.imwrite("./data/" + echantillon + "/cam_match_median" + str(window) +"_AV.png", blur2)
