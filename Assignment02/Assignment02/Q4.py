import numpy as np
from skimage import io,color,img_as_float32,img_as_ubyte
from matplotlib import pyplot as plt
from math import e,sqrt,radians,cos,sin
from scipy.signal import convolve2d
from scipy.optimize import minimize
import cv2
import random
import skimage

def smooth(image:np.ndarray,m:int)->np.ndarray:
    '''
    m : MxM smoothing window
    '''
    h = np.ones(shape=(m,m)).astype(np.float32)
    image = img_as_float32(image)
    res = convolve2d(image,h,mode='same')/(m**2)
    return img_as_ubyte(res)

def highBoostFiltering(image:np.ndarray,m:int,k:float)->np.ndarray:
    '''
    image : source image
    m : smoothing window size
    k : highboost filter co efficient
    '''
    h = np.ones(shape=(m,m)).astype(np.float32)/(m**2)
    image = img_as_float32(image)
    blurred = convolve2d(image,h,mode='same')
    gmask = image-blurred
    res = image + k*gmask
    return res,smooth(image,m)

def testHighBoostFiltering(m):
    img = io.imread('inputs/noisy.tif')
    clean = io.imread('inputs/characters.tif')
    clean = img_as_float32(clean)
    #Minimizing Scaling Constant
    obj = lambda k : sqrt(np.mean(np.square(highBoostFiltering(smooth(img,m),m,k)[0]-clean)))
    error = obj(1)
    errors = list()
    k=1
    for i in np.linspace(0,2,50):
        er = obj(i)
        errors.append(er)
        if er<error:
            error,k = er,i
    plt.subplot(121)
    plt.scatter(x=np.linspace(0,2,50),y=errors)
    res1,res2 = highBoostFiltering(smooth(img,m),m,k)
    plt.subplot(122)
    plt.title(label=f'{m}x{m},k = {k:.3f}')
    plt.imshow(res1,cmap=plt.cm.gray)
    plt.savefig("./outputs/Q4/"+str(m))
    plt.clf()

if __name__=="__main__":
    testHighBoostFiltering(5)
    testHighBoostFiltering(10)
    testHighBoostFiltering(15)
    # Apply Unsharp masking
    # img = io.imread('inputs/noisy.tif')
    # m=5
    # h = np.ones(shape=(m,m)).astype(np.float32)/(m**2)
    # gauss = cv2.filter2D(img, -1 , h)
    # res1 = cv2.addWeighted(gauss, 2, gauss, -1, 0)
    # plt.imshow(res1,cmap=plt.cm.gray)
    # plt.show()
