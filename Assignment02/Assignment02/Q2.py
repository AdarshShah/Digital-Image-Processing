import numpy as np
from skimage import io,color,img_as_float32,img_as_ubyte
from matplotlib import pyplot as plt
from math import e,sqrt,radians,cos,sin
from scipy.signal import convolve2d
from scipy.optimize import minimize
import cv2
import random

def subsample(image:np.ndarray,k:int=2)->np.ndarray:
    '''
    Reduces the image size by factor of k
    image : source image
    k : factor in int
    '''
    L,B = image.shape
    newimg = np.zeros(shape=(L//k,B//k))
    for l in range(L//k):
        for b in range(B//k):
            newimg[l][b]=image[k*l][k*b]
    return newimg.astype(int)
    
def upsample(image:np.ndarray,method:str,k:int)->np.ndarray:
    '''
    Upscale the image by a factor
    image : source image
    method : ['NNI','bilinear']
    k : factor in int
    '''
    if method=='NNI':
        return NNI(image,k)
    if method=='bilinear':
        return Bilinear(image,k)

#Nearest Neighbour Interpolation Algorithm
def NNI(image:np.ndarray,k:int)->np.ndarray:
    L,B = image.shape
    resimage = np.zeros(shape=(L*k,B*k))
    for l in range(k*L):
        for b in range(k*B):
            resimage[l,b]=image[l//k][b//k]
    return resimage.astype(int)

#Bilinear InterpolationAlgorithm
def Bilinear(image:np.ndarray,k:int)->np.ndarray:
    image = NNI(image,k)
    resimage = image.copy()
    L,B = image.shape
    for l in range(1,L-1):
        for b in range(1,B-1):
            A = np.array([[1,l-1,b-1,(l-1)*(b-1)],
                         [1,l+1,b-1,(l+1)*(b-1)],
                         [1,l-1,b+1,(l-1)*(b+1)],
                         [1,l+1,b+1,(l+1)*(b+1)]])
            I = np.array([[image[l-1,b-1]],[image[l+1,b-1]],[image[l-1,b+1]],[image[l+1,b+1]]])
            d = np.linalg.inv(A).dot(I)
            resimage[l,b]=d[0,0]+l*d[1][0]+b*d[2][0]+l*b*d[3][0]
    return np.round(resimage).astype(int)

def testImageUpsampling(method,files,k):
    for file in files:
        image = io.imread("inputs/"+file)
        try:
            L,B = image.shape
        except ValueError:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            L,B = image.shape
        image = image[:k*(L//k),:k*(B//k)]
        resimage = subsample(image,k)
        plt.subplot(121)
        plt.imshow(resimage,cmap=plt.cm.gray)
        plt.subplot(122)
        resimage1 = upsample(resimage,method,k)
        plt.imshow(resimage1,cmap=plt.cm.gray)
        error = sqrt(np.sum(np.square(np.subtract(image,resimage1))))/(L*B)
        plt.title(label=f'method = {method}, k = {k},MSE = {error:.5f}')
        plt.savefig("./outputs/"+"Q2/"+method+"_"+str(k)+"_"+file)
        plt.clf()

if __name__=="__main__":
    methods = ['NNI','bilinear']
    files = ['StoneFace.png','Bee.jpg']
    for m in methods:
        for k in [2,3]:
            testImageUpsampling(m,files,k)