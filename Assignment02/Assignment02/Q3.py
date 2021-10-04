import numpy as np
from skimage import io,color,img_as_float32,img_as_ubyte
from matplotlib import pyplot as plt
from math import e,sqrt,radians,cos,sin
from scipy.signal import convolve2d
from scipy.optimize import minimize
import cv2
import random

import skimage

#Rotation function
def rotate(image:np.ndarray,d:float,inter:str='NNI')->np.ndarray:
    try:
        L,B = image.shape
    except ValueError:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        L,B = image.shape
    d_=radians(d)
    d = 180-d
    M = np.max(image.shape)
    X = lambda a,b : int(-1*a*cos(radians(d))+b*sin(radians(d)))
    Y = lambda a,b : int(-1*a*sin(radians(d))-b*cos(radians(d)))
    res = np.zeros(shape=(M+M//2,M+M//2))
    res1 = np.zeros(shape=(M+M//2,M+M//2))
    a,b = X(0,0),Y(0,0)
    for i in range(L):
        for j in range(B):
            res[X(i,j),Y(i,j)] = image[i,j]
            a=min(a,X(i,j))
            b=min(b,Y(i,j))
    for i in range(M+M//2):
        for j in range(M+M//2):
            res1[i,j]=res[i+a,j+b]
    L,B = image.shape
    res1=res1[:int(abs(L*cos(d_))+abs(B*sin(d_))),:int(abs(L*sin(d_))+abs(B*cos(d_)))]
    L,B = res1.shape
    if inter == 'NNI':
        res1=upsample(res1,method='NNI',k=1)
    elif inter == 'bilinear':
        res1=upsample(res1,method='bilinear',k=1)
    return res1.astype(int)

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

def testRoation(d,file,method):
    image = io.imread("inputs/"+file)
    resimage = rotate(image,d,method)
    plt.title(label=f'method = {method}, deg = {d}')
    plt.imshow(resimage,cmap=plt.cm.gray)
    plt.savefig("./outputs/Q3/Rotate_"+str(d)+"_"+file)
    plt.clf()

if __name__=="__main__":
    file = "Bee.jpg"
    deg = [30,75,-45]
    for d in deg:
        testRoation(d,file,'NNI')
