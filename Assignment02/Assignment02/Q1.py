import numpy as np
from skimage import io,color,img_as_float32,img_as_ubyte
from matplotlib import pyplot as plt
from math import e,sqrt,radians,cos,sin
from scipy.signal import convolve2d
from scipy.optimize import minimize
import cv2
import random

def contrastEnhance(image:np.ndarray,method:str='FSCS',clip_limit:int=400)->np.ndarray:
    '''
    image : source image
    method : ['FSCS','NLCS','HE','CLAHE','CLAHE1']
    cliplimit : (optional) for CLAHE
    '''
    if method == 'FSCS':
        return FSCS(image)
    elif method == 'NLCS':
        return NLCS(image)
    elif method == 'HE':
        return HE(image)
    elif method == 'CLAHE':
        return CLAHE(image,clip_limit=clip_limit)
    elif method == 'CLAHE1':
        return CLAHE1(image,clip_limit=clip_limit)

#Full Scale Contrast Stretch
def FSCS(image:np.ndarray)->np.ndarray:
    image = img_as_float32(image)
    mn = np.min(image)
    mx = np.max(image)
    image = np.divide(np.subtract(image,mn),mx-mn)
    image = img_as_ubyte(image)
    return image

#Non Linear Contrast Stretch
def NLCS(image:np.ndarray)->np.ndarray:
    image = img_as_float32(image)
    image = np.log2(np.add(1,image))
    mn = np.min(image)
    mx = np.max(image)
    image = np.divide(np.subtract(image,mn),mx-mn)
    image = img_as_ubyte(image)
    image = FSCS(image)
    return image

#Histogram Equalization
def HE(image:np.ndarray)->np.ndarray:
    x,P = histogram(image,bins=256)
    sum = np.sum(P)
    for i in range(256):
        P[i]=P[i]/sum
    for i in range(1,256):
        P[i]+=P[i-1]
    func = lambda x : P[x]*255
    func = np.vectorize(func)
    image = np.round(func(image)).astype(int)
    image = FSCS(image)
    return image

#CLAHE
def CLAHE(image:np.ndarray,clip_limit:int=400)->np.ndarray:
    l,b = np.shape(image)
    l,b = l//8,b//8
    resimage = np.ones(shape=np.shape(image))
    for i in range(0,8):
        for j in range(0,8):
            img = image[i*l:(i+1)*l,j*b:(j+1)*b]
            _,P = histogram(img,bins=256)
            P = P.astype(int)
            clipped = 0
            for h in range(256):
                if P[h] > clip_limit:
                    clipped = clipped + P[h] - clip_limit
                    P[h] = clip_limit
            P = np.round(np.add(P,clipped/255))
            sum = np.sum(P)
            for k in range(256):
                P[k]=P[k]/sum
            for k in range(1,256):
                P[k]+=P[k-1]
            P = np.round(np.multiply(P,255))
            func = lambda x : P[x]
            func = np.vectorize(func)
            resimage[i*l:i*l+l,j*b:(j+1)*b] = func(img).astype(int)
    return resimage.astype(int)

#CLAHE with 25% more boundaries
def CLAHE1(image:np.ndarray,clip_limit:np.ndarray=400)->np.ndarray:
    l,b = np.shape(image)
    l,b = l//8,b//8
    resimage = np.ones(shape=np.shape(image))
    for i in range(0,8):
        for j in range(0,8):
            br,ur,bc,uc = i*l,i*l+l,j*b,j*b+b
            if i>0:
                br-=0.25*l
            if i<7:
                ur+=0.25*l
            if j>0:
                bc-=0.25*b
            if j<7:
                uc+=0.25*b
            br,ur,bc,uc = int(br),int(ur),int(bc),int(uc)
            img = image[br:ur,bc:uc]
            _,P = histogram(img,bins=256)
            P = P.astype(int)
            clipped = 0
            for h in range(256):
                if P[h] > clip_limit:
                    clipped = clipped + P[h] - clip_limit
                    P[h] = clip_limit
            P = np.round(np.add(P,clipped/255))
            sum = np.sum(P)
            for k in range(256):
                P[k]=P[k]/sum
            for k in range(1,256):
                P[k]+=P[k-1]
            P = np.round(np.multiply(P,255))
            func = lambda x : P[x]
            func = np.vectorize(func)
            resimage[i*l:i*l+l,j*b:(j+1)*b] = func(image[i*l:i*l+l,j*b:(j+1)*b]).astype(int)
    return resimage.astype(int)

#Histogram of image
def histogram(image,bins):
  hist = np.zeros(256)
  for i in image.ravel():
    hist[i]+=1
  width = 256/bins
  bc = [0]
  for i in range(0,bins):
    bc.append(bc[-1]+width)
  res = np.zeros(len(bc)-1)
  j=0
  for i in range(0,256):
    if i <= bc[j+1]:
      res[j] += hist[i]
    else:
      j+=1
      res[j] += hist[i]
  bcenter = []
  for i in range(0,bins):
    bcenter.append((bc[i]+bc[i+1])/2)
  bcenter = np.array(bcenter)
  return bcenter, res


#Test Function tests a particular contrast enhancement function on all files
def testContrastEnhancement(method:str,files):
    for file in files:
        image = io.imread("inputs/"+file)
        plt.subplot(221)
        plt.imshow(image,cmap=plt.cm.gray)
        x,y=histogram(image,bins=256)
        plt.subplot(222)
        plt.bar(x,y)
        resimg = contrastEnhance(image,method,clip_limit=1000)
        plt.subplot(223)
        plt.imshow(resimg,cmap=plt.cm.gray)
        x,y=histogram(resimg,bins=256)
        plt.subplot(224)
        plt.bar(x,y)
        plt.savefig("./outputs/"+"Q1/"+method+"_"+file)
        plt.clf()

'''
Tests All Methods on All Images and saves the result in output/Q1
'''
if __name__=='__main__':
    methods = ['FSCS','NLCS','HE','CLAHE','CLAHE1']
    files = ['IIScMainBuilding_LowContrast.png','LowLight_2.png','LowLight_3.png','Hazy.png','StoneFace.png']
    for m in methods:
        testContrastEnhancement(m,files)
    