import numpy as np
from skimage import io, filters, color
import cv2
import matplotlib.pyplot as plt
import timeit
from collections import deque, Counter

#input directory
dir = './Inputs/'

#output directory
outdir = './Outputs/'

# 1. Histogram

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

# Testing Histogram

def testHist():
  image = io.imread(dir+'Coins.png')
  bins = 256
  x,y = histogram(image,bins)
  plt.bar(x=x,height=y,width=256/bins)
  plt.savefig(outdir+'1_histogram.png')
  x = plt.hist(x=image.ravel(),bins=256,range=(0,256))

# 2. Otsu's Binarization

def otsu(image):
  t1 = timeit.default_timer()
  bc,hist = histogram(image,256)
  h,w = np.shape(image)
  p = np.divide(hist,h*w)
  mx = 0
  threshold = -1
  for t in range(0,255):
    w0 = np.sum(p[:t])
    w1 = np.sum(p[t:])
    u0,u1 = 0,0
    for i in range(0,t):
      u0 += i*p[i]/w0
    for i in range(t,256):
      u1 += i*p[i]/w1
    mx_ = w0*w1*(u0-u1)**2
    if mx_ > mx:
      threshold,mx = t,mx_
  t2 = timeit.default_timer()
  avg1 = (t2-t1)
  binimage = np.multiply(np.add(np.divide(np.sign(np.subtract(image,threshold+0.1)),2),0.5),255)
  t1=timeit.default_timer()
  filters.threshold_otsu(image,256)
  t2=timeit.default_timer()
  avg2=t2-t1
  return threshold, binimage, avg1, avg2

# Testing Otsu
def testOtsu():
  image = io.imread(dir+'Coins.png')
  threshold, binimage, t1, t2 = otsu(image)
  io.imshow(binimage,cmap=plt.cm.gray)
  io.imsave(fname=outdir+'2_otsu.png',arr=binimage)
  print(t1,t2,threshold)
  print(filters.threshold_otsu(image,256))

# 3. Foreground Extraction

def foreground(image,background):
  th,image,avg1,avg2 = otsu(image)
  image = np.divide(image,255)
  n_img = np.subtract(1,image)
  background[:,:,0]=np.multiply(background[:,:,0],n_img)
  background[:,:,1]=np.multiply(background[:,:,1],n_img)
  background[:,:,2]=np.multiply(background[:,:,2],n_img)
  background[:,:,0]=np.add(np.multiply(image,255),background[:,:,0])
  return background

# Testing

def testFE():
  image = io.imread(dir+'SingleColorText_Gray.png')
  background = io.imread(dir+'GrassBackground.png')
  r_img = foreground(image,background)
  io.imshow(r_img)
  io.imsave(fname=outdir+'3_foreground_extraction.png',arr=r_img)

# 4. Connected Components

def connected_components(image):
  th,image,t1,t2 = otsu(image)
  h,w = np.shape(image)
  visited = np.zeros(shape=np.shape(image))
  component = np.zeros(shape=np.shape(image))
  q = deque()
  k=0
  comp = []
  for i in range(0,h):
    for j in range(0,w):
      if visited[i][j] == 0 and image[i][j]>=th:
        k+=1
        s=0
        q.append((i,j))
        while len(q) > 0:
          cx,cy = q.popleft()
          visited[cx][cy]=1
          component[cx][cy]=k
          s+=1
          if cx-1>=0:
            if image[cx-1][cy] == image[cx][cy]:
              if visited[cx-1][cy] == 0:
                visited[cx-1][cy]=1
                q.append((cx-1,cy))
          if cy-1>=0:
            if image[cx][cy-1] == image[cx][cy]:
              if visited[cx][cy-1] == 0:
                visited[cx][cy-1]=1
                q.append((cx,cy-1))
          if cx+1<h :
            if image[cx+1][cy] == image[cx][cy] :
              if visited[cx+1][cy] == 0:
                visited[cx+1][cy]=1
                q.append((cx+1,cy))
          if cy+1<w :
            if image[cx][cy+1] == image[cx][cy]:
              if visited[cx][cy+1] == 0:
                visited[cx][cy+1]=1
                q.append((cx,cy+1))
        comp.append(s)
  ones=0
  for i in comp:
    if i == 3797:
      ones+=1
  return len(comp), ones

# Testing Connected Components

def testCC():
  image = io.imread(dir+'PiNumbers.png')
  print(connected_components(image))

# 5. Binary Morphology

def binaryMorphology(image):
  '''
  Used Squared Window and Majority function to de noise
  '''
  h,w = np.shape(image)
  new_img = np.zeros(shape=np.shape(image))
  d=5
  for x in range(d,h-d):
    for y in range(d,w-d):
      c = np.bincount(image[x-d:x+d+1,y-d:y+d+1].ravel())
      new_img[x][y] = np.argmax(c)
  return new_img.astype(int)

# Testing Binary Morphology

def testBM():
  image = io.imread(dir+'NoisyImage.png')
  new_img = binaryMorphology(image)
  io.imsave(fname=outdir+'5_binary_morphology.png',arr=new_img)

# 6. Maximally Stable External Regions

def set_0(image,x,y,delta):
  binimg = np.ones(shape=np.shape(image))
  h,w = np.shape(image)
  visited = np.zeros(shape=(h,w),dtype='int8')   
  Q = deque()
  Q.append((x,y))
  th = image[x][y]
  while len(Q) > 0:
    a,b = Q.popleft()
    binimg[a][b]=0
    visited[a][b]=1
    for x in [-1,1]:
      if a+x>=0 and a+x < h and image[a+x][b] >= th-delta and image[a+x][b] <= th+delta and visited[a+x][b] == 0:
        visited[a+x][b] = 1
        Q.append((a+x,b))
    for x in [-1,1]:
      if b+x>=0 and b+x < w and image[a][b+x] >= th-delta and image[a][b+x] <= th+delta and visited[a][b+x] == 0:
        visited[a][b+x] = 1
        Q.append((a,b+x))
  return binimg

def mser(image):
  h,w = np.shape(image)
  visited = np.zeros(shape=np.shape(image))
  delta = 3
  comps = []
  for i in range(1,h-1):
    for j in range(1,w-1):
      if visited[i][j]==0:
        Q = deque()
        Q.append((i,j))
        th = image[i][j]
        comp = 0
        while len(Q) > 0:
          a,b = Q.popleft()
          comp+=1
          visited[a][b]=1
          for x in [-1,1]:
            if a+x>=0 and a+x < h  and image[a+x][b] >= th-delta and image[a+x][b] <= th+delta and visited[a+x][b] == 0:
              visited[a+x][b] = 1
              Q.append((a+x,b))
          for x in [-1,1]:
            if b+x>=0 and b+x < w  and image[a][b+x] >= th-delta and image[a][b+x] <= th+delta and visited[a][b+x] == 0:
              visited[a][b+x] = 1
              Q.append((a,b+x))
        if comp <= 20000 and comp >= 2000:
          comps.append((i,j))
  mserimg = np.ones(shape=np.shape(image))
  mser_comps = len(comps)
  for comp in comps:
    x,y = comp
    mserimg = np.multiply(set_0(image,x,y,delta),mserimg)
  mserimg = np.multiply(mserimg,255).astype('uint8')
  otsu_th, otsu_binimg, t1, t2 = otsu(image)
  return mserimg,otsu_binimg,mser_comps

def testmser():
  image = io.imread(dir+'DoubleColorText_Gray.png')
  mser_img,otsu_img,c = mser(image)
  print(c)
  io.imshow(mser_img,cmap=plt.cm.gray)
  io.imsave(outdir+'6_mserimg.png',mser_img)
  io.imsave(outdir+'6_otsuimg.png',otsu_img)

if __name__=='__main__':
  testHist()
  print('<---------------------------------->')
  testOtsu()
  print('<---------------------------------->')
  testFE()
  print('<---------------------------------->')
  testCC()
  print('<---------------------------------->')
  testBM()
  print('<---------------------------------->')
  testmser()
