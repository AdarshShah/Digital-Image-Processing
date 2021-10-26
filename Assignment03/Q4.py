import numpy as np
from scipy.optimize import fmin

if __name__=='__main__':
    g = np.zeros((5,5))

    H = np.fromfunction(lambda x,y: (x-2)**2+(y-2)**2,(5,5))

    #IDFT
    h = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(H)))

    #Matrix in question a
    g[0][2] = g[2][0] = g[4][2] = g[2][4] = 1
    g[1][1] = g[3][1] = g[3][3] = g[1][3] = 1

    g[1][2] = g[2][1] = g[3][2] = g[2][3] = 2
    g[2][2] = -16

    #Objective function to minimize
    def obj(K):
        return np.sum(np.square(np.absolute(g-K[0]*h)))

    #minimization and displaying result
    print(fmin(func=obj,x0=[10]))
    xopt=fmin(func=obj,x0=[10])
    print(np.round(xopt*np.real(h),3))

    #Matrix in question b
    g[0][1] = g[0][3] = g[1][0] = g[1][4] = 1
    g[3][0] = g[3][4] = g[4][1] = g[4][3] = 1
    g[2][2] = -24

    #minimization and displaying result
    print(fmin(func=obj,x0=[10]))
    xopt=fmin(func=obj,x0=[10])
    print(np.round(xopt*np.real(h),3))





