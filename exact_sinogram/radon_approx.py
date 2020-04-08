import numpy as np
import numpy.matlib as npmat
import math
from scipy.interpolate import interp2d
from scipy.integrate import simps
from skimage.transform import rotate

def imrotate_new(img, theta):
    n    = img.shape[0]
    th   = np.deg2rad(theta)
    x_1n = np.arange(n)
    y_1n = x_1n.copy()
    mid  = (n+1)/2
    X, Y = np.meshgrid(x_1n, y_1n)
    X = X - mid
    Y = Y - mid
    xi = mid + X*np.cos(th) - Y*np.sin(th)
    yi = mid + X*np.sin(th) + Y*np.cos(th)
    
    f  = interp2d(X,Y,img)

    Ra = f(xi,yi)
    Ra[np.where(Ra == np.isnan)[0]] = 0.0
    
    return Ra
    
def my_radon(M, theta):
    '''
    M = image
    thetaval = list of angles
    
    TO DO:
    volendo si puo' ruotare la griglia per non usare imrotate
    '''
    m,n = M.shape
    theta_len = len(theta)
    R = np.zeros((m,theta_len))

    x, y = np.meshgrid(np.linspace(0,1,m), np.linspace(0,1,n))
    
    
    for i in range(theta_len):
        R_theta = rotate(M, 90 - theta[i])
        #R[ :, i] = np.sum(R_theta, axis = 1)
        R[ :,i] = simps( y = R_theta, x = x, axis = 1) 

    return R[::-1,:]
