#============================================================================================
# Gaussian Donut Median Filter Function
# Author: Gerald M
#
# This function peforms a gaussian donut weighted median filter process on an input image.
#============================================================================================

import numpy as np
import math
import scipy.stats
from im2col import im2col
from col2im import col2im

def gausskern(SHAPE,SIGMA):
    m,n = [(ss-1.)/2. for ss in SHAPE]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*SIGMA*SIGMA) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussmedfilt(A,RADIUS,SIGMA):
    N = (RADIUS*2)+1

    G = gausskern((N,N),SIGMA)
    G_norm = G/np.amax(G)
    g = gausskern((N,N),SIGMA/1.5)
    g_norm = g/np.amax(g)
    Gg = G_norm-g_norm
    Gg_norm = Gg/np.amax(Gg)

    Gg_col = Gg_norm.reshape((np.size(Gg_norm),1))

    A = np.lib.pad(A,((int(math.floor(N/2.)),int(math.floor(N/2.))),(int(math.floor(N/2.)),int(math.floor(N/2.)))),'constant')
    A_pad_shape = np.shape(A)
    A = im2col(A, (N, N))

    A = np.multiply(A,Gg_col)

    A = np.sort(A,0)

    A = A[int(math.floor(N*N/2)+1),:]

    A = col2im(A, (N, N), A_pad_shape[::-1])

    return A
