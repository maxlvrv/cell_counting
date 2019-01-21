import numpy
import math
import scipy.stats
from im2col import im2col
from col2im import col2im

def medfilt(A,RADIUS):
    N = (RADIUS*2)+1
    X = Y = RADIUS+1

    [COL, ROW] = numpy.meshgrid(numpy.linspace(1,N,N),numpy.linspace(1,N,N))

    M = (ROW-Y)**2 + (COL-X)**2 <= (RADIUS**2)+1
    M_col = M.reshape((numpy.size(M),1))

    A_pad = numpy.lib.pad(A,((int(math.floor(N/2.)),int(math.floor(N/2.))),(int(math.floor(N/2.)),int(math.floor(N/2.)))),'constant')
    A_col = im2col(A_pad, (N, N))

    A_weighted = numpy.multiply(A_col,M_col)

    A_sort = numpy.sort(A_weighted,0)

    A_median = A_sort[int(math.floor(N*N/2)+1),:]

    A_im = col2im(A_median, (N, N), numpy.shape(A_pad))

    return A_im
