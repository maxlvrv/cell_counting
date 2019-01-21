def col2im(A, BLKSZ, SIZE):
    return A.reshape(SIZE[1]-BLKSZ[1]+1,SIZE[0]-BLKSZ[0]+1)
      # Or simply B.reshape(nn-n+1,-1).T
      # Or B.reshape(mm-m+1,nn-n+1,order='F')
