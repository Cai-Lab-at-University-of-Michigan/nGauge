import numpy as np
from cython.parallel import prange

cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
def mask_array( unsigned char[:,:,:] out, long[:] shape, double[:,:] hull_eqs, long num_hull_eqs, unsigned char val ):    
    cdef int i_size = shape[0] #this fixes python -> cython type conversion
    cdef int j_size = shape[1]
    cdef int k_size = shape[2]
    
    cdef int i = 0
    cdef int cnt = 0
    
    cdef int j = 0 #prob broken
    cdef int k = 0
    
    #for i in range( shape[0] ):
    for i in prange(i_size, nogil=True):
        for j in range( shape[1] ):
            for k in range( shape[2] ):
                if point_in_hull( i, j, k, hull_eqs, num_hull_eqs ):
                    out[ i, j, k ] = val
                    cnt += 1
                #else: #uncomment for background clearing
                #    out[ i, j, k ] = 0
    #print( "Integrated:", cnt ) 
    return cnt

# hull.equations -> hull_eqs
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef int point_in_hull( int i, int j, int k, double [:,:] hull_eqs, long num_hull_eqs ) nogil:
    cdef double tolerance = 1e-6
    cdef double tmp = 0
    
    for n in range( num_hull_eqs ):
        tmp = i*hull_eqs[n][0]
        tmp += j*hull_eqs[n][1]
        tmp += k*hull_eqs[n][2]
        tmp += hull_eqs[n][3]
        if tmp > tolerance:
            return 0
    return 1
    