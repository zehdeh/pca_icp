from numpy import *
import numpy
from math import sqrt

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB
    print 'cov'
    print H

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    print t

    return R, t

# Test with random data

# Random rotation and translation
#R = mat(random.rand(3,3))
R = numpy.matrix('0 0 -1; 0 1 0; 1 0 0')
t = numpy.matrix('0; 0; 0')#mat(random.rand(3,1))

# make R a proper rotation matrix, force orthonormal
U, S, Vt = linalg.svd(R)
R = U*Vt

# remove reflection
if linalg.det(R) < 0:
   Vt[2,:] *= -1
   R = U*Vt

# number of points
n = 8

A = numpy.matrix('0 0 0; 2 0 0; 2 1 0; 0 1 0; 0 0 1; 2 0 1; 2 1 1; 0 1 1')#mat(random.rand(n,3));
B = R*A.T + tile(t, (1, n))
B = B.T;
#B = numpy.matrix('0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 2; 1 0 2; 1 1 2; 0 1 2')

# recover the transformation
ret_R, ret_t = rigid_transform_3D(A, B)

A2 = (ret_R*A.T) + tile(ret_t, (1, n))
A2 = A2.T

# Find the error
err = A2 - B

err = multiply(err, err)
err = sum(err)
rmse = sqrt(err/n);

print "Points A"
print A
print ""

print "Points A2"
print A2
print ""

print "Points B"
print B
print ""

print "Rotation"
print R
print ""

print "Translation"
print t
print ""

print "RMSE:", rmse
print "If RMSE is near zero, the function is correct!"
