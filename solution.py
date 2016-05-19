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
    #H = numpy.matrix("""0.132861 0.265722 0.797168;
    #0.265722 0.531443 0.265722;
    #0.797166 -0.265722 -0.132861
    #""")
    print 'covariance:'
    print H

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    print 'Vt.T:'
    print Vt.T

    print 'U.T:'
    print U.T

    print 'R:'
    print R

    # special reflection case
    if linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T
    print 'R:'
    print R

    #print t

    return R, t

# Test with random data

# Random rotation and translation
#R = mat(random.rand(3,3))
R = numpy.matrix('1 0 0; 0 0 -1; 0 1 0')
t = numpy.matrix('0; 0; 0')#mat(random.rand(3,1))

# make R a proper rotation matrix, force orthonormal
U, S, Vt = linalg.svd(R)
R = U*Vt
print "R:"
print R

# remove reflection
if linalg.det(R) < 0:
   Vt[2,:] *= -1
   R = U*Vt

# number of points
n = 56

A = numpy.matrix("""5.26665 3.39849 3.87546;
5.26665 3.39849 4.69051;
4.4516 3.39849 4.69051;
4.4516 3.39849 3.87546;
5.26665 4.21354 3.87546;
5.26665 4.21354 4.69051;
4.4516 4.21354 4.69051;
4.4516 4.21354 3.87546;
4.50866 3.45554 4.69051;
4.50866 4.15648 4.69051;
5.2096 3.45554 4.69051;
5.2096 4.15648 4.69051;
5.26665 3.45554 3.93252;
5.26665 4.15648 3.93252;
5.26665 3.45554 4.63346;
5.26665 4.15648 4.63346;
4.50866 3.45554 3.87546;
4.50866 4.15648 3.87546;
5.2096 3.45554 3.87546;
5.2096 4.15648 3.87546;
4.4516 3.45554 3.93252;
4.4516 4.15648 3.93252;
4.4516 3.45554 4.63346;
4.4516 4.15648 4.63346;
5.2096 4.21354 3.93252;
4.50866 4.21354 3.93252;
4.50866 4.21354 4.63346;
5.2096 4.21354 4.63346;
4.50866 3.39849 4.63346;
4.50866 3.39849 3.93252;
5.2096 3.39849 3.93252;
5.2096 3.39849 4.63346;
4.48742 4.15648 3.93252;
4.48742 3.45554 3.93252;
4.48742 4.15648 4.63346;
4.48742 3.45554 4.63346;
4.50866 3.45554 4.65469;
4.50866 4.15648 4.65469;
5.2096 3.45554 4.65469;
5.2096 4.15648 4.65469;
5.23083 3.45554 3.93252;
5.23083 4.15648 3.93252;
5.23083 3.45554 4.63346;
5.23083 4.15648 4.63346;
5.2096 4.17771 3.93252;
4.50866 4.17771 3.93252;
4.50866 4.17771 4.63346;
5.2096 4.17771 4.63346;
4.50866 3.45554 3.91128;
4.50866 4.15648 3.91128;
5.2096 3.45554 3.91128;
5.2096 4.15648 3.91128;
4.50866 3.43431 4.63346;
4.50866 3.43431 3.93252;
5.2096 3.43431 3.93252;
5.2096 3.43431 4.6334
""")
B = R*A.T# + tile(t, (1, n))
B = B.T;
#B = numpy.matrix('0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 2; 1 0 2; 1 1 2; 0 1 2')
print "Points A"
print A
print ""

print "Points B"
print B
print ""

# recover the transformation
ret_R, ret_t = rigid_transform_3D(A, B)

A2 = (ret_R*A.T) + tile(ret_t, (1, n))
A2 = A2.T

# Find the error
err = A2 - B

err = multiply(err, err)
err = sum(err)
rmse = sqrt(err/n);


print "Rotation"
print R
print ""

#print "Translation"
#print t
#print ""
#
#print "RMSE:", rmse
#print "If RMSE is near zero, the function is correct!"
