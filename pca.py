from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import numpy
import sys

class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
		self._verts3d = xs, ys, zs
	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
		FancyArrowPatch.draw(self, renderer)

def show_points(A, B, fig,ax):
	x,y,z = A.T
	ax.scatter(x.tolist(), y.tolist(), z.tolist(),marker='o')
	x2,y2,z2 = B.T
	ax.scatter(x2.tolist(), y2.tolist(), z2.tolist(),marker='^')
	plt.show()

R = numpy.matrix('0.52 0 0.85; 0 1 0; -0.85 0 0.52')
A = numpy.matrix('0.0 0.0 0.0; 2.0 0.0 0.0; 2.0 1.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 2.0 0.0 1.0; 2.0 1.0 1.0; 0.0 1.0 1.0')
B = R*A.T
B = B.T

N = A.shape[0];

C = numpy.cov(A)

mean_vectorA = mean(A, axis=0)
mean_vectorB = mean(B, axis=0)

AA = A - tile(mean_vectorA, (N, 1))
BB = B - tile(mean_vectorB, (N, 1))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

cov1 = cov(AA, rowvar=0)
cov2 = cov(BB, rowvar=0)
print 'Covariance 1:'
print cov1
print 'Covariance 2:'
print cov2

eig_val_cov1,eig_vec_cov1 = linalg.eig(cov1)
eig_val_cov2,eig_vec_cov2 = linalg.eig(cov2)

for v in eig_vec_cov1.T:
	un = mean_vectorA + v
	a = Arrow3D([mean_vectorA.item(0), un.item(0)], [mean_vectorA.item(1), un.item(1)], [mean_vectorA.item(2), un.item(2)], mutation_scale=20,lw=3)
	ax.add_artist(a)

for v in eig_vec_cov2.T:
	un = mean_vectorB + v
	a = Arrow3D([mean_vectorB.item(0), un.item(0)], [mean_vectorB.item(1), un.item(1)], [mean_vectorB.item(2), un.item(2)], mutation_scale=20,lw=3)
	ax.add_artist(a)

show_points(A,B,fig,ax)
