#include "cuda_objtest.h"

#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>

#include "util.h"
#include "svd.h"
#include "cuda_covariance.h"
#include "objloader.h"

int cuda_objTest() {
	std::vector<vec3> vertices1;
	const unsigned int numDimensions = 3;
	unsigned int numElements1 = loadObj("res/muscleman/obj/Kneel.000001.obj", &vertices1);
	//numElements1 = 5000;
	unsigned int numElements2 = numElements1;
	const unsigned int maxNumElements = std::max(numElements1, numElements2);

	std::vector<vec3> vertices2;
	loadObj("res/muscleman/obj/Kneel.000001.obj", &vertices2);

	//static_assert(numElements1 == numElements2, "The number of points do not match!");
	float* pointList1[numElements1*numDimensions];
	float* pointList2[numElements1*numDimensions];
	for(unsigned int i = 0; i < numElements1; i++) {
		pointList1[i*numDimensions] = &vertices1[0].x;
		pointList1[i*numDimensions + 1] = &vertices1[0].y;
		pointList1[i*numDimensions + 2] = &vertices1[0].z;

		pointList2[i*numDimensions] = &vertices2[0].x;
		pointList2[i*numDimensions + 1] = &vertices2[0].y;
		pointList2[i*numDimensions + 2] = &vertices2[0].z;
	}
	
	float testRotation[9] = {1,0,0,0,0,-1,0,1,0};
	rotateMatrix(numElements1, numDimensions, *pointList2, testRotation);
	/*
	const unsigned int numElements1 = 8;
	const unsigned int numElements2 = 8;
	const unsigned int maxNumElements = std::max(numElements1, numElements2);

	const unsigned int numDimensions = 3;
	float* pointList1 = new float[numElements1*numDimensions];
	float* pointList2 = new float[numElements2*numDimensions];
// 2x1x1 cube
	pointList1[0 + 0] = 0; pointList1[0 + 1] = 0; pointList1[0 + 2] = 0;
	pointList1[3 + 0] = 2; pointList1[3 + 1] = 0; pointList1[3 + 2] = 0;
	pointList1[6 + 0] = 2; pointList1[6 + 1] = 1; pointList1[6 + 2] = 0;
	pointList1[9 + 0] = 0; pointList1[9 + 1] = 1; pointList1[9 + 2] = 0;
	pointList1[12 + 0] = 0; pointList1[12 + 1] = 0; pointList1[12 + 2] = 1;
	pointList1[15 + 0] = 2; pointList1[15 + 1] = 0; pointList1[15 + 2] = 1;
	pointList1[18 + 0] = 2; pointList1[18 + 1] = 1; pointList1[18 + 2] = 1;
	pointList1[21 + 0] = 0; pointList1[21 + 1] = 1; pointList1[21 + 2] = 1;

	// 1x1x2 cube
	pointList2[0 + 0] = 0; pointList2[0 + 1] = 0; pointList2[0 + 2] = 0;
	pointList2[3 + 0] = 0; pointList2[3 + 1] = 0; pointList2[3 + 2] = 2;
	pointList2[6 + 0] = 0; pointList2[6 + 1] = 1; pointList2[6 + 2] = 2;
	pointList2[9 + 0] = 0; pointList2[9 + 1] = 1; pointList2[9 + 2] = 0;
	pointList2[12 + 0] = -1; pointList2[12 + 1] = 0; pointList2[12 + 2] = 0;
	pointList2[15 + 0] = -1; pointList2[15 + 1] = 0; pointList2[15 + 2] = 2;
	pointList2[18 + 0] = -1; pointList2[18 + 1] = 1; pointList2[18 + 2] = 2;
	pointList2[21 + 0] = -1; pointList2[21 + 1] = 1; pointList2[21 + 2] = 0;
	*/

	// CUDA specific starting from here

	float** d_pointList1 = getDevicePointList1();
	float** d_pointList2 = getDevicePointList2();
	//float* d_pointList1;
	//float* d_pointList2;
	float centroid1[3] = {0,0,0};
	float centroid2[3] = {0,0,0};

	std::cout << "CUDA:" << std::endl;
	cuda_initPointLists(numElements1, numDimensions, *pointList1, *pointList2);

	cuda_downloadPointList(numElements1, numDimensions, *pointList1, *d_pointList1);
	cuda_downloadPointList(numElements2, numDimensions, *pointList2, *d_pointList2);

	/*
	std::cout << "CUDA (before translation):" << std::endl;
	std::cout << "First:" << std::endl;
	printMatrix(numElements1, numDimensions, *pointList1);
	std::cout << "Second:" << std::endl;
	printMatrix(numElements2, numDimensions, *pointList2);
	*/

	cuda_findOriginDistance(numElements1, numDimensions, *d_pointList1, centroid1);
	cuda_findOriginDistance(numElements2, numDimensions, *d_pointList2, centroid2);

	std::cout << "Centroids:" << std::endl;
	std::cout << centroid1[0] << " " << centroid1[1] << " " << centroid1[2] << std::endl;
	std::cout << centroid2[0] << " " << centroid2[1] << " " << centroid2[2] << std::endl;

	cuda_translate(numElements1, numDimensions, *d_pointList1, centroid1);
	cuda_translate(numElements2, numDimensions, *d_pointList2, centroid2);

	cuda_downloadPointList(numElements1, numDimensions, *pointList1, *d_pointList1);
	cuda_downloadPointList(numElements2, numDimensions, *pointList2, *d_pointList2);

/*
	std::cout << "CUDA (after translation):" << std::endl;
	std::cout << "First:" << std::endl;
	printMatrix(numElements1, numDimensions, *pointList1);
	std::cout << "Second:" << std::endl;
	printMatrix(numElements2, numDimensions, *pointList2);
	*/

	/*
	float pointList1Transposed[numElements1*numDimensions];
	memset(pointList1Transposed, 0, sizeof(float)*numElements1*numDimensions);

	float* d_pointList1Transposed;
	gpuErrchk(cudaMalloc(&d_pointList1Transposed, bytes1));
	gpuErrchk(cudaMemset(d_pointList1Transposed, 0, bytes1));

	kernel_transpose<<<gridSize, blockSize>>>(numElements1, numDimensions, d_pointList1, d_pointList1Transposed);

	cudaMemcpy(pointList1Transposed, d_pointList1Transposed, bytes1, cudaMemcpyDeviceToHost);

	std::cout << "First transposed:" << std::endl;
	printMatrix(numDimensions, numElements1, pointList1Transposed);
	
	cudaFree(d_pointList1Transposed);
	*/
	float covariance[numDimensions*numDimensions];
	memset(covariance, 0, sizeof(float)*numDimensions*numDimensions);
	cuda_findCovariance(maxNumElements, numDimensions, *d_pointList1, *d_pointList2, covariance);

	std::cout << "Covariance: " << std::endl;
	printMatrix(numDimensions, numDimensions, covariance);

	rotationMatrix rotation;
	svdMethod(numDimensions, covariance, rotation);

	for(unsigned int i = 0; i < numDimensions*numDimensions; i++) {
		rotation[i] = (rotation[i] < 0.0001 && rotation[i] > -0.0001)?0:rotation[i];
	}

	std::cout << "Rotation: " << std::endl;
	printMatrix(numDimensions, numDimensions, rotation);
	std::cout << "Num Elements: " << numElements1 << std::endl;
	
	cuda_destroyPointList(*d_pointList1);
	cuda_destroyPointList(*d_pointList2);

	return 0;
}
