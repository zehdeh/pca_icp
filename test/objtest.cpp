#include "objtest.h"

#include <iostream>
#include <vector>
#include <cstring>

#include "util.h"
#include "objloader.h"
#include "svd.h"

#include "cuda.h"
#include "cuda_runtime.h"

int objTest() {
	std::vector<Point> vertices1;
	const unsigned int numDimensions = 3;
	unsigned int numElements1 = loadObj("res/muscleman/obj/Kneel.000001.obj", &vertices1);

	std::vector<Point> vertices2;
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
	rotateMatrix(numElements1, *pointList2, testRotation);

	/*
	std::cout << "First:" << std::endl;
	printMatrix(numElements1, numDimensions, *pointList1);
	std::cout << std::endl;
	std::cout << "Second:" << std::endl;
	printMatrix(numElements1, numDimensions, *pointList2);
	*/
	
	cudaEvent_t start;
	cudaEvent_t stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float distance1[3];
	cudaEventRecord(start);
	findOriginDistance(numElements1, *pointList1, distance1);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Computing centroid took " << milliseconds << " ms" << std::endl;

	float distance2[3];
	cudaEventRecord(start);
	findOriginDistance(numElements1, *pointList2, distance2);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Computing centroid took " << milliseconds << " ms" << std::endl;

	cudaEventRecord(start);
	translate(numElements1, *pointList1, distance1);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Translating took " << milliseconds << " ms" << std::endl;

	cudaEventRecord(start);
	translate(numElements1, *pointList2, distance2);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Translating took " << milliseconds << " ms" << std::endl;

	std::cout << "Centroids:" << std::endl;
	std::cout << distance1[0] << " " << distance1[1] << " " << distance1[2] << std::endl;
	std::cout << distance2[0] << " " << distance2[1] << " " << distance2[2] << std::endl;

	float covariance[numDimensions * numDimensions];
	memset(covariance, 0, sizeof(float)*numDimensions*numDimensions);


	cudaEventRecord(start);
	findCovariance(numElements1, numDimensions, *pointList1, *pointList2, covariance);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Computing covariance took " << milliseconds << " ms" << std::endl;

	std::cout << "Covariance:" << std::endl;
	printMatrix(numDimensions, numDimensions, covariance);

	rotationMatrix rotation;
	svdMethod(numDimensions, covariance, rotation);

	for(unsigned int i = 0; i < numDimensions*numDimensions; i++) {
		rotation[i] = (rotation[i] < 0.0001 && rotation[i] > -0.0001)?0:rotation[i];
	}

	std::cout << "Rotation:" << std::endl;
	printMatrix(numDimensions, numDimensions, rotation);
	std::cout << "Num Elements: " << numElements1 << std::endl;

	//std::cout << "RMSE:" << std::endl;
	//std::cout << matrixRMSE(numElements1, numDimensions, *pointList1, *pointList2) << std::endl;
	return 0;
}
