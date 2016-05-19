#include "objtest.h"

#include <iostream>
#include <vector>
#include <cstring>

#include "util.h"
#include "objloader.h"
#include "svd.h"

int objTest() {
	std::vector<vec3> vertices1;
	const unsigned int numDimensions = 3;
	const unsigned int numElements1 = loadObj("res/untitled.obj", &vertices1);

	std::vector<vec3> vertices2;
	loadObj("res/untitled.obj", &vertices2);

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

	float testRotation[9] = {0,0,-1,0,1,0,1,0,0};
	rotateMatrix(numElements1, numDimensions, *pointList2, testRotation);

	std::cout << "First:" << std::endl;
	printMatrix(numElements1, numDimensions, *pointList1);
	std::cout << std::endl;
	std::cout << "Second:" << std::endl;
	printMatrix(numElements1, numDimensions, *pointList2);

	float covariance[numDimensions * numDimensions];
	memset(covariance, 0, sizeof(float)*numDimensions*numDimensions);
	findCovariance(numElements1, numDimensions, *pointList1, *pointList2, covariance);

	std::cout << "Covariance:" << std::endl;
	printMatrix(numDimensions, numDimensions, covariance);

	rotationMatrix rotation;
	svdMethod(numDimensions, covariance, rotation);

	std::cout << "Rotation:" << std::endl;
	printMatrix(numDimensions, numDimensions, rotation);

	//std::cout << "RMSE:" << std::endl;
	//std::cout << matrixRMSE(numElements1, numDimensions, *pointList1, *pointList2) << std::endl;
	return 0;
}
