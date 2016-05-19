#include "cubestest.h"

#include <cstring>
#include <iostream>
#include "util.h"
#include "svd.h"

int cubesTest() {
	const unsigned int numElements = 8;
	const unsigned int numDimensions = 3;
	float* pointList1 = new float[numElements*numDimensions];
	float* pointList2 = new float[numElements*numDimensions];
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

	std::cout << "BEFORE" << std::endl;
	std::cout << "First:" << std::endl;
	printMatrix(numElements, numDimensions, pointList1);
	std::cout << std::endl;
	std::cout << "Second:" << std::endl;
	printMatrix(numElements, numDimensions, pointList2);

	float distance1[3];
	findOriginDistance(numElements, numDimensions, pointList1, distance1);
	translate(numElements, numDimensions, pointList1, distance1);
	std::cout << "Centroids:" << std::endl;
	std::cout << distance1[0] << " " << distance1[1] << " " << distance1[2] << std::endl;

	float distance2[3];
	findOriginDistance(numElements, numDimensions, pointList2, distance2);
	translate(numElements, numDimensions, pointList2, distance2);

	std::cout << distance2[0] << " " << distance2[1] << " " << distance2[2] << std::endl;

	std::cout << std::endl;
	std::cout << "AFTER CENTERING" << std::endl;
	std::cout << "First:" << std::endl;
	printMatrix(numElements, numDimensions, pointList1);
	std::cout << std::endl;
	std::cout << "Second:" << std::endl;
	printMatrix(numElements, numDimensions, pointList2);
	
	float covariance[numDimensions * numDimensions];
	memset(covariance, 0, sizeof(float)*numDimensions*numDimensions);
	findCovariance(numElements, numDimensions, pointList1, pointList2, covariance);

	std::cout << "Covariance:" << std::endl;
	printMatrix(numDimensions, numDimensions, covariance);

	rotationMatrix rotation;
	svdMethod(numDimensions, covariance, rotation);

	std::cout << std::endl;
	std::cout << "Rotation matrix:" << std::endl;
	printMatrix(3, 3, rotation);

	rotateMatrix(numElements, numDimensions, pointList1, rotation);

	std::cout << std::endl;
	std::cout << "AFTER TURNING" << std::endl;
	std::cout << "First:" << std::endl;
	printMatrix(numElements, numDimensions, pointList1);
	std::cout << std::endl;
	std::cout << "Second:" << std::endl;
	printMatrix(numElements, numDimensions, pointList2);

	return 0;
}
