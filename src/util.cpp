#include "util.h"

#include <iostream>
#include <cstring>

void rotateMatrix(const unsigned int numElements, const unsigned int numDimensions, float* const pointList, const float* const rotation) {
	float temp[numElements * numDimensions];
	memset(temp, 0, sizeof(float)*numDimensions*numElements);
	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			for(unsigned int k = 0; k < numDimensions; k++) {
				temp[i*numDimensions + j] += pointList[i*numDimensions + k]*rotation[j*numDimensions + k];
			}
		}
	}

	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			pointList[i*numDimensions + j] = temp[i*numDimensions + j];
		}
	}
}

void printMatrix(const unsigned int m, const unsigned int n, const float* const covariance) {
	std::cout << "[";
	for(unsigned int i = 0; i < m; i++) {
		std::cout << "[";
		for(unsigned int j = 0; j < n; j++) {
			std::cout << covariance[i*n + j];

			if(j < n - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]";

		if(i < m - 1) {
			std::cout << "," << std::endl;
		}
	}
	std::cout << "]" << std::endl;
}

float matrixRMSE(const unsigned int m, const unsigned int n, const float* matrix1, const float* matrix2) {
	float diff[m*n];
	memset(diff, 0, sizeof(float)*m*n);

	for(unsigned int i = 0; i < m; i++) {
		for(unsigned int j = 0; j < n; j++) {
			diff[i*n + j] = matrix2[i*n + j]*matrix2[i*n + j] - matrix1[i*n + j]*matrix1[i*n + j];
		}
	}

	long sum = 0;
	for(unsigned int i = 0; i < m*n; i++) {
		sum += diff[i];
		std::cout << diff[i] << std::endl;
	}

	return sum;

}

void findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const distance) {
	memset(distance, 0, sizeof(float)*numDimensions);
	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			distance[j] += pointList[i*numDimensions + j] / numElements;
		}
	}
}

void translate(const unsigned int numElements, const unsigned int numDimensions, float* const pointList, float* t) {
	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			pointList[i*numDimensions + j] -= t[j];
		}
	}
}

void transpose(const unsigned int numElements, const unsigned int numDimensions, float* const pointList, float* const pointListTransposed) {
	for(unsigned int i = 0; i < numElements * numDimensions; i++) {
		unsigned int j = i / numElements;
		unsigned int k = i % numElements;

		pointListTransposed[i] = pointList[k*numDimensions + j];
	}
}
