#include "util.h"

#include <iostream>
#include <cstring>

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

float MatrixRMSE(const unsigned int m, const unsigned int n, const float* matrix1, const float* matrix2) {
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
