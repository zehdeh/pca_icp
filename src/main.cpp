#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

int main() {
	cudaError_t err = cudaSuccess;

	unsigned int numElements = 5;
	unsigned int numDimensions = 3;
	float** pointList = new float*[numElements];

	// Initialize vectors with random data
	std::srand(std::time(0));
	//for(int i = 0; i < numElements; i++) {
	//	pointList[i] = new float[numDimensions];

		// pointList[i][0] = std::rand()*10;
		// pointList[i][1] = std::rand()*10;
		// pointList[i][2] = std::rand()*20;
	//}
	pointList[0] = new float[3]{4,2,0.6};
	pointList[1] = new float[3]{4.2,2.1,0.59};
	pointList[2] = new float[3]{3.9,2,0.58};
	pointList[3] = new float[3]{4.3,2.1,0.62};
	pointList[4] = new float[3]{4.1,2.2,0.63};

	// Compute mean
	float mean[3] = {0,0,0};
	for(int i = 0; i < numElements; i++) {
		mean[0] += pointList[i][0];
		mean[1] += pointList[i][1];
		mean[2] += pointList[i][2];
	}
	mean[0] = mean[0] / numElements;
	mean[1] = mean[1] / numElements;
	mean[2] = mean[2] / numElements;

	// Compute covariance matrix
	float** covariance = new float*[numDimensions];
	covariance[0] = new float[numDimensions];
	covariance[1] = new float[numDimensions];
	covariance[2] = new float[numDimensions];
	
	for(int i = 0; i < numDimensions; i++) {
		for(int j = 0; j < numDimensions; j++) {
			float numerator = 0;
			
			// Calculate sum over all points
			for(int k = 0; k < numElements; k++) {
				numerator += (pointList[k][i] - mean[i]) * (pointList[k][j] - mean[j]);
			}
			float denominator = numElements - 1;

			covariance[i][j] = numerator / denominator;
		}
	}

	covariance[0][0] = 2;
	covariance[0][1] = -12;
	covariance[1][0] = 1;
	covariance[1][1] = -5;
	float eigenvalue[3] = {1,1,1};
	float temp[3] = {0,0,0};
	for(int i = 0; i < 5; i++) {
		temp[0] = (covariance[0][0]*eigenvalue[0]) + (covariance[0][1]*eigenvalue[1]);
		temp[1] = (covariance[1][0]*eigenvalue[0]) + (covariance[1][1]*eigenvalue[1]);
		temp[2] = (covariance[2][0]*eigenvalue[0]) + (covariance[2][1]*eigenvalue[1]);

		eigenvalue[0] = temp[0];
		eigenvalue[1] = temp[1];
		eigenvalue[2] = temp[2];
	}

	std::cout << eigenvalue[0] << " " << eigenvalue[1] << " " << eigenvalue[2] << std::endl;


	/*
	for(int i = 0; i < numDimensions; i++) {
		for(int j = 0; j < numDimensions; j++) {
			std::cout << "    " << covariance[i][j];
		}
		std::cout << std::endl;
	}
	*/

	// Delete covariance matrix
	delete[] covariance[0];
	delete[] covariance[1];
	delete[] covariance[2];
	
	delete covariance;

	// Delete all vectors
	for(int i = 0; i < numElements; i++) {
		delete[] pointList[i];
	}

	delete pointList;
	return 0;
}
