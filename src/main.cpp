#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

int main() {
	cudaError_t err = cudaSuccess;

	unsigned int numElements = 100;
	unsigned int numDimensions = 3;
	float** pointList = new float*[numElements];

	// Initialize vectors with random data
	std::srand(std::time(0));
	for(int i = 0; i < numElements; i++) {
		pointList[i] = new float[numDimensions];

		pointList[i][0] = std::rand()*10;
		pointList[i][1] = std::rand()*10;
		pointList[i][2] = std::rand()*20;
	}

	// Compute mean
	float* mean = new float[3]{0,0,0};
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

	std::cout << mean[0] << " " << mean[1] << " " << mean[2] << std::endl;

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
