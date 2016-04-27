#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>

// Compute eigenvector of matrix with power iteration
void computeEigenvalue(float** const covariance, float* eigenvector, float* eigenvalue, const unsigned int numDimensions, const unsigned int numIterations) {
	float temp[numDimensions];
	for(int i = 0; i < numDimensions; i++) {
		temp[i] = 0;
	}

	float norm = 0;
	for(int k = 0; k < numIterations; k++) {
		for(int i = 0; i < numDimensions; i++) {
			for(int j = 0; j < numDimensions; j++) {
				temp[i] += covariance[i][j]*eigenvector[j];
			}
		}

		float normSq = 0;
		for(int l = 0; l < numDimensions; l++) {
			normSq += temp[l]*temp[l];
		}
		norm = sqrt(normSq);
		
		for(int i = 0; i < numDimensions; i++) {
			eigenvector[i] = temp[i] / norm;
		}
	}

	float numVec[3];
	for(int i = 0; i < numDimensions; i++) {
		numVec[i] = 0;
	}

	for(int i = 0; i < numDimensions; i++) {
		for(int j = 0; j < numDimensions; j++) {
			numVec[i] += covariance[i][j] * eigenvector[j];
		}
	}

	float denom = 0;
	for(int i = 0; i < numDimensions; i++) {
		denom += eigenvector[i]*eigenvector[i];
	}

	for(int i = 0; i < numDimensions; i++) {
		numVec[i] = numVec[i] / denom;
	}
	
	float sum = 0;
	for(int i = 0; i < numDimensions; i++) {
		sum += numVec[i]*numVec[i];
	}
	*eigenvalue = sqrt(sum);
}

int main() {
	cudaError_t err = cudaSuccess;

	unsigned int numElements = 5;
	unsigned int numDimensions = 3;
	float** pointList = new float*[numElements];

	// Initialize vectors with random data
	std::srand(std::time(0));
	for(int i = 0; i < numElements; i++) {
		pointList[i] = new float[numDimensions];
		for(int j = 0; j < numElements; j++) {
			pointList[i][j] = std::rand()%10;
		}
	}

	// Compute mean
	float mean[numDimensions];
	for(int i = 0; i < numElements; i++) {
		for(int j = 0; j < numDimensions; j++) {
			mean[j] += pointList[i][j];
		}
	}

	for(int i = 0; i < numDimensions; i++) {
		mean[i] = mean[i] / numElements;
	}

	// Compute covariance matrix
	float** covariance = new float*[numDimensions];
	for(int i = 0; i < numDimensions; i++) {
		covariance[i] = new float[numDimensions];
	}
	
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

	// Initialize eigenvector so it sums up to 1
	float eigenvector[numDimensions];
	float frac = 1.0 / numDimensions;
	for(int i = 0; i < numDimensions; i++) {
		eigenvector[i] = frac;
	}

	float eigenvalue = 0;
	computeEigenvalue(covariance,eigenvector,&eigenvalue,numDimensions,100);

	std::cout << "\033[1mDominant eigenvector:\033[0m" << std::endl;
	std::cout << "{";
	for(int i = 0; i < numDimensions; i++) {
		std::cout << eigenvector[i];

		if(i < numDimensions - 1) {
			std::cout << ",";
		}
	}
	std::cout << "}" << std::endl;

	std::cout << "\033[1mDominant eigenvalue:\033[0m" << std::endl;
	std::cout << eigenvalue << std::endl;

	std::cout << "\033[1mCovariance matrix:\033[0m" << std::endl;
	std::cout << "{";
	for(int i = 0; i < numDimensions; i++) {
		std::cout << "{";
		for(int j = 0; j < numDimensions; j++) {
			std::cout << covariance[i][j];

			if(j < numDimensions - 1) {
				std::cout << ",";
			}
		}
		std::cout << "}";

		if(i < numDimensions - 1) {
			std::cout << "," << std::endl;
		}
	}
	std::cout << "}" << std::endl;

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
