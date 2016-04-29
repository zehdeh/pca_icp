#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <math.h>

void computeCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const covariance) {
	// Compute mean
	float mean[numDimensions];
	// Make sure mean is zeroed
	memset(mean, 0, sizeof(float)*numDimensions);
	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			mean[j] += pointList[i*numDimensions + j];
		}
	}
	for(unsigned int i = 0; i < numDimensions; i++) {
		mean[i] = mean[i] / numElements;
	}

	// Compute covariance
	for(unsigned int i = 0; i < numDimensions; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			float numerator = 0;
			
			// Calculate sum over all points
			for(unsigned int k = 0; k < numElements; k++) {
				numerator += (pointList[k*numDimensions + i] - mean[i]) * (pointList[k*numDimensions + j] - mean[j]);
			}
			float denominator = numElements - 1;

			covariance[i*numDimensions + j] = numerator / denominator;

		}
	}
}

void printCovarianceMatrix(const unsigned int numDimensions, const float* const covariance) {
	std::cout << "\033[1mCovariance matrix:\033[0m" << std::endl;
	std::cout << "{";
	for(unsigned int i = 0; i < numDimensions; i++) {
		std::cout << "{";
		for(unsigned int j = 0; j < numDimensions; j++) {
			std::cout << covariance[i*numDimensions + j];

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
}

int main() {

	const unsigned int numElements = 8;
	const unsigned int numDimensions = 3;
	float* pointList1 = new float[numElements*numDimensions];
	float* pointList2 = new float[numElements*numDimensions];

	// 2x1x1 cube
	pointList1[0 + 0] = 0; pointList1[0 + 1] = 0; pointList1[0 + 2] = 0;
	pointList1[1 + 0] = 2; pointList1[1 + 1] = 0; pointList1[1 + 2] = 0;
	pointList1[1 + 0] = 2; pointList1[1 + 1] = 1; pointList1[1 + 2] = 0;
	pointList1[1 + 0] = 0; pointList1[1 + 1] = 1; pointList1[1 + 2] = 0;
	pointList1[1 + 0] = 0; pointList1[1 + 1] = 0; pointList1[1 + 2] = 1;
	pointList1[1 + 0] = 2; pointList1[1 + 1] = 0; pointList1[1 + 2] = 1;
	pointList1[1 + 0] = 2; pointList1[1 + 1] = 1; pointList1[1 + 2] = 1;
	pointList1[1 + 0] = 0; pointList1[1 + 1] = 1; pointList1[1 + 2] = 1;

	// 1x1x2 cube
	pointList2[0 + 0] = 0; pointList2[0 + 1] = 0; pointList2[0 + 2] = 0;
	pointList2[0 + 0] = 1; pointList2[0 + 1] = 0; pointList2[0 + 2] = 0;
	pointList2[0 + 0] = 1; pointList2[0 + 1] = 1; pointList2[0 + 2] = 0;
	pointList2[0 + 0] = 0; pointList2[0 + 1] = 1; pointList2[0 + 2] = 0;
	pointList2[0 + 0] = 0; pointList2[0 + 1] = 0; pointList2[0 + 2] = 2;
	pointList2[0 + 0] = 1; pointList2[0 + 1] = 0; pointList2[0 + 2] = 2;
	pointList2[0 + 0] = 1; pointList2[0 + 1] = 1; pointList2[0 + 2] = 2;
	pointList2[0 + 0] = 0; pointList2[0 + 1] = 1; pointList2[0 + 2] = 2;

	/*
	// Initialize vectors with random data
	std::srand(std::time(0));
	for(int i = 0; i < numElements; i++) {
		for(int j = 0; j < numDimensions; j++) {
			pointList1[i*numDimensions + j] = std::rand()%10;
		}
	}
	*/

	// Compute covariance matrix
	float* covariance1 = new float[numDimensions * numDimensions];
	float* covariance2 = new float[numDimensions * numDimensions];
	computeCovariance(numElements, numDimensions, pointList1, covariance1);
	computeCovariance(numElements, numDimensions, pointList2, covariance2);


	Eigen::Matrix3f eigenCovariance1 = Eigen::Map< Eigen::Matrix<float, 3, 3, Eigen::RowMajor> >(covariance1);
	Eigen::EigenSolver< Eigen::Matrix3f > es1(eigenCovariance1);

	Eigen::Matrix3f eigenCovariance2 = Eigen::Map< Eigen::Matrix<float, 3, 3, Eigen::RowMajor> >(covariance2);
	Eigen::EigenSolver< Eigen::Matrix3f > es2(eigenCovariance2);

	std::cout << "\033[1m First pointcloud:\033[0m" << std::endl;

	std::cout << "\033[1mEigenvectors:\033[0m" << std::endl;
	std::cout << es1.eigenvectors() << std::endl;
	std::cout << "\033[1mEigenvalues:\033[0m" << std::endl;
	std::cout << es1.eigenvalues() << std::endl;
	printCovarianceMatrix(numDimensions, covariance1);
	
	std::cout << "\033[1m Second pointcloud:\033[0m" << std::endl;

	std::cout << "\033[1mEigenvectors:\033[0m" << std::endl;
	std::cout << es2.eigenvectors() << std::endl;
	std::cout << "\033[1mEigenvalues:\033[0m" << std::endl;
	std::cout << es2.eigenvalues() << std::endl;
	printCovarianceMatrix(numDimensions, covariance2);

	// Delete covariance matrices
	delete covariance1;
	delete covariance2;

	// Delete all vectors
	delete pointList1;
	delete pointList2;

	return 0;
}
