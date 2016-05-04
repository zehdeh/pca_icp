#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <math.h>

#include "util.h"

void svdMethod(const unsigned int numElements, const unsigned int numDimensions, float* const pointList1, float* const pointList2) {
	float centroid1[numDimensions];
	memset(centroid1, 0, sizeof(float)*numDimensions);

	float centroid2[numDimensions];
	memset(centroid2, 0, sizeof(float)*numDimensions);

	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			centroid1[j] += pointList1[i*numDimensions + j] / numElements;
			centroid2[j] += pointList2[i*numDimensions + j] / numElements;
		}
	}

	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			pointList1[i*numDimensions + j] -= centroid1[j];
			pointList2[i*numDimensions + j] -= centroid2[j];
		}
	}

	float covariance[numDimensions * numDimensions];
	memset(covariance, 0, sizeof(float)*numDimensions*numDimensions);

	for(unsigned int k = 0; k < numElements; k++) {
		for(unsigned int i = 0; i < numDimensions; i++) {
			for(unsigned int j = 0; j < numDimensions; j++) {
				covariance[i*numDimensions + j] += pointList1[k*numElements + i] * pointList2[k*numElements + j];

			}
		}
	}

	Eigen::Matrix3f eigenCovariance = Eigen::Map< Eigen::Matrix<float, 3, 3, Eigen::RowMajor> >(covariance);
	Eigen::JacobiSVD< Eigen::MatrixXf > svd(eigenCovariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::MatrixXf R = svd.matrixU() * svd.matrixV();
	std::cout << "Rotation matrix: " << R << std::endl;

	float* rotation;
	Eigen::Map< Eigen::Matrix<float, 3, 3, Eigen::RowMajor> >(rotation, R.rows(), R.cols()) = R;
	printMatrix(numDimensions, numDimensions, rotation);

	float result[numElements*numDimensions];
}

int main() {

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
	pointList2[3 + 0] = 1; pointList2[3 + 1] = 0; pointList2[3 + 2] = 0;
	pointList2[6 + 0] = 1; pointList2[6 + 1] = 1; pointList2[6 + 2] = 0;
	pointList2[9 + 0] = 0; pointList2[9 + 1] = 1; pointList2[9 + 2] = 0;
	pointList2[12 + 0] = 0; pointList2[12 + 1] = 0; pointList2[12 + 2] = 2;
	pointList2[15 + 0] = 1; pointList2[15 + 1] = 0; pointList2[15 + 2] = 2;
	pointList2[18 + 0] = 1; pointList2[18 + 1] = 1; pointList2[18 + 2] = 2;
	pointList2[21 + 0] = 0; pointList2[21 + 1] = 1; pointList2[21 + 2] = 2;

	svdMethod(numElements, numDimensions, pointList1, pointList2);

	std::cout << "First:" << std::endl;
	printMatrix(numElements, numDimensions, pointList1);
	std::cout << "Second:" << std::endl;
	printMatrix(numElements, numDimensions, pointList1);
	return 0;

	/*
	// Initialize vectors with random data
	std::srand(std::time(0));
	for(int i = 0; i < numElements; i++) {
		for(int j = 0; j < numDimensions; j++) {
			pointList1[i*numDimensions + j] = std::rand()%10;
		}
	}
	*/
	doTranslation(numElements, numDimensions, pointList1, pointList2);


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
	printMatrix(numElements, numDimensions, pointList1);

	std::cout << "\033[1mEigenvectors:\033[0m" << std::endl;
	std::cout << es1.eigenvectors() << std::endl;
	std::cout << "\033[1mEigenvalues:\033[0m" << std::endl;
	std::cout << es1.eigenvalues() << std::endl;

	std::cout << "\033[1mCovariance matrix:\033[0m" << std::endl;
	printMatrix(numDimensions, numDimensions, covariance1);
	
	std::cout << "\033[1m Second pointcloud:\033[0m" << std::endl;
	printMatrix(numElements, numDimensions, pointList2);

	std::cout << "\033[1mEigenvectors:\033[0m" << std::endl;
	std::cout << es2.eigenvectors() << std::endl;
	std::cout << "\033[1mEigenvalues:\033[0m" << std::endl;
	std::cout << es2.eigenvalues() << std::endl;
	
	std::cout << "\033[1mCovariance matrix:\033[0m" << std::endl;
	printMatrix(numDimensions, numDimensions, covariance2);

	// Delete covariance matrices
	delete covariance1;
	delete covariance2;

	// Delete all vectors
	delete pointList1;
	delete pointList2;

	return 0;
}
