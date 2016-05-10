#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <math.h>

#include "util.h"
#include "objloader.h"

void centerPoints(const unsigned int numElements, const unsigned int numDimensions, float* const pointList1, float* const pointList2) {
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
}

void rotateMatrix(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, float* const pointList2, const float* const rotation) {
	memset(pointList2, 0, sizeof(float)*numDimensions*numElements);
	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			for(unsigned int k = 0; k < numDimensions; k++) {
				pointList2[i*numDimensions + j] += pointList1[i*numDimensions + k]*rotation[j*numDimensions + k];
				if(j == 0 && i == 0) {
					std::cout << pointList1[i*numDimensions + k] << " * " << rotation[j*numDimensions + k] << std::endl;
				}
			}
			if(j == 0 && i == 0) {
				std::cout << pointList2[i*numDimensions + j] << std::endl;
			}
		}
	}
}

void svdMethod(const unsigned int numElements, const unsigned int numDimensions, float* const pointList1, float* const pointList2) {
	centerPoints(numElements, numDimensions, pointList1, pointList2);

	std::cout << "First:" << std::endl;
	printMatrix(numElements, numDimensions, pointList1);

	float covariance[numDimensions * numDimensions];
	memset(covariance, 0, sizeof(float)*numDimensions*numDimensions);

	/*
	for(unsigned int k = 0; k < numElements; k++) {
		for(unsigned int i = 0; i < numDimensions; i++) {
			for(unsigned int j = 0; j < numDimensions; j++) {
				covariance[i*numDimensions + j] += pointList1[k*numElements + i] * pointList2[k*numElements + j];

			}
		}
	}*/
	//std::cout << "non-Transposed:" << std::endl;
	//printMatrix(numElements, numDimensions, pointList1);
	
	float pointList1Transposed[numElements*numDimensions];
	for(unsigned int i = 0; i < numElements * numDimensions; i++) {
		unsigned int j = i / numElements;
		unsigned int k = i % numElements;

		pointList1Transposed[i] = pointList1[k*numDimensions + j];
	}

	for(unsigned int i = 0;i < numDimensions; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			covariance[i * numDimensions + j]=0;
			for(unsigned int k = 0; k < numElements; k++) {
				covariance[i * numDimensions + j]=covariance[i * numDimensions + j]+pointList1Transposed[numElements*i + k] * pointList2[numDimensions*k + j];
			}
		}
	}

	std::cout << "Covariance:" << std::endl;
	printMatrix(numDimensions, numDimensions, covariance);

	Eigen::Matrix3f eigenCovariance = Eigen::Map< Eigen::Matrix<float, 3, 3, Eigen::RowMajor> >(covariance);
	Eigen::JacobiSVD< Eigen::MatrixXf > svd(eigenCovariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::MatrixXf R = svd.matrixU() * svd.matrixV();
	std::cout << "Rotation matrix: " << std::endl;

	float rotation[numDimensions*numDimensions];
	Eigen::Map< Eigen::Matrix<float, 3, 3> >(rotation, R.rows(), R.cols()) = R;
	printMatrix(numDimensions, numDimensions, rotation);
	std::cout << "determinant: " << R.determinant() << std::endl;

	return rotateMatrix(numElements, numDimensions, pointList1, pointList2, rotation);
}

int main() {
	/*
	std::vector<vec3> vertices1;
	const unsigned int numDimensions = 3;
	const unsigned int numElements1 = loadObj("res/untitled.obj", &vertices1);

	std::vector<vec3> vertices2;
	const unsigned int numElements2 = loadObj("res/untitled.obj", &vertices2);

	assert(numElements1 == numElements2);
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

	svdMethod(numElements1, numDimensions, *pointList1, *pointList2);
	*/

	const unsigned int numElements1 = 8;
	const unsigned int numDimensions = 3;
	float* pointList1 = new float[numElements1*numDimensions];
	float* pointList2 = new float[numElements1*numDimensions];

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

	std::cout << "First:" << std::endl;
	printMatrix(numElements1, numDimensions, pointList1);

	svdMethod(numElements1, numDimensions, pointList1, pointList2);

	std::cout << "First:" << std::endl;
	printMatrix(numElements1, numDimensions, pointList1);
	std::cout << std::endl;
	std::cout << "Second:" << std::endl;
	printMatrix(numElements1, numDimensions, pointList2);


	return 0;
}
