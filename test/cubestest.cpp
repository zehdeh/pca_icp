#include "cubestest.h"

#include <cstring>
#include <iostream>
#include "util.h"
#include "svd.h"
#include <Eigen/Eigenvalues>
#include <array>

struct eigen {
	float value;
	float vector[3];

	bool operator<(eigen const &other) const {
		return value < other.value;
	}
	void operator=(eigen const &e) {
		vector[0] = e.vector[0];
		vector[1] = e.vector[1];
		vector[2] = e.vector[2];
		value = e.value;
	}
};

void findInnerCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const covariance) {
	for(unsigned int i = 0; i < numDimensions; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			float numerator = 0;
			for(unsigned int k = 0; k < numElements; k++) {
				numerator += pointList[k*numDimensions + i]*pointList[k*numDimensions + j];
			}
			float denominator = numElements - 1;
			covariance[i*numDimensions + j] = numerator / denominator;
		}
	}
}

void findEigenvectors(float* const covariance, eigen* e) {
	Eigen::Matrix3f eigenCovariance = Eigen::Map< Eigen::Matrix<float, 3, 3, Eigen::RowMajor> >(covariance);
	Eigen::EigenSolver<Eigen::Matrix3f> es(eigenCovariance);

	std::cout << "Eigenvectors:" << std::endl;
	std::cout << es.eigenvectors() << std::endl;

	std::cout << "Eigenvalues:" << std::endl;
	std::cout << es.eigenvalues() << std::endl;

	float ev[9];
	for(int i = 0; i < 3; i++) {
		e[i].value = es.eigenvalues()[i].real();
		for(int j = 0; j < 3; j++) {
			e[i].vector[j] = es.eigenvectors().col(i)[j].real();
		}
	}

	for(unsigned int i = 1; i < 3; i++) {
		for(unsigned int j = 0; j < 2; j++) {
			if(e[j+1] < e[j]) {
				eigen tmp = e[j];
				e[j] = e[j+1];
				e[j+1] = tmp;
			}
		}
	}
}

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
	for(unsigned int i = 0; i < numElements*numDimensions; i++) {
		pointList2[i] = pointList1[i];
	}

	float testRotation[9] = {0.52,0,0.85,0,1,0,-0.85,0,0.52};
	rotateMatrix(numElements, numDimensions, pointList2, testRotation);

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
	
	float covariance1[numDimensions * numDimensions];
	memset(covariance1, 0, sizeof(float)*numDimensions*numDimensions);
	findInnerCovariance(numElements, numDimensions, pointList1, covariance1);
	printMatrix(numDimensions, numDimensions, covariance1);

	float covariance2[numDimensions * numDimensions];
	memset(covariance2, 0, sizeof(float)*numDimensions*numDimensions);
	findInnerCovariance(numElements, numDimensions, pointList2, covariance2);
	printMatrix(numDimensions, numDimensions, covariance2);

	eigen e1[3];
	findEigenvectors(covariance1, e1);

	std::cout << "Eigenvectors in C-Array:" << std::endl;
	for(unsigned int i = 0; i < 3; i++) {
		std::cout << "(";
		for(unsigned int j = 0; j < 3; j++) {
			std::cout << e1[i].vector[j] << " ";
		}
		std::cout << "|" << e1[i].value << ")" << std::endl;
	}
	std::cout << std::endl;

	eigen e2[3];
	findEigenvectors(covariance2, e2);
	std::cout << "Eigenvectors in C-Array:" << std::endl;
	for(unsigned int i = 0; i < 3; i++) {
		std::cout << "(";
		for(unsigned int j = 0; j < 3; j++) {
			std::cout << e2[i].vector[j] << " ";
		}
		std::cout << "|" << e2[i].value << ")" << std::endl;
	}
	/*
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
	*/

	return 0;
}
