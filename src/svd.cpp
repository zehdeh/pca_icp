#include "svd.h"
#include <Eigen/Dense>

#include "util.h"
//#include <Eigen/Eigenvalues>

void findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, const float* const pointList2, float* const covariance) {
	for(unsigned int i = 0;i < numDimensions; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			covariance[i * numDimensions + j]=0;
			for(unsigned int k = 0; k < numElements; k++) {
				// We're actually dealing with the transposed pointList1 here
				covariance[i * numDimensions + j]=covariance[i * numDimensions + j]+pointList1[numElements*i + k] * pointList2[numDimensions*k + j];
			}
		}
	}
}

void svdMethod(const unsigned int numElements, const unsigned int numDimensions, float* const pointList1, float* const pointList2, rotationMatrix rotation) {
	float pointList1Transposed[numElements*numDimensions];
	memset(pointList1Transposed, 0, sizeof(float)*numElements*numDimensions);
	transpose(numElements, numDimensions, pointList1, pointList1Transposed);

	float covariance[numDimensions * numDimensions];
	memset(covariance, 0, sizeof(float)*numDimensions*numDimensions);
	findCovariance(numElements, numDimensions, pointList1Transposed, pointList2, covariance);

	Eigen::Matrix3f eigenCovariance = Eigen::Map< Eigen::Matrix<float, 3, 3, Eigen::RowMajor> >(covariance);
	Eigen::JacobiSVD< Eigen::MatrixXf > svd(eigenCovariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::MatrixXf R = svd.matrixU() * svd.matrixV();

	Eigen::Map< Eigen::Matrix<float, 3, 3> >(rotation, R.rows(), R.cols()) = R;
}
