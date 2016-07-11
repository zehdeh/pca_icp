#ifndef UTIL_HEADER
#define UTIL_HEADER

void rotateMatrix(const unsigned int numElements, float* const pointList, const float* const rotation);
void printMatrix(const unsigned int m, const unsigned int n, const float* const covariance);
float matrixRMSE(const unsigned int m, const unsigned int n, const float* matrix1, const float* matrix2);
void findOriginDistance(const unsigned int numElements, const float* const pointList, float* const distance);
void translate(const unsigned int numElements, float* const pointList, float* t);
void transpose(const unsigned int numElements, const float* const pointList, float* const pointListTransposed);

#endif
