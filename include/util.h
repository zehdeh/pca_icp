#ifndef UTIL_HEADER
#define UTIL_HEADER

void printMatrix(const unsigned int m, const unsigned int n, const float* const covariance);

float MatrixRMSE(const unsigned int m, const unsigned int n, const float* matrix1, const float* matrix2);

#endif
