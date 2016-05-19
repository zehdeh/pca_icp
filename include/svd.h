#ifndef SVD_HEADER
#define SVD_HEADER

#include "types.h"

void findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, const float* const pointList2, float* const covariance);
void svdMethod(const unsigned int numDimensions, float* const covariance, rotationMatrix rotation);

#endif
