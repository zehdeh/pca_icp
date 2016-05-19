#ifndef CUDA_COVARIANCE_HEADER
#define CUDA_COVARIANCE_HEADER

void cuda_initPointLists(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, const float* const pointList2);
void cuda_destroyPointList(float* d_pointList);
void cuda_downloadPointList(const unsigned numElements, const unsigned int numDimensions, float* pointList, float* d_pointList);
void cuda_findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* centroid);
void cuda_translate(const unsigned int numElements, const unsigned int numDimensions, float* const d_pointList, float* centroid);
void cuda_findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const d_pointList1, const float* const d_pointList2, float* const covariance);

float** getDevicePointList1();
float** getDevicePointList2();

#endif
