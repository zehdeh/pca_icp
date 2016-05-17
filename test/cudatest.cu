#include "cudatest.h"

#include <iostream>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"

#include "util.h"

inline void gpuAssert(cudaError_t code, const char * file, int line, bool Abort=true) {
	if(code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
		if (Abort) exit(code);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void cuda_findOriginDistance(const unsigned int numElements, const float* pointList, float* centroid) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float sum[3];
	if(threadIdx.x == 0) {
		sum[0] = 0;
		sum[1] = 0;
		sum[2] = 0;
	}

	__syncthreads();
	if(i < numElements) {
		int sumIdx = i % 3;
		atomicAdd(&sum[sumIdx], pointList[i]);
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		atomicAdd(&centroid[0], sum[0]);
		atomicAdd(&centroid[1], sum[1]);
		atomicAdd(&centroid[2], sum[2]);
	}

	__syncthreads();
	if(blockIdx.x == 0 && threadIdx.x == 0) {
		centroid[0] = centroid[0] / numElements;
		centroid[1] = centroid[1] / numElements;
		centroid[2] = centroid[2] / numElements;
	}
}

__global__ void cuda_translate(const unsigned int numElements, float* pointList, float* vec) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	int vecIdx = i % 3;

	pointList[i] -= vec[vecIdx];
}

__global__ void cuda_transpose(const unsigned int numElements, float* pointList, float* pointListTransposed) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
}

int cudaTest() {
	const unsigned int numElements1 = 8;
	const unsigned int numElements2 = 8;
	const unsigned int maxNumElements = std::max(numElements1, numElements2);

	const unsigned int numDimensions = 3;
	float* pointList1 = new float[numElements1*numDimensions];
	float* pointList2 = new float[numElements2*numDimensions];
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

	cudaSetDevice(0);

	size_t bytes1 = numElements1*sizeof(float);
	size_t bytes2 = numElements2*sizeof(float);

	float* d_pointList1;
	float* d_pointList2;
	size_t bytesCentroid = numDimensions*sizeof(float*);
	float* d_centroid1;
	float* d_centroid2;
	float centroid1[3] = {0,0,0};
	float centroid2[3] = {0,0,0};
	gpuErrchk(cudaMalloc(&d_pointList1, bytes1));
	gpuErrchk(cudaMalloc(&d_pointList2, bytes2));

	gpuErrchk(cudaMemcpy(d_pointList1, pointList1, bytes1, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_pointList2, pointList2, bytes2, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_centroid1, bytesCentroid));
	gpuErrchk(cudaMalloc(&d_centroid2, bytesCentroid));

	gpuErrchk(cudaMemset(d_centroid1, 0, bytesCentroid));
	gpuErrchk(cudaMemset(d_centroid2, 0, bytesCentroid));

	int blockSize, gridSize;

	blockSize = 1024;
	gridSize = (int)ceil((float)maxNumElements/blockSize);

	cuda_findOriginDistance<<<gridSize, blockSize>>>(numElements1, d_pointList1, d_centroid1);
	cuda_findOriginDistance<<<gridSize, blockSize>>>(numElements2, d_pointList2, d_centroid2);

	cudaMemcpy(centroid1, d_centroid1, bytesCentroid, cudaMemcpyDeviceToHost);
	cudaMemcpy(centroid2, d_centroid2, bytesCentroid, cudaMemcpyDeviceToHost);

	std::cout << "CUDA:" << std::endl;
	std::cout << centroid1[0] << " " << centroid1[1] << " " << centroid1[2] << std::endl;
	std::cout << centroid2[0] << " " << centroid2[1] << " " << centroid2[2] << std::endl;

	cuda_translate<<<gridSize, blockSize>>>(numElements1, d_pointList1, d_centroid1);
	cuda_translate<<<gridSize, blockSize>>>(numElements2, d_pointList2, d_centroid1);

	cudaMemcpy(pointList1, d_pointList1, bytesCentroid, cudaMemcpyDeviceToHost);
	cudaMemcpy(pointList2, d_pointList2, bytesCentroid, cudaMemcpyDeviceToHost);

	std::cout << "CUDA (after translation):" << std::endl;
	std::cout << "First:" << std::endl;
	printMatrix(numElements1, numDimensions, pointList1);
	std::cout << "Second:" << std::endl;
	printMatrix(numElements2, numDimensions, pointList2);

	float* d_pointList1Transposed;
	gpuErrchk(cudaMalloc(&d_pointList1Transposed, bytes1));
	gpuErrchk(cudaMemset(d_pointList1Transposed, 0, bytes1));

	cuda_transpose<<<gridSize, blockSize>>>(numElements1, d_pointList1, d_pointList1Transposed);
	
	cudaFree(d_pointList1Transposed);
	
	cudaFree(d_centroid1);
	cudaFree(d_centroid2);

	cudaFree(d_pointList1);
	cudaFree(d_pointList2);

	return 0;
}
