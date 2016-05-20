#include "cuda_covariance.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "cuda_util.h"

__global__ void kernel_findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const centroid);
__global__ void kernel_translate(const unsigned int numElements, const unsigned int numDimensions, float* pointList, float* vec);
__global__ void kernel_transpose(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const pointListTransposed);
__global__ void kernel_findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, const float* const pointList2, float* const covariance);

//FIXME: Do not rely on static variables here
float* d_pointList1;
float* d_pointList2;

float** getDevicePointList1() {
	return &d_pointList1;
}
float** getDevicePointList2() {
	return &d_pointList2;
}

void cuda_initPointLists(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, const float* const pointList2) {
	size_t bytes = numDimensions*numElements*sizeof(float);

	gpuErrchk(cudaSetDevice(0));

	gpuErrchk(cudaMalloc(&d_pointList1, bytes));
	gpuErrchk(cudaMemcpy(d_pointList1, pointList1, bytes, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_pointList2, bytes));
	gpuErrchk(cudaMemcpy(d_pointList2, pointList2, bytes, cudaMemcpyHostToDevice));
}
void cuda_destroyPointList(float* d_pointList) {
	cudaFree(d_pointList);
}

void cuda_downloadPointList(const unsigned numElements, const unsigned int numDimensions, float* const pointList, float* d_pointList) {
	size_t bytes = numDimensions*numElements*sizeof(float);

	gpuErrchk(cudaMemcpy(pointList, d_pointList, bytes, cudaMemcpyDeviceToHost));
}

void cuda_findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* const d_pointList, float* centroid) {
	size_t bytesCentroid = numDimensions*sizeof(float);
	float* d_centroid;

	gpuErrchk(cudaMalloc(&d_centroid, bytesCentroid));
	gpuErrchk(cudaMemset(d_centroid, 0, bytesCentroid));

	int blockSize, gridSize;
	blockSize = 1024;
	gridSize = (int)ceil((float)numDimensions*numElements/blockSize);

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kernel_findOriginDistance<<<gridSize, blockSize>>>(numElements, numDimensions, d_pointList, d_centroid);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Translating took " << milliseconds << " ms" << std::endl;

	gpuErrchk(cudaMemcpy(centroid, d_centroid, bytesCentroid, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_centroid));
}

void cuda_translate(const unsigned int numElements, const unsigned int numDimensions, float* const d_pointList, float* const centroid) {
	size_t bytesCentroid = numDimensions*sizeof(float);
	float* d_centroid;

	gpuErrchk(cudaMalloc(&d_centroid, bytesCentroid));
	gpuErrchk(cudaMemcpy(d_centroid, centroid, bytesCentroid, cudaMemcpyHostToDevice));

	int blockSize, gridSize;
	blockSize = 1024;
	gridSize = (int)ceil((float)numDimensions*numElements/blockSize);

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kernel_translate<<<gridSize, blockSize>>>(numElements, numDimensions, d_pointList, d_centroid);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Translating took " << milliseconds << " ms" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//gpuErrchk(cudaMemcpy(centroid, d_centroid, bytesCentroid, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_centroid));
}

void cuda_findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const d_pointList1, const float* const d_pointList2, float* const covariance) {
	float* d_covariance;
	size_t bytesCovariance = sizeof(float)*numDimensions*numDimensions;

	// Our kernel copies a 3x3 matrix from shared memory so we need at least 9 threads
	// FIXME: Use exception here
	if(numElements <= 3) {
		std::cout << "Does NOT work with 3 or less elements!" << std::endl;
		return;
	}

	gpuErrchk(cudaMalloc(&d_covariance, bytesCovariance));
	gpuErrchk(cudaMemset(d_covariance, 0, bytesCovariance));

	int blockSize, gridSize;
	blockSize = 1024;
	gridSize = (int)ceil((float)numDimensions*numDimensions*numElements/blockSize);
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kernel_findCovariance<<<gridSize, blockSize>>>(numElements, numDimensions, d_pointList1, d_pointList2, d_covariance);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Computing covariance took " << milliseconds << " ms" << std::endl;

	gpuErrchk(cudaMemcpy(covariance, d_covariance, bytesCovariance, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_covariance));
}

__global__ void kernel_findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const centroid) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

/*
	if(i < numElements * numDimensions) {
		int sumIdx = i % 3;
		atomicAdd(&centroid[sumIdx], pointList[i]);
	}
	*/

	__shared__ float sum[3];
	if(threadIdx.x == 0) {
		sum[0] = 0;
		sum[1] = 0;
		sum[2] = 0;
	}

	__syncthreads();
	if(i < numElements * numDimensions) {
		int sumIdx = i % 3;
		atomicAdd(&sum[sumIdx], pointList[i] / numElements);
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		atomicAdd(&centroid[0], sum[0]);
		atomicAdd(&centroid[1], sum[1]);
		atomicAdd(&centroid[2], sum[2]);
	}
}

__global__ void kernel_translate(const unsigned int numElements, const unsigned int numDimensions, float* pointList, float* vec) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < numDimensions*numElements) {
		int vecIdx = i % numDimensions;

		pointList[i] -= vec[vecIdx];
	}
}

//FIXME: Not used anymore!
__global__ void kernel_transpose(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const pointListTransposed) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < numDimensions*numElements) {
		const unsigned int j = i / numElements;
		const unsigned int k = i % numElements;

		pointListTransposed[i] = pointList[k*3 + j];
	}
}

__global__ void kernel_findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, const float* const pointList2, float* const covariance) {
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float s_covariance[9];
	if(i < numDimensions*numDimensions*numElements) {
		if(threadIdx.x < 9) {
			s_covariance[threadIdx.x] = 0;
		}
		__syncthreads();

		const unsigned int cov_j = i % numElements;
		const unsigned int cov_i = i / numElements;
		const unsigned int x = cov_i % numDimensions;
		const unsigned int y = cov_i / numDimensions;

		const float elem = pointList1[y+cov_j*numDimensions]*pointList2[x + cov_j*numDimensions];
		atomicAdd(&s_covariance[y*numDimensions + x], elem);
		
		__syncthreads();
		if(threadIdx.x < 9) {
			atomicAdd(&covariance[threadIdx.x], s_covariance[threadIdx.x]);
		}
	}
}
