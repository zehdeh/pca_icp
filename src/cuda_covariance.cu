#include "cuda_covariance.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "cuda_util.h"

__global__ void kernel_findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* pointList, float* centroid);
__global__ void kernel_translate(const unsigned int numElements, const unsigned int numDimensions, float* pointList, float* vec);
__global__ void kernel_transpose(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const pointListTransposed);
__global__ void kernel_findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, const float* const pointList2, float* const covariance);

float* d_pointList1;
float* d_pointList2;

float** getDevicePointList1() {
	return &d_pointList1;
}
float** getDevicePointList2() {
	return &d_pointList1;
}

void cuda_initPointList(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* d_pointList) {
	size_t bytes = numDimensions*numElements*sizeof(float);

	gpuErrchk(cudaSetDevice(0));

	gpuErrchk(cudaMalloc(&d_pointList, bytes));
	gpuErrchk(cudaMemcpy(d_pointList, pointList, bytes, cudaMemcpyHostToDevice));
}
void cuda_destroyPointList(float* d_pointList) {
	cudaFree(d_pointList);
}

void cuda_downloadPointList(const unsigned numElements, const unsigned int numDimensions, float* const pointList, float* d_pointList) {
	size_t bytes = numDimensions*numElements*sizeof(float);

	gpuErrchk(cudaMemcpy(pointList, d_pointList, bytes, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(pointList, d_pointList, bytes, cudaMemcpyDeviceToHost));
}

void cuda_findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* const d_pointList, float* centroid) {
	size_t bytesCentroid = numDimensions*sizeof(float);
	float* d_centroid;
	std::cout << bytesCentroid << std::endl;

	gpuErrchk(cudaMalloc(&d_centroid, bytesCentroid));
	gpuErrchk(cudaMemset(d_centroid, 0, bytesCentroid));

	int blockSize, gridSize;
	blockSize = 1024;
	gridSize = (int)ceil((float)numDimensions*numElements/blockSize);

	kernel_findOriginDistance<<<gridSize, blockSize>>>(numElements, numDimensions, d_pointList, d_centroid);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

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

	kernel_translate<<<gridSize, blockSize>>>(numElements, numDimensions, d_pointList, d_centroid);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//gpuErrchk(cudaMemcpy(centroid, d_centroid, bytesCentroid, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_centroid));
}

void cuda_findCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const d_pointList1, const float* const d_pointList2, float* const covariance) {
	float* d_covariance;
	size_t bytesCovariance = sizeof(float)*numDimensions*numDimensions;

	gpuErrchk(cudaMalloc(&d_covariance, bytesCovariance));
	gpuErrchk(cudaMemset(d_covariance, 0, bytesCovariance));

	int blockSize, gridSize;
	blockSize = 1024;
	gridSize = (int)ceil((float)numDimensions*numDimensions*numElements/blockSize);

	kernel_findCovariance<<<gridSize, blockSize>>>(numElements, numDimensions, d_pointList1, d_pointList2, d_covariance);

	gpuErrchk(cudaMemcpy(covariance, d_covariance, bytesCovariance, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_covariance));
}

__global__ void kernel_findOriginDistance(const unsigned int numElements, const unsigned int numDimensions, const float* pointList, float* centroid) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float sum[3];
	if(threadIdx.x == 0) {
		sum[0] = 0;
		sum[1] = 0;
		sum[2] = 0;
	}

	__syncthreads();
	if(i < numElements * numDimensions) {
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

	if(i < numDimensions*numDimensions*numElements) {

		const unsigned int cov_j = i % numElements;
		const unsigned int cov_i = i / numElements;
		const unsigned int x = cov_i % numDimensions;
		const unsigned int y = cov_i / numDimensions;

		__syncthreads();

		const float elem = pointList1[y+cov_j*numDimensions]*pointList2[x + cov_j*numDimensions];
		atomicAdd(&covariance[y*numDimensions + x], elem);
		
		__syncthreads();
	}
}
