#ifndef CUDA_UTIL_HEADER
#define CUDA_UTIL_HEADER

#include <iostream>

inline void gpuAssert(cudaError_t code, const char * file, int line, bool Abort=true) {
	if(code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),file,line);
		if (Abort) exit(code);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#endif
