#ifndef UTIL_HEADER
#define UTIL_HEADER

float vectorAngle(const unsigned int numDimensions, float* const vec1, float* const vec2) {
	float angle = 0;
	for(unsigned int i = 0; i < numDimensions; i++) {
		angle += vec1[i]*vec2[i];
	}

	return acos(angle);
}

void doTranslation(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList1, float* const pointList2) {
	float centroid1[numDimensions];
	memset(centroid1, 0, sizeof(float)*numDimensions);

	float centroid2[numDimensions];
	memset(centroid2, 0, sizeof(float)*numDimensions);

	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			centroid1[j] += pointList1[i*numDimensions + j] / numElements;
			centroid2[j] += pointList2[i*numDimensions + j] / numElements;
		}
	}

	float translationVector[numDimensions];
	memset(translationVector, 0, sizeof(float)*numDimensions);
	for(unsigned int i = 0; i < numDimensions; i++) {
		translationVector[i] = centroid2[i] - centroid1[i];
	}

	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			pointList2[i*numDimensions + j] = pointList2[i*numDimensions + j] + translationVector[j];
		}
	}
}

void computeCentroid(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const centroid) {
	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			centroid[j] += pointList[i*numDimensions + j] / numElements;
		}
	}
}

void computeCovariance(const unsigned int numElements, const unsigned int numDimensions, const float* const pointList, float* const covariance) {
	// Compute mean
	float mean[numDimensions];
	// Make sure mean is zeroed
	memset(mean, 0, sizeof(float)*numDimensions);
	for(unsigned int i = 0; i < numElements; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			mean[j] += pointList[i*numDimensions + j];
		}
	}
	for(unsigned int i = 0; i < numDimensions; i++) {
		mean[i] = mean[i] / numElements;
	}

	// Compute covariance
	for(unsigned int i = 0; i < numDimensions; i++) {
		for(unsigned int j = 0; j < numDimensions; j++) {
			float numerator = 0;
			
			// Calculate sum over all points
			for(unsigned int k = 0; k < numElements; k++) {
				numerator += (pointList[k*numDimensions + i] - mean[i]) * (pointList[k*numDimensions + j] - mean[j]);
			}
			float denominator = numElements - 1;

			covariance[i*numDimensions + j] = numerator / denominator;
		}
	}
}

void printMatrix(const unsigned int m, const unsigned int n, const float* const covariance) {
	std::cout << "{";
	for(unsigned int i = 0; i < m; i++) {
		std::cout << "{";
		for(unsigned int j = 0; j < n; j++) {
			std::cout << covariance[i*n + j];

			if(j < n - 1) {
				std::cout << ",";
			}
		}
		std::cout << "}";

		if(i < m - 1) {
			std::cout << "," << std::endl;
		}
	}
	std::cout << "}" << std::endl;
}

#endif
