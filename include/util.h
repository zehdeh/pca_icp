#ifndef UTIL_HEADER
#define UTIL_HEADER

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
