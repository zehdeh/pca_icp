#include "kdtree.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <ctime>

class kdnode {
private:
	unsigned int axis;
	float location;
	kdnode* leftChild;
	kdnode* rightChild;
};

kdnode* buildKdTree(const unsigned int numElements, const unsigned int numDimensions, const float* pointList, const unsigned int depth) {
	unsigned int dimension = depth % numDimensions;
}

void quicksort(const unsigned int numElements, const unsigned int numDimensions, float* pointList, unsigned int dimension, const unsigned int startIdx) {
	std::cout << "QUICKSORT between " << startIdx << " and " << numElements << std::endl;
	int medianIdx = std::min(numElements - 1,std::max(startIdx+1,startIdx + std::rand() % (numElements - startIdx)));
	for(unsigned int i = startIdx; i < medianIdx; i++) {
		unsigned int j = numElements - 1 - i;
		std::cout << "Comparing " << i << " with " << j << std::endl;
		if(pointList[i*numDimensions + dimension] > pointList[j*numDimensions + dimension]) {
			std::cout << "Swapping " << i << " with " << j << std::endl;

			std::swap(pointList[i*numDimensions + 0], pointList[j*numDimensions + 0]);
			std::swap(pointList[i*numDimensions + 1], pointList[j*numDimensions + 1]);
			std::swap(pointList[i*numDimensions + 2], pointList[j*numDimensions + 2]);
		}
	}

	if(medianIdx - startIdx > 1) {
		quicksort(medianIdx, numDimensions, pointList, dimension, startIdx);
	}
	if(numElements - medianIdx > 1) {
		quicksort(numElements, numDimensions, pointList, dimension, medianIdx);
	}
}
void quicksort(const unsigned int numElements, const unsigned int numDimensions, float* pointList, unsigned int dimension) {
	std::srand(std::time(0));

	quicksort(numElements, numDimensions, pointList, dimension, 0);
}
