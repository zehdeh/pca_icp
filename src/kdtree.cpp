#include "kdtree.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <vector>


class kdNode {
public:
	kdNode(float median) : location(median) {}
private:
	unsigned int axis;
	float location;
};
class kdTreeNode : public kdNode {
public:
	kdTreeNode(float median) : kdNode(median) {}
	void setLeftChild(kdNode* node) { leftChild = node; }
	void setRightChild(kdNode* node) { rightChild = node; }
private:
	kdNode* leftChild;
	kdNode* rightChild;
};

class kdLeafNode : public kdNode {
public:
	kdLeafNode(float median) : kdNode(median) {}
	void setLeftChild(float* f) { leftChild = f; }
	void setRightChild(float* f) { rightChild = f; }
private:
	float* leftChild;
	float* rightChild;
};

kdNode* buildKdTree(const unsigned int numElements, const unsigned int numDimensions, float* pointList, const unsigned int depth) {
	unsigned int dimension = depth % numDimensions;
	if(numElements == 2) {
		kdLeafNode* node = new kdLeafNode(pointList[0]);
		node->setLeftChild(&pointList[0]);
		node->setRightChild(&pointList[1]);
		return node;
	}
	if(numElements == 1) {
		kdLeafNode* node = new kdLeafNode(pointList[0]);
		node->setLeftChild(&pointList[0]);
		return node;
	}

	quicksort(numElements, numDimensions, pointList, dimension);
	float median = pointList[(numElements / 2)*numDimensions + dimension];

	kdTreeNode* node = new kdTreeNode(median);
	
	std::vector<float> leftPoints;
	std::vector<float> rightPoints;
	for(unsigned int i = 0; i < numElements; i++) {
		if(pointList[i*numDimensions + dimension] < median) {
			leftPoints.push_back(pointList[i*numDimensions + dimension]);
		} else {
			rightPoints.push_back(pointList[i*numDimensions + dimension]);
		}
	}

	if(leftPoints.size() > 0) {
		node->setLeftChild(buildKdTree(leftPoints.size(), numDimensions, leftPoints.data(), depth+1));
	}
	if(rightPoints.size() > 0) {
		node->setRightChild(buildKdTree(rightPoints.size(), numDimensions, rightPoints.data(), depth+1));
	}

	return node;
}

void quicksort(const unsigned int numElements, const unsigned int numDimensions, float* pointList, unsigned int dimension, const unsigned int startIdx) {
	//std::cout << "QUICKSORT between " << startIdx << " and " << numElements << std::endl;
	int medianIdx = std::min(numElements - 1,std::max(startIdx+1,startIdx + std::rand() % (numElements - startIdx)));
	for(unsigned int i = startIdx; i < medianIdx; i++) {
		unsigned int j = numElements - 1 - i;
		if(j <= i) break;
		//std::cout << "Comparing " << i << " with " << j << std::endl;
		if(pointList[i*numDimensions + dimension] > pointList[j*numDimensions + dimension]) {
			//std::cout << "Swapping " << i << " with " << j << std::endl;

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
