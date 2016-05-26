#include "kdtree.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <ctime>

KdTree::KdTree(std::vector<point>& pts) {
	const unsigned int maxLevels = 99;
	m_id = 0;
	m_points = &pts;
	root = new KdNode();
	root->setId(m_id++);

	root->indices.resize(pts.size());
	for(unsigned int i = 0; i < pts.size(); i++) {
		root->indices[i] = i;
	}

	std::vector<KdNode*> toVisit;
	toVisit.push_back(root);

	while(toVisit.size()) {
		std::vector<KdNode*> nextSearch;

		while(toVisit.size()) {
			KdNode* node = toVisit.back();
			toVisit.pop_back();

			if(node->getLevel() < maxLevels) {
				if(node->indices.size() > 1) {
					KdNode* left = new KdNode();
					KdNode* right = new KdNode();

					split(node, left, right);

					{
						std::vector<unsigned int> dummy;
						node->indices.swap(dummy);
					}

					node->setLeft(left);
					node->setRight(right);

					if(left->indices.size()) {
						nextSearch.push_back(left);
					}
					if(right->indices.size()) {
						nextSearch.push_back(right);
					}
				}
			}
		}
		toVisit = nextSearch;
	}
}

void KdTree::split(KdNode* current, KdNode* left, KdNode* right) {
	std::vector<point>& points = *m_points;
	m_currentAxis = current->getLevel() % 3;

	std::sort(current->indices.begin(), current->indices.end(), [this](const int a, const int b) {
		std::vector<point>& points = *(this->m_points);

		return points[a].coords[this->m_currentAxis] < points[b].coords[this->m_currentAxis];
	});

	int mid = current->indices[current->indices.size() / 2];
	current->setSplitValue(points[mid].coords[m_currentAxis]);

	left->setParent(current);
	right->setParent(current);

	left->setLevel(current->getLevel() + 1);
	right->setLevel(current->getLevel() + 1);

	for(unsigned int i = 0; i < current->indices.size(); i++) {
		int idx = current->indices[i];

		if(points[idx].coords[m_currentAxis] < current->getSplitValue()) {
			left->indices.push_back(idx);
		} else {
			right->indices.push_back(idx);
		}
	}
}
/*
kdNode* buildKdTree(const unsigned int numElements, const unsigned int numDimensions, float** pointList, const unsigned int depth) {
	unsigned int dimension = depth % numDimensions;
	if(numElements == 0) {
		return NULL;
	}

	quicksort(numElements, numDimensions, pointList, dimension);
	float median = (*pointList)[(numElements / 2)*numDimensions + dimension];

	kdTreeNode* node = new kdTreeNode(median);
	
	std::vector<float*> leftPoints;
	std::vector<float*> rightPoints;
	for(unsigned int i = 0; i < numElements; i++) {
		if((*pointList)[i*numDimensions + dimension] < median) {
			leftPoints.push_back(pointList[i*numDimensions + dimension]);
		} else {
			rightPoints.push_back(pointList[i*numDimensions + dimension]);
		}
	}

	if(leftPoints.size() > 0) {
		if(leftPoints.size() == 1) {
			node->setLeftChild(new kdLeafNode(leftPoints[0]));
		} else {
			node->setLeftChild(buildKdTree(leftPoints.size(), numDimensions, leftPoints.data(), depth+1));
		}
	}
	if(rightPoints.size() > 0) {
		if(leftPoints.size() == 1) {
			node->setRightChild(new kdLeafNode(rightPoints[0]));
		} else {
			node->setRightChild(buildKdTree(rightPoints.size(), numDimensions, rightPoints.data(), depth+1));
		}
	}

	return node;
}
*/

void quicksort(const unsigned int numElements, const unsigned int numDimensions, float** pointList, unsigned int dimension, const unsigned int startIdx) {
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
void quicksort(const unsigned int numElements, const unsigned int numDimensions, float** pointList, unsigned int dimension) {
	std::srand(std::time(0));

	quicksort(numElements, numDimensions, pointList, dimension, 0);
}
