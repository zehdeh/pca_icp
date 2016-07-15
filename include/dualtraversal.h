#ifndef DUALTRAVERSAL_HEADER
#define DUALTRAVERSAL_HEADER

#include <vector>
#include <utility>

#include "kdtree.h"

//TODO: Find better name
struct KdNode2 {
	KdNode2() : isLeaf(false), minDistance(9999), parentIdx(-1) {}
	unsigned int rightChild;
	unsigned int pointIdx;
	float minDistance;
	int parentIdx;

	char SplitDim;
	bool isLeaf;
	std::pair<float, float> boundaries[3];
	void print(const std::vector<KdNode2>& nodes, const std::vector<Point>& points, unsigned int depth, unsigned int ownIdx) const {
		if(isLeaf) {
			std::cout << "leaf (" << ownIdx << "," << pointIdx << ") " << points[pointIdx] << std::endl;
		} else {
			std::cout << "node (" << ownIdx << ",x:{" << boundaries[0].first << "," << boundaries[0].second 
			<< "},y:{" << boundaries[1].first << "," << boundaries[1].second 
			<< "},z:{" << boundaries[2].first << "," << boundaries[2].second << "})" << std::endl;
			for(unsigned int i = 0; i < depth; i++) std::cout << "   ";
			nodes[ownIdx+1].print(nodes, points, depth + 1, ownIdx + 1);
			for(unsigned int i = 0; i < depth; i++) std::cout << "   ";
			nodes[rightChild].print(nodes, points, depth + 1, rightChild);
		}
	}
};

std::vector<KdNode2> makeKdLeafTree(const std::vector<Point>& points);
void cpu_findNnDual(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
	const std::vector<Point>& points, const std::vector<Point>& queries, 
	std::vector<unsigned int>& results);
void cpu_findNnDualPrioritized(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<unsigned int>& results);
void cuda_findNnDual(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<unsigned int>& results);

#endif
