#ifndef KDTREE_HEADER
#define KDTREE_HEADER

#include <types.h>

#include <iostream>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>

std::ostream& operator<<(std::ostream& os, const Point& point);
struct KdNode
{
	const static char X = 0;
	const static char Y = 1;
	const static char Z = 2;
	const static char None = 3;

	int PointIdx;
	int RightChildIdx;
	char SplitDim;
};

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

void cuda_increaseStackSize();
float randF(const float min, const float max);
void dumpPart(std::vector<KdNode> &nodes, const std::vector<Point> &points,const int fromNode, const int toNode, const char splitDim);
std::vector<KdNode> makeKdTree(const std::vector<Point> &points);
unsigned int findNnBruteForce(const std::vector<Point> &points, const Point& query);
unsigned int cpu_findNnKd(const std::vector<KdNode> &nodes, const std::vector<Point> &points, const Point &query);
void cuda_findNnKd(const std::vector<KdNode> &nodes, const std::vector<Point> &points, const std::vector<Point> &queries, std::vector<int>& kdResultsGpu);
std::vector<KdNode2> makeKdLeafTree(const std::vector<Point>& points);
void findNnDual(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
	const std::vector<Point>& points, const std::vector<Point>& queries, 
	std::vector<int>& results);
void findNnDualPrioritized(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<int>& results);

#endif
