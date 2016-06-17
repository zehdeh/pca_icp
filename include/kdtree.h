#ifndef KDTREE_HEADER
#define KDTREE_HEADER

#include <iostream>
#include <vector>

struct Point {
	float x;
	float y;
	float z;
};
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

void cuda_increaseStackSize();
float randF(const float min, const float max);
void dumpPart(std::vector<KdNode> &nodes, const std::vector<Point> &points,const int fromNode, const int toNode, const char splitDim);
std::vector<KdNode> makeKdTree(const std::vector<Point> &points);
unsigned int findNnBruteForce(const std::vector<Point> &points, const Point& query);
unsigned int cpu_findNnKd(const std::vector<KdNode> &nodes, const std::vector<Point> &points, const Point &query);
void cuda_findNnKd(const std::vector<KdNode> &nodes, const std::vector<Point> &points, const std::vector<Point> &queries, std::vector<int>& kdResultsGpu);

#endif
