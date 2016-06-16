#include "kdtree.h"

#include "Tools.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>

using namespace std;

#define THREADS_PER_BLOCK 512


//#define VERBOSE

const char KdNode::X;
const char KdNode::Y;

struct CudaPoint : Point {
	__host__ __device__ CudaPoint(const Point& p) {
		x = p.x;
		y = p.y;
	}
	__host__ __device__ float operator[](const unsigned int index) const
	{
		return *((float*) this + index);
	}

	__host__ __device__ float & operator[](const unsigned int index)
	{
		return *((float*) this + index);
	}

	__host__ __device__ CudaPoint operator-(const Point& other) const
	{
		Point result;
		result.x = x - other.x;
		result.y = y - other.y;

		return result;
	}

	__host__ __device__ float length() const
	{
		return sqrtf(x * x + y * y);
	}
};

std::ostream& operator<<(std::ostream& os, const Point& point)
{
	os << "(" << point.x << "," << point.y << ")";
	return os;
}

struct NodeComparator {
	char SplitDim;
	const vector<Point> &Points;

	NodeComparator(const char splitDim, const vector<Point> &points) :
			SplitDim(splitDim), Points(points)
	{
	}

	bool operator()(const KdNode &i, const KdNode &j)
	{
		return (((const CudaPoint)Points[i.PointIdx])[SplitDim] < ((const CudaPoint)Points[j.PointIdx])[SplitDim]);
	}
};

void cuda_increaseStackSize() {
	cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024);
}

float randF(const float min = 0.0f, const float max = 1.0f)
{
	int randI = rand();
	float randF = (float) randI / (float) RAND_MAX;
	float result = min + randF * (max - min);

	return result;
}

void dumpPart(vector<KdNode> &nodes, const vector<Point> &points,
		const int fromNode, const int toNode, const char splitDim)
{
#ifdef VERBOSE
	cout << "Index " << fromNode << " to " << toNode << ": ";
	for (unsigned int n = fromNode; n < toNode; n++)
	{
		if (n > fromNode)
			cout << ", ";
		cout << points[nodes[n].PointIdx][splitDim];
	}
	cout << endl;
#endif
}

void makeKdTree(vector<KdNode> &nodes, const vector<Point> &points,
		const int fromNode, const int toNode, const char splitDim)
{
	// Check base case
	if (fromNode + 1 == toNode)
	{
		dumpPart(nodes, points, fromNode, toNode, splitDim);
		return;
	}

	// Sort all nodes currently processed
	sort(nodes.begin() + fromNode, nodes.begin() + toNode,
			NodeComparator(splitDim, points));

	// Check special case: just one child
	if (fromNode + 2 == toNode)
	{
		// Make the larger node the right child of the smaller node
		nodes[fromNode].RightChildIdx = fromNode + 1;
		nodes[fromNode].SplitDim = splitDim;

		// Done: Right child is correctly set to leaf because all nodes are initialized as leafs.
		dumpPart(nodes, points, fromNode, toNode, splitDim);
		return;
	}

	// Recursive case: At least three nodes -> one central one which has at least two children

	// Find the index where the nodes are split
	int splitNode = (fromNode + toNode) / 2;

	// Pull the split node to the front
	int tmp = nodes[fromNode].PointIdx;
	nodes[fromNode].PointIdx = nodes[splitNode].PointIdx;
	nodes[splitNode].PointIdx = tmp;

	// Setup the split node
	nodes[fromNode].RightChildIdx = splitNode + 1;
	nodes[fromNode].SplitDim = splitDim;

	dumpPart(nodes, points, fromNode, toNode, splitDim);

	// Recursively process left and right part
	char newSplitDim = (splitDim == KdNode::X) ? KdNode::Y : KdNode::X;
	makeKdTree(nodes, points, fromNode + 1, splitNode + 1, newSplitDim);
	makeKdTree(nodes, points, splitNode + 1, toNode, newSplitDim);
}

vector<KdNode> makeKdTree(const vector<Point> &points)
{
	// Create a node for every point
	vector<KdNode> nodes(points.size());
	for (unsigned int n = 0; n < nodes.size(); n++)
	{
		nodes[n].PointIdx = n;
		nodes[n].RightChildIdx = -1;
		nodes[n].SplitDim = KdNode::None;
	}

	makeKdTree(nodes, points, 0, nodes.size(), KdNode::X);

	return nodes;
}

unsigned int findNnBruteForce(const vector<Point> &points, const Point& query)
{
	unsigned int bestIdx = 0;
	for (unsigned int p = 1; p < points.size(); p++)
		if (((const CudaPoint)points[p] - (const CudaPoint)query).length() < ((CudaPoint)points[bestIdx] - (CudaPoint)query).length())
			bestIdx = p;

	return bestIdx;
}

__host__ __device__ unsigned int findNnKd(const KdNode* nodes,
		const Point* points, const unsigned int currentNodeIdx,
		unsigned int bestPointIdx, const Point &query)
{
	const CudaPoint &cudaQuery = (const CudaPoint)query;
	const KdNode &currentNode = nodes[currentNodeIdx];
	const CudaPoint &currentPoint = (const CudaPoint)points[currentNode.PointIdx];
	const CudaPoint &bestPoint = (const CudaPoint)points[bestPointIdx];

	// Check if current node is closer
	if ((currentPoint - cudaQuery).length() < (bestPoint - cudaQuery).length())
		bestPointIdx = nodes[currentNodeIdx].PointIdx;

	// Base case
	if (nodes[currentNodeIdx].SplitDim == KdNode::None)
		return bestPointIdx;

	// First check nearer child. If query point is to close to border, also check other child (backtracking)
	if (cudaQuery[currentNode.SplitDim] < currentPoint[currentNode.SplitDim]
			&& currentNode.RightChildIdx != currentNodeIdx + 1)
	{
		bestPointIdx = findNnKd(nodes, points, currentNodeIdx + 1, bestPointIdx,
				query);
		const Point &newBestPoint = points[bestPointIdx];

		// Backtracking
		if ((cudaQuery - newBestPoint).length()
				>= fabsf(
						cudaQuery[currentNode.SplitDim]
								- currentPoint[currentNode.SplitDim]))
			bestPointIdx = findNnKd(nodes, points, currentNode.RightChildIdx,
					bestPointIdx, query);
	}
	else
	{
		// Right node exists always!
		bestPointIdx = findNnKd(nodes, points, currentNode.RightChildIdx,
				bestPointIdx, query);
		const Point &newBestPoint = points[bestPointIdx];

		//Backtracking
		if ((cudaQuery - newBestPoint).length()
				>= fabsf(
						cudaQuery[currentNode.SplitDim]
								- currentPoint[currentNode.SplitDim])
				&& currentNode.RightChildIdx != currentNodeIdx + 1)
			bestPointIdx = findNnKd(nodes, points, currentNodeIdx + 1,
					bestPointIdx, query);
	}

	return bestPointIdx;
}

unsigned int cpu_findNnKd(const vector<KdNode> &nodes, const vector<Point> &points,
		const Point &query)
{
	return findNnKd(&nodes[0], &points[0], 0, nodes[0].PointIdx, query);
}

__global__ void findNnKd(int* nns, const Point* points, const int numPoints,
		const KdNode* nodes, const int numNodes, const Point* queries,
		const int numQueries)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > numQueries)
		return;

	nns[idx] = findNnKd(nodes, points, 0, nodes[0].PointIdx, queries[idx]);
}

void cuda_findNnKd(const std::vector<KdNode> &nodes, const std::vector<Point> &points, const std::vector<Point> &queries, std::vector<int>& kdResultsGpu) {
	__int64_t kdTimeGpu = 0;
	__int64_t start;
	Point* gPoints;
	KdNode* gNodes;
	Point* gQueries;
	int* gNns;
	cudaMalloc(&gPoints, sizeof(Point) * points.size());
	cudaMalloc(&gNodes, sizeof(KdNode) * nodes.size());
	cudaMalloc(&gQueries, sizeof(Point) * queries.size());
	cudaMalloc(&gNns, queries.size() * sizeof(int));
	cudaMemcpy(gPoints, &(points[0]), sizeof(Point) * points.size(),
			cudaMemcpyHostToDevice);
	cudaMemcpy(gNodes, &(nodes[0]), sizeof(KdNode) * nodes.size(),
			cudaMemcpyHostToDevice);
	cudaMemcpy(gQueries, &(queries[0]), sizeof(Point) * queries.size(),
			cudaMemcpyHostToDevice);
	int blocksPerGrid =
			queries.size() % THREADS_PER_BLOCK == 0 ?
					queries.size() / THREADS_PER_BLOCK :
					queries.size() / THREADS_PER_BLOCK + 1;
	start = continuousTimeNs();
	findNnKd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(gNns, gPoints, points.size(),
			gNodes, nodes.size(), gQueries, queries.size());
	cudaDeviceSynchronize();
	kdTimeGpu = continuousTimeNs() - start;
	cudaMemcpy(&(kdResultsGpu[0]), gNns, sizeof(int) * queries.size(),
			cudaMemcpyDeviceToHost);
	cudaFree(gPoints);
	cudaFree(gNodes);
	cudaFree(gQueries);
	cudaFree(gNns);

	std::cout << "GPU KD time: " << kdTimeGpu << std::endl;
}

