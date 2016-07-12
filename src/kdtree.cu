#include "kdtree.h"

#include "Tools.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <stack>
#include <queue>

#include <nvbio/basic/priority_queue.h>
#include <nvbio/basic/vector_view.h>

using namespace std;

#define THREADS_PER_BLOCK 512


//#define VERBOSE

const char KdNode::X;
const char KdNode::Y;

struct CudaPoint : Point {
	__host__ __device__ CudaPoint(const Point& p) {
		x = p.x;
		y = p.y;
		z = p.z;
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
		result.z = z - other.z;

		return result;
	}

	__host__ __device__ float length() const
	{
		return sqrtf(x * x + y * y + z * z);
	}
};

std::ostream& operator<<(std::ostream& os, const Point& point)
{
	os << "(" << point.x << "," << point.y << "," << point.z << ")";
	return os;
}

struct workItem {
	workItem(const unsigned int queryNodeIdx,
		const unsigned int nodeIdx) : queryNodeIdx(queryNodeIdx), nodeIdx(nodeIdx) {}

	unsigned int queryNodeIdx;
	unsigned int nodeIdx;
};

struct prioritizedWorkItem : workItem {
	prioritizedWorkItem(const unsigned int queryNodeIdx,
		const unsigned int nodeIdx, const float priority) : workItem(queryNodeIdx, nodeIdx), priority(priority) {}
	float priority;
};

bool operator<(const prioritizedWorkItem& lhs, const prioritizedWorkItem& rhs) {
	return lhs.priority < rhs.priority;
}

struct PriorityComparator {
	bool operator()(const prioritizedWorkItem& lhs, const prioritizedWorkItem& rhs) {
		return lhs.priority < rhs.priority;
	}
};

struct NodeComparator {
	char SplitDim;
	const vector<Point> &Points;

	NodeComparator(const char splitDim, const vector<Point> &points) :
		SplitDim(splitDim), Points(points) {
	}

	bool operator()(const KdNode &i, const KdNode &j) {
		return (((const CudaPoint)Points[i.PointIdx])[SplitDim] < ((const CudaPoint)Points[j.PointIdx])[SplitDim]);
	}
};

struct PointIndexComparator {
	char splitDim;
	const std::vector<Point> &points;
	const std::vector<unsigned int> &pointIndices;

	PointIndexComparator(const char splitDim, const vector<Point> &points, const vector<unsigned int> &pointIndices) :
		splitDim(splitDim), points(points), pointIndices(pointIndices) {
		}

	bool operator()(const unsigned int &i, const unsigned int &j)
	{
		return (((const CudaPoint)points[i])[splitDim] < ((const CudaPoint)points[j])[splitDim]);
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
#ifdef VERBOSE2
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
	//char newSplitDim = (splitDim == KdNode::X) ? KdNode::Y : KdNode::X;
	char newSplitDim = (splitDim + 1) % 3;
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

unsigned int makeKdLeafTree(const std::vector<Point>& points, std::vector<unsigned int> pointIndices, std::vector<KdNode2>& nodes,
	const char splitDim, const unsigned int currentNodeIdx, const int parentIdx) {
	KdNode2& currentNode = nodes[currentNodeIdx];
	currentNode.parentIdx = parentIdx;

	if(pointIndices.size() == 1) {
		currentNode.pointIdx = pointIndices[0];
		currentNode.isLeaf = true;
		for(unsigned int i = 0; i < 3; i++) {
			currentNode.boundaries[i].first = currentNode.boundaries[i].second = ((CudaPoint)points[currentNode.pointIdx])[i];
		}
	/*} else if(pointIndices.size() == 2) {
		currentNode.leftChild = (((CudaPoint)points[pointIndices[0]])[splitDim]<((CudaPoint)points[pointIndices[1]])[splitDim])?pointIndices[0]:pointIndices[1];
		currentNode.rightChild = (((CudaPoint)points[pointIndices[0]])[splitDim]>((CudaPoint)points[pointIndices[1]])[splitDim])?pointIndices[0]:pointIndices[1];
		
		for(unsigned int i = 0; i < 3; i++) {
			currentNode.boundaries[i].first = std::min(((CudaPoint)points[pointIndices[0]])[i],((CudaPoint)points[pointIndices[1]])[i]);
			currentNode.boundaries[i].second = std::max(((CudaPoint)points[pointIndices[0]])[i],((CudaPoint)points[pointIndices[1]])[i]);
		}

		currentNode.isLeaf = true;*/
	} else {
		sort(pointIndices.begin(), pointIndices.end(), PointIndexComparator(splitDim, points, pointIndices));

		unsigned int splitIndex = pointIndices.size() / 2;
		char newSplitDim = (splitDim + 1) % 3;
		std::vector<unsigned int> pointIndicesLeft(pointIndices.begin(), pointIndices.begin() + splitIndex);
		makeKdLeafTree(points, pointIndicesLeft, nodes, newSplitDim, currentNodeIdx + 1, currentNodeIdx);

		std::vector<unsigned int> pointIndicesRight(pointIndices.begin() + splitIndex, pointIndices.end());
		currentNode.rightChild = makeKdLeafTree(points, pointIndicesRight, nodes, newSplitDim, currentNodeIdx + (2*pointIndicesLeft.size()), currentNodeIdx);

		for(unsigned int i = 0; i < 3; i++) {
			currentNode.boundaries[i].first = std::min(nodes[currentNodeIdx + 1].boundaries[i].first,nodes[currentNode.rightChild].boundaries[i].first);
			currentNode.boundaries[i].second = std::max(nodes[currentNodeIdx + 1].boundaries[i].second,nodes[currentNode.rightChild].boundaries[i].second);
		}
	}

	return currentNodeIdx;
}

std::vector<KdNode2> makeKdLeafTree(const std::vector<Point>& points) {
	std::vector<unsigned int> pointIndices(points.size());
	std::vector<KdNode2> nodes(2*points.size()-1);
	for(unsigned int i = 0; i < points.size(); i++) {
		pointIndices[i] = i;
	}
	const unsigned int rootIdx = makeKdLeafTree(points, pointIndices, nodes, KdNode::X, 0, -1);
#ifdef VERBOSE
	nodes[rootIdx].print(nodes, points, 0, rootIdx);
#endif

	return nodes;
}

float maxDescendantDistance(const std::vector<KdNode2>& queryNodes, const unsigned int nodeIdx, const float* const distances) {
	const KdNode2& node = queryNodes[nodeIdx];
	if(node.isLeaf) {
		return distances[node.pointIdx];
	} else {
		return std::max(maxDescendantDistance(queryNodes, nodeIdx + 1, distances), maxDescendantDistance(queryNodes, node.rightChild, distances));
	}
}

float minNodeDistance(const KdNode2& query, const KdNode2& node) {
	float distance = 0;
	for(int i = 0; i < 3; i++) {
		float qMin = std::min(query.boundaries[i].first, query.boundaries[i].second);
		float qMax = std::max(query.boundaries[i].first, query.boundaries[i].second);
		float nMin = std::min(node.boundaries[i].first, node.boundaries[i].second);
		float nMax = std::max(node.boundaries[i].first, node.boundaries[i].second);
		if(qMax < nMin) {
			float d = nMin - qMax;
			distance += d*d;
		} else if(nMax < qMin){
			float d = qMin - nMax;
			distance += d*d;
		}
	}
	return sqrtf(distance);
}

void printChildren(const std::vector<KdNode2>& queryNodes, const unsigned int nodeIdx, const float* const distances) {
	const KdNode2& node = queryNodes[nodeIdx];
	if(node.isLeaf) {
		std::cout << node.pointIdx << " ";
	} else {
		printChildren(queryNodes, nodeIdx + 1, distances);
		printChildren(queryNodes, node.rightChild, distances);
	}
}

void dualBaseCase(const CudaPoint& query, const CudaPoint& point,
	const unsigned int currentQueryIdx, const unsigned int currentPointIdx,
	unsigned int* Nns, float* distances) {

	const float distance = (point - query).length();
	if(distance < distances[currentQueryIdx]) {
		Nns[currentQueryIdx] = currentPointIdx;
		distances[currentQueryIdx] = distance;
	}
}

void dualTreeStep(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
	const std::vector<Point>& points, const std::vector<Point>& queries,
	const unsigned int currentNodeIdx, const unsigned int currentQueryNodeIdx,
	unsigned int* Nns, float* distances,
	std::stack< workItem >& stack) {
#ifdef VERBOSE
	std::cout << "q: " << currentQueryNodeIdx << " n: " << currentNodeIdx << std::endl;
#endif

	const KdNode2& currentQueryNode = queryNodes[currentQueryNodeIdx];
	const KdNode2& currentNode = nodes[currentNodeIdx];
	
	const float nodeDistance = minNodeDistance(currentQueryNode, currentNode);
	const float maxDescendant = maxDescendantDistance(queryNodes, currentQueryNodeIdx, distances);
	if(nodeDistance > maxDescendant) {
#ifdef VERBOSE
		std::cout << "Pruning query node " << currentQueryNodeIdx << (currentQueryNode.isLeaf?"(leaf)":"") 
			<< " with node " << currentNodeIdx << (currentNode.isLeaf?"(leaf)":"") << " (" << nodeDistance << " > " << maxDescendant << ")" << std::endl;
		std::cout << "Skipping comparison of queries " << std::endl;
		printChildren(queryNodes, currentQueryNodeIdx, distances);
		std::cout << std::endl << " with points " << std::endl;
		printChildren(nodes, currentNodeIdx, distances);
		std::cout << std::endl;
#endif
		return;
	}

	if(currentQueryNode.isLeaf && currentNode.isLeaf) {
#ifdef VERBOSE
		std::cout << "Comparing " << currentQueryNode.pointIdx << " with " << currentNode.pointIdx << std::endl;
#endif
		dualBaseCase(queries[currentQueryNode.pointIdx], points[currentNode.pointIdx], 
			currentQueryNode.pointIdx, currentNode.pointIdx,
			Nns, distances);
	} else if(currentQueryNode.isLeaf && !currentNode.isLeaf) {
#ifdef VERBOSE
		std::cout << "pushing (" << currentQueryNodeIdx << "," << (currentNodeIdx + 1) << ")" << std::endl;
		std::cout << "pushing (" << currentQueryNodeIdx << "," << currentNode.rightChild << ")" << std::endl;
#endif
		stack.push(workItem(currentQueryNodeIdx,currentNodeIdx + 1));
		stack.push(workItem(currentQueryNodeIdx,currentNode.rightChild));
	} else if(!currentQueryNode.isLeaf && currentNode.isLeaf) {
#ifdef VERBOSE
		std::cout << "pushing (" << (currentQueryNodeIdx + 1) << "," << currentNodeIdx << ")" << std::endl;
		std::cout << "pushing (" << currentQueryNode.rightChild << "," << currentNodeIdx << ")" << std::endl;
#endif
		stack.push(workItem(currentQueryNodeIdx + 1,currentNodeIdx));
		stack.push(workItem(currentQueryNode.rightChild,currentNodeIdx));
	} else {
#ifdef VERBOSE
		std::cout << "pushing (" << (currentQueryNodeIdx + 1) << "," << (currentNodeIdx + 1) << ")" << std::endl;
		std::cout << "pushing (" << currentQueryNode.rightChild << "," << currentNode.rightChild << ")" << std::endl;
		std::cout << "pushing (" << (currentQueryNodeIdx + 1) << "," << currentNode.rightChild << ")" << std::endl;
		std::cout << "pushing (" << currentQueryNode.rightChild << "," << currentNodeIdx + 1 << ")" << std::endl;
#endif
		stack.push(workItem(currentQueryNodeIdx + 1,currentNodeIdx + 1));
		stack.push(workItem(currentQueryNodeIdx + 1,currentNode.rightChild));
		stack.push(workItem(currentQueryNode.rightChild,currentNodeIdx + 1));
		stack.push(workItem(currentQueryNode.rightChild,currentNode.rightChild));
	}
}

void findNnDual(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<int>& results) {
	
	unsigned int Nns[queries.size()];
	float distances[queries.size()];
	std::stack< workItem > stack;
	memset(Nns, 0, sizeof(Nns));
	for(unsigned int i = 0; i < queries.size(); i++) {
		distances[i] = 9999;
	}

	stack.push(workItem(0, 0));
	while(!stack.empty()) {
		workItem work = stack.top();
		stack.pop();
		dualTreeStep(nodes, queryNodes, points, queries, work.nodeIdx, work.queryNodeIdx, Nns, distances, stack);
	}
	results.insert(results.begin(), &Nns[0], &Nns[sizeof(Nns) / sizeof(unsigned int)]);
}

float score(const std::vector<KdNode2>& queryNodes,
	const unsigned int currentQueryNodeIdx, const KdNode2& currentNode,
	const float* distances) {

	const KdNode2& currentQueryNode = queryNodes[currentQueryNodeIdx];

	const float nodeDistance = minNodeDistance(currentQueryNode, currentNode);
	const float maxDescendant = maxDescendantDistance(queryNodes, currentQueryNodeIdx, distances);
	if(nodeDistance < maxDescendant) {
		return maxDescendant - nodeDistance;
	}

	return 9999;
}

void dualTreeStepPrioritized(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
	const std::vector<Point>& points, const std::vector<Point>& queries,
	const unsigned int currentNodeIdx, const unsigned int currentQueryNodeIdx,
	unsigned int* Nns, float* distances,
	std::priority_queue<prioritizedWorkItem> stack) {

	const KdNode2& currentQueryNode = queryNodes[currentQueryNodeIdx];
	const KdNode2& currentNode = nodes[currentNodeIdx];
	
	float s = 0;
	if(currentQueryNode.isLeaf && currentNode.isLeaf) {
		dualBaseCase(queries[currentQueryNode.pointIdx], points[currentNode.pointIdx], 
			currentQueryNode.pointIdx, currentNode.pointIdx,
			Nns, distances);
	} else if(currentQueryNode.isLeaf && !currentNode.isLeaf) {
		s = score(queryNodes, currentQueryNodeIdx, nodes[currentNodeIdx + 1], distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx,currentNodeIdx + 1,1/s));
		}
		s = score(queryNodes, currentQueryNodeIdx, nodes[currentNode.rightChild], distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx,currentNode.rightChild,1/s));
		}
	} else if(!currentQueryNode.isLeaf && currentNode.isLeaf) {
		s = score(queryNodes, currentQueryNodeIdx + 1, currentNode, distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx + 1,currentNodeIdx, 1/s));
		}
		s = score(queryNodes, currentQueryNode.rightChild, currentNode, distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNode.rightChild,currentNodeIdx, 1/s));
		}
	} else {
		s = score(queryNodes, currentQueryNodeIdx + 1, nodes[currentNodeIdx + 1], distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx + 1,currentNodeIdx + 1,1/s));
		}
		s = score(queryNodes, currentQueryNodeIdx + 1, nodes[currentNode.rightChild], distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx + 1,currentNode.rightChild,1/s));
		}
		s = score(queryNodes, currentQueryNode.rightChild, nodes[currentNodeIdx + 1], distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNode.rightChild,currentNodeIdx + 1,1/s));
		}
		s = score(queryNodes, currentQueryNode.rightChild, nodes[currentNode.rightChild], distances);
		if(s < 9999) {
			stack.push(prioritizedWorkItem(currentQueryNode.rightChild,currentNode.rightChild,1/s));
		}
	}
}

void findNnDualPrioritized(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<int>& results) {
	
	nvbio::vector_view<prioritizedWorkItem*> priorityVector;
	nvbio::priority_queue<prioritizedWorkItem, nvbio::vector_view<prioritizedWorkItem*>, PriorityComparator > test(priorityVector);
	
	unsigned int Nns[queries.size()];
	float distances[queries.size()];
	std::priority_queue<prioritizedWorkItem> stack;
	memset(Nns, 0, sizeof(Nns));
	for(unsigned int i = 0; i < queries.size(); i++) {
		distances[i] = 9999;
	}

	stack.push(prioritizedWorkItem(0, 0, 0));
	while(!stack.empty()) {
		prioritizedWorkItem work = stack.top();
		stack.pop();
		dualTreeStepPrioritized(nodes, queryNodes, points, queries, work.nodeIdx, work.queryNodeIdx, Nns, distances, stack);
	}
	results.insert(results.begin(), &Nns[0], &Nns[sizeof(Nns) / sizeof(unsigned int)]);
}


