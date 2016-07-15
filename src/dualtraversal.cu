#include "dualtraversal.h"
#include <stack>
#include <queue>

#include "cuda_util.h"
#include "Tools.h"

#define INFTY 9999
#define THREADS_PER_BLOCK 512

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

struct PointIndexComparator {
	char splitDim;
	const std::vector<Point> &points;
	const std::vector<unsigned int> &pointIndices;

	PointIndexComparator(const char splitDim, const std::vector<Point> &points, const std::vector<unsigned int> &pointIndices) :
		splitDim(splitDim), points(points), pointIndices(pointIndices) {
		}

	bool operator()(const unsigned int &i, const unsigned int &j)
	{
		return (((const CudaPoint)points[i])[splitDim] < ((const CudaPoint)points[j])[splitDim]);
	}
};

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

__host__ __device__ float maxDescendantDistance(const KdNode2* queryNodes, const unsigned int nodeIdx, const float* const distances) {
	const KdNode2& node = queryNodes[nodeIdx];
	if(node.isLeaf) {
		return distances[node.pointIdx];
	} else {
		return fmax(maxDescendantDistance(queryNodes, nodeIdx + 1, distances), maxDescendantDistance(queryNodes, node.rightChild, distances));
	}
}

__host__ __device__ float minNodeDistance(const KdNode2& query, const KdNode2& node) {
	float distance = 0;
	for(int i = 0; i < 3; i++) {
		float qMin = fmin(query.boundaries[i].first, query.boundaries[i].second);
		float qMax = fmax(query.boundaries[i].first, query.boundaries[i].second);
		float nMin = fmin(node.boundaries[i].first, node.boundaries[i].second);
		float nMax = fmax(node.boundaries[i].first, node.boundaries[i].second);
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

void printChildren(const KdNode2* queryNodes, const unsigned int nodeIdx, const float* const distances) {
	const KdNode2& node = queryNodes[nodeIdx];
	if(node.isLeaf) {
		std::cout << node.pointIdx << " ";
	} else {
		printChildren(queryNodes, nodeIdx + 1, distances);
		printChildren(queryNodes, node.rightChild, distances);
	}
}

__host__ __device__ void dualBaseCase(const CudaPoint& query, const CudaPoint& point,
	const unsigned int currentQueryIdx, const unsigned int currentPointIdx,
	unsigned int* Nns, float* distances) {

	const float distance = (point - query).length();
	if(distance < distances[currentQueryIdx]) {
		Nns[currentQueryIdx] = currentPointIdx;
		distances[currentQueryIdx] = distance;
	}
}

__host__ __device__ unsigned int dualTreeStep(const KdNode2* nodes, const KdNode2* queryNodes,
	const Point* points, const Point* queries,
	const unsigned int currentQueryNodeIdx,
	const unsigned int currentNodeIdx,
	unsigned int* Nns, float* distances,
	unsigned int* const queue, const unsigned int queueEnd) {
	unsigned int queueAdvance = 0;
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
		return 0;
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
		//stack.push(workItem(currentQueryNodeIdx,currentNodeIdx + 1));
		//stack.push(workItem(currentQueryNodeIdx,currentNode.rightChild));
		queue[queueEnd+queueAdvance] = currentQueryNodeIdx;
		queueAdvance++;
		queue[queueEnd+queueAdvance] = currentNodeIdx + 1;
		queueAdvance++;

		queue[queueEnd+queueAdvance] = currentQueryNodeIdx;
		queueAdvance++;

		queue[queueEnd+queueAdvance] = currentNode.rightChild;
		queueAdvance++;
	} else if(!currentQueryNode.isLeaf && currentNode.isLeaf) {
#ifdef VERBOSE
		std::cout << "pushing (" << (currentQueryNodeIdx + 1) << "," << currentNodeIdx << ")" << std::endl;
		std::cout << "pushing (" << currentQueryNode.rightChild << "," << currentNodeIdx << ")" << std::endl;
#endif
		//stack.push(workItem(currentQueryNodeIdx + 1,currentNodeIdx));
		//stack.push(workItem(currentQueryNode.rightChild,currentNodeIdx));
		queue[queueEnd+queueAdvance] = currentQueryNodeIdx + 1;
		queueAdvance++;
		queue[queueEnd+queueAdvance] = currentNodeIdx;
		queueAdvance++;

		queue[queueEnd+queueAdvance] = currentQueryNode.rightChild;
		queueAdvance++;
		queue[queueEnd+queueAdvance] = currentNodeIdx;
		queueAdvance++;
	} else {
#ifdef VERBOSE
		std::cout << "pushing (" << (currentQueryNodeIdx + 1) << "," << (currentNodeIdx + 1) << ")" << std::endl;
		std::cout << "pushing (" << currentQueryNode.rightChild << "," << currentNode.rightChild << ")" << std::endl;
		std::cout << "pushing (" << (currentQueryNodeIdx + 1) << "," << currentNode.rightChild << ")" << std::endl;
		std::cout << "pushing (" << currentQueryNode.rightChild << "," << currentNodeIdx + 1 << ")" << std::endl;
#endif
		//stack.push(workItem(currentQueryNodeIdx + 1,currentNodeIdx + 1));
		//stack.push(workItem(currentQueryNodeIdx + 1,currentNode.rightChild));
		//stack.push(workItem(currentQueryNode.rightChild,currentNodeIdx + 1));
		//stack.push(workItem(currentQueryNode.rightChild,currentNode.rightChild));
		
		queue[queueEnd+queueAdvance] = currentQueryNodeIdx + 1;
		queueAdvance++;
		queue[queueEnd+queueAdvance] = currentNodeIdx + 1;
		queueAdvance++;

		queue[queueEnd+queueAdvance] = currentQueryNodeIdx + 1;
		queueAdvance++;
		queue[queueEnd+queueAdvance] = currentNode.rightChild;
		queueAdvance++;

		queue[queueEnd+queueAdvance] = currentQueryNode.rightChild;
		queueAdvance++;
		queue[queueEnd+queueAdvance] = currentNodeIdx + 1;
		queueAdvance++;

		queue[queueEnd+queueAdvance] = currentQueryNode.rightChild;
		queueAdvance++;
		queue[queueEnd+queueAdvance] = currentNode.rightChild;
		queueAdvance++;
	}

	return queueAdvance;
}

void cpu_findNnDual(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<unsigned int>& results) {
	
	unsigned int Nns[queries.size()];
	float distances[queries.size()];

	//FIXME: Worst-case space estimate
	size_t queueSize = queryNodes.size() * nodes.size();
	unsigned int * queue = new unsigned int[queueSize];
	unsigned int currentItemCtr = 0;
	queue[0] = 0;
	queue[1] = 0;
	unsigned int queueEnd = 2;

	memset(Nns, 0, sizeof(Nns));
	for(unsigned int i = 0; i < queries.size(); i++) {
		distances[i] = INFTY;
	}

	while(currentItemCtr < (queueEnd - 1)) {
		const unsigned int queueAdvance = dualTreeStep(&nodes[0], &queryNodes[0], &points[0], &queries[0], queue[currentItemCtr], queue[currentItemCtr + 1], Nns, distances, queue, queueEnd);
		queueEnd += queueAdvance;
		currentItemCtr += 2;
	}
	results.insert(results.begin(), &Nns[0], &Nns[sizeof(Nns) / sizeof(unsigned int)]);
	delete queue;
}

__global__ void findNnDual(unsigned int* nns, const Point* const points, const unsigned int numPoints,
	const KdNode2* const nodes, const unsigned int numNodes,
	const Point* const queries, const unsigned int numQueries,
	const KdNode2* const queryNodes, const unsigned int numQueryNodes,
	float* distances, unsigned int* const queue, unsigned int* queueEnd, unsigned int* currentItemCtr) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

/*
	int i = 0;
	while(*currentItemCtr < (*queueEnd - 1) && nns[999] < 10) {
		if((i+idx*2) < (*queueEnd - 1)) {
			const unsigned int queueAdvance = dualTreeStep(nodes, queryNodes, points, queries, queue[i+idx*2], queue[i+idx*2+1], nns, distances, queue, *queueEnd);
			atomicAdd(queueEnd, queueAdvance);
			atomicAdd(currentItemCtr,2);
			nns[999]++;
		}
		i += blockDim.x*gridDim.x;
	}
*/
	if(idx == 0) {
		const unsigned int queueAdvance = dualTreeStep(nodes, queryNodes, points, queries, queue[0], queue[1], nns, distances, queue, *queueEnd);
		atomicAdd(queueEnd, queueAdvance);
		atomicAdd(currentItemCtr, 2);
	}
	__syncthreads();

	if(idx < 4) {
		const unsigned int queueAdvance = dualTreeStep(nodes, queryNodes, points, queries, queue[idx*2], queue[idx*2+1], nns, distances, queue, *queueEnd);
		atomicAdd(currentItemCtr, 2);
		atomicAdd(queueEnd, queueAdvance);
	}
	__syncthreads();
}

float score(const KdNode2* queryNodes,
	const unsigned int currentQueryNodeIdx, const KdNode2& currentNode,
	const float* distances) {

	const KdNode2& currentQueryNode = queryNodes[currentQueryNodeIdx];

	const float nodeDistance = minNodeDistance(currentQueryNode, currentNode);
	const float maxDescendant = maxDescendantDistance(queryNodes, currentQueryNodeIdx, distances);
	if(nodeDistance < maxDescendant) {
		return maxDescendant - nodeDistance;
	}

	return INFTY;
}

void dualTreeStepPrioritized(const KdNode2* nodes, const KdNode2* queryNodes,
	const Point* points, const Point* queries,
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
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx,currentNodeIdx + 1,1/s));
		}
		s = score(queryNodes, currentQueryNodeIdx, nodes[currentNode.rightChild], distances);
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx,currentNode.rightChild,1/s));
		}
	} else if(!currentQueryNode.isLeaf && currentNode.isLeaf) {
		s = score(queryNodes, currentQueryNodeIdx + 1, currentNode, distances);
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx + 1,currentNodeIdx, 1/s));
		}
		s = score(queryNodes, currentQueryNode.rightChild, currentNode, distances);
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNode.rightChild,currentNodeIdx, 1/s));
		}
	} else {
		s = score(queryNodes, currentQueryNodeIdx + 1, nodes[currentNodeIdx + 1], distances);
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx + 1,currentNodeIdx + 1,1/s));
		}
		s = score(queryNodes, currentQueryNodeIdx + 1, nodes[currentNode.rightChild], distances);
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNodeIdx + 1,currentNode.rightChild,1/s));
		}
		s = score(queryNodes, currentQueryNode.rightChild, nodes[currentNodeIdx + 1], distances);
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNode.rightChild,currentNodeIdx + 1,1/s));
		}
		s = score(queryNodes, currentQueryNode.rightChild, nodes[currentNode.rightChild], distances);
		if(s < INFTY) {
			stack.push(prioritizedWorkItem(currentQueryNode.rightChild,currentNode.rightChild,1/s));
		}
	}
}

void cpu_findNnDualPrioritized(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<unsigned int>& results) {
	
	unsigned int Nns[queries.size()];
	float distances[queries.size()];
	std::priority_queue<prioritizedWorkItem> stack;
	memset(Nns, 0, sizeof(Nns));
	for(unsigned int i = 0; i < queries.size(); i++) {
		distances[i] = INFTY;
	}

	stack.push(prioritizedWorkItem(0, 0, 0));
	while(!stack.empty()) {
		prioritizedWorkItem work = stack.top();
		stack.pop();
		dualTreeStepPrioritized(&(nodes[0]), &(queryNodes[0]), &(points[0]), &(queries[0]), work.nodeIdx, work.queryNodeIdx, Nns, distances, stack);
	}
	results.insert(results.begin(), &Nns[0], &Nns[sizeof(Nns) / sizeof(unsigned int)]);
}

void cuda_findNnDual(const std::vector<KdNode2>& nodes, const std::vector<KdNode2>& queryNodes,
		const std::vector<Point>& points, const std::vector<Point>& queries,
		std::vector<unsigned int>& results) {

	__int64_t dualTimeGpu = 0;
	__int64_t start;
	
	Point* gPoints;
	KdNode2* gNodes;

	Point* gQueries;
	KdNode2* gQueryNodes;

	unsigned int* gNns;
	float* gDistances;

	unsigned int* gQueue;
	unsigned int* gQueueEnd;
	unsigned int* gCurrentItemCtr;

	unsigned int queueEnd = 2;

	size_t queueSize = queryNodes.size() * nodes.size() * sizeof(unsigned int);
	unsigned int queue[100];

	cudaMalloc(&gPoints, sizeof(Point) * points.size());
	cudaMalloc(&gNodes, sizeof(KdNode2) * nodes.size());
	cudaMalloc(&gQueries, sizeof(Point) * queries.size());
	cudaMalloc(&gQueryNodes, sizeof(KdNode2) * queryNodes.size());
	cudaMalloc(&gDistances, queries.size() * sizeof(float));
	cudaMalloc(&gQueue, queueSize);
	cudaMalloc(&gQueueEnd, sizeof(unsigned int));
	cudaMalloc(&gCurrentItemCtr, sizeof(unsigned int));
	gpuErrchk(cudaMalloc(&gNns, queries.size() * sizeof(int)));
	gpuErrchk(cudaMemset(gNns, 0, sizeof(unsigned int)*queries.size()));

	gpuErrchk(cudaMemset(gQueue, 0, queueSize));
	gpuErrchk(cudaMemset(gCurrentItemCtr, 0, sizeof(unsigned int)));

	cudaMemcpy(gPoints, &(points[0]), sizeof(Point) * points.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(gNodes, &(nodes[0]), sizeof(KdNode2) * nodes.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(gQueries, &(queries[0]), sizeof(KdNode2) * queries.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(gQueryNodes, &(queryNodes[0]), sizeof(KdNode2) * queryNodes.size(), cudaMemcpyHostToDevice);

	gpuErrchk(cudaMemcpy(gQueueEnd, &queueEnd, sizeof(unsigned int), cudaMemcpyHostToDevice));

	int blocksPerGrid =
			queries.size() % THREADS_PER_BLOCK == 0 ?
					queries.size() / THREADS_PER_BLOCK :
					queries.size() / THREADS_PER_BLOCK + 1;

	start = continuousTimeNs();
	
	findNnDual<<<blocksPerGrid, THREADS_PER_BLOCK>>>(gNns, gPoints, points.size(),
		gNodes, nodes.size(),
		gQueries, queries.size(),
		gQueryNodes, queryNodes.size(),
		gDistances, gQueue,
		gQueueEnd, gCurrentItemCtr);
	cudaDeviceSynchronize();

	dualTimeGpu = continuousTimeNs() - start;

	unsigned int currentItemCtr = 0;
	size_t resultSize = sizeof(unsigned int) * queries.size();
	gpuErrchk(cudaMemcpy(&(results[0]), gNns, resultSize, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(&currentItemCtr, gCurrentItemCtr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&queueEnd, gQueueEnd, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	std::cout << "CurrentItemCtr: " << currentItemCtr << std::endl;
	std::cout << "QueueEnd: " << queueEnd << std::endl;

	gpuErrchk(cudaMemcpy(&queue, gQueue, sizeof(unsigned int)*100, cudaMemcpyDeviceToHost));
	for(unsigned int i = 0; i < 100; i++) {
		std::cout << "queue[" << i << "]: " << queue[i] << std::endl;
	}

	cudaFree(gPoints);
	cudaFree(gNodes);
	cudaFree(gQueries);
	cudaFree(gNns);
	cudaFree(gDistances);
	cudaFree(gQueue);
	cudaFree(gQueueEnd);
	cudaFree(gCurrentItemCtr);
	std::cout << "GPU Dual time: " << dualTimeGpu << std::endl;
}
