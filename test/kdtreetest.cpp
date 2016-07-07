#include "kdtreetest.h"

#include "Tools.h"

#include "util.h"
#include "kdtree.h"

/// Numbers for testing
//#define NUM_POINTS 20
//#define NUM_QUERIES 10

/// Numbers for benchmarking
#define NUM_POINTS 1000
#define NUM_QUERIES 1000

#define BF // Don't use for benchmarking!
#define KD
#define KD_GPU

int kdTreeTest() {
	// Increase the cuda stack size for more recursion levels.
	cuda_increaseStackSize();

	// Create random point data
	std::vector<Point> points(NUM_POINTS);
	for (unsigned int p = 0; p < NUM_POINTS; p++)
	{
		points[p].x = randF(-1.0f, 1.0f);
		points[p].y = randF(-1.0f, 1.0f);
		points[p].z = randF(-1.0f, 1.0f);
	}
	std::vector<KdNode2> spaceNodes = makeKdLeafTree(points);
	return 0;
	std::vector<KdNode> nodes = makeKdTree(points);

	std::vector<Point> queries(NUM_QUERIES);
	for (unsigned int q = 0; q < NUM_QUERIES; q++)
	{
		queries[q].x = randF(-1.0f, 1.0f);
		queries[q].y = randF(-1.0f, 1.0f);
		queries[q].z = randF(-1.0f, 1.0f);
	}
	std::vector<KdNode> query_nodes = makeKdTree(queries);

	// Init timing variables
	__int64_t bfTimeCpu = 0;
	__int64_t kdTimeCpu = 0;
	__int64_t start;

	// BF CPU
#ifdef BF
	std::vector<int> bfResults(queries.size());
	start = continuousTimeNs();
	for (unsigned int q = 0; q < queries.size(); q++)
		bfResults[q] = findNnBruteForce(points, queries[q]);
	bfTimeCpu += continuousTimeNs() - start;
#endif

	// KD CPU
#ifdef KD
	std::vector<int> kdResults(queries.size());
	start = continuousTimeNs();
	for (unsigned int q = 0; q < queries.size(); q++)
		kdResults[q] = cpu_findNnKd(nodes, points, queries[q]);
	kdTimeCpu += continuousTimeNs() - start;
#endif

	// GPU
#ifdef KD_GPU
	std::vector<int> kdResultsGpu(queries.size());
	cuda_findNnKd(nodes, points, queries, kdResultsGpu);
#endif

	// Verification
	for (unsigned int q = 0; q < queries.size(); q++)
	{
#ifdef VERBOSE
		std::cout << queries[q] << "   BF: " << bfResults[q] << " "
				<< (queries[q] - points[bfResults[q]]).length() << "   KD: "
				<< kdResults[q] << " "
				<< (queries[q] - points[kdResults[q]]).length() << std::endl;
#endif
#ifdef BF
#ifdef KD
		if (bfResults[q] != kdResults[q])
			std::cout << "CPU KD Tree error!" << std::endl;
#endif
#ifdef KD_GPU
		if (bfResults[q] != kdResultsGpu[q])
			std::cout << "GPU KD Tree error!" << std::endl;
#endif
#endif
#ifdef KD
#ifdef KD_GPU
		if (kdResults[q] != kdResultsGpu[q])
			std::cout << "CPU/GPU KD Tree results differ!" << std::endl;
#endif
#endif
	}

	// Timing
	std::cout << "BF time: " << bfTimeCpu << std::endl;
	std::cout << "KD time: " << kdTimeCpu << std::endl;

	return 0;
}
