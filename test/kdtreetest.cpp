#include "kdtreetest.h"

#include "Tools.h"

#include "util.h"
#include "kdtree.h"
#include "dualtraversal.h"

#include <cmath>

/// Numbers for testing
#define NUM_POINTS 1000
#define NUM_QUERIES 1000

/// Numbers for benchmarking

#define BF // Don't use for benchmarking!
#define KD
#define KD_GPU
#define DUAL

float distance2(const Point p1, const Point p2) {
	float x = p2.x - p1.x;
	float y = p2.y - p1.y;
	float z = p2.z - p1.z;

	return sqrtf(x*x + y*y + z*z);
}

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

	std::vector<KdNode> nodes = makeKdTree(points);

	std::vector<Point> queries(NUM_QUERIES);
	for (unsigned int q = 0; q < NUM_QUERIES; q++)
	{
		queries[q].x = randF(-1.0f, 1.0f);
		queries[q].y = randF(-1.0f, 1.0f);
		queries[q].z = randF(-1.0f, 1.0f);
	}

	std::vector<KdNode> query_nodes = makeKdTree(queries);

	std::vector<KdNode2> query_dualNodes = makeKdLeafTree(queries);
	std::vector<KdNode2> dualNodes = makeKdLeafTree(points);

	/*
	for(unsigned int i = 0; i < queries.size(); i++) {
		std::cout << i << std::endl;
		std::cout << queries[i].x << " " << queries[i].y << " " << queries[i].z << std::endl;
		std::cout << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
	}
	*/

	// Init timing variables
	__int64_t bfTimeCpu = 0;
	__int64_t kdTimeCpu = 0;
	__int64_t dualTimeCpu = 0;
	__int64_t priorizedDualTimeCpu = 0;
	__int64_t start;

	std::cout << "Built trees. Now searching neighbors" << std::endl;
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

	// KD GPU
#ifdef KD_GPU
	std::vector<int> kdResultsGpu(queries.size());
	cuda_findNnKd(nodes, points, queries, kdResultsGpu);
#endif

#ifdef DUAL
	// Dual CPU
	std::vector<unsigned int> dualResults(queries.size());
	start = continuousTimeNs();
	cpu_findNnDual(dualNodes, query_dualNodes, points, queries, dualResults);
	dualTimeCpu += continuousTimeNs() - start;
#endif

	// Dual GPU
	std::vector<unsigned int> dualResultsGpu(queries.size());
	cuda_findNnDual(dualNodes, query_dualNodes, points, queries, dualResultsGpu);

	// Dual Prioritized CPU
	std::vector<unsigned int> prioritizedDualResults(queries.size());
	start = continuousTimeNs();
	cpu_findNnDualPrioritized(dualNodes, query_dualNodes, points, queries, prioritizedDualResults);
	priorizedDualTimeCpu += continuousTimeNs() - start;

	unsigned int noErrors = 0;
	unsigned int noErrorsPrioritized = 0;
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
		if(bfResults[q] != dualResults[q] && distance2(queries[q], points[bfResults[q]]) != distance2(queries[q],points[dualResults[q]])) {
			std::cout << "CPU Dual Tree error!" << std::endl;
			std::cout << q << " correct neighbor: " << bfResults[q] << " wrong neighbor: " << dualResults[q] << std::endl;
			std::cout << "Correct distance: " << distance2(queries[q], points[bfResults[q]]) << std::endl;
			std::cout << "Wrong distance: " << distance2(queries[q], points[dualResults[q]]) << std::endl;
			noErrors++;
		}

		if(bfResults[q] != prioritizedDualResults[q] && distance2(queries[q], points[bfResults[q]]) != distance2(queries[q],points[dualResults[q]])) {
			std::cout << "CPU Priority Dual Tree error!" << std::endl;
			std::cout << q << " correct neighbor: " << bfResults[q] << " wrong neighbor: " << prioritizedDualResults[q] << std::endl;
			noErrorsPrioritized++;
		}

		if(bfResults[q] != dualResultsGpu[q] && distance2(queries[q], points[bfResults[q]]) != distance2(queries[q],points[dualResultsGpu[q]])) {
			std::cout << "GPU Dual Tree error!" << std::endl;
			std::cout << q << " correct neighbor: " << bfResults[q] << " wrong neighbor: " << dualResultsGpu[q] << std::endl;
		}
	}
	std::cout << "No Dual Errors: " << noErrors << std::endl;
	std::cout << "No Dual Prioritized Errors: " << noErrorsPrioritized << std::endl;

	// Timing
	std::cout << "BF time: " << bfTimeCpu << std::endl;
	std::cout << "KD time: " << kdTimeCpu << std::endl;
	std::cout << "Dual time: " << dualTimeCpu << std::endl;
	std::cout << "Priorized dual time: " << priorizedDualTimeCpu << std::endl;

	return 0;
}
