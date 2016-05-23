#include "kdtreetest.h"

#include "util.h"
#include "kdtree.h"

int kdTreeTest() {
	const int numElements = 8;
	const int numDimensions = 3;

	float pointList[numElements*numDimensions] = {
	9,1,0,
	7,2,0,
	6,3,0,
	5,4,0,
	4,5,0,
	3,6,0,
	2,7,0,
	1,8,0};


	buildKdTree(numElements, numDimensions, pointList, 0);
	//quicksort(numElements, numDimensions, pointList, 0);

	//printMatrix(numElements, numDimensions, pointList);


	//quicksort(numElements, numDimensions, pointList, 1);
	//printMatrix(numElements, numDimensions, pointList);
	return 0;
}
