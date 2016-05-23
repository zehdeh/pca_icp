#ifndef KDNODE_HEADER
#define KDNODE_HEADER

class kdNode;
kdNode* buildKdTree(const unsigned int numElements, const unsigned int numDimensions, float* pointList, const unsigned int depth);
void quicksort(const unsigned int numElements, const unsigned int numDimensions, float* pointList, unsigned int dimension);

#endif
