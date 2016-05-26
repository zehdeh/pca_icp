#ifndef KDNODE_HEADER
#define KDNODE_HEADER

#include <vector>

struct point {
	float coords[3];
};

class KdNode {
public:
	void setId(const unsigned int i) { id = i; }
	const unsigned int getId() const { return id; }
	const unsigned int getLevel() const { return level; }
	void setLevel(unsigned int lvl) { level = lvl; }
	std::vector<unsigned int> indices;
	KdNode* const getLeft() const { return left; }
	KdNode* const getRight() const { return right; }
	void setLeft(KdNode* const l) {
		left = l; 
		_left = left->getId();
	}
	void setRight(KdNode* const r) {
		right = r; 
		_right = right->getId();
	}
	void setSplitValue(float s) { splitValue = s; }
	float getSplitValue() const { return splitValue; }
	void setParent(KdNode* const p) {
		parent = p;
		_parent = parent->getId();
	}
private:
	unsigned int level;
	KdNode* parent;
	KdNode* left;
	KdNode* right;
	int _parent, _left, _right;
	unsigned int id;
	float splitValue;
};

class KdTree {
public:
	KdTree(std::vector<point>& pts);
private:
	void split(KdNode* current, KdNode* left, KdNode* right);
	std::vector<point>* m_points;
	KdNode* root;
	int m_id;
	unsigned int m_currentAxis;
};

//kdNode* buildKdTree(const unsigned int numElements, const unsigned int numDimensions, float** pointList, const unsigned int depth);
void quicksort(const unsigned int numElements, const unsigned int numDimensions, float** pointList, unsigned int dimension);

#endif
