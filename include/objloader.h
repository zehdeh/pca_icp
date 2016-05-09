#ifndef OBJLOADER_HEADER
#define OBJLOADER_HEADER

#include <vector>

struct vec3 {
	float x;
	float y;
	float z;
};

int loadObj(const char * path, std::vector<vec3>* vertices);

#endif
