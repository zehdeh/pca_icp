#include "objloader.h"

#include <stdio.h>
#include <string.h>

int loadObj(const char* path, std::vector<vec3>* vertices) {
	FILE* file = fopen(path, "r");

	int numElements = 0;
	while(1) {
		char lineHeader[128];
		int res = fscanf(file, "%s", lineHeader);
		if(res == EOF) break;
		if(strcmp(lineHeader, "v") == 0) {
			vec3 v;
			fscanf(file, "%f %f %f\n", &v.x, &v.y, &v.z);
			vertices->push_back(v);
			numElements++;
		}
	}

	return numElements;
}
