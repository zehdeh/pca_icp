#ifndef OBJLOADER_HEADER
#define OBJLOADER_HEADER

#include <vector>
#include "types.h"

int loadObj(const char * path, std::vector<Point>* vertices);

#endif
