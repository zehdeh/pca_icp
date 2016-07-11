#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <math.h>

#include "cubestest.h"
#include "objtest.h"
#include "pcltest.h"
#include "cuda_objtest.h"
#include "kdtreetest.h"
#include "kdobjtest.h"

int main(int argc, char** argv) {
	//return pclTest(argc, argv);
	//return cubesTest();
	//objTest();
	//std::cout << "=====================" << std::endl;
	//cuda_objTest();
	//kdTreeTest();
	kdObjTest();

	return 0;
}
