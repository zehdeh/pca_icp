#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <math.h>

#include "cubestest.h"
#include "objtest.h"
#include "pcltest.h"
#include "cudatest.h"



int main(int argc, char** argv) {
	//return pclTest(argc, argv);
	//return cubesTest();
	//return cudaTest();
	return objTest();
}
