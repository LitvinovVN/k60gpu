#include <iostream>

#include "tests/testFunctions.h"


int main()
{
	std::cout << "Hello CMake." << std::endl;
	
	testFunction1(2, 5);
	testFunction2();
	testFunction3();

	return 0;
}
