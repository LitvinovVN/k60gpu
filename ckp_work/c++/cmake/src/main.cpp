#include <iostream>
#include <chrono>

#include "testFunctions.cpp"

int main()
{
	std::cout << "Hello CMake." << std::endl;
	
	auto result = testFunction1(2, 5);
	std::cout << "testFunction1(2, 5) returns " << result << std::endl;

	return 0;
}
