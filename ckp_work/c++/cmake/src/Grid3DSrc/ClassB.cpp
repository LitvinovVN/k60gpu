#ifndef CLASS_B
#define CLASS_B

#include <iostream>
#include "ClassA.cpp"

struct ClassB
{
	double value;
	ClassA valueClassA{};

	void print()
	{
		std::cout << "ClassB.value = " << value << std::endl;
		std::cout << "ClassB.valueClassA = " << valueClassA.value << std::endl;
	}
};

#endif