#ifndef CLASS_A
#define CLASS_A

#include <iostream>

struct ClassA {
	double value;

	void print()
	{
		std::cout << "ClassA.value = " << value << std::endl;
	}
};

#endif