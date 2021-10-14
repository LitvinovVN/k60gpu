#include "testFunctions.h"

#include "../Grid3DSrc/ClassA.cpp"
#include "../Grid3DSrc/ClassB.cpp"


void testFunction1(int a, int b)
{
	std::cout << "testFunction1(int a, int b): " << (a + b) << std::endl;
}


void testFunction2()
{
	ClassA c1{555};
	c1.print();
}


void testFunction3()
{
	ClassB c1;
	c1.value = 777.7;
	c1.valueClassA.value = 888.8;
	c1.print();
}