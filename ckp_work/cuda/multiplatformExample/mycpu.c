#include "myplatform.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#ifdef WIN_PLATFORM
  #include <windows.h>
#else
  #include <unistd.h>
  #include <pthread.h>
  #include <sched.h>
#endif

#include "mycpu.h"

// Internal functions:

double MyFun(double x);

// CPU count:

int MyGetCPUCount()
{
#ifdef WIN_PLATFORM
  int i = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);
#else
  int i = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  return i;
}

// CPU process:

int MyCPUProcess(int mp, int mt, int i1, int i2, double a, double h, double *s)
{
  int i;
  *s = 0;

  for (i=i1; i<=i2; i++) {
    double x = a + h * (1.0 * i - 0.5);
    *s += MyFun(x) * h;
  }

  return 0;
}

// Job function:

double MyFun(double x)
{
  double c = cos(x);
  c = c * tan(x-0.25);
  c = c * tan(x-0.5);
  c = c * exp(-x);
  c = c * log(1.25+x);
  return 4.0 / (1.0 + x * x * (2.0 - c) / (2.0 + c));
}
