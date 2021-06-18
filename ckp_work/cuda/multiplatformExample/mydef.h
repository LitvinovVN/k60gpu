// My global definitions:

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#include "myplatform.h"

#ifdef WIN_PLATFORM
  #include <windows.h>
  #include <sys/timeb.h>
  #include <time.h>
#else
  #include <unistd.h>
  #include <pthread.h>
  #include <sched.h>
  #include <sys/time.h>
#endif

#include <mpi.h>
#include <omp.h>

// My types:

#define BUF_MAX_CHR      65536
#define BUF_MAX_INT      16384
#define BUF_MAX_DBL       8192

typedef union tag_buffer_t {
  char   cdata[BUF_MAX_CHR];
  int    idata[BUF_MAX_INT];
  double ddata[BUF_MAX_DBL];
} buffer_t;

typedef struct tag_App_Config_t {
// Network & threads
  int np, mp, lpname;
  int nt, mt, ng, mg;
  int ngb, ngt;
  int mode, np_max, nt_max, ng_max;
  char pname[MPI_MAX_PROCESSOR_NAME];
  double tick, t1, t2, t3;
  MPI_Group gr0;
  MPI_Comm cm0;
  void* CritSect;
// Input & output
  int ier, lp, lname;
  char* basename;
  char* geomname;
  char* meshname;
  double smem;
  FILE* Fi;
  FILE* Fo;
  FILE* Fp;
} App_Config_t;

// CPU & GPU threads:

typedef struct tag_ThreadDt_t {
  int mt;
} ThreadDt_t;

#ifdef WIN_PLATFORM
  #define ThreadHn_t HANDLE
  #define ThreadId_t DWORD
  #define CritSect_t CRITICAL_SECTION
  #define MyThreadLock EnterCriticalSection
  #define MyThreadUnLock LeaveCriticalSection
  #define MyThreadFun_t DWORD WINAPI
#else
  #define ThreadHn_t pthread_t
  #define ThreadId_t int
  #define CritSect_t pthread_mutex_t
  #define MyThreadLock pthread_mutex_lock
  #define MyThreadUnLock pthread_mutex_unlock
  #define MyThreadFun_t void*
#endif

MyThreadFun_t MyJobThr(void* d);

void MyThreadWork(App_Config_t *AppCfg);
