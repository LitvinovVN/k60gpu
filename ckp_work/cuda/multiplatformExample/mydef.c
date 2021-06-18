// My global definitions:

#include "mydef.h"

void MyThreadWork(App_Config_t *AppCfg)
{
  static ThreadHn_t *ThreadHnArray; 
  static ThreadId_t *ThreadIdArray;
  static ThreadDt_t *ThreadDtArray;
  static CritSect_t CriticalSection; 
  int i, mp, nt;

  mp = AppCfg->mp;
  nt = AppCfg->nt;

#ifdef WIN_PLATFORM
  if (!(ThreadHnArray = (ThreadHn_t *) HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, nt*sizeof(ThreadHn_t))))
    MyNetErr("Not enough memory",mp,1);

  if (!(ThreadDtArray = (ThreadDt_t *) HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, nt*sizeof(ThreadDt_t))))
    MyNetErr("Not enough memory",mp,2);

  if (!(ThreadIdArray = (ThreadId_t *) HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, nt*sizeof(ThreadId_t))))
    MyNetErr("Not enough memory",mp,3);

  if (!InitializeCriticalSectionAndSpinCount(&CriticalSection,0x00000400)) 
    MyNetErr("Can not create critical section",mp,4);
#else
  if (!(ThreadHnArray = (ThreadHn_t *) malloc(nt*sizeof(ThreadHn_t))))
    MyNetErr("Not enough memory",mp,1);

  if (!(ThreadDtArray = (ThreadDt_t *) malloc(nt*sizeof(ThreadDt_t))))
    MyNetErr("Not enough memory",mp,2);

  if (pthread_mutex_init(&CriticalSection, NULL) != 0)
    MyNetErr("Can not create critical section",mp,4);
#endif

  AppCfg->CritSect = (void*)&CriticalSection;

  for (i=0; i<nt; i++) {
    (ThreadDtArray+i)->mt=i;

#ifdef WIN_PLATFORM
    ThreadHnArray[i] = CreateThread(NULL,0,MyJobThr,&ThreadDtArray[i],0,&ThreadIdArray[i]);
    if (ThreadHnArray[i] == NULL)
      MyNetErr("Can not create thread",mp,5);
#else
    if (pthread_create(ThreadHnArray+i,0,MyJobThr,(void*)(ThreadDtArray+i)))
      MyNetErr("Can not create thread",mp,5);
#endif
  }

#ifdef WIN_PLATFORM
  WaitForMultipleObjects(nt,ThreadHnArray,TRUE,INFINITE);

  for (i=0; i<nt; i++)
    CloseHandle(ThreadHnArray[i]);
#else
  for (i=0; i<nt; i++)
    pthread_join(ThreadHnArray[i],0);
#endif

#ifdef WIN_PLATFORM
  HeapFree(GetProcessHeap(),0,ThreadHnArray);
  HeapFree(GetProcessHeap(),0,ThreadDtArray);
  HeapFree(GetProcessHeap(),0,ThreadIdArray);
  DeleteCriticalSection(&CriticalSection);
#else
  free(ThreadHnArray);
  free(ThreadDtArray);
#endif

  return;
}
