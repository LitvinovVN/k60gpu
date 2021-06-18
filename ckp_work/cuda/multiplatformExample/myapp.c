#define MAIN_FILE 1
#define CUDA_C_PREF 

#include "mydef.h"
#include "mycom.h"
#include "mynet.h"
#include "mycpu.h"
#include "mygpu.h"

#include "myapp.h"

static App_Config_t AppCfgVal;
static App_Param_t  AppParVal;

static App_Config_t* AppCfg = &AppCfgVal;
static App_Param_t*  AppPar = &AppParVal;

// Main thread function:

MyThreadFun_t MyJobThr(void* d)
{
  int err, i, i1, i2, nc, nn, mm;
  double s;
  int np = AppCfg->np;
  int mp = AppCfg->mp;
  int nt = AppCfg->nt;
  int mt = ((ThreadDt_t *)d)->mt;
  int ngb = AppCfg->ngb;
  int ngt = AppCfg->ngt;
  CritSect_t *CritSect = (CritSect_t *)AppCfg->CritSect;

// Subdomain:

  nn = np * nt;
  mm = mp * nt + mt;
  MyRange(nn,mm,1,AppPar->ni,&i1,&i2,&nc);

  fprintf(stderr,"[%d,%d]: i1=%d i2=%d nc=%d\n",mp,mt,i1,i2,nc);

// Computations:

  if (AppCfg->mode==0) {
    err = MyCPUProcess(mp,mt,i1,i2,AppPar->a,AppPar->h,&s);
    fprintf(stderr,"[%d,%d]: CPU compute retcode is %d\n",mp,mt,err);
  }
  else {
    err = MyGPUProcess(mp,mt,ngb,ngt,i1,i2,AppPar->a,AppPar->h,&s);
    fprintf(stderr,"[%d,%d]: GPU compute retcode is %d\n",mp,mt,err);
  }

// Join of local results:

  MyThreadLock(CritSect);

  AppPar->sum += s;
  AppPar->nsum++;

  MyThreadUnLock(CritSect);

  if (mt==0) {
    double sum0;

    Met:
    if (AppPar->nsum<nt) {
      double s = MyDelay(10);
      goto Met;
    }

    if (np>1) {
      AppCfg->t2 = MPI_Wtime();
      sum0 = AppPar->sum;
      MPI_Reduce(&sum0, &AppPar->sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      AppCfg->t2 = MPI_Wtime() - AppCfg->t2;
    }

    if (mp==0)
      fprintf(stderr,"nsum=%d sum=%le\n",AppPar->nsum,AppPar->sum);
  }

  return 0;
}

// Main program:

int main(int argc, char *argv[])
{
  MyAppInit(argc,argv,AppCfg,AppPar);

  AppCfg->t3 = MPI_Wtime();

  MyThreadWork(AppCfg);

  AppCfg->t3 = MPI_Wtime() - AppCfg->t3;

  MyAppDone(AppCfg,AppPar);

  return 0;
}

// Initialization:

int MyAppInit(int argc, char *argv[], App_Config_t *AppCfg, App_Param_t *AppPar)
{
  int i;

// Equipment:

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &AppCfg->np);
  MPI_Comm_rank(MPI_COMM_WORLD, &AppCfg->mp);
  MPI_Get_processor_name(AppCfg->pname, &AppCfg->lname);
  MPI_Barrier(MPI_COMM_WORLD);

  AppCfg->nt_max = MyGetCPUCount();
  AppCfg->ng_max = MyGetGPUCount();

  fprintf(stderr,"Netsize: %d, process: %d, system: %s, cpu_count: %d, gpu_count: %d\n",
    AppCfg->np,AppCfg->mp,AppCfg->pname,AppCfg->nt_max,AppCfg->ng_max);

  MPI_Barrier(MPI_COMM_WORLD);

  if (AppCfg->nt_max<1 || AppCfg->ng_max<1)
    MyNetErr("Bad count of devices",AppCfg->mp,10);

  if (AppCfg->ng_max > 3) AppCfg->ng_max = 1;

// Command arguments:

  AppCfg->basename = NULL;

  AppCfg->mode = 0;
  AppCfg->nt   = 1;
  AppCfg->ng   = 0;
  AppCfg->ngb  = 0;
  AppCfg->ngt  = 0;

  if (AppCfg->mp == 0) {
    fprintf(stderr,"[%d]: argc=%d\n",AppCfg->mp,argc);
    for (i=0; i<argc; i++)
      fprintf(stderr,"[%d]: arg[%d]=`%s`\n",AppCfg->mp,i,argv[i]);

    if (argc<2) {
      AppCfg->basename = "myapp";
      AppCfg->lname    = strlen(AppCfg->basename);
      fprintf(stderr,"Usage: %s [<basename> [<mode> [<cpu_threads> [<gpu_blocks> [<gpu_threads>]]]]]\n",argv[0]);
    }
    else {
      AppCfg->lname = strlen(argv[1]);
      AppCfg->basename = (char *)malloc(sizeof(char)*AppCfg->lname);
      strcpy(AppCfg->basename,argv[1]);
    }

    AppCfg->lname = strlen(AppCfg->basename);

    if (argc>2) {
      i = sscanf(argv[2],"%d",&AppCfg->mode);
      if (AppCfg->mode<1) AppCfg->mode = 0; // CPU calculations
      else                AppCfg->mode = 1; // GPU calculations
    }

    if (argc>3) {
      i = sscanf(argv[3],"%d",&AppCfg->nt);
      if (AppCfg->nt<1) AppCfg->nt = 1;
    }

    if (argc>4) {
      i = sscanf(argv[4],"%d",&AppCfg->ngb);
      if (AppCfg->ngb<1) AppCfg->ngb = 1;
    }

    if (argc>5) {
      i = sscanf(argv[5],"%d",&AppCfg->ngt);
      if (AppCfg->ngt<1) AppCfg->ngt = 1;
    }
  }

  if (AppCfg->np > 1) {
    MPI_Bcast(&AppCfg->lname,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&AppCfg->mode,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&AppCfg->nt  ,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&AppCfg->ng  ,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&AppCfg->ngb ,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&AppCfg->ngt ,1,MPI_INT,0,MPI_COMM_WORLD);

    if (AppCfg->mp > 0)
      AppCfg->basename = (char *)malloc(sizeof(char)*AppCfg->lname);

    MPI_Bcast(AppCfg->basename,AppCfg->lname,MPI_CHAR,0,MPI_COMM_WORLD);
  }

  if (AppCfg->mode==0) {
    if (AppCfg->nt > AppCfg->nt_max) AppCfg->nt = AppCfg->nt_max;
  }
  else {
    if (AppCfg->nt > AppCfg->ng_max) AppCfg->nt = AppCfg->ng_max;
    AppCfg->ng = AppCfg->nt;
  }

  fprintf(stderr,"[%d]: basename=%s mode=%d nt=%d ng=%d ngb=%d ngt=%d\n",
    AppCfg->mp,AppCfg->basename,AppCfg->mode,AppCfg->nt,AppCfg->ng,AppCfg->ngb,AppCfg->ngt);

  MPI_Barrier(MPI_COMM_WORLD);

  if (AppCfg->mode > 0)
    MyGPUInfo(AppCfg->mp);

  MPI_Barrier(MPI_COMM_WORLD);

// Input of parameters:

  if (AppCfg->mp == 0) {
    char str[1024];

    sprintf(str,"%s.ini",AppCfg->basename);
    fprintf(stderr,"[%d]: open file %s\n",AppCfg->mp,str);

    i = fopen_m(&AppCfg->Fi,str,"rt");
    if (i != 0) MyNetErr("File not opened",AppCfg->mp,i);

    i = fget_dbl(AppCfg->Fi,16,&AppPar->a);
    if (i != 0) MyNetErr("Error of read",AppCfg->mp,i);
    fprintf(stderr,"[%d]: a=%le\n",AppCfg->mp,AppPar->a);

    i = fget_dbl(AppCfg->Fi,16,&AppPar->b);
    if (i != 0) MyNetErr("Error of read",AppCfg->mp,i);
    fprintf(stderr,"[%d]: b=%le\n",AppCfg->mp,AppPar->b);

    i = fget_int(AppCfg->Fi,16,&AppPar->ni);
    if (i != 0) MyNetErr("Error of read",AppCfg->mp,i);
    fprintf(stderr,"[%d]: ni=%d\n",AppCfg->mp,AppPar->ni);

    fclose_m(&AppCfg->Fi);
  }

  if (AppCfg->np > 1) {
    MPI_Bcast(AppPar,sizeof(App_Param_t),MPI_CHAR,0,MPI_COMM_WORLD);
  }

// Init of work variables:

  AppCfg->t1 = 0;
  AppCfg->t2 = 0;
  AppCfg->t3 = 0;

  AppPar->sum  = 0;
  AppPar->nsum = 0;
  AppPar->h    = (AppPar->b - AppPar->a) / AppPar->ni;

// Return:

  fprintf(stderr,"[%d]: Init is OK\n",AppCfg->mp);

  return 0;
}

// Finalization:

int MyAppDone(App_Config_t *AppCfg, App_Param_t *AppPar)
{
  AppCfg->t1 = AppCfg->t3 - AppCfg->t2;

  fprintf(stderr,"mode=%d np=%d nt=%d ng=%d ngb=%d ngt=%d t1=%le t2=%le t3=%le mp=%d node=%s\n",
    AppCfg->mode, AppCfg->np, AppCfg->nt, AppCfg->ng, AppCfg->ngb, AppCfg->ngt,
    AppCfg->t1, AppCfg->t2, AppCfg->t3, AppCfg->mp, AppCfg->pname);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}