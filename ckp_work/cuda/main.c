#define MAIN_FILE 1
#define CUDA_C_PREF
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <mpi.h>
#include "cudaf.h"
// MPI error messages:
void mpierr(char *msg, int mp, const int n);
void mpierr(char *msg, int mp, const int n)
{
 fprintf(stderr,"Process %d message: %s\n",mp,msg);
 MPI_Abort(MPI_COMM_WORLD,n);
}
// Network:
static int np, mp, lname;
static char pname[MPI_MAX_PROCESSOR_NAME];
void MyRange(int np, int mp, int ia, int ib, int *i1, int *i2, int *nc);
void MyRange(int np, int mp, int ia, int ib, int *i1, int *i2, int *nc)
{
 if (np<2) { *i1=ia; *i2=ib; *nc=ib-ia+1; }
 else {
 int ni, mi, nn;
 nn = ib - ia + 1; ni = nn / np; mi = nn - ni * np;
 if (mp+1<=mi)
 { *i1 = ia + mp * (ni+1); *i2 = *i1 + ni; }
 else
 { *i1 = ia + mi * (ni+1) + (mp-mi) * ni; *i2 = *i1 + ni - 1; }
 *nc = *i2 - *i1 + 1;
 }
 return;
}
// Job parameters:
static int mode = 0;
static int ni = 1024 * 1024 * 1024;
static double a = 0;
static double b = 1;
static double h;
static double sum = 0;
double MyFun(double x);
double MyFun(double x)
{
 double c = cos(x);
 c = c * tan(x-0.25);
 c = c * tan(x-0.5);
 c = c * exp(-x);
 c = c * log(1.25+x);
 return 4.0 / (1.0 + x * x * (2.0 - c) / (2.0 + c));
}
// CPU & GPU threads:
typedef struct tag_data_t {
 int nt, mt;
 double *sum;
} data_t;
static int nt, ng, ngb, ngt, nt_max, nd_max; 

static data_t *ThreadDtArray;
static pthread_t *ThreadHnArray;
static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
void* myjobt(void* d);
void* myjobt(void* d)
{
 int err, i, i1, i2, nc, nn, mm;
 int k, kk = 20;
 double s = 0;
 data_t* dd = (data_t *)d;
 int nt = dd->nt;
 int mt = dd->mt;
// Subdomain:
 nn = np * nt;
 mm = mp * nt + mt;
 MyRange(nn,mm,1,ni,&i1,&i2,&nc);
 fprintf(stderr,"[%04d,%04d,%04d]: np=%d nt=%d nn=%d\n",mp,mt,mm,np,nt,nn);
 fprintf(stderr,"[%04d,%04d,%04d]: i1=%d i2=%d nc=%d\n",mp,mt,mm,i1,i2,nc);
 if (mode==0) {
 for (k=0; k<kk; k++)
 for (i=i1; i<=i2; i++) {
 double x = a + h * (1.0 * i - 0.5);
 s += MyFun(x) * h;
 }
 s /= (1.0*kk);
 }
 else {
 err = MyCudaProcess(mp,mt,ngb,ngt,i1,i2,a,h,&s);
 fprintf(stderr,"[%04d,%04d,%04d]: cuda compute retcode is %d\n",mp,mt,mm,err);
 }
 pthread_mutex_lock(&mut); // lock
 *dd->sum += s;
 pthread_mutex_unlock(&mut); // unlock
 return 0;
}
void ThreadWork();
void ThreadWork()
{
 int i;
 if (!(ThreadHnArray = (pthread_t*) malloc(nt*sizeof(pthread_t))))
 mpierr("Not enough memory",mp,1);
 if (!(ThreadDtArray = (data_t*) malloc(nt*sizeof(data_t))))
 mpierr("Not enough memory",mp,2);
 for (i=0; i<nt; i++){
 (ThreadDtArray+i)->nt=nt;
 (ThreadDtArray+i)->mt=i;
 (ThreadDtArray+i)->sum = &sum;
 if (pthread_create(ThreadHnArray+i,0,myjobt,(void*)(ThreadDtArray+i)))
 mpierr("Can not create thread",mp,3);
 }
 for (i=0; i<nt; i++)
 if (pthread_join(ThreadHnArray[i],0))
 mpierr("Can not close thread",mp,4); 

 free(ThreadHnArray);
 free(ThreadDtArray);
 return;
}
int MyGetCPUCount();
int MyGetCPUCount()
{
 int i = sysconf(_SC_NPROCESSORS_ONLN);
 return i;
}
int main(int argc, char *argv[])
{
 int i;
 double t0,t1,t2;
 MPI_Init(&argc, &argv);
 MPI_Barrier(MPI_COMM_WORLD);
 MPI_Comm_size(MPI_COMM_WORLD, &np);
 MPI_Comm_rank(MPI_COMM_WORLD, &mp);
 MPI_Get_processor_name(pname, &lname);
 MPI_Barrier(MPI_COMM_WORLD);
 nt_max = MyGetCPUCount();
 nd_max = MyCudaDevCount();
 fprintf(stderr,
 "Netsize: %d, process: %d, system: %s, cpu_count: %d, gpu_count: %d\n",
 np,mp,pname,nt_max,nd_max);
 MPI_Barrier(MPI_COMM_WORLD);
// if (nd_max > 0)
// MyCudaInfo(mp);
 if (nd_max<1 || nt_max<1)
 mpierr("Bad count of devices",mp,10);
 mode = 0; // Computation mode
 nt = 1; // Thread or GPU number
 ngb = 1; // GPU block number
 ngt = 1; // GPU total threads number
 if (mp==0)
 fprintf(stderr,
 "Usage: %s <mode> <cpu_threads> <gpu_blocks> <gpu_threads>\n",argv[0]);
 if (argc>1) {
 i = sscanf(argv[1],"%d",&mode);
 if (mode<1) mode = 0; // CPU calculations
 else mode = 1; // GPU calculations
 }
 if (argc>2) {
 i = sscanf(argv[2],"%d",&nt);
 if (nt<1) nt = 1;
 }
 if (argc>3) {
 i = sscanf(argv[3],"%d",&ngb);
 if (ngb<1) ngb = 1;
 }
 if (argc>4) {
 i = sscanf(argv[4],"%d",&ngt); 

 if (ngt<1) ngt = 1;
 }
 if (mode==0) {
 if (nt>nt_max) nt = nt_max;
 }
 else {
 if (nt>nd_max) nt = nd_max;
 }
 ng = ngb * ngt;
 h = (b-a) / ni;
 MPI_Barrier(MPI_COMM_WORLD);
 t0 = MPI_Wtime();
 ThreadWork();
 t1 = MPI_Wtime();
 if (np>1) {
 double sum0;
 MPI_Barrier(MPI_COMM_WORLD);
 sum0 = sum;
 MPI_Reduce(&sum0, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 t2 = MPI_Wtime();
 }
 else {
 t2 = t1;
 }
 fprintf(stderr,
 "mode=%d np=%d nt=%d ng=%d ngb=%d ngt=%d sum=%le "
 "t1=%le t2=%le t3=%le mp=%d node=%s\n",
 mode, np, nt, ng, ngb, ngt, sum, t1-t0, t2-t1, t2-t0, mp, pname);
 MPI_Barrier(MPI_COMM_WORLD);
 MPI_Finalize();
 return 0;
}