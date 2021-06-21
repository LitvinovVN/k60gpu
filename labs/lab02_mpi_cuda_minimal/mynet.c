#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "mpi.h"
#include "mynet.h"

void MyNetErr(const char *msg, int mp, const int n)
{
  fprintf(stderr,"Process %d message: %s\n",mp,msg);
  MPI_Abort(MPI_COMM_WORLD,n);
}

int MyNetInit(int* argc, char*** argv, int* np, int* mp,
              int* nl, char* pname, double* tick)
{
  int i;

  i = MPI_Init(argc,argv);
  if (i != 0){
    fprintf(stderr,"MPI initialization error");
    exit(i);
  }

  MPI_Comm_size(MPI_COMM_WORLD,np);
  MPI_Comm_rank(MPI_COMM_WORLD,mp);
  MPI_Get_processor_name(pname,nl);

  *tick = MPI_Wtick();

  return 0;
}

void MyRange(int np, int mp, const int ia, const int ib, int *i1, int *i2, int *nc)
{
  if (np<2) {
    *i1=ia; *i2=ib; *nc=ib-ia+1;
  }
  else {
    int ni, mi, nn;
    nn = ib - ia + 1; ni = nn / np;  mi = nn - ni * np;
    if (mp+1<=mi)
      { *i1 = ia + mp * (ni+1); *i2 = *i1 + ni; }
    else
      { *i1 = ia + mi * (ni+1) + (mp-mi) * ni; *i2 = *i1 + ni - 1; }
    *nc = *i2 - *i1 + 1;
  }
  return;
}

void MyIndexes(int np, const int ia, const int ib, int* ind)
{
  int mp, i1, i2, nc;

  for (mp=0; mp<np; mp++) {
    MyRange(np,mp,ia,ib,&i1,&i2,&nc);
    ind[3*mp  ] = i1;
    ind[3*mp+1] = i2;
    ind[3*mp+2] = nc;
  }

  return;
}

void My2DGrid(int np, int mp, int n1, int n2,
              int *np1, int *np2, int *mp1, int *mp2)
{
  if (np<2) {
    *np1 = 1; *np2 = 1;
    *mp1 = 0; *mp2 = 0;
  }
  else {
    int i, j, k;
    double s = sqrt(((double)np));

    if (n1 <= n2) {
      k = (int)floor(s*((double)n1/(double)n2));
      for (i=1; i<=k+1; i++)
        for (j=1; j<=np; j++)
          if (i*j == np) {
            if (i<j) {*np1 = i; *np2 = j;}
            else     {*np1 = j; *np2 = i;}
          }
    }
    else {
      k = (int)floor(s*((double)n2/(double)n1));
      for (i=1; i<=k+1; i++)
        for (j=1; j<=np; j++)
          if (i*j == np) {
            if (i<j) {*np2 = i; *np1 = j;}
            else     {*np2 = j; *np1 = i;}
          }
    }

    *mp2 = mp / (*np1);
    *mp1 = mp % (*np1);
  }

  return;
}

int BndAExch1D(int mp_l, int nn_l, double* ss_l, double* rr_l,
               int mp_r, int nn_r, double* ss_r, double* rr_r)
{
  int i, n, m;
  static MPI_Status Sta[4];
  static MPI_Request Req[4];
  MPI_Request *R;

  R = Req;
  m = 0;
  n = 0;

  if (mp_l>=0) {
    MPI_Irecv(rr_l,nn_l,MPI_DOUBLE,mp_l,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_r>=0) {
    MPI_Irecv(rr_r,nn_r,MPI_DOUBLE,mp_r,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_l>=0) {
    MPI_Isend(ss_l,nn_l,MPI_DOUBLE,mp_l,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_r>=0) {
    MPI_Isend(ss_r,nn_r,MPI_DOUBLE,mp_r,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (m>0) {
    MPI_Waitall(m,Req,Sta);

    for (i=0; i<m; i++)
      if (Sta[i].MPI_ERROR != 0) n++;
  }

  return n;
}

int BndAExch2D(int mp_l, int nn_l, double* ss_l, double* rr_l,
               int mp_r, int nn_r, double* ss_r, double* rr_r,
               int mp_b, int nn_b, double* ss_b, double* rr_b,
               int mp_t, int nn_t, double* ss_t, double* rr_t)
{
  int i, n, m;
  static MPI_Status Sta[8];
  static MPI_Request Req[8];
  MPI_Request *R;

  R = Req;
  m = 0;
  n = 0;

  if (mp_l>=0) {
    MPI_Irecv(rr_l,nn_l,MPI_DOUBLE,mp_l,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_r>=0) {
    MPI_Irecv(rr_r,nn_r,MPI_DOUBLE,mp_r,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_b>=0) {
    MPI_Irecv(rr_b,nn_b,MPI_DOUBLE,mp_b,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_t>=0) {
    MPI_Irecv(rr_t,nn_t,MPI_DOUBLE,mp_t,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++; 
  }

  if (mp_l>=0) {
    MPI_Isend(ss_l,nn_l,MPI_DOUBLE,mp_l,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_r>=0) {
    MPI_Isend(ss_r,nn_r,MPI_DOUBLE,mp_r,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (mp_b>=0) {
    MPI_Isend(ss_b,nn_b,MPI_DOUBLE,mp_b,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++; 
  }

  if (mp_t>=0) {
    MPI_Isend(ss_t,nn_t,MPI_DOUBLE,mp_t,MY_TAG,MPI_COMM_WORLD,R);
    R++; m++;
  }

  if (m>0) {
    MPI_Waitall(m,Req,Sta);

    for (i=0; i<m; i++)
      if (Sta[i].MPI_ERROR != 0) n++;
  }

  return n;
}
