CUDA_C_PREF int MyCudaDevCount();
CUDA_C_PREF int MyCudaSetDev(int device);
CUDA_C_PREF int MyCudaProcess(int mp, int mt,
 int ngb, int ngt,
 int i1, int i2,
 double a, double h,
 double *s);