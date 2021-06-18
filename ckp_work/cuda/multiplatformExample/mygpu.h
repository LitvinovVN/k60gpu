CUDA_C_PREF int MyGetGPUCount();

CUDA_C_PREF int MyGPUInfo(int rank);

CUDA_C_PREF int MyGPUSetDev(int device);

CUDA_C_PREF int MyGPUProcess(int mp, int mt,
                             int ngb, int ngt,
                             int i1, int i2,
                             double a, double h,
                             double *s);
