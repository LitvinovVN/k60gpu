extern "C"
void thread_proc(int tnum, int rank);

extern "C"
void testThreads(int rank);

extern "C"
void sum2Arrays(double* a, double* b, double* c_par);