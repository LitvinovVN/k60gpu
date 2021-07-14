extern "C"
void testSum2Arrays(int mpi_rank, int mpi_size,
                    size_t cpuThreadsPerNode, size_t numElementsPerThread,
                    size_t numGpu, size_t nBlocks, size_t nThreads, size_t numElementsPerGpuThread);