@call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x64
@set ND_INC=C:\Program Files\Microsoft HPC Pack 2008 SDK\Include\
@set path=C:\Program Files\Microsoft HPC Pack 2008 SDK\Bin;%path%
@set path=C:\Program Files\Microsoft HPC Pack 2008 SDK\NetworkDirect\Bin\amd64;%path%
@set path=C:\Program Files\Microsoft HPC Pack 2008 SDK\NetworkDirect\Bin\amd64\mpi;%path%
@set LIB=C:\Program Files\Microsoft HPC Pack 2008 SDK\Lib\amd64;%LIB%
@set LIBPATH=C:\Program Files\Microsoft HPC Pack 2008 SDK\Lib\amd64;%LIBPATH%
@set INCLUDE=C:\Program Files\Microsoft HPC Pack 2008 SDK\Include;%INCLUDE%
mpiexec %1 %2 %3 %4 %5 %6 %7 %8 %9
