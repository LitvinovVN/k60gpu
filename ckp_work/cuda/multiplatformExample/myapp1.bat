@del *.obj

@echo #define WIN_PLATFORM > myplatform.h

@call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x64
@set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0
@set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64
@set PATH=%CUDA_PATH%\open64\bin;%CUDA_PATH%\bin;%CUDA_PATH%\lib\x64;%PATH%
@set LIB=%CUDA_LIB_PATH%;%LIB%
@set LIBPATH=%CUDA_LIB_PATH%;%LIBPATH%
@set INCLUDE=%CUDA_PATH%\include;%INCLUDE%

nvcc.exe -arch sm_20 --ptxas-options=-v -c mygpu.cu

@set ND_INC=C:\Program Files\Microsoft HPC Pack 2008 SDK\Include\
@set path=C:\Program Files\Microsoft HPC Pack 2008 SDK\Bin;%path%
@set path=C:\Program Files\Microsoft HPC Pack 2008 SDK\NetworkDirect\Bin\amd64;%path%
@set path=C:\Program Files\Microsoft HPC Pack 2008 SDK\NetworkDirect\Bin\amd64\mpi;%path%
@set LIB=C:\Program Files\Microsoft HPC Pack 2008 SDK\Lib\amd64;%LIB%
@set LIBPATH=C:\Program Files\Microsoft HPC Pack 2008 SDK\Lib\amd64;%LIBPATH%
@set INCLUDE=C:\Program Files\Microsoft HPC Pack 2008 SDK\Include;%INCLUDE%

cl /O2 /openmp myapp.c mydef.c mycom.c mynet.c mycpu.c mygpu.obj msmpi.lib cudart.lib /link /OUT:myapp1.exe
