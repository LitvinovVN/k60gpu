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

@set CCP_HOME=C:\Program Files\Microsoft HPC Pack 2012\
@set CCP_INC=C:\Program Files\Microsoft HPC Pack 2012 SDK\Include\
@set CCP_JOBTEMPLATE=Default
@set CCP_LIB32=C:\Program Files\Microsoft HPC Pack 2012 SDK\Lib\i386\
@set CCP_LIB64=C:\Program Files\Microsoft HPC Pack 2012 SDK\Lib\amd64\
@set CCP_SDK=C:\Program Files\Microsoft HPC Pack 2012 SDK\
@set MSMPI_INC=C:\Program Files\Microsoft HPC Pack 2012\Inc\
@set MSMPI_LIB32=C:\Program Files\Microsoft HPC Pack 2012\Lib\i386\
@set MSMPI_LIB64=C:\Program Files\Microsoft HPC Pack 2012\Lib\amd64\
@set LIB=C:\Program Files\Microsoft HPC Pack 2012\Lib\amd64;C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Lib\x64;%LIBPATH%
@set LIBPATH=C:\Program Files\Microsoft HPC Pack 2012\Lib\amd64;C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Lib\x64;%LIBPATH%
@set INCLUDE=C:\Program Files\Microsoft HPC Pack 2012\Inc;%INCLUDE%

cl /O2 /openmp myapp.c mydef.c mycom.c mynet.c mycpu.c mygpu.obj msmpi.lib cudart.lib /link /OUT:myapp2.exe
