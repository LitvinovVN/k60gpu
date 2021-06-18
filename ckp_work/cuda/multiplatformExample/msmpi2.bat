@call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x64
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
mpiexec %1 %2 %3 %4 %5 %6 %7 %8 %9
