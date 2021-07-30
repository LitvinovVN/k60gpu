clear

echo '------ Stopping task myapp6.1: mcancel myapp6.1 -------'
mcancel myapp6.1

echo '------ Removing temporary files -------'
rm -r myapp6.*
rm *.o myapp6
echo '------ Temporary files removed ------'