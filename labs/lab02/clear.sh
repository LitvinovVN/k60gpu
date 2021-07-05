clear

echo '------ Stopping task myapp.1: mcancel myapp.1 -------'
mcancel myapp2.1
echo '------ Stopping task myapp.2: mcancel myapp.2 -------'
mcancel myapp2.2
echo '------ Stopping task myapp.3: mcancel myapp.3 -------'
mcancel myapp2.3

echo '------ Removing temporary files -------'
rm -r myapp2.*
rm *.o myapp2
echo '------ Temporary files removed ------'