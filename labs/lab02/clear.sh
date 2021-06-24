clear

echo '------ Stopping task myapp.1: mcancel myapp.1 -------'
mcancel myapp.1
echo '------ Stopping task myapp.2: mcancel myapp.2 -------'
mcancel myapp.2
echo '------ Stopping task myapp.3: mcancel myapp.3 -------'
mcancel myapp.3

echo '------ Removing temporary files -------'
rm -r myapp.*
rm *.o myapp
echo '------ Temporary files removed ------'