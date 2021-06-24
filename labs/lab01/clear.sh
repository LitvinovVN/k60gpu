clear

echo '------ Stopping task myapp.1: mcancel myapp.1 -------'
mcancel myapp.1
echo '------ Removing temporary files -------'
rm -r myapp.1
rm *.o myapp
echo '------ Temporary files removed ------'