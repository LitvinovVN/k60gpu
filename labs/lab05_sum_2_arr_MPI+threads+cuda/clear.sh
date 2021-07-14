clear

echo '------ Stopping task myapp5.1: mcancel myapp4.1 -------'
mcancel myapp5.1

echo '------ Removing temporary files -------'
rm -r myapp5.*
rm *.o myapp5
echo '------ Temporary files removed ------'