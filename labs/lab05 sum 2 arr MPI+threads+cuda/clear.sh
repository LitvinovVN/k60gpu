clear

echo '------ Stopping task myapp5.1: mcancel myapp5.1 -------'
mcancel myapp5.1
echo '------ Stopping task myapp5.2: mcancel myapp5.2 -------'
mcancel myapp5.2
echo '------ Stopping task myapp5.3: mcancel myapp5.3 -------'
mcancel myapp5.3

echo '------ Removing temporary files -------'
rm -r myapp5.*
rm *.o myapp5
echo '------ Temporary files removed ------'