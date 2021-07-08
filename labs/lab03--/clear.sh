clear

echo '------ Stopping task myapp.1: mcancel myapp.1 -------'
mcancel myapp3.1
echo '------ Stopping task myapp.2: mcancel myapp.2 -------'
mcancel myapp3.2
echo '------ Stopping task myapp.3: mcancel myapp.3 -------'
mcancel myapp3.3

echo '------ Removing temporary files -------'
rm -r myapp3.*
rm *.o myapp3
echo '------ Temporary files removed ------'