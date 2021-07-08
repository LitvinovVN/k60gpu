clear

echo '------ Stopping task myapp3.1: mcancel myapp3.1 -------'
mcancel myapp3.1
echo '------ Stopping task myapp3.2: mcancel myapp3.2 -------'
mcancel myapp3.2
echo '------ Stopping task myapp3.3: mcancel myapp3.3 -------'
mcancel myapp3.3

echo '------ Removing temporary files -------'
rm -r myapp3.*
rm *.o myapp3
echo '------ Temporary files removed ------'