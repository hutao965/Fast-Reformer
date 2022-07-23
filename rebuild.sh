rm bin/Release/*
cd build
rm -rf CMakeFiles/
cmake --build . --config Release
cd ..