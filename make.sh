rm -rf build bin
mkdir build
cd build
cmake ../
cmake --build . --config Release
cd ..