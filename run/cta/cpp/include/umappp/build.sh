git clone https://github.com/libscran/umappp.git
cd umappp
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DUMAPPP_TESTS=OFF
cmake --build . --target install
