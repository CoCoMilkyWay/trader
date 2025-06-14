#!/bin/bash

set -e

# rm -rf build

mkdir -p build
cd build

if [ ! -f build.ninja ] && [ ! -f Makefile ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

if [ -f build.ninja ]; then
    ninja
elif [ -f Makefile ]; then
    make -j$(nproc)
else
    echo "No known build system found."
    exit 1
fi

objdump -p Pipeline.dll | grep "DLL Name"

cp Pipeline.dll ../Pipeline.pyd
