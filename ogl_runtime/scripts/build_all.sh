#!/bin/bash

bash scripts/build_protos.sh
if [ -f "build" ];then
    echo "--- build directory exist ---"
else
    mkdir build
fi
cd build && cmake .. && make -j`nproc`
