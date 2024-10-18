#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Error: Not enough arguments provided."
    echo "Usage: $0 [run|debug] [run|comiple]"
    exit 1
fi

cd experiments

if [ "$2" == "compile" ]; then

    nvcc -I /usr/include/eigen3 -O3 -g -lineinfo -rdc=true -lcudadevrt -Xcompiler -fpermissive -Xcompiler -msse4.2 -arch=sm_75  leco_int_template.cu ../src/bpacking.cu ../src/varintencode.cu ../src/varintdecode.cu -lgmp -o leco_int_template_cu
fi

if [ "$1" == "run" ]; then

    ./leco_int_template_cu books_200M_uint32 100000 6 8 1 32

elif [ "$1" == "debug" ]; then 

    compute-sanitizer --tool memcheck ./leco_int_template_cu books_200M_uint32 100000 6 8 1 32

fi

