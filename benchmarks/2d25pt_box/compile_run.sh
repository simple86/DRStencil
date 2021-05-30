#!/bin/bash
ARCH=sm_80
name=$1
nvcc cu/${name}.cu -maxrregcount=128 -arch=${ARCH} -O3 -o bin/${name} -ccbin=g++ -std=c++11 -Xcompiler "-fPIC -fopenmp -O3 -fno-strict-aliasing" --use_fast_math -Xptxas "-dlcm=cg" 
CUDA_VISIBLE_DEVICES=1 ncu --kernel-id ::dr_2d25pt_box:10 --csv --set full bin/${name} > prof/${name}.csv 

