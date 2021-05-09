#!/usr/bin/bash
count=0
dev_num=2
for file in `ls cu | grep cu`
do
    count=$((count + 1))
    name=${file%.cu}
    if [ `find prof -name "${name}.csv" | wc -l` -eq 0 ]
    then 
        echo "${count}: ${name}"
        nvcc cu/${file} -O3 -arch=sm_80 -o bin/${name} -ccbin=g++ -std=c++11 -Xcompiler "-fPIC -fopenmp -O3 -fno-strict-aliasing" --use_fast_math -Xptxas "-dlcm=cg" 
        CUDA_VISIBLE_DEVICES=$((count % dev_num)) ncu --kernel-id ::j3d7pt:2 --csv --set full bin/${name} > prof/${name}.csv 
    fi
done

wait
#bash getGpuMetrics.sh
