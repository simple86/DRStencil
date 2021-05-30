cp ../gpuMetrics.csv .
for log in `ls prof`
do
    python3 getGpuMetrics.py ${log%.csv}
done
