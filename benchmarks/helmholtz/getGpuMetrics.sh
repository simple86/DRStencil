cp ../gpuMetrics.csv .
for log in `ls prof`
do
    python getGpuMetrics.py ${log%.csv}
done
