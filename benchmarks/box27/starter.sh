#!/usr/bin/bash
starttime=`date +'%Y-%m-%d %H:%M:%S'`

mkdir cu bin prof
python tuning.py
cp common.hpp cu/
./compile_run.sh
bash getGpuMetrics.sh

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
echo ${endtime} >> tuning-time.log
echo "running time: "$((end_seconds-start_seconds))"s" >> tuning-time.log
