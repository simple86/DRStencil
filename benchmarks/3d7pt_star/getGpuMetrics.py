import os
import sys

def main(stencilName=""):
	logfile = open('prof/'+str(stencilName)+'.csv', 'r')
	idx = -1
	counter = 0
    ## nsight-compute v2020.3
	metrics = ["DRAM Frequency","SM Frequency","Elapsed Cycles","Memory [%]","SOL DRAM","Duration","SOL L1/TEX Cache","SOL L2 Cache","SM Active Cycles","SM [%]","Executed Ipc Active","Executed Ipc Elapsed","Issue Slots Busy","Issued Ipc Active","SM Busy","Memory Throughput","Mem Busy","Max Bandwidth","L1/TEX Hit Rate","L2 Hit Rate","Mem Pipes Busy","One or More Eligible","Issued Warp Per Scheduler","No Eligible","Active Warps Per Scheduler","Eligible Warps Per Scheduler","Warp Cycles Per Issued Instruction","Warp Cycles Per Executed Instruction","Avg. Active Threads Per Warp","Avg. Not Predicated Off Threads Per Warp","Avg. Executed Instructions Per Scheduler","Executed Instructions","Avg. Issued Instructions Per Scheduler","Issued Instructions","Logical Links","Physical Links","Block Size","Function Cache Configuration","Grid Size","Registers Per Thread","Shared Memory Configuration Size","Driver Shared Memory Per Block","Dynamic Shared Memory Per Block","Static Shared Memory Per Block","Threads","Waves Per SM","Block Limit SM","Block Limit Registers","Block Limit Shared Mem","Block Limit Warps","Theoretical Active Warps per SM","Theoretical Occupancy","Achieved Occupancy","Achieved Active Warps Per SM","Branch Instructions Ratio","Branch Instructions","Branch Efficiency","Avg. Divergent Branches"]
	values = [stencilName]
	duration = 0.0
	#flag = False
	for line in logfile:
		#if flag == False:
		#	values.append('\"' + line.split()[2])
		#	flag = True
		if counter == len(metrics):
			counter = 0
		if line.find('Metric Value') > -1:
			idx = line.split(',').index('\"Metric Value\"')
		elif idx > -1 :
			seg = line.split('\",')
			if idx < len(seg) and seg[idx-2].find(metrics[counter]) > -1:
				values.append(seg[idx])
				if metrics[counter] == "Duration":
					duration = seg[idx][1:]
					durationLog = open('duration.log', 'a')
					durationLog.write(str(duration) + '\n')
					durationLog.close()

				counter += 1
	logfile.close()
	fileout = open('gpuMetrics.csv', 'a')
	fileout.write('\"')
	for val in values:
		fileout.write(str(val) + '",')
	fileout.write('\r\n')
	fileout.close()

if __name__ == '__main__':
	main(sys.argv[1])
