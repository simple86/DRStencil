import itertools
from functools import reduce
import os
import sys
import pickle
import subprocess
maxThreadsPerBlockLg2 = 10 # 1024

def FilterParams(spaceVector):
    order = 2 
    step, dist, blockSize, sn, s_unroll, m_threshold = spaceVector
    if dist > step * order or dist < (step - 1) * order:
        return False
    if step * order * 2 >= min(blockSize[0], blockSize[1]):
        return False
    return True

def cfgToCommandLine(spaceVector):
    step, dist, blockSize, sn, s_unroll, m_threshold = spaceVector
    cmd = " --bx {0} --by {1} --sn {2} --stream-unroll {3}".format(blockSize[0], blockSize[1], sn, s_unroll)
    cmd += " --step {0} --dist {1}".format(step, dist)
    cmd += " --merge-forward {0}".format(m_threshold)
    return cmd

def cfgToString(spaceVector):
    step, dist, blockSize, sn, s_unroll, m_threshold = spaceVector
    cmd = "fu{5}d{6}bx{0}y{1}sn{2}u{3}m{4}".format(blockSize[0], blockSize[1], sn, s_unroll, m_threshold, step, dist)
    return cmd

def searchSpace():

    blockSize = itertools.product([2**i for i in range(4, maxThreadsPerBlockLg2)], repeat=2)
    for paraVector in filter(FilterParams, itertools.product(
      [step for step in range (2, 4)], # time steps to fuse
      [dist for dist in range (3, 7)], # Dist
      filter(lambda x: x[0] * x[1] <= 2 ** (maxThreadsPerBlockLg2),
        blockSize), # block size
      [2 ** snlg2 for snlg2 in range (3, 6)], # length of stream block
      [4], # Stream unrollFactors
	  [4], # threshold for merging forward
     )):
        config = cfgToCommandLine(paraVector)
        conf_str = cfgToString(paraVector)
        ## .stc varies from stencils
        cmd = " ".join(["./fsstencil", \
          config, " --check -o ./cu/"+conf_str+".cu j3d7pt.stc >> /dev/null"])
        os.system(cmd)

if __name__ == '__main__':
    searchSpace()
