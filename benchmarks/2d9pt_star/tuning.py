import itertools
from functools import reduce
import os
import random
import sys
import pickle
import datetime
import time
maxThreadsPerBlockLg2 = 10 # 1024
maxShmPerBlockLg2 = 15 # 32K
order = 2

def FilterParams(spaceVector):
    step, dist, blockSize, streaming, sn, s_unroll, blockMergeX, mergeFactorX, blockMergeY, mergeFactorY, m_threshold, prefetch = spaceVector
    shmemUsage = 0
    if streaming:
        shmemUsage = (step * order + 1) * (mergeFactorX * blockSize[0])
    else:
        shmemUsage = (mergeFactorX * blockSize[0]) * (mergeFactorY * blockSize[1])
    ## using too much shared memory
    if shmemUsage > 2 ** (maxShmPerBlockLg2 - 3):
        return False
    ## dist too big or too small
    if dist > step * order or dist < (step - 1) * order:
        return False
    ## not covering halo region
    if step * order * 2 >= blockSize[0] * mergeFactorX:
        return False

    if streaming:
        if blockSize[1] > 1:
            return False
        if mergeFactorY > 1:
            return False
    else:
        if sn > 8 or s_unroll > 4:
            return False
        if blockSize[0] * blockSize[1] > 2 ** maxThreadsPerBlockLg2:
            return False
        if step * order * 2 >= blockSize[1] * mergeFactorY:
            return False

    if blockMergeX and mergeFactorX == 1:
        return False
    if blockMergeY and mergeFactorY == 1:
        return False
    return True


def cfgToCommandLine(spaceVector):
    step, dist, blockSize, streaming, sn, s_unroll, blockMergeX, mergeFactorX, blockMergeY, mergeFactorY, m_threshold, prefetch = spaceVector
    cmd = " --step {0} --dist {1} --bx {2}".format(step, dist, blockSize[0])

    if streaming:
        cmd += " --streaming --sn {0} --stream-unroll {1}".format(sn, s_unroll)
        if blockMergeY:
            cmd += " --block-merge-y {0}".format(mergeFactorY)
        else:
            cmd += " --cyclic-merge-y {0}".format(mergeFactorY)

    if blockMergeX:
        cmd += " --block-merge-x {0}".format(mergeFactorX)
    else:
        cmd += " --cyclic-merge-x {0}".format(mergeFactorX)
    cmd += " --merge-forward {0}".format(m_threshold)
    if prefetch and streaming:
        cmd += " --prefetch"

    return cmd


def cfgToString(spaceVector):
    step, dist, blockSize, streaming, sn, s_unroll, blockMergeX, mergeFactorX, blockMergeY, mergeFactorY, m_threshold, prefetch = spaceVector
    if streaming:
        cmd = "fu{0}d{1}bx{2}sn{3}u{4}".format(step, dist, blockSize[0], sn, s_unroll)
    else:
        cmd = "fu{0}d{1}bx{2}y{3}".format(step, dist, blockSize[0], blockSize[1])

    cmd += ("bmx" if blockMergeX else "cmx") + str(mergeFactorX)
    if not streaming:
        cmd += ("bmy" if blockMergeY else "cmy") + str(mergeFactorY)
    cmd += "mf{0}".format(m_threshold)
    if prefetch and streaming:
        cmd += "p"

    return cmd


def getElapsedTime (start, end):
    return (end - start).seconds + (end - start).microseconds / 1e6


def getMetrics(stencilName, startTime, best):
    logfile = open('prof/'+str(stencilName)+'.csv', 'r')
    idx = -1
    counter = 0
    for line in logfile:
        if line.find('Metric Value') > -1:
            idx = line.split(',').index('\"Metric Value\"')
        elif idx > -1 :
            seg = line.split('\",')
            if idx < len(seg) and seg[idx-2].find("Duration") > -1:
                duration = seg[idx][1:].replace(',','')
                if duration.isdigit():
                    if int(best) > int(duration):
                        best = int(duration)
                        durationLog = open('duration.log', 'a')
                        currentTime = datetime.datetime.now()
                        durationLog.write (str((currentTime-startTime).seconds) + ' s, ' + str(best) + '\n')
                        durationLog.close()
                counter += 1
                break

    logfile.close()
    return best


def searchSpace():

    startTime = datetime.datetime.now()
    best = 1e12
    paras = []

    blockSize = itertools.product([2**i for i in range(0, 10)], repeat=2)
    for paraVector in filter(FilterParams, itertools.product(
      [step for step in range (2, 3)], # time steps to fuse
      [dist for dist in range (1, 5)], # Dist
      filter(lambda sz: sz[0] * sz[1] < 2 ** maxThreadsPerBlockLg2, 
        blockSize), # blockSize
      [False, True], # streaming
      [2 ** sn for sn in range (3, 7)], # length of stream block
      [4, 8], # Stream unrollFactors
      [False, True], # cyclic(False) or block(True) merging for dimension x
      [2 ** mFactorLg2 for mFactorLg2 in range (0, 3)], # merge factor for dimension x
      [False, True], # cyclic(False) or block(True) merging for dimension y
      [2 ** mFactorLg2 for mFactorLg2 in range (0, 3)], # merge factor for dimension y
      [5], # threshold for merging forward
      [False, True], # prefetch
     )):
        paras.append (paraVector)

    random.shuffle (paras)
    cnt = 0
    for paraVector in paras:
        cnt += 1
        config = cfgToCommandLine(paraVector)
        conf_str = cfgToString(paraVector)
        ## .stc varies from stencils
        cmd = " ".join(["./drstencil", \
          config, " --check -o ./cu/"+conf_str+".cu 2d9pt_star.stc"])
        os.system(cmd)
        print ("{0}/{1}: {2}".format(cnt, len(paras), conf_str))
        os.system("./compile_run.sh " + conf_str)
        best = getMetrics (conf_str, startTime, best)

        currentTime = datetime.datetime.now()
        if (currentTime-startTime).seconds > 3600:
            break


    durationLog = open('duration.log', 'a')
    currentTime = datetime.datetime.now()
    durationLog.write (str((currentTime-startTime).seconds) + ' s, ' + str(best) + '\n')
    durationLog.close()

if __name__ == '__main__':
    searchSpace()
