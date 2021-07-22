#!/usr/bin/python3
import subprocess
import time
import random
import fcntl
import os
import torch

'''
Wait until a gpu is available
to wait in the queue, you just need to use wait4FreeGPU to wait
until you get a gpu device id,
then you are supposed to invoke registerGPU,
if you get True response, feel free to use the GPU,
and if you failed to register (i.e. False returned),
then you have to wait4FreeGPU again.
'''

def get_free_mem_GPU():
    result = subprocess.check_output(['nvidia-smi', \
            '--query-gpu=memory.free,memory.used', \
            '--format=csv,nounits,noheader'])
    result = result.decode('utf-8')
    result = [[int(y) for y in x.strip().split(',')] \
            for x in result.strip().split('\n')]
    return result 

#def checkFreeGPU(mem_MB, used_MB, check_list=None):

def wait4FreeGPU(mem_MB, used_MB, \
        wall=0, interval=60, check_list=None):
    starttime = time.time()
    while wall<=0 or time.time() - starttime <= wall:
        mem = get_free_mem_GPU()
        if check_list is None:
            check_list = range(len(mem))
        for index in random.sample(check_list, len(check_list)):
            if mem[index][0] >= mem_MB and mem[index][1] <= used_MB:
                return index
        time.sleep(interval)
    return -1


class Locker(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.is_locked = False

    def lock(self):
        status = True
        if self.is_locked:
            # already locked
            return self.is_locked
        self.lock_file = open(self.file_name, 'w')
        try:
            fcntl.lockf(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            status = False
            self.lock_file.close()
        self.is_locked = status
        return status

    def lock_block(self):
        status = True
        if self.is_locked:
            # already locked
            return self.is_locked
        self.lock_file = open(self.file_name, 'w')
        try:
            fcntl.lockf(self.lock_file, fcntl.LOCK_EX)
        except IOError:
            status = False
            self.lock_file.close()
        self.is_locked = status
        return status


    def unlock(self):
        if self.is_locked:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.is_locked = False
        return True

def autoGPU():
    GPU_MEM_MIN = 8000 #MB
    GPU_MEM_USED_MAX = 100 #MB
    filelock_name='/tmp/waitGPU666.lock'
    INTERVAL = 30
    locker = Locker(filelock_name)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        check_list = os.environ["CUDA_VISIBLE_DEVICES"].strip().split(',')
        check_list = list(map(int, check_list))
        print('=> wait for ' + str(check_list) + ' only')
    else:
        check_list = None
    while True:
        GPU = wait4FreeGPU(GPU_MEM_MIN, GPU_MEM_USED_MAX, \
                wall=-1, interval=INTERVAL, check_list=check_list)
        if locker.lock():
            print('=> lock acquired, will use GPU ' + str(GPU))
            break
        print('=> failed to acquire lock, will try again soon')
        time.sleep(INTERVAL)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU)
    # occupy required gpu memory
    foo = torch.zeros((int(GPU_MEM_MIN*1024**2/4),), \
            dtype=torch.float32, device='cuda')
    del foo
    locker.unlock()
    print('=> lock released')

if __name__ == '__main__':
    autoGPU()
    print(torch.cuda.memory_allocated(), torch.cuda.memory_cached())
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(), torch.cuda.memory_cached())
    print('done')
