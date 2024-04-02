import multiprocessing
import time
import subprocess

import os

env = {}

env['DMLC_PS_ROOT_URI'] = '127.0.0.1'
env['DMLC_PS_ROOT_PORT'] = '9092'
env['DMLC_NUM_SERVER'] = '1'
env['DMLC_NUM_WORKER'] = '1'
env['SCALING_TIMES'] = '1'
env['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

script = os.path.join(os.path.dirname(__file__), 'cifar10_dist.py')
cmd = 'python ' + script


MAX_SCALING_TIMES = 5

def run(env):
    for k,v in env.items():
        os.environ[k] = v
    if env['DMLC_ROLE'] == 'worker':
        os.system(cmd)
    else:
        os.system('python -c "import mxnet"')
    

for i in range(MAX_SCALING_TIMES):
    env['SCALING_TIMES'] = str(i)
    # load schedule
    env['DMLC_ROLE'] = 'scheduler'
    multiprocessing.Process(target=run, args=(env,)).start()
    
    # load server
    env['DMLC_ROLE'] = 'server'
        
    for j in range(int(env['DMLC_NUM_SERVER'])):
        multiprocessing.Process(target=run, args=(env,)).start()
        
    # load worker
    env['DMLC_ROLE'] = 'worker'
    p_worker = []
    for j in range(int(env['DMLC_NUM_WORKER'])):
        os.environ['WORKER_NAME'] = 'WORKER_' + str(j)
        p = multiprocessing.Process(target=run, args=(env,))
        p.start()
        p_worker.append(p)
    
    for p in p_worker:
        p.join()
    
    for p in multiprocessing.active_children():
        p.terminate()
    
    env['DMLC_NUM_WORKER'] = str(int(env['DMLC_NUM_WORKER']) + 1)

    
    
    