import inspect
import math
import re
import paddle
import os
import signal
import socket
import subprocess
import sys
import random
from logging import getLogger
import paddle
import torch
from paddle.distributed import fleet
import logging
import numpy as np
import pickle
from logger import create_logger


logger = getLogger()

def get_optimizer(parameter_list, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}
        optim_params['lr'] = 0.00001
    # AdamOptimizer
    if method == 'adam':
        optim_fn = paddle.optimizer.Adam
    elif method == 'sgd':
        optim_fn = paddle.optimizer.SGD
    elif method == 'adamW':
        optim_fn = paddle.optimizer.AdamW
    # SGD
    else:
        raise Exception("We only support sgd and adam now. Feel free to add yours!")
    assert 'lr' in optim_params
    clip = paddle.nn.ClipGradByNorm(clip_norm=5.0)
    return optim_fn(learning_rate=optim_params['lr'], parameters=parameter_list, grad_clip=clip)



def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        logger.warning("Not the master process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    logger.warning("Signal handler installed.")


def init_distributed_mode(params):
    """
    Handle single/multi node multi-card case. Not sure if paddle supports slurm environment.
    """
    # multi-GPU job (local or multi-node)
    if params.multi_gpu:
        # read environment variables
        params.gpu_id = int(os.getenv("FLAGS_selected_gpus", "-1"))
        assert params.gpu_id >= 0, "gpu_id is not correctly set from env vars!"
        paddle.set_device("gpu")
    # local job (single GPU)
    elif torch.cuda.is_available():
        params.gpu_id = 0
        paddle.set_device("gpu")
    else:
        params.gpu_id = -1

    # summary
    print("gpu_id: %i" % params.gpu_id)
    params.is_master = (params.gpu_id == 0) or (params.gpu_id == -1)
    # initialize multi-GPU
    if params.multi_gpu and paddle.distributed.get_world_size() > 1:
        # paddle.distributed.init_parallel_env()
        print("Initializing PaddlePaddle distributed ...")
        role = fleet.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

def initialize_exp(params):
    """
    Initialize the experiment:
    - dump parameters
    - create a logger
    """
    # # dump parameters
    # get_dump_path(params)
    # pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    assert params.qlm_steps is not None or params.rr_steps is not None
    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


#############################################
#
# Below are helper functions for finetuning
#
#############################################

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def format_seconds(seconds):
    t = float(seconds)
    hour = int(t // 3600)
    t %= 3600
    minutes = int(t // 60)
    t %= 60
    seconds = int(t)
    return f"{hour:02}:{minutes:02}:{seconds:02}"


class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.adjustedTime = format_seconds(record.relativeCreated / 1000)
        return super(CustomFormatter, self).format(record)
