import logging
import torch
import numpy as np
import random
import os
import json
import yaml
from munch import Munch
from datetime import datetime
from functools import wraps
import time
import colorama
from colorama import Fore, Style
colorama.init()


def timer(func):
    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        args = [str(arg) for arg in args if type(arg) in {int, float, str}]
        args = ', '.join(args)
        print(Fore.GREEN+'Timer: '+Style.RESET_ALL+f'{func.__name__}({args}) {total_time:.2f}s')
        return result
    return timer_wrapper


def init_saved_path(path):
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = path + '/' + name
    os.makedirs(model_save_path, exist_ok=True)
    return model_save_path


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 输出到文件
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # 输出到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_device(device_type):
    local_rank = 0
    if device_type == 'multigpu':
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif device_type == 'gpu':
        device = "cuda:0"
    else:
        device = torch.device("cpu")
    return device, local_rank


def load_lines(file, remove_empty=True):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    if remove_empty:
        lines = list(filter(None, lines))
    return lines


def store_lines(lines, file):
    with open(file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def store_json(obj, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj


def store_yaml(obj, file):
    with open(file, 'w') as f:
        yaml.safe_dump(dict(obj), f)


def load_yaml(file):
    with open(file, 'r') as f:
        obj = yaml.safe_load(f)
    obj = Munch.fromDict(obj)
    return obj


"""简易的并行处理"""
try:
    gpu_num = os.environ['CUDA_VISIBLE_DEVICES'].count(',')+1
except Exception:
    gpu_num = 8
use_gpu = True

pbars = []  # 并行处理的进度条


def equal_split(all_items, all_ids, n, split_manner):
    """将all_items, all_ids均分成块，每块的形式都是[(id,item),...]"""
    total = len(all_items)
    if all_ids == None:
        all_ids = list(range(total))
    assert split_manner in ['chunk', 'turn']
    if split_manner == 'chunk':
        indices = np.array_split(range(total), n)
    elif split_manner == 'turn':
        indices = [list(range(total)[i::n]) for i in range(n)]
    items = [[(all_ids[i], all_items[i]) for i in inds] for inds in indices]  # (id, doc)
    return items


def single_run_cpu(func, docs):
    """用func逐一处理docs"""
    docs, pos = docs
    # pos = multiprocessing.current_process()._identity[0] - 1  # 0,1,2..., 但每次运行的Pool都会累加
    pbar = pbars[pos]
    res = []
    for id, doc in docs:
        res.append(func(doc))
        pbar.update()
    return res


def single_run_infer(use_gpu, func, docs):
    """先用func初始化模型，再将docs分块，调用func去处理"""
    # func(model/device, is, docs)
    # docs: [(id, doc)]
    docs, pos = docs
    pbar = pbars[pos]
    res = []
    if use_gpu:
        device = torch.device("cuda:"+str(pos))
    else:
        device = torch.device("cpu")
    model = func(device, None)

    if len(docs) < 100:
        split_docs = equal_split([d[1] for d in docs], [d[0] for d in docs], len(docs), split_manner='chunk')
    else:
        split_docs = equal_split([d[1] for d in docs], [d[0] for d in docs], 100, split_manner='chunk')
    for sub_docs in split_docs:
        sub_docs, sub_ids = [d[1] for d in sub_docs], [d[0] for d in sub_docs]
        pred = func(model, sub_docs)
        res.extend(pred)
        pbar.update()
    return res


def parallel_run(func, all_docs, num_proc=8, split_manner='chunk'):  # all_doc是一个list, func对一个doc做处理
    """并行处理

    Args:
        func: 对一个doc做处理的函数
        all_docs: 所有需要被处理的doc
        num_proc: 进程数量
        split_manner: chunk/turn 分块分配/轮流分配

    Return:
        当func不是模型批量处理时，返回结果等价于 [func(doc) for doc in all_docs]
        当func是模型批量处理时，返回结果类似，只不过func是批量处理的
    """
    import functools
    from tqdm import tqdm
    import multiprocessing
    global pbars
    num_proc = min(num_proc, len(all_docs))
    if use_gpu:
        num_proc = min(num_proc, gpu_num)
    split_docs = equal_split(all_docs, None, num_proc, split_manner)

    ori_func = func.func if isinstance(func, functools.partial) else func
    if ori_func.__code__.co_varnames[0] == 'model_or_device':
        single_run = functools.partial(single_run_infer, use_gpu)
        pbars = [tqdm(total=min(100, len(docs)), desc=str(docs[0][0]).rjust(5, '0'), position=pos) for pos, docs in enumerate(split_docs)]
    else:
        single_run = single_run_cpu
        pbars = [tqdm(total=len(docs), desc=str(docs[0][0]).rjust(5, '0'), position=pos, mininterval=1.0) for pos, docs in enumerate(split_docs)]

    results = []
    with multiprocessing.Pool(num_proc) as p:
        pids = list(range(num_proc))
        assert len(pids) == len(split_docs)
        for single_res in p.imap(functools.partial(single_run, func), list(zip(split_docs, pids))):
            results.extend(single_res)
    print('\n'*num_proc+'PARALLEL FINISH')
    return results
