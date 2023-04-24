import argparse
import functools
import math
from deepthulac.seg.model import LacModel
from deepthulac.utils import init_distributed, set_seed, set_logger, init_saved_path, load_yaml, DistributedInfo
from deepthulac.seg.data_format import *
from deepthulac.seg.eval import SegEvaluator
from deepthulac.eval.test import run_test
from deepthulac.utils import log, print_green
import warnings
import sys

""" Training examples:
CUDA_VISIBLE_DEVICES=2 python deepthulac/seg/train.py
accelerate launch --gpu_ids 0 --num_processes 1 --mixed_precision fp16 deepthulac/seg/train.py
accelerate launch --gpu_ids 6,7 --num_processes 2 --mixed_precision bf16 deepthulac/seg/train.py
accelerate launch --gpu_ids 2,3,4,5 --num_processes 4 --mixed_precision bf16 deepthulac/seg/train.py
accelerate launch --gpu_ids 0,1,2,3,4,5 --num_processes 6 --mixed_precision fp16 deepthulac/seg/train.py
"""
warnings.filterwarnings('ignore')
dinfo = init_distributed()
set_seed(42)


def run_train(config_file):
    config = load_yaml(config_file)
    if dinfo.is_main:
        saved_path = init_saved_path(config.saved_path)
        print_green(saved_path)
        log_dir = saved_path + '/train.log'
        set_logger(log_dir)
        logging.info(' '.join(sys.argv[:]))
    else:
        saved_path = None

    data_strategy = getattr(config, 'data_strategy', 'shuffle_batches')
    train_loaders = []
    batch_order, loop_times, repeat_times = [], 0, []
    for i, dataset in enumerate(config.train_datasets):
        dataset_config = config.train_datasets[dataset]
        dataset_config.name = dataset
        dataset_config.split = 'train'
        dataset_config.samples_num = 1600 if config.part_data else 0
        train_loader = build_dataloader(dataset_config, config.heads, batch_size=config.batch_size)
        train_loaders.append(train_loader)
        if data_strategy=='continuous_batches':
            continuous_batches = dataset_config.continuous_batches
            batch_order.extend([i]*continuous_batches)
            loop_times = max(loop_times, math.ceil(len(train_loader)/continuous_batches))
        elif data_strategy=='shuffle_batches':
            batch_order.extend([i]*len(train_loader)*dataset_config.repeat_times)
        elif data_strategy=='shuffle_samples':
            repeat_times.append(dataset_config.repeat_times)
    if data_strategy == 'continuous_batches':
        batch_order = batch_order * loop_times
    elif data_strategy == 'shuffle_batches':
        random.shuffle(batch_order)
    elif data_strategy == 'shuffle_samples':
        train_loaders = [build_mixed_dataloader(train_loaders, repeat_times)]
        batch_order = [0]*len(train_loaders[0])

    dev_loaders = []
    for dataset in config.dev_datasets:
        dataset_config = config.dev_datasets[dataset]
        dataset_config.name = dataset
        dataset_config.split = 'test'
        dataset_config.samples_num = 60 if config.part_data else 120  # 这个会影响训练过程中的eval和最终eval结果不一样的问题
        dev_loader = build_dataloader(dataset_config, config.heads, batch_size=config.batch_size)
        dev_loaders.append(dev_loader)

    log('loading model')
    
    if hasattr(config, 'load_seg_pretrain'):
        load_path = config.load_seg_pretrain # +'/'+config.saved_path.split('/')[-1]
        if not os.path.exists(load_path):
            seg_pretrain = LacModel.load(config.load_seg_pretrain)
            model = LacModel(config, DistributedInfo(device='cpu'), config.pretrained_bert_model)
            model.bert = seg_pretrain.bert
            model.seg_head = seg_pretrain.seg_head
            model.heads['seg'] = model.seg_head
            model.save(load_path)
        model = LacModel.load(load_path, dinfo.device)
        model.model_config = config
    else:
        model = LacModel(config, dinfo, config.pretrained_bert_model)
        

    # model = LacModel.load(path, 'cuda:0', use_f16=False) # 加载训练好的模型
    evaluator = functools.partial(SegEvaluator.eval_model, split_long=False)
    log('training')

    model.fit(config, train_loaders, dev_loaders, saved_path, evaluator, batch_order)

    def deepthulac_api(task, sents):
        # TODO: split_long的影响非常巨大，要把split_long的逻辑改一下，让split之后尽量长
        return model.seg(sents, batch_size=64, split_long=False, post_vmvd=False)[task]['res']  # seg_pos头也一样从pos里拿结果即可

    if dinfo.is_main:
        can_pos = 'pos' in model.heads or 'seg_pos' in model.heads
        run_test(deepthulac_api, saved_path+'/test.log', can_pos)
    # 上传到huggingface
    # model.quantize_float16()
    # model.save_to_hub('chengzl18/deepthulac-seg', organization='chengzl18', commit_message='test', exist_ok=True)


if __name__ == "__main__":
    CONFIG_PATH = './deepthulac/seg/configs'
    DEFAULT_CONFIG_FILE = 'bound_seg/seg_punc_dict.yaml'
    # DEFAULT_CONFIG_FILE = 'seg/seg_punc_dict.yaml'

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_path', default=CONFIG_PATH+'/'+DEFAULT_CONFIG_FILE, type=str)
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    run_train(args.config_path)
