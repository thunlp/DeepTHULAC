import functools
import os
import logging
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from munch import Munch
from deepthulac.seg.model import Seg
from deepthulac.utils import set_seed, set_logger, set_device, init_saved_path, load_yaml
from deepthulac.seg.data_format import *
from deepthulac.seg.eval import SegEvaluator
import warnings

warnings.filterwarnings('ignore')
set_seed(42)

if "LOCAL_RANK" in os.environ:
    # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 run.py
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run.py
    device_type = 'multigpu'
else:
    device_type = 'gpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

device, local_rank = set_device(device_type)

device_config = Munch()
device_config.device_type = device_type
device_config.local_rank = local_rank
device_config.device = device

data_dir = './data/seg'


def run_demo(model_dir):
    sentences = ['在石油化工发达的国家已大幅取代了乙炔水合法。', '这件和服务必于今日裁剪完毕']
    model = Seg.load(model_dir, device_config, use_f16=False)
    results = model.seg(sentences)
    print(results)


def run_test(model_dir):
    from sklearn.model_selection import train_test_split

    log_dir = model_dir + '/test.log'
    set_logger(log_dir)

    batch_size = 32
    model = Seg.load(model_dir, device_config, use_f16=False)
    logging.info(f"--Load model from {model_dir}")
    evaluator = functools.partial(SegEvaluator.eval_model, split_long=False)
    mode = model.mode

    dataset = format_trainset_by_special_token(f'{data_dir}/test.txt', mode, pair=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logging.info('testing pku')
    evaluator(dataloader, model, save_result=True, saved_path=model_dir, grain=0.5)

    dataset = format_testset(f'{data_dir}/msr_test.txt', mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logging.info('testing msr')
    evaluator(dataloader, model, save_result=True, saved_path=model_dir, grain=0.5)

    dataset = format_testset(f'{data_dir}/my_test.txt', mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logging.info('testing custom')
    evaluator(dataloader, model, save_result=True, saved_path=model_dir, grain=0.5)

    dataset = format_trainset_by_special_token(f'{data_dir}/train_corpus.txt', mode, sample_num=473686, pair=True)
    _, dataset = train_test_split(dataset, test_size=0.001, random_state=0)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    logging.info('testing corpus')
    evaluator(dataloader, model, save_result=True, saved_path=model_dir, grain=0.5)


def run_train():
    from sklearn.model_selection import train_test_split
    
    config = load_yaml('./config.yaml')
    saved_path = init_saved_path('./output/seg')
    log_dir = saved_path + '/train.log'
    set_logger(log_dir)

    batch_size = config.batch_size
    part_data = config.part_data
    mode = 'BMES' if len(config.labels) == 4 else 'EN'

    train_file = f'{data_dir}/train.txt'  # 'data/all_segment.txt'
    if part_data:
        samples = format_trainset_by_special_token(train_file, mode, 1600, pair=True)
        samples = samples[:1600]
        dev_split_size = 0.1
    else:
        samples = format_trainset_by_special_token(train_file, mode, 1581341_00, pair=True)  # 473686
        dev_split_size = 0.001
    samples_train, samples_dev = train_test_split(samples, test_size=dev_split_size, random_state=0)
    store_samples(samples_dev, mode, 'test_samples.txt')
    store_samples(samples_train, mode, 'train_samples.txt')

    if device_type == 'multigpu':
        train_loader = DataLoader(samples_train, batch_size=batch_size, sampler=DistributedSampler(samples_train, shuffle=False), num_workers=4)
    else:
        train_loader = DataLoader(samples_train, batch_size=batch_size, num_workers=4)
    dev_loader = DataLoader(samples_dev, batch_size=batch_size, num_workers=4)

    model = Seg(config, device_config)
    model = model.to(device)
    # model = BertSeg.load('path to model', device, use_f16=False))  # 继续训练
    evaluator = functools.partial(SegEvaluator.eval_model, split_long=False)
    model.fit(config, train_loader, dev_loader, saved_path, evaluator)
    if local_rank == 0:
        run_test(saved_path)
        run_demo(saved_path)
    # 上传到huggingface
    # model.quantize_float16()
    # model.save_to_hub('chengzl18/deepthulac-seg', organization='chengzl18', commit_message='test', exist_ok=True)


if __name__ == "__main__":
    run_train()