from deepthulac.seg.seg_utils import *
from deepthulac.utils import load_lines, store_json, store_lines
from deepthulac.seg.model import LacModel
import logging
from deepthulac.utils import set_logger
from deepthulac.utils import load_lines, store_lines, timer
from typing import Callable
import re
from deepthulac.eval.cases import *


def f1_score(pred_spans, true_spans):
    """ 通过span列表计算指标 """
    # span是前闭后闭的闭区间
    # 每个span前加上句子编号，构成唯一标识
    y_pred, y_true = [], []
    for i, ys in enumerate(pred_spans):
        y_pred.extend([(i, y) for y in ys])  # y[0], y[1] -> y
    for i, ys in enumerate(true_spans):
        y_true.extend([(i, y) for y in ys])
    y_pred, y_true = set(y_pred), set(y_true)
    n_pred, n_true, n_correct = len(y_pred), len(y_true), len(y_pred & y_true)
    p = n_correct / n_pred if n_pred > 0 else 0
    r = n_correct / n_true if n_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return score, p, r


def accuracy(pred_labels, true_labels):
    acc, total = 0, 0
    for p, t in zip(pred_labels, true_labels):
        for pc, tc in zip(p, t):
            if pc == tc:
                acc += 1
            total += 1
    return acc/total


def eval_res(fileformat, pred_res, true_res, gi, pred_labels=None):
    """从运行结果和正确结果计算指标"""

    if fileformat in {'seg', 'punc', 'pos'}:
        if gi:
            return gi_wrong_num(pred_res, true_res)
        else:
            res2spans = segs2spans if fileformat in {'seg', 'punc'} else pos2spans
            pred_spans = [res2spans(segs) for segs in pred_res]
            true_spans = [res2spans(segs) for segs in true_res]
            logging.info('true: '+' '.join(true_res[0]))
            logging.info('pred: '+' '.join(pred_res[0]))
            if(pred_labels):
                logging.info(''.join(pred_labels[0]))
            return f1_score(pred_spans, true_spans)
    elif fileformat in {'seghard'}:
        # pred_res是分词/词性标注结果, true_res是{must_sep:[], must_join:[]}
        if(pred_labels):
            logging.info('\n'.join([' '.join(pred_res[idx]) + '\n' + ''.join(pred_labels[idx]) for idx in range(len(pred_res))]))
        else:
            logging.info('\n'.join([' '.join(p) for p in pred_res]))
        res2spans = segs2spans
        pred_spans = [res2spans(segs) for segs in pred_res]
        acc, total = 0, len(pred_res)
        for ps, ts in zip(pred_spans, true_res):
            ok = True
            starts = {p[0] for p in ps}
            for t in ts['must_sep']:
                if t not in starts:
                    ok = False
            for t in ts['must_join']:
                if t in starts:
                    ok = False
            if ok:
                acc += 1
        return acc/total, acc, total
    elif fileformat == 'dict':
        res2spans = segs2spans
        pred_spans = [res2spans(segs) for segs in pred_res]
        acc, total = 0, len(pred_res)
        for ps, ts in zip(pred_spans, true_res):
            ok = True
            ps = set(ps)
            for t in ts:
                if t not in ps:
                    ok = False
            if ok:
                acc += 1
        return acc/total, acc, total


def eval_file(task, pred_file, true_file):
    """从运行结果输出文件和正确结果文件计算指标
    seg的存储结果： 空格分隔
    pos的存储结果： 空格分隔，每个词的词性用_相连
    """
    if task in {'seg', 'punc', 'pos'}:
        pred_res = [line.split(' ') for line in load_lines(pred_file)]
        true_res = [line.split(' ') for line in load_lines(true_file)]
        return eval_res(pred_res, true_res)
    elif task == 'dict':
        pass


def eval_api(api: Callable, fileformat: str, test_file: str, gi: bool = False):
    """用test_loader测试api的指标

    Args:
        api (Callable): 分词api, 传入句子列表, 返回句子分词列表
        fileformat (str): 测试文件格式
        test_file (str): 测试文件名
    """

    if fileformat in {'seg', 'punc'}:
        sents = [line.replace(' ', '') for line in load_lines(test_file)]
        true_res = [line.split(' ') for line in load_lines(test_file)]
    elif fileformat == 'pos':
        sents = [''.join(
            [word.rsplit('_', maxsplit=1)[0] for word in line.split(' ')]
        ) for line in load_lines(test_file)]
        true_res = [line.split(' ') for line in load_lines(test_file)]
    elif fileformat == 'seghard':
        # 只保留关键词位置
        lines = [line.replace(' ', '') for line in load_lines(test_file)]
        sents = [line.replace('|', '').replace('&', '') for line in lines]
        true_res = []  # sep表示此位置一定分开，join表示一定合并
        for line in lines:
            i = 0
            req = {'must_sep': [], 'must_join': []}
            for char in line:
                if char == '|':
                    req['must_sep'].append(i)
                elif char == '&':
                    req['must_join'].append(i)
                else:
                    i += 1
            true_res.append(req)
    elif fileformat == 'dict':
        sents, true_res = [], []
        for line in load_lines(test_file):
            key, sent = line.split(' ', maxsplit=1)
            req = [(m.start(), m.end()-1) for m in re.finditer(key, sent)]  # 必须包含的span
            sents.append(sent)
            true_res.append(req)
    task = fileformat
    if fileformat in {'dict', 'seghard'}:
        task = 'seg'
    pred_res = api(task, sents)
    if isinstance(pred_res, tuple):
        return eval_res(fileformat, pred_res[0], true_res, gi, pred_labels=pred_res[1])
    else:
        return eval_res(fileformat, pred_res, true_res, gi)


def gi_wrong_num(pred_res, true_res, output_file: str = ''):
    # granularity independent
    # len(target_lines)表示错误的个数
    target_lines = []
    for result, ans in zip(pred_res, true_res):
        sentence = ''.join(ans)
        assert ''.join(ans) == ''.join(result), ans+'\n'+result

        ans_blocker = ''
        result_blocker = ''
        ans_idx = 0
        result_idx = 0

        add_tag = 0  # 0 is ans, 1 is result

        ans_block_counter = 0
        ans_block_count_list = []
        ans_block_list = []
        result_block_counter = 0
        result_block_count_list = []
        result_block_list = []

        block_idx = 0

        while (add_tag == 0 and ans_idx < len(ans)) or (add_tag == 1 and result_idx < len(result)):
            if add_tag:
                result_blocker += result[result_idx]
                result_idx += 1
                result_block_list.append(block_idx)
                ans_block_counter += 1
            else:
                ans_blocker += ans[ans_idx]
                ans_idx += 1
                ans_block_list.append(block_idx)
                result_block_counter += 1
            if ans_blocker == result_blocker:
                ans_blocker = ''
                result_blocker = ''
                ans_block_count_list.append(ans_block_counter)
                result_block_count_list.append(result_block_counter)
                ans_block_counter = 0
                result_block_counter = 0
                block_idx += 1
            if len(result_blocker) < len(ans_blocker):
                add_tag = 1
            else:
                add_tag = 0
        print_tag = 0
        for i in range(len(ans_block_count_list)):
            if ans_block_count_list[i] != 1 and result_block_count_list[i] != 1:
                print_tag = 1
                ans[ans_block_list.index(i)] += '( ! )'
                result[result_block_list.index(i)] += '( ! )'
                break
        if print_tag:
            target_lines.append(f"{sentence}\n{ans}\n{result}\n")
    if output_file:
        store_lines(target_lines, output_file)
    acc, total = len(true_res) - len(target_lines), len(true_res)
    return acc/total, acc, total


def run_test(api, log_file: str, can_pos):
    from prettytable import PrettyTable
    DATA_DIR = 'data'
    set_logger(log_file)
    files = [
        # pos
        'pos_ours_test_2000.txt',

        # seghard
        'seghard_hard_test_109.txt',

        # seg
        'seg_ours_test_2000.txt',
        'seg_oursseg_test_1614.txt',
        # 'seg_pku_test_1951.txt',
        'seg_msr_test_3985.txt',
        'seg_cityu_test_1492.txt',
        'seg_as_test_14432.txt',
        'seg_cbtsample_test_2000.txt',
        'seg_nlpccweibo_test_2052.txt',

        # dict
        'dict_wantwords_test_2000.txt',

        # seg-gi
        ('seg_ours_test_2000.txt', True),
    ]

    scores = {
        'pos': [],
        'seg': [],
        'seghard': [],
        'dict': [],
        'seg-gi': []
    }
    for file in files:
        gi = False
        if isinstance(file, tuple):
            file = file[0]
            gi = True
        fileformat = file.split('_')[0]

        if fileformat == 'pos' and not can_pos:
            continue

        score = eval_api(api, fileformat, f'{DATA_DIR}/{file}', gi)
        if fileformat in {'seghard', 'dict'} or (fileformat == 'seg' and gi):
            logging.info(f'{file}\tacc: {score[0]*100:.2f}\tacc/total: {score[1]}/{score[2]}')
        else:
            logging.info(f'{file}\tf1: {score[0]*100:.2f}\tp/r: {score[1]*100:.2f}/{score[2]*100:.2f}')

        scores[fileformat+'-gi' if gi else fileformat].append(f'{score[0]*100:.2f}')

    table = PrettyTable(['pos', 'seghard', 'ours/oursseg/msr/cityu/as/cbt/nlpcc',  'dict', 'seg-gi'])
    table.add_row(['/'.join(scores['pos']), '/'.join(scores['seghard']), '/'.join(scores['seg']), '/'.join(scores['dict']), '/'.join(scores['seg-gi'])])
    logging.info(log_file+"\n"+str(table))

# from deepthulac.eval.third_party_api import ltp_

if __name__ == "__main__":
    from deepthulac.seg.model import LacModel
    from deepthulac.eval.old_model import SemiSeg
    import argparse

    path = 'output/punc_dict/2023-02-18_02-56-14'
    path = '/data03/private/chengzhili/deepthulac_dev_before2_26/dsb'
    path = '/data03/private/chengzhili/segmentation/output/seg/2023-02-27_04-42-02'
    # path = '/data03/private/chengzhili/segmentation/output/seg/2022-10-07_10-32-24 四分类 crf old'
    path = '/data03/private/chengzhili/segmentation/output/bound_punc_dict/2023-02-27_05-39-17/3276832'
    path = '/data03/private/chengzhili/segmentation/output/bound_punc_dict/2023-02-27_05-43-23/3276832'
    path = '/data03/private/chengzhili/segmentation/output/bound_punc_dict/2023-02-27_05-43-23/3276832'
    path = '/data03/private/chengzhili/segmentation/output/bound_punc_dict/2023-02-27_05-31-35 不用extra'
    path = '/data03/private/chengzhili/segmentation/output/bound_punc_dict/2023-02-27_05-41-11 111full/13107232'
    path = '/data03/private/chengzhili/segmentation/output/dsb_old/5808016'
    path = '/data03/private/chengzhili/segmentation/output/dsb_old/supervised'
    path = '/data03/private/chengzhili/segmentation/output/_good_checkpoint/sup_pretrain'
    path = '/data03/private/chengzhili/segmentation/output/seg_punc_dict/2023-03-05_01-51-41/9011200'
    path = 'output/seg_punc_dict/2023-03-05_02-06-24/9011200'
    path = 'output/bound_seg_punc_dict/2023-03-05_03-00-20/6553600'
    path = 'output/bound_seg_punc_dict/2023-03-05_03-03-24/8601600'
    path = 'output/seg_punc_dict_pretrain/2023-03-06_04-32-47/2867200'
    path = '/data03/private/chengzhili/segmentation/output/_good_checkpoint/sup_pretrain'
    # path = 'output/seg_punc_dict_pretrain/2023-03-06_04-31-08/3276800'

    #from ltp import LTP
    #ltp = LTP('LTP/base2', cache_dir='.cache', local_files_only=True)
#
#
    #def ltp_(task, sentences):
    #    task='seg'
    #    words, pos = ltp.pipeline(sentences, tasks=["cws", "pos"],).to_tuple()
    #    words, pos = words[0], pos[0]
    #    if task == 'seg':
    #        return '/'.join(words)
    #    elif task == 'pos':
    #        return '  '.join([w+''+str.lower(p) for w, p in zip(words, pos)])
    #run_test(ltp_, 'output/_third_party/ltp.log', False)
    #exit(0)
    # 测试的模型路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=path, type=str)
    args = parser.parse_args()
    saved_path = args.model_path

    print('loading...')
    device = 'cuda:3'  # 'cuda:5'
    log_file = saved_path+'/test.log'

    def deepthulac_api(task, sents):
        # TODO: split_long的影响非常巨大，要把split_long的逻辑改一下，让split之后尽量长
        result = model.seg(sents, batch_size=64, split_long=False, post_vmvd=False, vote_pattern=None)  # seg_pos头也一样从pos里拿结果即可
        # logging.info(result[task]['labels']) # TODO: 改成全在这里logging
        # return result[task]['res']
        return (result[task]['res'], result[task]['labels'])

    def old_deepthulac_api(task, sents):
        return model.seg(sents, batch_size=64, split_long=False)

    example_sents = hard_cases+long_cases
    if 'old' in saved_path:  # old是semi_supervised_punct分支的模型
        model = SemiSeg.load(saved_path, device, use_f16=False)
        print(model.seg(example_sents, batch_size=1))
        api = old_deepthulac_api
    else:
        model = LacModel.load(saved_path, device, use_f16=False)
        example_segs = model.seg(example_sents)['seg']['res']
        for seg in example_segs:
            print('/'.join(seg))
        if 'punc' in model.heads:
            print(model.seg(example_sents, punc=True)['seg']['res'])
        api = deepthulac_api
    can_pos = isinstance(model, LacModel) and ('pos' in model.heads or 'seg_pos' in model.heads)
    run_test(api, log_file, can_pos)
