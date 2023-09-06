import functools
from torch.utils.data import DataLoader, IterableDataset
import re
from typing import List, Callable, Optional
from typeguard import typechecked
from tqdm import tqdm
from deepthulac.seg.cut_sent import split_sentence
from deepthulac.utils import get_dinfo, load_lines, parallel_run, store_lines, load_json, store_json, timer, log
from deepthulac.seg.seg_utils import *
import os
from distutils.version import StrictVersion
import platform
if StrictVersion(platform.python_version()) >= StrictVersion('3.8'):
    from typing import Literal
else:
    from typing_extensions import Literal
import random
import logging
from deepthulac.utils import timer


class LacDataset(IterableDataset):
    @typechecked
    def __init__(self, corpus_name: str, path: str, line_processor: Callable, lines_num: int = 0, file_order_reverse=False):
        """流式处理的数据集初始化

        Args:
            path (str): 文件名或文件夹名，每个文件按行存储，一行一个样例
            line_processor (Callable): 对每行文本进行处理，得到对应训练格式的样本
            lines (int, optional): 处理的行数，如果为0，表示使用全部的行
        """
        import os
        if os.path.isdir(path):
            self.files = [os.path.join(path, file) for file in sorted(os.listdir(path))]
            if file_order_reverse:
                self.files = self.files[::-1]
        else:
            self.files = [path]
        total = sum([int(file.rsplit('.', maxsplit=1)[0].split('_')[-1]) for file in self.files])
        if lines_num == 0 or lines_num > total:
            lines_num = total
        self.line_processor = line_processor
        self.lines_num = lines_num
        self.corpus_name = corpus_name

    def __iter__(self):
        index = -1
        for file in self.files:
            logging.info(f'{file} start')
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    index += 1
                    if index == self.lines_num:
                        return
                    train_sample = self.line_processor(line.strip('\n'))  # 这个处理很快, 主要是collate慢
                    if train_sample:
                        yield train_sample+(self.corpus_name, )
            logging.info(f'{file} end')

    def __len__(self):
        return self.lines_num


class MixedLacDataset(IterableDataset):
    @typechecked
    def __init__(self, datasets: List[IterableDataset], repeat_times):
        self.datasets = datasets
        self.sample_nums = [len(datasets[i])*repeat_times[i] for i in range(len(datasets))]
        self.length = sum(self.sample_nums)

    def __iter__(self):
        data_iters = [iter(dataset) for dataset in self.datasets]
        unused_nums, unused_sum = np.array(self.sample_nums), self.length
        # 以随机的次序 取各个datasets的line
        while unused_sum:
            i = np.random.choice(len(self.datasets), p=unused_nums/unused_sum)
            unused_nums[i] -= 1
            unused_sum -= 1
            try:
                yield next(data_iters[i])
            except StopIteration:
                data_iters[i] = iter(self.datasets[1])
                yield next(data_iters[i])

    def __len__(self):
        return self.length


def make_pair(chars, labels):
    # TODO: remove
    return [(char, label) for char, label in zip(chars, labels)]

# TODO: label全部改成list 禁止使用字符串，不然后面还要分情况处理


MAX_LEN = 511


def seg_line_processor(line, mode: Literal["BMES", "ME"]):
    # NOTE: 分词数据的表示 用单个空格表示分隔，句首末都没有空格
    """ file: 空格分隔的分词训练数据的一行，处理为模型训练的标签
    """
    chars = line.replace(" ", "")
    labels = segs2labels(line.split(' '), mode)
    assert len(labels) == len(chars)
    assert len(chars) <= MAX_LEN
    return (chars, labels)


def punc_line_processor(line, mode: Literal["BMES", "ME"], use_partial):
    # unified: 标点的标签是否处理得和分词意义统一，即处理为B/S E/S U S
    # M标签变成了U
    (chars, labels) = seg_line_processor(line, mode)
    if use_partial:  # 没有punc头，并且希望和分词保持一样的标签
        labels = [{'M': 'U', 'B': 'B/S', 'E': 'E/S', 'S': 'S'}[label] for label in labels]
    else:  # B=B/S, E=E/S的标签体系
        """
        # 2023 06 10
        import random
        new_labels = []
        for i in range(len(labels)):
            if labels[i] == 'M' and all([label=='M' for label in labels[i:i+2]]): # 认为4字以下可以视为词，连续2个以上字符是M，四字以上，M->U
                labels[i] = 'U'
                for j in range(i+1, len(labels)):
                    if labels[j] != 'M':
                        break
                    labels[j] = 'U'
        """
        labels = ['U' if label == 'M' else label for label in labels]
    return (chars, labels)


def get_vocab_chars_for_dict():
    global vocab_chars
    try:
        return vocab_chars
    except:
        vocab_chars = load_json('data_raw/vocab_chars.json')
        return vocab_chars


def dict_line_processor(line, mode: Literal["BMES", "ME"], extra_info=False):
    # NOTE: 词典数据的表示 词[空格]句，其中句中没有空格

    def get_label(key: str, sentence: str) -> str:
        pieces = sentence.split(key)
        labels = ''
        if mode == 'BMES':
            if(len(key) == 1):
                w_label = 'S'
            else:
                w_label = 'B'+'M'*(len(key)-2)+'E'
            if extra_info:
                labels = w_label.join(['' if len(piece) == 0 else 'S' if len(piece) == 1 else 'b'+'U'*(len(piece)-2)+'e' for piece in pieces])
                labels = ['B/S' if label == 'b' else 'E/S' if label == 'e' else label for label in labels]
            else:
                labels = w_label.join(['U'*len(piece) for piece in pieces])
        else:
            w_label = (len(key)-1)*'M'+'E'  # NOTE: 只给ME就等价于 连续、断开
            labels = w_label.join(['U'*(len(piece)-1)+'E' if len(piece) >= 1 else '' for piece in pieces])
        return list(labels)

    key, line = line.split(' ', maxsplit=1)  # NOTE: 字典训练的line允许了空格
    # 词典匹配不靠谱
    if len(key) == len(line):  # 单独的词不要了or len(key)>4 or  or :
        return None
    if len(key) > 4:  # 长词不要了，这很重要，这里四字以上的不要了
        return None
    if len(set(key)) != len(key):  # 有相同字的不要了，例如把把
        return None
    if not all([c in get_vocab_chars_for_dict() for c in key]):  # 有UNK的全部过滤掉，想要模型在不知道的情况下默认分=>UNK时要分
        if len(key) > 1:  # 等于1是单字，这个可以有
            return None
    chars = line
    labels = get_label(key, line)
    assert len(labels) == len(chars)
    assert len(chars) <= MAX_LEN
    return (chars, labels)


def pos_line_processor(line, mode):
    # TODO: 按mode分情况
    """ file: 空格分隔的分词训练数据，将超长的样例按最大长度切分为多个样例，
    切分处加入分隔标记，其余token保持原label，
    文本存储为.words文件，标签存储为.label文件
    """
    chars, labels = pos2labels(line.split(' '))
    assert len(chars) <= MAX_LEN
    return (chars, labels)


def seg_pos_line_processor(line, mode):
    # TODO: 按mode分情况
    """ file: 空格分隔的分词训练数据，将超长的样例按最大长度切分为多个样例，
    切分处加入分隔标记，其余token保持原label，
    文本存储为.words文件，标签存储为.label文件
    """
    splits = line.split(' ')
    chars, pos_labels = pos2labels(splits)
    seg_labels = segs2labels([sp.split('_')[0] for sp in splits], 'BMES')  # mode='BMES'
    assert len(chars) == len(pos_labels) and len(chars) == len(seg_labels)
    seg_pos_labels = [s+p for s, p in zip(seg_labels, pos_labels)]
    assert len(chars) <= MAX_LEN
    return (chars, seg_pos_labels)


def store_samples(samples, samples_mode, file):
    # TODO: remove
    """ 将pairs用空格分隔存为文件 """
    sents, labels = [x[0] for x in samples], [x[1] for x in samples]
    spans = [labels2spans(l, samples_mode) for l in labels]
    segs = [spans2segs(s, p) for (s, p) in zip(sents, spans)]
    store_lines([' '.join(w) for w in segs], file)


@timer
@typechecked
def format_raw(file: str, filetype: Literal['seg', 'pos', 'dict'] = 'seg', save_dir='data'):
    """ file: 空格分隔的分词训练数据中，将超长的样例拆分为长度合法的多个样例 """
    # filetype如果是seg，则全为空格分词。若为pos则全为空格分词，每个字用_表示词性后缀，禁止文本中有空格或_
    from sklearn.model_selection import train_test_split

    def split_long_sample(text: str):
        # 将文本分成内容长度不超过MAX_LEN的patch，每个patch首尾都没有空格
        patches = []
        if filetype == 'pos':
            seg_pos = text.split(' ')
            for sp in seg_pos:
                assert sp.count('_') == 1 and not sp.startswith('_'), sp  # 禁止文本中有空格或_
            text = ' '.join([sp.split('_')[0] for sp in seg_pos])
            pos = [sp.split('_')[1] for sp in seg_pos]

        # TODO: 不要用split_sentence了，重新写！split_sentence里面的操作太不清晰了
        for sent in split_sentence(text, skip_punc_intraword=True, use_vertical_bar=True):
            # 注意 '\u2002' == ' ' 会被strip掉，导致sent没有\u2002，而text有，造成后面assert的错误
            # sent = sent.strip()  # split_sentence会将空格strip掉，但有个特例：'。 ”'->['。', ' ”']，因此还是需要手动strip
            sent = sent.strip(' ')
            if len(sent)-sent.count(' ') <= MAX_LEN:  # 内容长度不超过最大长度的句子，直接作为一个patch
                patches.append(sent)
            else:  # 内容超过最大长度，每个词作为一个patch
                patches.extend(sent.split(' '))

        # 将连续的patch合并为chunk，使每个chunk尽量长且不超过MAX_LEN
        chunks = []
        chunk, chunk_len = patches[0], len(patches[0])-patches[0].count(' ')
        for patch in patches[1:]:
            assert len(patch.replace(' ', '')) <= MAX_LEN
            patch_len = len(patch)-patch.count(' ')
            if chunk_len + patch_len > MAX_LEN:
                chunks.append(chunk)
                chunk, chunk_len = patch, patch_len
            else:
                chunk, chunk_len = chunk+' '+patch, chunk_len+patch_len
        chunks.append(chunk)
        # '58×29 cm .  22 7/8×11 3/8 in .' 中\u2002会导致不一致
        assert sum([len(chunk)-chunk.count(' ') for chunk in chunks]) == len(text)-text.count(' ')
        assert sum([len(chunk) for chunk in chunks])+(len(chunks)-1) == len(text)
        # 对齐地恢复pos和seg
        if filetype == 'pos':
            idx = 0
            for i in range(len(chunks)):
                words = chunks[i].split(' ')
                chunks[i] = ' '.join([words[j]+'_'+pos[idx+j] for j in range(len(words))])
                idx += len(words)
            assert idx == len(pos)

        return chunks

    lines = load_lines(file)
    new_lines = []
    for line in tqdm(lines):
        line = re.sub(r' +', ' ', line).strip()
        if not line:
            continue
        if filetype == 'dict' and len(line)-(line.index(' ')+1) > MAX_LEN:  # TODO: 处理过长词典例句
            print(line)
            continue
        if len(line)-line.count(' ') > MAX_LEN:
            new_lines.extend(split_long_sample(line))
        else:
            new_lines.append(line)
    os.makedirs(save_dir, exist_ok=True)
    prefix, suffix = file.rsplit('/', maxsplit=1)[-1].split('.')
    store_lines(new_lines, f'{save_dir}/{prefix}_{len(new_lines)}.{suffix}')
    if 'all' in file:
        train_lines, test_lines = train_test_split(new_lines, test_size=min(2000, int(0.01*len(new_lines))), random_state=0)
        store_lines(train_lines, f'{save_dir}/{prefix}_{len(train_lines)}.{suffix}'.replace('all', 'train'))
        store_lines(test_lines, f'{save_dir}/{prefix}_{len(test_lines)}.{suffix}'.replace('all', 'test'))
    print(f'format {file} ok')


def build_dataloader(config, heads: List[str], batch_size: int):
    # config: dataset config

    # 同一个数据集可以有不同的processor处理方式
    # 数据集文件命名：数据集保存形式_数据集名_[train/dev/test]_数量.txt
    #   fileformat  | seg   punc        pos             dict    seghard      （punc和seg一样空格分隔）
    #   processor   | seg   punc        pos/seg_pos     dict    [无，仅测试]
    #   head/task   | seg   punc/seg    pos/seg_pos     seg     seg          （这里省略了bound，seg能做的，bound头也可以）
    processors = {
        'seg': functools.partial(seg_line_processor, mode=config.mode),
        'punc': functools.partial(seg_line_processor, mode=coonfig.mode) if getattr(config, 'use_M', False) else functools.partial(punc_line_processor, mode=config.mode, use_partial=False),
        'pos': functools.partial(pos_line_processor, mode=config.mode) if 'seg_pos' not in heads else functools.partial(seg_pos_line_processor, mode=config.mode),
        'dict': functools.partial(dict_line_processor, mode=config.mode, extra_info=getattr(config, 'extra_info', False))  # NOTE: 字典现在用ME更好！
    }
    # task等价于head
    tasks = {
        'seg': 'seg',
        'punc': 'seg' if 'punc' not in heads else 'punc',
        'pos': 'pos' if 'seg_pos' not in heads else 'seg_pos',
        'dict': 'seg'
    }
    data_dir = config.dir if hasattr(config, 'dir') else './data'
    filename = [f for f in os.listdir(data_dir) if f.startswith(config.name+'_'+config.split)][0]
    log(f'build {config.name}={filename}')

    dataset_type = config.name.split('_')[0]
    dataset = LacDataset(config.name, f'{data_dir}/{filename}', processors[dataset_type], lines_num=config.samples_num, file_order_reverse=getattr(config, 'file_order_reverse', False))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    data_loader.task = tasks[dataset_type]
    if config.split == 'train' and 'bound' in heads:
        data_loader.task = 'bound'
    return data_loader


def build_mixed_dataloader(loaders, repeat_times):
    task, batch_size = loaders[0].task, loaders[0].batch_size
    assert all([loader.task == task for loader in loaders])
    dataset = MixedLacDataset([loader.dataset for loader in loaders], repeat_times)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    data_loader.task = task
    return data_loader


def generate_fake():
    # TODO: 新词测试集
    # TODO: 难词测试集
    # TODO: 只对多义字进行，“星星” 是可以连着的
    def gen_next(c, choices):
        if True:  # random.random() < 0.9:
            return c
        else:
            return random.choice(choices)

    def gen_lines(choices, num):
        lines = []
        for i in range(num):
            fake_line = ''
            c = random.choice(choices)
            for j in range(int(random.random()**2*6)+3):
                c = gen_next(c, choices)
                fake_line = fake_line+' '+c
            lines.append(fake_line.strip())
        return lines

    c500 = load_lines('data_raw/dict/通用规范汉字表/常用字500.txt')[0]
    c2500 = load_lines('data_raw/dict/通用规范汉字表/常用字2500.txt')[0]
    c8000 = load_lines('data_raw/dict/通用规范汉字表/一级字.txt')[0]\
        + load_lines('data_raw/dict/通用规范汉字表/二级字.txt')[0]\
        + load_lines('data_raw/dict/通用规范汉字表/三级字.txt')[0]
    fake_lines = []
    fake_lines.extend(gen_lines(c500, 100000))
    fake_lines.extend(gen_lines(c2500, 100000))
    # fake_lines.extend(gen_lines(c8000, 10000))
    random.shuffle(fake_lines)
    store_lines(fake_lines, f'data/seg_fake_train_{len(fake_lines)}.txt')
    # exit(0)


# generate_fake()
symbols = [r'!', r'"', r'#', r'\$', r'%', r'&', r"'", r'\(', r'\)', r'\*', r'\+', r',', r'\-', r'\.', r'/', r':', r';', r'<', r'=', r'>', r'\?', r'@', r'\[', r'\\', r'\]', r'\^', r'_', r'`', r'\{', r'\|', r'\}', r'~', r'！', r'？', r'。', r'，', r'：', r'；', r'、', r'〃', r'《', r'》', r'【', r'】', r'〰', r'–', r'—', r'‘', r'’', r'“', r'”', r'﹏', r'…', r' ', r'\n']
# 小数和分数不要分，原则上可少分不可多分
# -需要转义，/不需要转移。需要转义的全部字符为* . ? + $ ^ [ ] ( ) { } \ |
# https://m.runoob.com/python/python-reg-expressions.html
pattern_str = re.compile(r"(——+|\.\.+|(?:\.(?=\D))|(?:/(?=\D))|[!\"#\$%&'\(\)\*\+,\-:;<=>\?@\[\\\]\^_`\{\|\}~！？。，：；、〃《》【】〰–—‘’“”﹏… \n])")

if __name__ == "__main__":
    def process_punc_train(raw_file: str, single_file: bool, output_corpus_name: str, split_test=False, concat=True):
        # 处理成标准分词文件
        """ single_file表示这一个文件就是整个数据集
        """
        if single_file:
            output_dir = 'data_raw'
        else:
            rs_output_dir = f'data_raw/punc_{output_corpus_name}-rs_train'
            rm_output_dir = f'data_raw/punc_{output_corpus_name}-rm_train'
            os.makedirs(rs_output_dir, exist_ok=True)
            os.makedirs(rm_output_dir, exist_ok=True)
        lines = load_lines(raw_file, remove_empty=False)

        punc_removed_lines = []
        punc_reserved_lines = []
        sample_text = ''
        for i, line in tqdm(enumerate(lines)):
            if line == '[EOS]':
                continue
            if concat:
                if line == '' or (i == len(lines)-1):
                    sample_text = sample_text.replace(' ', ' ')
                    punc_removed_lines.append(pattern_str.sub(' ', sample_text))
                    punc_reserved_lines.append(pattern_str.sub(r' \1 ', sample_text))
                    sample_text = ''
                else:
                    if not sample_text:
                        sample_text = line
                    elif pattern_str.match(sample_text[-1]):
                        sample_text += line
                    else:
                        sample_text += '|'+line  # 为何不行？？
            else:
                if line:
                    punc_removed_lines.append(pattern_str.sub(' ', line))
                    punc_reserved_lines.append(pattern_str.sub(r' \1 ', line))

        if single_file:
            split = 'all' if split_test else 'train'
            store_lines(punc_removed_lines, f'{output_dir}/punc_{output_corpus_name}-rm_{split}.txt')
            store_lines(punc_reserved_lines, f'{output_dir}/punc_{output_corpus_name}-rs_{split}.txt')
        else:
            file_id = raw_file.rsplit('.', maxsplit=1)[0].split('/')[-1]
            store_lines(punc_removed_lines, f'{rm_output_dir}/{file_id}.txt')
            store_lines(punc_reserved_lines, f'{rs_output_dir}/{file_id}.txt')

    def seg_remove_punc(raw_file, output_corpus_name):
        lines = load_lines(raw_file)
        punc_removed_lines = []
        for line in tqdm(lines):
            punc_removed_lines.append(pattern_str.sub(' ', line))
        random.shuffle(lines)
        store_lines(punc_removed_lines, f'data_raw/punc_{output_corpus_name}_train.txt')
        random.shuffle(punc_removed_lines)
        store_lines(punc_removed_lines, f'data_raw/seg_{output_corpus_name}-rm_train.txt')
        random.shuffle(punc_removed_lines)
        store_lines(punc_removed_lines, f'data_raw/punc_{output_corpus_name}-rm_train.txt')

    def format_all():
        format_raw('data_raw/dict_wantwords_all.txt', filetype='dict')

        process_punc_train('data_raw/punc_baidubaike0_all.txt')
        format_raw('data_raw/punc_baidubaike0rm_all.txt')
        format_raw('data_raw/punc_baidubaike0rs_all.txt')

        format_raw('data_raw/pos_ours_all.txt', filetype='pos')
        format_raw('data_raw/pos_oursvmvd_all.txt', filetype='pos')
        format_raw('data_raw/seg_ours_all.txt')
        format_raw('data_raw/seg_pku_train.txt')
        format_raw('data_raw/seg_pku_test.txt')
        format_raw('data_raw/seg_msr_train.txt')
        format_raw('data_raw/seg_msr_test.txt')

    @timer
    def test_lacdataset():
        path = 'data/seg_ours_train_1611208.txt'
        dataset = LacDataset('seg', path, seg_line_processor, lines_num=0)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
        print(next(iter(dataloader.dataset)))
        for batch in dataloader:
            pass
            # print(batch)
        print(len(dataloader))

    format_raw('data_raw/pos_guiyuan_train.txt')
    exit(0)
    format_raw('data_raw/seg_zuozhuan-a_test.txt')
    format_raw('data_raw/seg_zuozhuan-b_test.txt')
    exit(0)
    test_lacdataset()
    exit(0)

    # seg_remove_punc('data_raw/seg_zuozhuan_train.txt', 'zuozhuan')
    format_raw('data_raw/seg_zuozhuan-rm_train.txt')
    exit(0)
    format_raw('data_raw/seg_zuozhuan_train.txt')
    exit(0)
    # format_all()
    # format_all()
    # test_lacdataset()

    assert '\u2002' == ' '
    corpus_name = 'baidubaike'

    def process_one_file(i):
        print(i, 'start')
        process_punc_train(f'/data01/private/hujinyi/bert-wwm/train_data/txt_data/chunk_{i}.txt', single_file=False, output_corpus_name=corpus_name)
        print(i, 'process ok')
        format_raw(f'data_raw/punc_{corpus_name}-rm_train/chunk_{i}.txt', save_dir=f'data/punc_{corpus_name}-rm_train')
        format_raw(f'data_raw/punc_{corpus_name}-rs_train/chunk_{i}.txt', save_dir=f'data/punc_{corpus_name}-rs_train')
        print(i, 'format ok')

    exit(0)
    seg_remove_punc('data/seg_ours_train_1611208.txt', output_corpus_name='ours')
    format_raw(f'data_raw/punc_ours_train.txt', save_dir=f'data')
    format_raw(f'data_raw/seg_ours-rm_train.txt', save_dir=f'data')
    format_raw(f'data_raw/punc_ours-rm_train.txt', save_dir=f'data')
    exit(0)
    parallel_run(process_one_file, list(range(165, 181+1)), num_proc=17)
    format_raw('data/seperate_data/seg_as_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_cityu_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_ctb_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_hanyuyuliaoku_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_keben_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_msr_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_msr-no-digit_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_nlpcc2016_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_rmrb2014_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_pku_all.txt', save_dir='data/seperate_data')
    format_raw('data/seperate_data/seg_sanku_all.txt', save_dir='data/seperate_data')
