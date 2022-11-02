import re
from tqdm import tqdm
from deepthulac.seg.cut_sent import split_sentence
from deepthulac.utils import load_lines, store_lines, timer
from deepthulac.seg.seg_utils import *


def make_pair(chars, labels):
    return [(char, label) for char, label in zip(chars, labels)]


@timer
def format_trainset_by_special_token(file, mode='BMES', sample_num=1000_000_000, MAX_LEN=500, pair=False):
    """ file: 空格分隔的分词训练数据，将超长的样例按最大长度切分为多个样例，
    切分处加入分隔标记，其余token保持原label，
    文本存储为.words文件，标签存储为.label文件
    """
    assert mode in {'BMES', 'EN'}
    sep_word = '@'
    sep_label = 'S' if mode == 'BMES' else 'E'

    def get_label(word):
        if mode == 'BMES':
            if len(word) == 1:
                return 'S'
            elif len(word) == 2:
                return 'BE'
            else:
                return 'B'+'M'*(len(word) - 2)+'E'
        else:
            return 'N'*(len(word) - 1) + 'E'

    def split_long_sample(text, sep):
        length = MAX_LEN - 5
        chunks = []
        for start in range(0, len(text), length):
            chunks.append(text[start:start+length]+sep)
        chunks[-1] = chunks[-1][:-1]
        return chunks

    lines = load_lines(file)
    char_lines, label_lines = [], []
    count_long = 0
    for line in tqdm(lines[:sample_num]):
        line = re.sub(r' +', ' ', line).strip()
        chars = line.replace(" ", "")
        labels = ''.join([get_label(word) for word in line.split(' ')])
        assert len(labels) == len(chars)
        if len(chars) > MAX_LEN:
            char_lines.extend(split_long_sample(chars, sep_word))
            label_lines.extend(split_long_sample(labels, sep_label))
            count_long += 1
        else:
            char_lines.append(chars)
            label_lines.append(labels)
    assert len(char_lines) == len(label_lines)
    print(f"file:{file} lines:{len(lines)} long_lines:{count_long}")
    store_lines(char_lines, f'{file}.seg.train.char')
    store_lines(label_lines, f'{file}.seg.train.label')
    if pair:
        return make_pair(char_lines, label_lines)
    return char_lines, label_lines


def store_samples(samples, samples_mode, file):
    """ 将pairs用空格分隔存为文件 """
    sents, labels = [x[0] for x in samples], [x[1] for x in samples]
    spans = [labels2spans(l, samples_mode) for l in labels]
    segs = [spans2segs(s, p) for (s, p) in zip(sents, spans)]
    store_lines([' '.join(w) for w in segs], file)


@timer
def format_trainset_by_split_text(file: str, MAX_LEN=511):
    """ file: 空格分隔的分词训练数据中，将超长的样例拆分为长度合法的多个样例 """

    def split_long_sample(text):
        # 将文本分成内容长度不超过MAX_LEN的patch，每个patch首尾都没有空格
        patches = []
        for sent in split_sentence(text):
            sent = sent.strip()  # split_sentence会将空格strip掉，但有个特例：'。 ”'->['。', ' ”']，因此还是需要手动strip
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
        assert sum([len(chunk)-chunk.count(' ') for chunk in chunks]) == len(text)-text.count(' ')
        assert sum([len(chunk) for chunk in chunks])+(len(chunks)-1) == len(text)
        return chunks

    lines = load_lines(file)
    new_lines = []
    for line in tqdm(lines):
        line = re.sub(r' +', ' ', line).strip()
        if len(line)-line.count(' ') > MAX_LEN:
            new_lines.extend(split_long_sample(line))
        else:
            new_lines.append(line)
    store_lines(new_lines, file + '.seg.train')
    return new_lines


@timer
def format_testset(file: str, MAX_LEN=511):
    """ 统一格式：空格分词，一个样例一行 """
    lines = load_lines(file)
    new_lines = []
    for line in tqdm(lines):
        line = re.sub(r' +', ' ', line).strip()
        new_lines.append(line)
    store_lines(new_lines, file + '.seg.test')
    return new_lines


if __name__ == "__main__":
    format_trainset_by_split_text('data/train.txt')
    format_trainset_by_special_token('data/train.txt')
