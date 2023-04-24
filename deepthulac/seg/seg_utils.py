import numpy as np
import torch
from torch import Tensor
from typing import List


def segs2spans(segs):
    """ 将分词转换为词的范围 """
    spans = []
    pos = 0
    for seg in segs:
        spans.append((pos, pos+len(seg)-1))
        pos += len(seg)
    return spans


def spans2segs(sent: str, spans):
    """ 将词的范围转换为分词 """
    return [sent[span[0]:span[1]+1] for span in spans]


def labels2spans(labels: List[str], mode='BMES', vote_pattern = None):
    assert type(labels) == list
    """ 将标签列表转换为词的范围 """
    if mode == 'BMES':
        assert 'U' not in labels
        if vote_pattern is None:
            prev_tag = labels[0]
            begin_offset = 0
            chunks = []
            for i in range(1, len(labels)):
                tag = labels[i]
                if prev_tag + tag not in {'BM', 'BE', 'MM', 'ME'}:
                    chunks.append((begin_offset, i - 1))
                    begin_offset = i
                prev_tag = tag
            chunks.append((begin_offset, len(labels)-1))
        else:
            seg_from_front = []
            seg_from_after = []
            for label in labels[1:]:
                if label in ['B','S']:
                    seg_from_after.append(1)
                else:
                    seg_from_after.append(0)
            for label in labels[:-1]:
                if label in ['E','S']:
                    seg_from_front.append(1)
                else:
                    seg_from_front.append(0)
            if vote_pattern == 'and':
                seg = [seg_from_front[idx] & seg_from_after[idx] for idx in range(len(labels)-1)]
            elif vote_pattern == 'or':
                seg = [seg_from_front[idx] | seg_from_after[idx] for idx in range(len(labels)-1)]
            else:
                assert False
            begin_offset = 0
            chunks = []
            for i in range(len(labels)-1):
                if seg[i] == 1:
                    chunks.append((begin_offset, i))
                    begin_offset = i+1
            chunks.append((begin_offset, len(labels)-1))
        return chunks
    elif mode == 'EN':
        e_pos = np.argwhere(np.array(labels) == 'E').squeeze(-1)
        s_pos = [0] + list(e_pos[:-1]+1)
        return [(s, e) for s, e in zip(s_pos, e_pos)]
    assert False


def pos2spans(pos: List[str]):
    # TODO: 统一把word称呼为seg
    """ 将词_词性标注转换为词的范围_词性（唯一标识） """
    spans = []

    idx = 0
    for sp in pos:
        s, p = sp.rsplit('_', maxsplit=1)
        spans.append(f'{idx}_{idx+len(s)-1}_{p}')
        idx += len(s)
    return spans


def spans2pos(sent: str, spans, pos):
    """ 将词的范围转换为分词加词性标注 """
    segs = [sent[span[0]:span[1]+1] for span in spans]
    idx = 0
    for i in range(len(segs)):
        seg = segs[i]
        segs[i] += '_'+pos[idx]
        idx += len(seg)
    return segs


def segs2labels(segs: List[str], mode='BMES') -> List[str]:
    """ 将空格分隔的分词数据转换为标签 """
    def get_label(word):
        if mode == 'BMES':
            if len(word) == 1:
                return ['S']
            elif len(word) == 2:
                return ['B', 'E']
            else:
                return ['B']+['M']*(len(word) - 2)+['E']
        else:
            return ['N']*(len(word) - 1) + ['E']
    labels = []
    for word in segs:
        labels.extend(get_label(word))
    return labels


def pos2labels(pos, return_sent=True):
    """ 将分词加词性标注转换为标签 """
    word_tags = [tuple(word_tag.split('_')) for word_tag in pos]
    words = [word_tag[0] for word_tag in word_tags]
    sent = ''.join(words)
    labels = []
    for (word, tag) in word_tags:
        labels.extend([tag]*len(word))
    assert len(labels) == len(sent)
    if return_sent:
        return sent, labels
    return labels


def ids2partial_label(ids, labels_num) -> Tensor:
    """
    Examples::

        >>> ids2partial_label(torch.tensor([9, 12, 13]) + 4, 4)
        tensor([[1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 1, 1]], dtype=torch.uint8)
    """
    # 用pytorch的接口简化了multi_label的表示，https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    # label id和multi label tensor（tensor的每一位为0或1，表示对应label是否是可能的标签）间的简易转换
    ids -= labels_num
    mask = 2**torch.arange(labels_num, dtype=ids.dtype, device=ids.device)
    return ids.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def partial_label2id(partial_labels: List[int]) -> int:  # partial_labels[i]等于0说明labels[i]不可能，等于1说明可能
    """
    Examples::

        >>> partial_label = [0, 0, 0, 1]
        >>> label_id = partial_label2id(partial_label)
        >>> partial_label = ids2partial_label(torch.tensor(label_id), len(partial_label))
        tensor([0, 0, 0, 1], dtype=torch.uint8)
    """
    # https://gist.github.com/nebil/b0cee3e049b0afd4722b948d3e013ff6
    lable_id = 0
    for i in partial_labels[::-1]:
        lable_id = (lable_id << 1) | i
    return lable_id + len(partial_labels)


def parse_label2id(t, label2id):  # 将label字符串转换为label_id，考虑用/表示partial label的情况
    if t == 'U':  # 未知标签，不进行loss计算。NOTE: U这个标签只会在训练时出现并使用，即config文件里不允许出现U标签，U表示该位置不计算loss，并不表示一种分类
        return -1
    elif t in label2id:  # 正常标签
        return label2id.get(t)
    else:  # NOTE 约定用/分隔表示partial label
        partial = [0] * len(label2id)
        for label in t.split('/'):
            partial[label2id.get(label)] = 1
        return partial_label2id(partial)
