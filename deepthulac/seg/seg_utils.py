import numpy as np


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


def labels2spans(labels, mode='BMES'):
    """ 将标签列表转换为词的范围 """
    if mode == 'BMES':
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
        return chunks
    elif mode == 'EN':
        e_pos = np.argwhere(np.array(labels) == 'E').squeeze(-1)
        s_pos = [0] + list(e_pos[:-1]+1)
        return [(s, e) for s, e in zip(s_pos, e_pos)]
    assert False
