from cyac import Trie
from tqdm import tqdm
from deepthulac.utils import load_lines

vm = {'必', '不必', '不当', '不得', '不该', '不敢', '不会', '不可', '不肯', '不能', '不配', '不让', '不想', '不须', '不需', '不许', '不要', '不宜', '不应', '不愿', '不准', '当', '得', '得以', '该', '敢', '会', '可', '可能', '可以', '肯', '乐得', '乐意', '能', '能够', '配', '情愿', '让', '想', '须', '须要', '需', '需要', '许', '要', '宜', '应', '应当', '应该', '愿', '愿意', '允许', '值得', '只能', '准'}
vd = {'出', '出来', '出去', '到', '过', '过来', '过去', '回', '回来', '回去', '进', '进来', '进去', '开', '开来', '来', '起', '起来', '去', '上', '上来', '上去', '下', '下来', '下去'}
vvd = {'出 到', '出 过', '到 过', '过 回', '过 回来', '过 回去', '过 来', '过 上', '开 起', '开 上', '开 下', '开 来', '开 去'}


class UserDict:
    def __init__(self, file) -> None:
        words = load_lines(file)
        self.trie = Trie()
        for word in words:
            self.trie.insert(word)

    def _adjust_seg(self, segs):
        i = 0
        while i < len(segs):
            cand = segs[i]
            cands = []  # n个片段的字符串，n>=2
            for j in range(i + 1, len(segs)):
                cand += segs[j]
                if not self.trie.predict(cand):
                    break
                cands.append(cand)
            for j in range(len(cands)-1, -1, -1):
                cand = cands[j]
                if cand in self.trie:
                    segs[i] = cand
                    del segs[i+1:i+j+2]
                    break
            i += 1
        return segs

    def adjust_seg(self, pred_segs, num_workers=12):
        from multiprocessing import Pool
        with Pool(num_workers) as p:
            res = p.map(self._adjust_seg, pred_segs)
        return res


def _adjust_pos(poss):
    seg, pos = [], []
    for seg_pos in poss:
        s, p = seg_pos.rsplit('_', maxsplit=1)
        seg.append(s)
        pos.append(p)

    i = 0
    while i < len(seg)-1:
        if pos[i] == 'v' and pos[i+1] == 'v':
            if seg[i] == seg[i+1]:
                i += 2  # 两个词一样【例如"来 来"】，跳过这两个词，不处理
                continue
            if seg[i] in vm:
                pos[i] = 'vm'
                i += 1
                continue
            if seg[i+1] in vd:
                if (seg[i] not in vd) or (seg[i]+' '+seg[i+1] in vvd):
                    pos[i+1] = 'vd'
                    i += 2
                    continue
        i += 1
    return [s+'_'+p for s, p in zip(seg, pos)]
# 下面分错了
# 对_p 儿童_n 创造力_n 的_u 发展_v 起_vd 着_u 不同_a 的_u 作用_n

def adjust_pos(pred_poss, num_workers=12):
    res = [_adjust_pos(pred_pos) for pred_pos in pred_poss]
    return res
    from multiprocessing import Pool
    with Pool(num_workers) as p:
        res = p.map(_adjust_pos, pred_poss)
    return res
