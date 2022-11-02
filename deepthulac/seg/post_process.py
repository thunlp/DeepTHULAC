from cyac import Trie
from tqdm import tqdm
from deepthulac.utils import load_lines


class UserDict:
    def __init__(self, file) -> None:
        words = load_lines(file)
        self.trie = Trie()
        for word in words:
            self.trie.insert(word)

    def adjust_seg(self, segs):
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

    def adjust_segs(self, pred_segs, num_workers=12):
        from multiprocessing import Pool
        with Pool(num_workers) as p:
            res = p.map(self.adjust_seg, pred_segs)
        return res
