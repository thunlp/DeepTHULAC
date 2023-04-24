import re

_SEPARATOR = r'#@#'
# . 可以匹配任意单个字符
# .*?表示匹配任意字符到下一个符合条件的字符 https://blog.csdn.net/WuLex/article/details/88563332
# ?=往后看，只有这样才匹配
# TODO: p = re.sub(r'([。？！]”；|”；|；|：|;|[。？！][”）]?)', r'\1\n', p).strip('\n')
# 原版 _RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)
# BUG：等 ... ... ” -> ['等 ...', '...'] BUGFIX: split_sentence会去掉最后一个英文引号 ... ... "
_RE_SENTENCE = re.compile(r'(.+?[\.!?])(?=\s+)|(.+?)(?=\n|$)', re.UNICODE)  # 修改后，防止删掉一些单个标点
_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)  # Dr. Li
_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)  # p.m.
_UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + _SEPARATOR + r'(\w)', re.UNICODE)
_UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + _SEPARATOR + r'(\w)', re.UNICODE)


def _replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def split_sentence(text, best=True, skip_punc_intraword=False, use_vertical_bar=False):
    assert _SEPARATOR not in text
    if skip_punc_intraword:  # 处理已经分好词的文件, DEPRECATED, 已经不用了，乱七八糟
        text = re.sub(r'([。！？?]) ', r"\1\n ", text)
        text = re.sub(r'(\.{6}) ', r"\1\n ", text)
        text = re.sub(r'(…{2}) ', r"\1\n ", text)
        text = re.sub(r'([。！？?][”’]) ', r'\1\n ', text)
        if use_vertical_bar:
            text = re.sub(r'(\|) ', r"\1\n ", text)
    else:
        text = re.sub(r'([。！？?])([^”’])', r"\1\n\2", text)
        text = re.sub(r'(\.{6})([^”’])', r"\1\n\2", text)
        text = re.sub(r'(…{2})([^”’])', r"\1\n\2", text)
        text = re.sub(r'([。！？?][”’])([^，。！？?])', r'\1\n\2', text)
    for chunk in text.split("\n"):
        # chunk = chunk.strip()
        if not chunk:
            continue
        if not best:
            yield chunk
            continue
        processed = _replace_with_separator(chunk, _SEPARATOR, [_AB_SENIOR, _AB_ACRONYM])
        sents = list(_RE_SENTENCE.finditer(processed))
        if not sents:
            yield chunk
            continue
        for sentence in sents:
            sentence = _replace_with_separator(sentence.group(), r" ", [_UNDO_AB_SENIOR, _UNDO_AB_ACRONYM])
            yield sentence


def split_batch(batch_text, max_len=500):
    '''现将句子按句子完结符号切分，如果切分完后一个句子长度超过限定值，再对该句子进行切分'''
    input_id2raw_id = []
    input_text = []
    for i, text in enumerate(batch_text):
        for sent in split_sentence(text):  # 注意：\n \t等特殊符号会被清理 TODO: 进一步改进
            if len(sent) <= max_len:
                input_text.append(sent)
                input_id2raw_id.append(i)
            else:
                for pos in range(0, len(sent), max_len):
                    input_text.append(sent[pos: pos + max_len])
                    input_id2raw_id.append(i)
    # TODO: 最好能再合起来使之不超过限定值的情况下尽量长 BUG: 一定要这么做！
    return input_text, input_id2raw_id


def split_batch_with_group_texts(batch_text, max_len=500):
    def split_long_sample(text: str):  # TODO: 和dataformat里面整合一下
        # 将文本分成内容长度不超过MAX_LEN的patch，每个patch首尾都没有空格
        patches = []

        for sent in split_sentence(text):
            if len(sent) <= max_len:  # 内容长度不超过最大长度的句子，直接作为一个patch
                patches.append(sent)
            else:
                for pos in range(0, len(sent), max_len):
                    patches.append(sent[pos: pos + max_len])

        # 将连续的patch合并为chunk，使每个chunk尽量长且不超过MAX_LEN
        chunks = []
        chunk, chunk_len = patches[0], len(patches[0])
        for patch in patches[1:]:
            if chunk_len + len(patch) > max_len:
                chunks.append(chunk)
                chunk, chunk_len = patch, len(patch)
            else:
                chunk, chunk_len = chunk+patch, chunk_len+len(patch)
        chunks.append(chunk)

        return chunks
    input_text = []
    input_id2raw_id = []
    for i, text in enumerate(batch_text):
        chunks = split_long_sample(text)
        input_text.extend(chunks)
        input_id2raw_id.extend([i]*len(chunks))
    return input_text, input_id2raw_id


def restore_batch(results, input_id2raw_id, accumulate_span=False):
    ori_results = [[] for _ in range(len(set(input_id2raw_id)))]
    for input_id, res in enumerate(results):
        raw_id = input_id2raw_id[input_id]
        if accumulate_span and ori_results[raw_id]:  # 对于span，要序号要统一加上前面的长度
            start = ori_results[raw_id][-1][-1]+1
            res = [(r[0]+start, r[1]+start) for r in res]
        ori_results[raw_id].extend(res)
    return ori_results
