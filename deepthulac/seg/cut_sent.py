import re

_SEPARATOR = r'@'
_RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)
_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
_UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + _SEPARATOR + r'(\w)', re.UNICODE)
_UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + _SEPARATOR + r'(\w)', re.UNICODE)


def _replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def split_sentence(text, best=True):
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
    # TODO: 最好能再合起来使之不超过限定值的情况下尽量长
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
