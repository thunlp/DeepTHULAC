"""
!pip install lac hanlp jieba fastNLP==0.5.5 fastHan ltp -i https://pypi.tuna.tsinghua.edu.cn/simple
!pip install paddlepaddle-gpu==2.4.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
!pip install 
"""
from deepthulac.legacy import thulac
from LAC import LAC
from fastHan import FastHan
import hanlp
import jieba
import jieba.posseg as pseg
import paddle
from ltp import LTP

paddle.enable_static()
jieba.enable_paddle()


# THULAC
print('THULAC')
thulac_seg = thulac(seg_only=True)
thulac_pos = thulac()


def thulac_(sentence, task='pos'):
    if task == 'seg':
        return thulac_seg.cut(sentence, text=True).replace(' ', '/')
    elif task == 'pos':
        return thulac_pos.cut(sentence, text=True).replace('_', '').replace(' ','  ')


# Baidu LAC
print('baidu LAC')
baidu_seg = LAC(mode='seg')
baidu_lac = LAC(mode='lac')


def baidu_lac_(sentence, task='pos'):
    if task == 'seg':
        return '/'.join(baidu_seg.run(sentence))
    elif task == 'pos':
        res = baidu_lac.run(sentence)
        return '  '.join([w+str.lower(p) for w, p in zip(res[0], res[1])])


# FastHan
print('fasthan')
fasthan_pos = FastHan()
def fasthan_(sentence, task='pos'):
    res = fasthan_pos(sentence,target="POS")
    if task == 'seg':
        return '/'.join([w for w, p in res[0]])
    elif task == 'pos':
        return '  '.join([w+str.lower(p) for w, p in res[0]])


# HANLP
print('hanlp')
hanlp_pos = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH, devices=[])
# hanlp_pos = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_UDEP_SDP_CON_ELECTRA_SMALL_ZH, devices=[])


def hanlp_(sentence, task='pos'):
    res = hanlp_pos([sentence])
    words, pos = res['tok/fine'][0], res['pos/pku'][0]
    if task == 'seg':
        return '/'.join(words)
    elif task == 'pos':
        return '  '.join([w+''+str.lower(p) for w, p in zip(words, pos)])


print('hanlp coarse')
hanlp_tok_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH, devices=[])


def hanlp_coarse_(sentence, task='pos'):
    if task == 'seg':
        return '/'.join(hanlp_tok_coarse(sentence))
    elif task == 'pos':
        return '/'.join(hanlp_tok_coarse(sentence))


print('hanlp fine')
hanlp_tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH, devices=[])


def hanlp_fine_(sentence, task='pos'):
    if task == 'seg':
        return '/'.join(hanlp_tok_fine(sentence))
    elif task == 'pos':
        return '/'.join(hanlp_tok_fine(sentence))


# jieba
print('jieba gru')


def jieba_gru_(sentence, task='pos'):
    if task == 'seg':
        return '/'.join(jieba.cut(sentence, use_paddle=True))
    elif task == 'pos':
        words = pseg.cut(sentence, use_paddle=True)
        return '  '.join([w + str.lower(p) for w, p in words])


print('jieba')


def jieba_(sentence, task='pos'):
    if task == 'seg':
        return '/'.join(jieba.cut(sentence, cut_all=False))
    elif task == 'pos':
        words = pseg.cut(sentence)
        return '  '.join([w + str.lower(p) for w, p in words])


# LTP
print('LTP')
# 第一次需要下载
# ltp = LTP('LTP/base2', force_download=True, cache_dir='.cache')  # ltp-4.1.5.post2
ltp = LTP('LTP/base2', cache_dir='.cache', local_files_only=True)


def ltp_(sentence, task='pos'):
    words, pos = ltp.pipeline([sentence], tasks=["cws", "pos"],).to_tuple()
    words, pos = words[0], pos[0]
    if task == 'seg':
        return '/'.join(words)
    elif task == 'pos':
        return '  '.join([w+''+str.lower(p) for w, p in zip(words, pos)])


def api(api_name: str, sent):
    api_map = {
        'thulac': thulac_,
        'baidu_lac': baidu_lac_,
        'fasthan': fasthan_,
        'hanlp': hanlp_,
        'hanlp_coarse': hanlp_coarse_,
        'hanlp_fine': hanlp_fine_,
        'jieba_gru': jieba_gru_,
        'jieba': jieba_,
        'ltp': ltp_
    }
    return api_map[api_name](sent)


def batched_api(api, sents):
    pred = []
    batch_size = 128  # NOTE: 要根据显存来定，太大了CUDA内存会不够
    for i in range(0, len(sents), batch_size):
        batch_sents = sents[i:min(i+batch_size, len(sents))]
        pred.extend(api(batch_sents))
    return pred