<h2 align="center" style="font-size:2em;font-weight:bold">DeepTHULAC：中文词法分析工具包
</h2>




------

<p align="center">
  <a href="#项目介绍">项目介绍</a> •
  <a href="https://huggingface.co/spaces/chengzl18/DeepTHULAC">在线演示🤗</a> •
  <a href="#安装">安装</a> •
  <a href="#使用方式">使用方式</a> •
  <a href="#todo">TODO</a> •
</p>



## 项目介绍

DeepTHULAC由清华大学自然语言处理与社会人文计算实验室开发的中文词法分析工具包。

DeepTHULAC基于实验室自研BERT，并利用我们整合的目前世界上规模最大的人工分词和词性标注中文语料库训练，模型标注准确性高。我们从整合出的分词语料中随机选取一部分作为测试集，F1指标可以达到 97.6％，其中包含了其他公开的分词语料。目前已完成中文分词功能，词性标注功能正在开发中。

## 安装

1.  安装 [pytorch](https://pytorch.org/get-started/locally/)

2. ```bash
   pip install deepthulac
   ```

## 使用方式

### 分词

```python
from deepthulac import LacModel
lac = LacModel.load(path='', device='cuda:0') # 加载模型，path为模型文件夹路径，空表示自动从huggingface下载，device设置为cuda或cpu

# 句子分词
sents = ['在石油化工发达的国家已大幅取代了乙炔水合法。', '他在衬衫外套了件外套，出门去了。']
results = lac.seg(sents, show_progress_bar=False)['seg']['res']
print(results)

# 文件分词
from deepthulac.utils import load_lines, store_lines
results = lac.seg(load_lines('lines.txt'), batch_size=256)['seg']['res']
store_lines([' '.join(w) for w in results], 'results.txt')
```

如果由于网络问题无法自动下载模型，可以[从这里手动下载](https://cloud.tsinghua.edu.cn/d/58ad34f5cc1c40a19071/)，path设置为模型路径（如果是Windows系统，路径形如`'X:\\...\\deepthulac-seg-model'`）。

### 加入用户词表
安装依赖包 `pip install cyac`。

```python
text = '醋酸氟轻松是一种皮质类固醇，主要用于治疗皮肤病，减少皮肤炎症和缓解瘙痒。'
print(lac.seg([text])['seg']['res']) # 醋酸, 氟, 轻松, ...

# 加入用户自己提供的词表, 一行一个词，例如化学词表.txt中包含词'醋酸氟轻松'
lac.add_user_dict('化学词表.txt')
print(lac.seg([text])['seg']['res']) # 醋酸氟轻松, ...
```

### 使用THULAC

工具包也集成了[THULAC](https://github.com/thunlp/THULAC-Python)，准确度比DeepTHULAC低，但速度更快。

```python
from deepthulac.legacy import thulac
lac = thulac.load(cache_dir='./cache')
results = lac.seg(sents)
print(results)
```

## TODO

* 词性标注功能开发
* 分词模型性能调优和方法改进
* 模型蒸馏压缩，加速
* 提供多粒度的分词功能
