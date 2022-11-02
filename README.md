<h2 align="center" style="font-size:2em;font-weight:bold">DeepTHULAC：中文词法分析工具包
</h2>




------

<p align="center">
  <a href="#项目介绍">项目介绍</a> •
  <a href="#安装">安装</a> •
  <a href="#使用方式">使用方式</a> •
  <a href="#todo">TODO</a> •
</p>



## 项目介绍

DeepTHULAC由清华大学自然语言处理与社会人文计算实验室开发的中文词法分析工具包。

DeepTHULAC基于实验室自研BERT，并利用我们整合的目前世界上规模最大的人工分词和词性标注中文语料库训练，模型标注准确性高。目前已完成中文分词功能，词性标注功能正在开发中。

## 安装

1.  安装 [pytorch](https://pytorch.org/get-started/locally/)

2. ```
   pip install deepthulac
   ```

## 使用方式

### 分词

```python
from deepthulac.seg.model import Seg
from deepthulac.utils import load_lines, store_lines

# 加载模型
lac = Seg.load(device='cuda:0') # cuda或cpu

# 句子分词
results = lac.seg([
    '在石油化工发达的国家已大幅取代了乙炔水合法。',
    '这件和服务必于今日裁剪完毕'
], show_progress_bar=False)
print(results)

# 小文件分词
results = lac.seg(load_lines('lines.txt'), batch_size=256)
store_lines([' '.join(w) for w in results], 'results.txt')
```

### 加入用户词表

```python
text = '醋酸氟轻松是一种皮质类固醇，主要用于治疗皮肤病，减少皮肤炎症和缓解瘙痒。'
print(lac.seg([text])) # 醋酸氟, 轻松, ...

# 加入用户自己提供的词表, 例如化学词表.txt中包含词'醋酸氟轻松'
lac.add_user_dict('化学词表.txt')
print(lac.seg([text])) # 醋酸氟轻松, ...
```

## TODO

* 词性标注功能开发
* 分词模型性能调优和方法改进
* 模型蒸馏压缩，加速
* 提供多粒度的分词功能
