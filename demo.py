from deepthulac import LacModel

# 加载模型
lac = LacModel.load(path='', device='cuda:0') # cuda或cpu

# 句子分词
sents = [
    '在石油化工发达的国家已大幅取代了乙炔水合法。',
    '他往衬衫外套了件外套，出门去了。'
]
results = lac.seg(sents, show_progress_bar=False)['seg']['res']
print(results)

# 文件分词
from deepthulac.utils import load_lines, store_lines
results = lac.seg(load_lines('lines.txt'), batch_size=256)['seg']['res']
store_lines([' '.join(w) for w in results], 'results.txt')

### 加入用户词表
text = '醋酸氟轻松是一种皮质类固醇，主要用于治疗皮肤病，减少皮肤炎症和缓解瘙痒。'
print(lac.seg([text])['seg']['res']) # 醋酸氟, 轻松, ...

# 加入用户自己提供的词表, 例如化学词表.txt中包含词'醋酸氟轻松'
lac.add_user_dict('化学词表.txt')
print(lac.seg([text])['seg']['res']) # 醋酸氟轻松, ...

# 我们也集成了
from deepthulac.legacy import thulac
lac = thulac.load(cache_dir='./cache')
results = lac.seg(sents)
print(results)