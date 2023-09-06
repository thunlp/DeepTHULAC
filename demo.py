from deepthulac import LacModel

# 加载模型
lac = LacModel.load(path='', device='cuda:0') # cuda或cpu

# 句子分词
sents = [
    "我爱北京天安门",
    '360公司创始人、董事长兼CEO周鸿祎在个人微博上晒出了清华大学研究生录取通知书，配文称“终于考上了，希望360智脑帮助我顺利毕业”。',
    '在国家科委支持下，袁隆平的水稻研究在1970至2000年代屡有突破，使他获联合国多项奖项和获封澳门科技大学荣誉博士，于2019年更获颁共和国勋章',
    '英国科学家艾萨克·牛顿出版《自然哲学的数学原理》，阐述运动定律和万有引力定律。',
    '英国罗斯林研究所研究的多莉出生，是世界上首只克隆成年体细胞成功的哺乳动物。',
    '忽如一夜春风来，千树万树梨花开。',
    '世有伯乐，然后有千里马。',
    "今天跟集美出去逛街啦，搞点神仙甜品778茶百道yyds，真的绝绝子～",
    "当你开始在一颦一笑中，发现每一个人可爱的模样，你便懂得了欣赏，当你在春花秋月更替中，学会了珍惜和看淡无常，你便懂得了生命。",
    '在石油化工发达的国家已大幅取代了乙炔水合法。',
    "他在衬衫外套了件外套，出门去了。",
    '这名研究生命大，躲过一劫。',
]
results = lac.seg(sents, show_progress_bar=True)['seg']['res']
for res in results:
    print(res)

# 文件分词
from deepthulac.utils import load_lines, store_lines
results = lac.seg(load_lines('lines.txt'), batch_size=256)['seg']['res']
store_lines([' '.join(w) for w in results], 'results.txt')

text = '醋酸氟轻松是一种皮质类固醇，主要用于治疗皮肤病，减少皮肤炎症和缓解瘙痒。'
print(lac.seg([text])['seg']['res']) # 醋酸, 氟, 轻松, ...

# 加入用户自己提供的词表, 一行一个词，例如化学词表.txt中包含词'醋酸氟轻松'
lac.add_user_dict('化学词表.txt')
print(lac.seg([text])['seg']['res']) # 醋酸氟轻松, ...
