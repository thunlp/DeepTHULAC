from typing import List, Optional, Union
from torch import softmax
from torch import nn
import torch
import os
import logging
import warnings
import shutil

from transformers.models.bert.modeling_bert import logger
from torchcrf import CRF

from tqdm import tqdm
import logging
from munch import Munch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from deepthulac.seg.post_process import UserDict, adjust_pos
from deepthulac.seg.seg_utils import *
from deepthulac.seg.cut_sent import restore_batch, split_batch, split_batch_with_group_texts
from deepthulac.seg.data_format import make_pair
from deepthulac.utils import DistributedInfo, load_yaml, store_yaml
from deepthulac.eval.cases import *
import torch.nn.functional as F

import functools
import random


def kd_ce_loss(logits_S, logits_T, temperature):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


class MLP(nn.Module):
    def __init__(self, dims: List[int]):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i != len(dims)-1:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=0.0))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Head(nn.Module):
    def __init__(self, bert_config, labels: List[str], head_config) -> None:
        super().__init__()
        num_labels = len(labels)
        self.num_labels = num_labels
        # self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)  # dropout是不是可以调一下？
        # self.classifier = nn.Linear(bert_config.hidden_size, num_labels)
        # NOTE: 多任务不能用单层线性了，要用MLP，否则会极大限制能力？？
        # self.classifier = MLP([bert_config.hidden_size, 512, 256, num_labels])
        if head_config.layers_num == 1:
            self.classifier = MLP([bert_config.hidden_size, num_labels])
        else:
            self.classifier = MLP([bert_config.hidden_size, 512, 256, num_labels])

        self.dropout = nn.Dropout(p=getattr(head_config, 'dropout', 0.1))

        self.use_crf = head_config.use_crf
        if self.use_crf:
            self.crf = CRF(num_labels, batch_first=True)

        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}

    def forward(self, x):
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs

    def decode(self, outputs, mask):  # outputs：每个字符所有类别得分(batch_size, max_len),  mask：对应是否不是pad
        # 仅用于测试，decode from head emissions
        if self.use_crf:
            if len(self.label2id) == 4:
                labels = self.crf.decode(outputs, mask=mask)  # (batch_size, max_len - padded_len)
            else:
                grain = 0.5  # TODO: grain接口
                if grain == 0.5:
                    labels = self.crf.decode(outputs, mask=mask)
                else:
                    outputs = softmax(outputs, dim=-1)
                    outputs = outputs[:, :, 0] > grain
                    labels = []
                    for output, label_mask in zip(outputs, mask):
                        labels.append([0 if output[i] else 1 for i in range(sum(label_mask)-1)]+[0])  # 最后一个一定是End
        else:
            outputs = softmax(outputs, dim=-1)
            outputs = torch.argmax(outputs, dim=-1)
            labels = []
            for output, label_mask in zip(outputs, mask):
                labels.append([output[i].item() for i in range(sum(label_mask))])
        labels = [[self.id2label.get(idx) for idx in indices] for indices in labels]
        return labels

    def partial_label_loss(self, logits, labels):
        # 参考 https://github.com/mikigom/DNPL-PyTorch/blob/master/train.py#L124
        full, partial = labels < self.num_labels, labels >= self.num_labels
        full_loss = nn.CrossEntropyLoss(reduction='sum')(logits[full], labels[full])
        sum_p = (F.softmax(logits[partial], dim=1) * ids2partial_label(labels[partial], self.num_labels)).sum(-1)  # 所有可能标签的概率
        partial_loss = -torch.sum(torch.log(sum_p.clamp(0., 1.) + 1e-10))
        return (full_loss + partial_loss) / len(logits)

    def loss_func_with_crf(self, logits, labels):
        loss_mask = labels.gt(-1)  # NOTE padded_label=-1
        loss = self.crf(logits, labels, loss_mask) * (-1)
        return loss

    def loss_func_no_crf(self, logits, labels, teacher_model=None, teacher_input_ids=None, task=None):
        loss_mask = labels.gt(-1)
        loss_mask = loss_mask.view(-1) == 1
        # 只对label实际存在的位置计算loss
        active_logits = logits.view(-1, self.num_labels)[loss_mask]
        active_labels = labels.view(-1)[loss_mask]

        if not torch.any(labels >= self.num_labels):  # 所有的标签都没有partial label，NOTE: 可以去掉？
            loss_fct = nn.CrossEntropyLoss()  # NOTE: = softmax + entropy
        else:
            loss_fct = self.partial_label_loss

        loss = loss_fct(active_logits, active_labels)

        if teacher_model:
            teacher_model.eval()
            alpha = 0.95
            loss_fct = kd_ce_loss
            with torch.no_grad():
                # 词表不一样，不能用同一份input_data
                logits_teacher = teacher_model.heads[task](teacher_model.forward_backbone(teacher_input_ids))
            active_logits_teacher = logits_teacher.view(-1, self.num_labels)[loss_mask]
            loss_soft = loss_fct(active_logits, active_logits_teacher, temperature=1)
            # https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/model/net.py
            # loss_soft = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(active_logits_teacher/T, dim=1)) * ( T * T)
            loss = loss_soft*alpha + loss*(1. - alpha)
        return loss

    def loss_func(self, logits, labels, teacher_model=None, teacher_input_ids=None, task=None):
        if self.use_crf:
            return self.loss_func_with_crf(logits, labels)
        else:
            return self.loss_func_no_crf(logits, labels, teacher_model, teacher_input_ids, task)


class BoundDetectHead(nn.Module):
    def __init__(self, bert_config, head_config) -> None:
        super().__init__()
        if head_config.layers_num == 1:
            self.classifier_left_bound = MLP([bert_config.hidden_size, 2])
            self.classifier_right_bound = MLP([bert_config.hidden_size, 2])
        else:
            self.classifier_left_bound = MLP([bert_config.hidden_size, 512, 256, 2])
            self.classifier_right_bound = MLP([bert_config.hidden_size, 512, 256, 2])
        # TODO: 加dropout？
        self.use_crf = False

        # 三个bit表示:  ([不确定性]) [左边是否是边界] [右边是否是边界]
        self.label2id = {'B': 0b10, 'E': 0b01, 'M': 0b00, 'S': 0b11, 'B/S': 0b110, 'E/S': 0b101}
        self.id2label = {self.label2id[label]: label for label in self.label2id}

    def forward(self, x):
        outputs = torch.cat([self.classifier_left_bound(x), self.classifier_right_bound(x)], dim=-1)
        return outputs

    def decode(self, outputs, mask):  # outputs：每个字符所有类别得分(batch_size, max_len),  mask：对应是否不是pad
        # 仅用于测试，decode from head emissions
        left_outputs, right_outputs = torch.split(outputs, [2, 2], dim=-1)
        left_labels, right_labels = torch.argmax(left_outputs, dim=-1), torch.argmax(right_outputs, dim=-1)  # (batch_size, seq_len) 1表示分,0表示不分
        outputs = (left_labels << 1)+right_labels  # labels
        labels = []
        for output, label_mask in zip(outputs, mask):
            labels.append([output[i].item() for i in range(sum(label_mask))])
        labels = [[self.id2label.get(idx) for idx in indices] for indices in labels]

        return labels

    def compute_loss(self, logits, labels):
        loss_mask = labels.gt(-1)
        loss_mask = loss_mask.view(-1) == 1
        # 只对label实际存在的位置计算loss
        active_logits = logits.view(-1, 2)[loss_mask]
        active_labels = labels.view(-1)[loss_mask]
        loss = nn.CrossEntropyLoss()(active_logits, active_labels)  # TODO: 试一下 BCEWithLogitsLoss
        return loss

    def loss_func(self, logits, labels, teacher_model=None, teacher_input_ids=None, task=None):
        left_labels = (labels & 0b10) >> 1  # 取出左边界的二分类值 # (batch_size, seq_len)
        left_labels[labels.lt(0)] = -1  # 不确定的U标签
        left_labels[labels == 0b101] = -1  # 左边界不确定

        right_labels = (labels & 0b01)
        right_labels[labels.lt(0)] = -1
        right_labels[labels == 0b110] = -1

        left_logits, right_logits = torch.split(logits, [2, 2], dim=-1)  # (batch_size, seq_len, 4)
        # logits = torch.cat([left_logits, right_logits],dim=1) # (batch_size, seq_len+seq_len, 2)
        # labels = torch.cat([left_labels, right_labels],dim=1)
        # return self.compute_loss(logits, labels)
        return (self.compute_loss(left_logits, left_labels) + self.compute_loss(right_logits, right_labels))/2


class LacModel(nn.Module):
    def __init__(self, config, dinfo, pretrained_path):
        # pretrained_path is pretrained model path or huggingface model or checkpoint
        bert = AutoModel.from_pretrained(pretrained_path)
        custom = os.path.exists(pretrained_path)
        if custom:
            self.pretrained_path = pretrained_path
            tokenizer = BertTokenizer.from_pretrained(f'{self.pretrained_path}/vocab.txt')
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert_model)
        super(LacModel, self).__init__()

        self.bert = bert
        self.tokenizer: AutoTokenizer = tokenizer
        if 'heads' not in config:
            config.heads = ['seg', 'pos']

        self.heads = {}

        # 历史版本兼容
        if 'head_config' not in config:
            config.head_config = Munch()
            config.head_config.use_crf = config.use_crf
            config.head_config.layers_num = 3

        if 'seg' in config.heads:
            self.seg_head = Head(self.bert.config, config.seg_labels, head_config=config.head_config)
            self.heads['seg'] = self.seg_head
        self.seg_mode = 'BMES' if len(config.seg_labels) == 4 else 'EN'
        if 'punc' in config.heads:  # 标点符号
            self.punc_head = Head(self.bert.config, config.punc_labels, head_config=config.head_config)
            self.heads['punc'] = self.punc_head
        if 'pos' in config.heads:
            self.pos_head = Head(self.bert.config, config.pos_labels, head_config=config.head_config)
            self.heads['pos'] = self.pos_head
        if 'seg_pos' in config.heads:
            seg_pos_labels = [sl+pl for pl in config.pos_labels for sl in config.seg_labels]
            self.seg_pos_head = Head(self.bert.config, seg_pos_labels, head_config=config.head_config)
            self.heads['seg_pos'] = self.seg_pos_head
        if 'bound' in config.heads:
            self.bound_head = BoundDetectHead(self.bert.config, head_config=config.head_config)
            self.heads['bound'] = self.bound_head

        self.model_config = config
        self.dinfo: DistributedInfo = dinfo

        self.onnx_model = None
        self.post_process = None

        self.corpus_map = {
            'seg_as': '[unused2]',
            'seg_cityu': '[unused3]',
            'seg_cnc': '[unused4]',  # 没有
            'seg_ctb': '[unused5]',
            'seg_msr': '[unused6]',
            'seg_pku': '[unused7]',
            'seg_hanyuyuliaoku': '[unused8]',
            'seg_keben': '[unused9]',
            'seg_nlpcc': '[unused10]',
            'seg_rmrb': '[unused11]',
            'seg_sanku': '[unused12]',
        }
        self.to(dinfo.device)

    def seg(self, sentences, verbose=False, grain=0.5, batch_size=16, split_long=True, show_progress_bar=True, punc=False, corpus_name=None, post_vmvd=False, vote_pattern=None):
        # TODO: 写入文件进行分词，不然数据量太大会崩？
        if split_long:
            sentences, input_id2raw_id = split_batch_with_group_texts(sentences)

        test_loader = DataLoader([(sent, corpus_name) for sent in sentences], batch_size=batch_size, shuffle=False)
        test_loader.collate_fn = functools.partial(self.collate_fn, 'any')
        model = self
        model.eval()

        res = {
            'seg': {'emissions': [], 'labels': [], 'spans': [], 'res': []},  # emission分类的各类分数, label分类的类别，span每个词的范围，seg最终结果
            'punc': {'emissions': [], 'labels': [], 'spans': [], 'res': []},
            'bound': {'emissions': [], 'labels': [], 'spans': [], 'res': []},
            'pos': {'emissions': [], 'labels': [], 'res': []},  # pos最终结果
            'seg_pos': {'emissions': [], 'labels': [], 'res': []},
            'ner': {}
        }
        seg, pos = res['seg'], res['pos']
        sents = []  # sents是不是和sentences一样？？
        with torch.no_grad():
            for batch_samples in tqdm(test_loader, disable=not show_progress_bar):
                if self.onnx_model:
                    input_ids, batch_sents = batch_samples
                else:
                    input_ids, batch_sents = model.batch_to_device(batch_samples)
                sents.extend(batch_sents)
                label_masks = input_ids[:, 1:].ne(0)  # NOTE: pad_token_id是0 (batch_size, max_len)
                if self.onnx_model:  # TODO: 这个不兼容了吧
                    with torch.no_grad():
                        inputs = {
                            'input_ids': input_ids.cpu().numpy()
                        }
                        outputs = self.onnx_model.run(None, inputs)[0]
                        outputs = torch.tensor(outputs, device=input_ids.device)
                else:
                    backbone_outputs = model.forward_backbone(input_ids)  # (batch_size, max_len, num_labels)

                for task in self.heads:
                    outputs = model.heads[task](backbone_outputs)
                    if verbose:
                        res[task]['emissions'].extend([outputs[i][label_masks[i]].cpu().numpy().tolist() for i in range(len(outputs))])
                    labels = model.heads[task].decode(outputs, label_masks)  # (batch_size, max_len - padded_len)
                    res[task]['labels'].extend(labels)
        # 如果punc，测试会自动使用punc头的结果作为punc和seg的结果，从seg里拿结果
        if punc or ('seg' not in self.heads and 'seg_pos' not in self.heads):
            seg = res['punc']
            res['seg'] = res['punc']
        # 如果有seg_pos头，测试会自动使用seg_pos头作为pos结果，从pos里拿结果
        if res['seg_pos']['labels']:
            for seg_pos_res in res['seg_pos']['labels']:
                seg['labels'].append([sp[0] for sp in seg_pos_res])
                pos['labels'].append([sp[1:] for sp in seg_pos_res])
        # 如果有bound头，测试会自动使用bound头作为seg结果，从seg里拿结果
        if res['bound']['labels']:
            seg = res['bound']
            res['seg'] = res['bound']

        # seg任务输出
        if seg['labels']:
            spans = [labels2spans(labels, model.seg_mode, vote_pattern=vote_pattern) for labels in seg['labels']]
            seg['res'] = [spans2segs(s, p) for (s, p) in zip(sents, spans)]
            if split_long:
                seg['labels'] = restore_batch(seg['labels'], input_id2raw_id)
                seg['res'] = restore_batch(seg['res'], input_id2raw_id)

            if verbose:
                assert self.post_process is None
                seg['spans'] = spans
                if split_long:
                    seg['spans'] = restore_batch(seg['spans'], input_id2raw_id, accumulate_span=True)
                    seg['emissions'] = restore_batch(seg['emissions'], input_id2raw_id)

            if self.post_process:
                assert not verbose
                seg['res'] = self.post_process.adjust_seg(seg['res'])

        # pos任务输出
        if pos['labels']:
            pos['res'] = [spans2pos(s, ps, pp) for (s, ps, pp) in zip(sents, spans, res['pos']['labels'])]
            if split_long:
                pos['labels'] = restore_batch(pos['labels'], input_id2raw_id)
                pos['res'] = restore_batch(pos['res'], input_id2raw_id)

            # TODO：不使用model_config.pos_labels
            if 'vm' not in self.model_config.pos_labels and post_vmvd:
                pos['res'] = adjust_pos(pos['res'])
        return res

    def forward_backbone(self, input_ids):
        outputs = self.bert(input_ids, attention_mask=input_ids.gt(0))
        outputs = outputs[0]

        if len(self.model_config.seg_labels) == 2 and False:  # 用差异作为分割依据 Constrastive Search Constrastive Seg
            outputs = F.normalize(outputs, p=2, dim=-1)
            outputs = outputs[:, 1:, :]*outputs[:, :-1, :]
            logits: torch.Tensor = outputs.sum(-1, keepdim=True)-0.97
            logits = torch.cat((torch.zeros_like(logits), logits), -1)
            return logits

        outputs = outputs[:, 1:, :]  # 去除[CLS]标签位置，获得与label对齐的outputs
        return outputs

    def forward(self, task, input_ids, labels=None, teacher_model=None, teacher_input_ids=None):
        logits = self.heads[task](self.forward_backbone(input_ids))
        if labels is None:
            return (logits,)
        else:
            loss = self.heads[task].loss_func(logits, labels, teacher_model, teacher_input_ids, task)
            return (loss, logits)

    def batch_to_device(self, batch):
        device = self.dinfo.device
        for i in range(len(batch)):
            if isinstance(batch[i], torch.Tensor):
                batch[i] = batch[i].to(device)
        return batch

    def collate_fn(self, task, batch):
        # STEP1: tokens和labels都变成id
        sent_only = (task == 'any')  # none表示没有真实label标签, 数据集只有文本输入
        if sent_only:
            sents, corpus_names = [x[0] for x in batch], [x[1] for x in batch]
        else:
            sents, labels, corpus_names = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]  # "李学勤", ['S', 'B', 'E'] -> [101, 3330, 2110, 1249]
            labels = [[parse_label2id(t, self.heads[task].label2id) for t in tag] for tag in labels]

        input_ids = []
        for sent, corpus_name in zip(sents, corpus_names):
            tokens = [self.tokenizer.tokenize(char) for char in sent]  # NOTE 主要做大小写转换
            corpus_tag = self.corpus_map.get(corpus_name, '[CLS]')
            tokens = [corpus_tag] + [token[0] if token else '[UNK]' for token in tokens]  # NOTE 一些非法字符例如 '' tokenize会变空[], 在convert_tokens_to_ids会导致这个位置缺失
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))  # 不等于tokenizer.encode(tokens)[:-1]
            """ NOTE: tokenize + convert_tokens_to_ids 与 encode 的不同
                由于label和sentence是逐字对齐的，不能用encode或encode_plus
                区别一，input_id的值会不同：    1996 -> 1, 9, 9, 6 | 1, ##9, ##9, ##6
                区别二，input_id的长度会不同：  12月 -> 1, 2, 月 | 12, 月
                ['[CLS]', '目', '前', '状', '况', '１', '９', '９', '６', '年', '１', '２', '月', '退', '休', '后', '忙', '于', '搞', '宣', '传', '作', '报', '告']
                [101, 4680, 1184, 4307, 1105, 8029, 8037, 8037, 8034, 2399, 8029, 8030, 3299, 6842, 828, 1400, 2564, 754, 3018, 2146, 837, 868, 2845, 1440]
                [101, 4680, 1184, 4307, 1105, 8029, 9174, 9174, 9234, 2399, 10351, 3299, 6842, 828, 1400, 2564, 754, 3018, 2146, 837, 868, 2845, 1440]
            """

        # STEP2: input_ids和labels按最大长度进行pad
        batch_size = len(batch)
        max_length = max([len(ids) for ids in input_ids])
        for i in range(batch_size):
            padded_length = max_length - len(input_ids[i])  # 始终是input_ids比labels长1（加了CLS）
            input_ids[i] += [0]*padded_length  # NOTE padded_input_id=0
            if not sent_only:
                labels[i] += [-1]*padded_length  # NOTE padded_label=-1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        if sent_only:
            return [input_ids, sents]
        else:
            labels = torch.tensor(labels, dtype=torch.long)
            return [input_ids, labels, sents]

    def fit(self, train_config, train_loaders, dev_loaders, model_dir, evaluator, batch_order: List[int], teacher_model=None):
        """
        batch_order has the same length with the steps in one epoch. batch_order[i]代表第i个step使用哪个训练集的batch训练
        """
        from transformers.optimization import get_cosine_schedule_with_warmup, AdamW  # import error on Apple M1
        if not hasattr(train_config, 'epoch_num'):
            train_config.epoch_num = 1
        epoch_num = train_config.epoch_num
        learning_rate = train_config.learning_rate
        weight_decay = train_config.weight_decay
        clip_grad = train_config.clip_grad
        batch_size = train_config.batch_size

        model = self
        # 设置optimizer
        # model.named_parameters(): [bert, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = []
        for task in model.heads:
            if task == 'bound':
                classifier_optimizer += list(model.heads[task].classifier_left_bound.named_parameters())+list(model.heads[task].classifier_right_bound.named_parameters())
            else:
                classifier_optimizer += list(model.heads[task].classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': learning_rate * 5, 'weight_decay': weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': learning_rate * 5, 'weight_decay': 0.0},
        ]
        for task in model.heads:
            if model.heads[task].use_crf:
                optimizer_grouped_parameters += [{'params': model.heads[task].crf.parameters(), 'lr': learning_rate * 5}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)

        steps_per_epoch = len(batch_order)
        training_steps = epoch_num * steps_per_epoch

        if hasattr(train_config, 'eval_steps'):
            class LinearSteps:
                def __contains__(self, item):
                    return item > 0 and item % train_config.eval_steps == 0
            eval_steps = LinearSteps()
        else:
            if hasattr(train_config, 'eval_exp_base'):
                exp_base = train_config.eval_exp_base
            else:
                exp_base = 50

            class ExpSteps: # 从base按2的指数倍
                def __contains__(self, item):
                    return (item > 0 and item % exp_base == 0 and (item//exp_base) & ((item//exp_base) - 1) == 0) or (item % 12800 == 0 and item != 0)
            eval_steps = ExpSteps()

        if not hasattr(train_config, 'warmup_steps'):
            train_config.warmup_steps = 0
        warmup_steps = train_config.warmup_steps 
        warmup_steps = warmup_steps if type(warmup_steps) == int else int(warmup_steps * steps_per_epoch)  # 2 * len(train_loader)
        logging.info(f'warmup_steps: {warmup_steps}')
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

        accelerator = self.dinfo.accelerator
        is_main = self.dinfo.is_main
        if accelerator:
            # model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
            model, optimizer = accelerator.prepare(model, optimizer)
            for idx, loader in enumerate(train_loaders):
                train_loaders[idx] = accelerator.prepare(loader)
                train_loaders[idx].task = loader.task

        def save_checkpoint(path):
            if accelerator:
                accelerator.unwrap_model(model).save(path)
            else:
                model.save(path)
            logging.info("best model saved.")

        def evaluate_all():  # 以最后一个dev_loader的指标为准
            examples = self.seg(hard_cases+funny_cases+long_cases+ancient_cases, batch_size=1, split_long=False)['seg']['res']
            logging.info('\n'+'\n'.join(['/'.join(segs) for segs in examples]))
            for dev_loader in dev_loaders:
                score = evaluator(dev_loader, self, save_result=False, saved_path=model_dir)
            return score

        if is_main:
            best_f1 = evaluate_all()

        for train_loader in train_loaders:
            train_loader.collate_fn = functools.partial(self.collate_fn, train_loader.task)
        data_iterators = [iter(dataloader) for dataloader in train_loaders]
        for dev_loader in dev_loaders:
            dev_loader.collate_fn = functools.partial(self.collate_fn, dev_loader.task)
        if teacher_model:
            train_loader_teacher = DataLoader(train_loader.dataset, batch_size=batch_size, num_workers=4)
            train_loader_teacher.collate_fn = teacher_model.collate_fn
            train_loader_teacher = iter(train_loader_teacher)

        logging.info("--Start Training--")
        for epoch in range(1, epoch_num + 1):
            model.train()
            train_losses = 0
            bar = tqdm(total=steps_per_epoch, disable=not is_main)

            # 实现Multitask learning训练方法 https://towardsdatascience.com/when-multi-task-learning-meet-with-bert-d1c49cc40a0c
            for i, train_idx in enumerate(batch_order):  # batch是哪个train_loader的batch
                data_iterator = data_iterators[train_idx]
                try:
                    batch_samples = next(data_iterator)
                except StopIteration:
                    data_iterators[train_idx] = iter(train_loaders[train_idx])
                    batch_samples = next(data_iterators[train_idx])

                if accelerator:
                    input_ids, labels, sents = batch_samples
                    # print(self.tokenizer.convert_ids_to_tokens(input_ids[0]))
                    # 由于只broadcast tensor，sent为空，见accelerate.data_loader.DataLoaderDispatcher._fetch_batches
                else:
                    input_ids, labels, sents = self.batch_to_device(batch_samples)

                if teacher_model:
                    teacher_input_ids, _, _ = self.batch_to_device(next(train_loader_teacher))
                    assert torch.equal(teacher_input_ids.gt(0).sum(-1), input_ids.gt(0).sum(-1))  # 内容token的长度一样
                else:
                    teacher_input_ids = None

                loss = model(train_loaders[train_idx].task, input_ids, labels=labels, teacher_model=teacher_model, teacher_input_ids=teacher_input_ids)[0]
                train_losses += loss.item()
                model.zero_grad()
                if accelerator:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_grad)  # gradient clipping
                optimizer.step()  # 更新参数
                lr_scheduler.step()
                bar.update()

                if accelerator:
                    accelerator.wait_for_everyone()
                if i in eval_steps and is_main:
                    logging.info(f'lr: {round(lr_scheduler.get_last_lr()[0],7)*1e5:.2f}e-5')
                    save_checkpoint(str(model_dir)+'/'+str(i*batch_size))
                    evaluate_all()

            if is_main:
                train_loss = float(train_losses) / steps_per_epoch
                logging.info(f"epoch: {epoch}, train loss: {train_loss}")
                f1 = evaluate_all()
                if f1 - best_f1 > 1e-5:
                    best_f1 = f1
                    save_checkpoint(model_dir)
        if is_main:
            logging.info("--Complete trainning--")

    def add_user_dict(self, file: str):
        self.post_process = UserDict(file)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        logger.info(f"save model to {path}")
        store_yaml(self.model_config, os.path.join(path, "config.yaml"))
        self.bert.config.save_pretrained(path)
        self.tokenizer.save_vocabulary(path)
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))

    @classmethod
    def load(cls, path: str = '', device: Union[str, torch.device] = 'cpu', use_f16=False, cache_dir=None):
        from transformers import logging
        from huggingface_hub import snapshot_download
        logging.set_verbosity_error()
        if not path:
            path = snapshot_download(repo_id="chengzl18/deepthulac-seg", cache_dir=cache_dir)
        dinfo = DistributedInfo(device=device)
        config = load_yaml(os.path.join(path, "config.yaml"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = cls(config, dinfo, pretrained_path=path)
        # TODO: 这还需要load和todevice吗
        model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=dinfo.device))
        model.to(device=dinfo.device)
        if device == 'cpu':
            use_f16 = False
        if use_f16:
            model.quantize_f16()
        return model

    def quantize_f16(self):
        self.half()

    def export_to_onnx(self):
        self.quantize_f16()
        # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb
        export_model_path = '.cache/temp.onnx'
        device = self.dinfo.device

        sentences = ['我爱北京天安门。']
        test_dataset = make_pair(sentences, ['E'*len(sent) for sent in sentences])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_loader.collate_fn = self.collate_fn
        data = next(iter(test_loader))
        inputs = {
            'input_ids': data[0].to(device)
        }

        self.eval()
        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(self,                                            # model being run
                              args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
                              f=export_model_path,                              # where to save the model (can be a file or file-like object)
                              verbose=False,
                              do_constant_folding=True,                         # whether to execute constant folding for optimization
                              input_names=['input_ids'],                        # the model's input names
                              output_names=['logits'],                    # the model's output names
                              dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                            'logits': symbolic_names})

    def init_onnx(self):
        import onnxruntime as ort
        import psutil
        self.export_to_onnx()
        self.crf.cpu()
        sess_options = ort.SessionOptions()
        sess_options.optimized_model_filepath = ".cache/temp.onnx"
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        self.onnx_model = ort.InferenceSession(".cache/temp.onnx", sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    def print_size(self):
        torch.save(self.state_dict(), ".cache/temp")
        print('Size (MB):', os.path.getsize(".cache/temp")/1e6)
        os.remove('.cache/temp')

    def save_to_hub(self,
                    repo_name: str,
                    organization: Optional[str] = None,
                    private: Optional[bool] = None,
                    commit_message: str = "A new DeepTHULAC model.",
                    local_model_path: Optional[str] = None,
                    exist_ok: bool = False,
                    replace_model_card: bool = False,
                    train_datasets: Optional[List[str]] = None):
        """ 将模型上传到 HuggingFace Hub """
        from huggingface_hub import HfApi, HfFolder, Repository
        import stat
        import tempfile
        from distutils.dir_util import copy_tree
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("You must login to the Hugging Face hub on this computer by typing `huggingface-cli login`.")

        if '/' in repo_name:
            splits = repo_name.split('/', maxsplit=1)
            if organization is None or organization == splits[0]:
                organization = splits[0]
                repo_name = splits[1]
            else:
                raise ValueError("You passed and invalid repository name: {}.".format(repo_name))

        endpoint = "https://huggingface.co"
        repo_url = HfApi(endpoint=endpoint).create_repo(
            token,
            repo_name,
            organization=organization,
            private=private,
            repo_type=None,
            exist_ok=exist_ok,
        )
        full_model_name = repo_url[len(endpoint)+1:].strip("/")

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info("Create repository and clone it if it exists")
            repo = Repository(tmp_dir, clone_from=repo_url)

            if local_model_path:
                copy_tree(local_model_path, tmp_dir)
            else:
                create_model_card = replace_model_card or not os.path.exists(os.path.join(tmp_dir, 'README.md'))
                self.save(tmp_dir)

            large_files = []
            for root, dirs, files in os.walk(tmp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, tmp_dir)

                    if os.path.getsize(file_path) > (5 * 1024 * 1024):
                        large_files.append(rel_path)

            if len(large_files) > 0:
                logger.info("Track files with git lfs: {}".format(", ".join(large_files)))
                repo.lfs_track(large_files)

            logger.info("Push model to the hub. This might take a while")
            push_return = repo.push_to_hub(commit_message=commit_message)

            def on_rm_error(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    os.unlink(path)
                except:
                    pass

            try:
                for f in os.listdir(tmp_dir):
                    shutil.rmtree(os.path.join(tmp_dir, f), onerror=on_rm_error)
            except Exception as e:
                logger.warning("Error when deleting temp folder: {}".format(str(e)))
                pass

        return push_return
