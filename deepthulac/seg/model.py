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

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
from munch import Munch
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from torch.utils.data import DataLoader
from deepthulac.seg.post_process import UserDict
from deepthulac.seg.seg_utils import *
from deepthulac.seg.cut_sent import restore_batch, split_batch
from deepthulac.seg.data_format import make_pair
from deepthulac.utils import load_yaml, set_device, store_yaml
import torch.nn.functional as F


def kd_ce_loss(logits_S, logits_T, temperature):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


class Seg(nn.Module):
    def __init__(self, config, device_config, checkpoint_dir=None, pretrained_model_dir='./pretrained'):
        num_labels = len(config.labels)

        custom = (config.pretrained_bert_model == 'custom_pretrained')
        if checkpoint_dir:
            bert = AutoModel.from_pretrained(checkpoint_dir)
        else:
            bert = AutoModel.from_pretrained(pretrained_model_dir if custom else config.pretrained_bert_model)
        if custom:
            self.pretrained_model_dir = checkpoint_dir if checkpoint_dir else pretrained_model_dir
            tokenizer = BertTokenizer.from_pretrained(f'{self.pretrained_model_dir}/vocab.txt')
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert_model)
        super(Seg, self).__init__()

        self.bert = bert
        self.tokenizer: AutoTokenizer = tokenizer
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert.config.hidden_size, num_labels)

        # crf具体实现 https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea
        self.crf = CRF(num_labels, batch_first=True)

        self.label2id = {label: i for i, label in enumerate(config.labels)}
        self.id2label = {_id: _label for _label, _id in list(self.label2id.items())}
        self.mode = 'BMES' if len(config.labels) == 4 else 'EN'
        self.model_config = config
        self.device_config = device_config

        self.onnx_model = None
        self.post_process = None

    def seg(self, sentences, verbose=False, grain=0.5, batch_size=16, split_long=True, show_progress_bar=True):
        if split_long:
            sentences, input_id2raw_id = split_batch(sentences)

        test_dataset = make_pair(sentences, ['E'*len(sent) for sent in sentences])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loader.collate_fn = self.collate_fn
        model = self
        model.eval()

        pred_labels, pred_emissions, sents = [], [], []
        with torch.no_grad():
            for batch_samples in tqdm(test_loader, disable=not show_progress_bar):
                if self.onnx_model:
                    input_ids, labels, batch_sents = batch_samples
                else:
                    input_ids, labels, batch_sents = model.batch_to_device(batch_samples)
                sents.extend(batch_sents)
                label_masks = labels.gt(-1)  # (batch_size, max_len)
                if self.onnx_model:
                    with torch.no_grad():
                        inputs = {
                            'input_ids': input_ids.cpu().numpy()
                        }
                        outputs = self.onnx_model.run(None, inputs)[0]
                        outputs = torch.tensor(outputs, device=input_ids.device)
                else:
                    outputs = model(input_ids)[0]  # (batch_size, max_len, num_labels)

                if verbose:
                    pred_emissions.extend([outputs[i][label_masks[i]].cpu().numpy().tolist() for i in range(len(outputs))])
                if self.model_config.use_crf:
                    if len(model.model_config.labels) == 4:
                        labels = model.crf.decode(outputs, mask=label_masks)  # (batch_size, max_len - padded_len)
                    else:
                        if grain == 0.5:
                            labels = model.crf.decode(outputs, mask=label_masks)
                        else:
                            outputs = softmax(outputs, dim=-1)
                            outputs = outputs[:, :, 0] > grain
                            labels = []
                            for output, label_mask in zip(outputs, label_masks):
                                labels.append([0 if output[i] else 1 for i in range(sum(label_mask)-1)]+[0])  # 最后一个一定是End
                else:
                    outputs = softmax(outputs, dim=-1)
                    outputs = torch.argmax(outputs, dim=-1)
                    labels = []
                    for output, label_mask in zip(outputs, label_masks):
                        labels.append([output[i].item() for i in range(sum(label_mask))])

                pred_labels.extend([[model.id2label.get(idx) for idx in indices] for indices in labels])

        pred_spans = [labels2spans(labels, model.mode) for labels in pred_labels]
        pred_segs = [spans2segs(s, p) for (s, p) in zip(sents, pred_spans)]
        if split_long:
            pred_labels = restore_batch(pred_labels, input_id2raw_id)
            pred_spans = restore_batch(pred_spans, input_id2raw_id, accumulate_span=True)
            pred_segs = restore_batch(pred_segs, input_id2raw_id)
        if verbose:
            assert self.post_process is None
            if split_long:
                pred_emissions = restore_batch(pred_emissions, input_id2raw_id)
            return pred_labels, pred_spans, pred_segs, pred_emissions
        else:
            if self.post_process:
                pred_segs = self.post_process.adjust_segs(pred_segs)
            return pred_segs

    def f(self, input_ids):
        outputs = self.bert(input_ids, attention_mask=input_ids.gt(0))
        outputs = outputs[0]

        if len(self.model_config.labels) == 2 and False:  # 用差异作为分割依据
            outputs = F.normalize(outputs, p=2, dim=-1)
            outputs = outputs[:, 1:, :]*outputs[:, :-1, :]
            logits: torch.Tensor = outputs.sum(-1, keepdim=True)-0.97
            logits = torch.cat((torch.zeros_like(logits), logits), -1)
            return logits

        outputs = outputs[:, 1:, :]  # 去除[CLS]标签位置，获得与label对齐的outputs
        outputs = self.dropout(outputs)  # TODO: dropout调参
        logits = self.classifier(outputs)
        return logits

    def forward(self, input_ids, labels=None, teacher_model=None, teacher_input_ids=None):
        logits = self.f(input_ids)
        outputs = (logits,)
        if labels is not None:
            if self.model_config.use_crf:
                loss_mask = labels.gt(-1)  # NOTE padded_label=-1
                loss = self.crf(logits, labels, loss_mask) * (-1)
            else:
                num_labels = len(self.model_config.labels)
                loss_mask = labels.gt(-1)
                loss_mask = loss_mask.view(-1) == 1
                # 只对label实际存在的位置计算loss
                active_logits = logits.view(-1, num_labels)[loss_mask]
                active_labels = labels.view(-1)[loss_mask]
                loss_fct = nn.CrossEntropyLoss()  # NOTE: = softmax + entropy
                loss = loss_fct(active_logits, active_labels)

                if teacher_model:
                    teacher_model.eval()
                    alpha = 0.95
                    loss_fct = kd_ce_loss
                    with torch.no_grad():
                        # 词表不一样，不能用同一份input_data
                        logits_teacher = teacher_model.f(teacher_input_ids)
                    active_logits_teacher = logits_teacher.view(-1, num_labels)[loss_mask]
                    loss_soft = loss_fct(active_logits, active_logits_teacher, temperature=1)
                    # https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/model/net.py
                    # loss_soft = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(active_logits_teacher/T, dim=1)) * ( T * T)
                    loss = loss_soft*alpha + loss*(1. - alpha)

            outputs = (loss,) + outputs
        return outputs

    def batch_to_device(self, batch):
        device = self.device_config.device
        input_ids, labels, sents = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        return input_ids, labels, sents

    def collate_fn(self, batch):
        # TODO label应该在这里处理, dataset应该直接是分词数据就行

        # STEP1: tokens和labels都变成id
        sents, labels = [x[0] for x in batch], [x[1] for x in batch]  # "李学勤", ['S', 'B', 'E'] -> [101, 3330, 2110, 1249]

        input_ids = []
        for sent in sents:
            tokens = [self.tokenizer.tokenize(char) for char in sent]  # NOTE 主要做大小写转换
            tokens = ['[CLS]'] + [token[0] if token else '[UNK]' for token in tokens]  # NOTE 一些非法字符例如 '' tokenize会变空[], 在convert_tokens_to_ids会导致这个位置缺失
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))  # 不等于tokenizer.encode(tokens)[:-1]
            """ NOTE: tokenize + convert_tokens_to_ids 与 encode 的不同
                由于label和sentence是逐字对齐的，不能用encode或encode_plus
                区别一，input_id的值会不同：    1996 -> 1, 9, 9, 6 | 1, ##9, ##9, ##6
                区别二，input_id的长度会不同：  12月 -> 1, 2, 月 | 12, 月
                ['[CLS]', '目', '前', '状', '况', '１', '９', '９', '６', '年', '１', '２', '月', '退', '休', '后', '忙', '于', '搞', '宣', '传', '作', '报', '告']
                [101, 4680, 1184, 4307, 1105, 8029, 8037, 8037, 8034, 2399, 8029, 8030, 3299, 6842, 828, 1400, 2564, 754, 3018, 2146, 837, 868, 2845, 1440]
                [101, 4680, 1184, 4307, 1105, 8029, 9174, 9174, 9234, 2399, 10351, 3299, 6842, 828, 1400, 2564, 754, 3018, 2146, 837, 868, 2845, 1440]
            """
        labels = [[self.label2id.get(t) for t in tag] for tag in labels]

        # STEP2: input_ids和labels按最大长度进行pad
        batch_size = len(batch)
        max_length = max([len(ids) for ids in input_ids])
        for i in range(batch_size):
            padded_length = max_length - len(input_ids[i])  # 始终是input_ids比labels长1（加了CLS）
            input_ids[i] += [0]*padded_length  # NOTE padded_input_id=0
            labels[i] += [-1]*padded_length  # NOTE padded_label=-1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return [input_ids, labels, sents]

    def fit(self, train_config, train_loader, dev_loader, model_dir, evaluator, teacher_model=None):
        from transformers.optimization import get_cosine_schedule_with_warmup, AdamW # import error on Apple M1 

        epoch_num = 1
        learning_rate = train_config.learning_rate
        weight_decay = train_config.weight_decay
        clip_grad = train_config.clip_grad
        batch_size = train_config.batch_size

        device_type = self.device_config.device_type
        local_rank = self.device_config.local_rank
        device = self.device_config.device

        train_loader.collate_fn = self.collate_fn
        dev_loader.collate_fn = self.collate_fn
        if teacher_model:
            train_loader_teacher = DataLoader(train_loader.dataset, batch_size=batch_size, num_workers=4)
            train_loader_teacher.collate_fn = teacher_model.collate_fn
            train_loader_teacher = iter(train_loader_teacher)
        model = self

        # 设置optimizer
        # model.named_parameters(): [bert, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
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
            {'params': model.crf.parameters(), 'lr': learning_rate * 5}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2 * len(train_loader), num_training_steps=epoch_num * len(train_loader))

        if device_type == 'multigpu':
            model = DistributedDataParallel(model, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)

        def save_checkpoint(path):
            #  选择一个进程保存
            if local_rank == 0:
                if device_type == 'multigpu':
                    model.module.save(path)
                else:
                    model.save(path)
                logging.info("best model saved.")

        best_f1 = evaluator(dev_loader, self, save_result=False, saved_path=model_dir)

        logging.info("--Start Training--")
        for epoch in range(1, epoch_num + 1):
            model.train()
            train_losses = 0
            bar = tqdm(total=len(train_loader))
            for idx, batch_samples in enumerate(train_loader):
                input_ids, labels, sents = self.batch_to_device(batch_samples)

                if teacher_model:
                    teacher_input_ids, _, _ = self.batch_to_device(next(train_loader_teacher))
                    assert torch.equal(teacher_input_ids.gt(0).sum(-1), input_ids.gt(0).sum(-1))  # 内容token的长度一样
                else:
                    teacher_input_ids = None

                loss = model(input_ids, labels=labels, teacher_model=teacher_model, teacher_input_ids=teacher_input_ids)[0]
                train_losses += loss.item()
                model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_grad)  # gradient clipping
                optimizer.step()  # 更新参数
                scheduler.step()
                bar.update()

                if idx > 0 and idx % 2048 == 0:
                    save_checkpoint(model_dir+'/'+str((idx+1)*batch_size))
                    evaluator(dev_loader, self, save_result=False, saved_path=model_dir)

            train_loss = float(train_losses) / len(train_loader)
            logging.info(f"epoch: {epoch}, train loss: {train_loss}")

            if local_rank == 0:
                f1 = evaluator(dev_loader, self, save_result=False, saved_path=model_dir)
                if f1 - best_f1 > 1e-5:
                    best_f1 = f1
                    save_checkpoint(model_dir)
        logging.info("--Complete trainning--")

    def add_user_dict(self, file: str):
        self.post_process = UserDict(file)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        logger.info(f"save model to {path}")
        store_yaml(self.model_config, os.path.join(path, "config.yaml"))
        if self.model_config.pretrained_bert_model == 'custom_pretrained':  # 自己预训练的模型存一下config.json
            shutil.copyfile(f'{self.pretrained_model_dir}/config.json', os.path.join(path, "config.json"))
            shutil.copyfile(f'{self.pretrained_model_dir}/vocab.txt', os.path.join(path, "vocab.txt"))
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))

    @classmethod
    def load(cls, path: str = '', device: Union[str, Munch] = 'cpu', use_f16=True, cache_dir=None):
        from transformers import logging
        from huggingface_hub import snapshot_download
        logging.set_verbosity_error()
        if not path:
            path = snapshot_download(repo_id="chengzl18/deepthulac-seg", cache_dir=cache_dir)
        if device == 'cpu':
            use_f16 = False
        if type(device) == str:
            assert device == 'cpu' or device.startswith('cuda')
            device_type = 'cpu' if device == 'cpu' else 'gpu'

            _, local_rank = set_device(device_type)  # TODO: device ?

            device_config = Munch()
            device_config.device_type = device_type
            device_config.local_rank = local_rank
            device_config.device = device
        else:
            device_config = device
        config = load_yaml(os.path.join(path, "config.yaml"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = cls(config, device_config, checkpoint_dir=path)
        model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=device_config.device))
        model.to(device=device_config.device)
        if use_f16:
            model.quantize_f16()
        return model

    def quantize_f16(self):
        self.half()

    def export_to_onnx(self):
        self.quantize_f16()
        # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb
        export_model_path = '.cache/temp.onnx'
        device = self.device_config.device

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
