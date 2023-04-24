import os
import logging
import numpy as np
from deepthulac.seg.model import LacModel
from deepthulac.utils import load_lines, store_json, store_lines
from deepthulac.seg.seg_utils import *
from deepthulac.eval.test import eval_res, f1_score, accuracy

def save_bad_cases(sents, pred_labels, true_labels, saved_file):
    lines = []
    for i, (s, p, t) in enumerate(zip(sents, pred_labels, true_labels)):
        pp, tt = ' '.join(list(p)), ' '.join(list(t))
        if pp != tt:
            lines.append(f'id: {i}\nsent: {s}\npred: {pp}\ntrue: {tt}\n')
    store_lines(lines, saved_file)


def save_segs(pred_segs, saved_file):
    lines = [' '.join(segs) for segs in pred_segs]
    # 把太长而拆开的句子最后合并起来
    for i in range(len(lines)):
        if lines[i][-1] != '@':
            lines[i] += '\n'
        else:
            lines[i] = lines[i][:-1]
    lines = ''.join(lines).split('\n')
    store_lines(lines, saved_file)


def analyse_results(sents, pred_emissions, pred_labels, true_labels, saved_path):
    saved_path = f'{saved_path}/analyse'
    os.makedirs(saved_path, exist_ok=True)

    def save_verbose_results():
        """ 将各个分类概率写入文件 """
        lines = []
        for i in range(len(sents)):
            line = ''
            for j in range(len(sents[i])):
                top1, top2 = sorted(pred_emissions[i][j], reverse=True)[:2]
                all_prob_values = ' '.join(['{0:>6}'.format('{:.2f}'.format(x)) for x in pred_emissions[i][j]])
                top2_prob_diff = '{:.2f}'.format(top1-top2)
                error_hint = '\t!' if true_labels[i][j] != pred_labels[i][j] else ''
                line += f'{sents[i][j]}\t{true_labels[i][j]} {pred_labels[i][j]}\t[{all_prob_values}]\t{top2_prob_diff}{error_hint}\n'
            lines.append(line+'\n')
        print(f'{saved_path}/details.txt')
        store_lines(lines, f'{saved_path}/details.txt')

    # 将所有样本的信息汇总成一维
    all_true, all_pred, all_emission, sentid = [], [], [], []
    for i in range(len(sents)):
        all_true.extend(true_labels[i])
        all_pred.extend(pred_labels[i])
        all_emission.extend(pred_emissions[i])
        sentid.extend([i]*len(sents[i]))

    def calc_mean_var():
        """ 计算四种标签的均值方差 """
        label_topvals = {'B': [], 'M': [], 'E': [], 'S': []}
        label_right_topvals = {'B': [], 'M': [], 'E': [], 'S': []}
        label_wrong_topvals = {'B': [], 'M': [], 'E': [], 'S': []}
        for i in range(len(all_emission)):
            top1, top2 = sorted(all_emission[i], reverse=True)[:2]
            val = top1-top2
            label = all_true[i]
            label_topvals[label].append(val)
            if all_true[i] != all_pred[i]:
                label_wrong_topvals[label].append(val)
            else:
                label_right_topvals[label].append(val)
        lines = []
        for c, vals in [('right', label_right_topvals), ('wrong', label_wrong_topvals), ('all  ', label_topvals)]:
            for label in ['B', 'M', 'E', 'S']:
                m = np.mean(vals[label])
                v = np.var(vals[label])
                lines.append('{} {} 均值{:.2f}, 方差{:.2f}'.format(c, label, m, v))
        store_lines(lines, f'{saved_path}/mean_var.txt')

    def draw_top2_diff_distribution():
        import matplotlib.pyplot as plt

        def draw_hist(x, file, colors='red'):
            plt.figure()
            plt.hist(x, 50, histtype='barstacked', stacked=True, color=colors, alpha=0.75)  # density=True 绘制概率密度值
            plt.xlabel('top2 diff')
            plt.savefig(file)
        # 统计错误率和top2概率差值的分布
        top2_diff, top2_diff_wrong, top2_diff_right = [], [], []
        for i in range(len(all_emission)):
            top1, top2 = sorted(all_emission[i], reverse=True)[:2]
            val = top1-top2
            top2_diff.append(val)
            if all_true[i] != all_pred[i]:  # NOTE: 标签预测错误和概率最高不完全一致
                top2_diff_wrong.append(val)
            else:
                top2_diff_right.append(val)

        draw_hist(top2_diff_wrong,  f'{saved_path}/top2_diff_wrong.png')
        draw_hist(top2_diff_right, f'{saved_path}/top2_diff_right.png')
        draw_hist([top2_diff_wrong, top2_diff_right], f'{saved_path}/top2_diff_stacked.png', colors=['red', 'green'])

    save_verbose_results()
    # calc_mean_var()
    draw_top2_diff_distribution()


class SegEvaluator:
    @staticmethod
    def eval_model(dev_loader, model: LacModel, save_result, saved_path, split_long, grain=0.5) -> float:
        dataset = dev_loader.dataset
        # TODO: 接口太混乱了
        if dev_loader.task == 'seg' or dev_loader.task == 'punc':
            punc = dev_loader.task == 'punc'
            if type(dataset) == list and type(dataset[0]) == str and (not save_result):  # 直接用分词的句子进行评价
                lines = dataset
                sents = [line.replace(' ', '') for line in lines]
                true_segs = [line.split(' ') for line in lines]
                pred_segs = model.seg(sents, split_long=split_long, punc=punc)
                f1, p, r = eval_res(dev_loader.task, pred_segs, true_segs)
            else:  # 用标签进行评价
                # 训练的时候必须用这里的 # TODO: 接口混乱，再整理
                sents, true_labels = [x[0] for x in dataset], [x[1] for x in dataset]
                true_spans = [labels2spans(list(labels), model.seg_mode) for labels in true_labels]

                verbose = True
                seg = model.seg(sents, verbose=verbose, grain=grain, split_long=split_long, punc=punc)[dev_loader.task]
                pred_labels, pred_spans, pred_results, pred_emissions = seg['labels'], seg['spans'], seg['res'], seg['emissions']

                if save_result:
                    save_bad_cases(sents, pred_labels, true_labels, saved_path + '/bad_case.txt')
                    save_segs(pred_results, saved_path + '/res.txt')
                    print(sents, pred_emissions, pred_labels, true_labels)
                    analyse_results(sents, pred_emissions, pred_labels, true_labels, saved_path)

                f1, p, r = f1_score(pred_spans, true_spans)
            logging.info("{}, f1 score: {}, precision: {}, recall: {}".format(dev_loader.task, f1, p, r))
            return f1
        elif dev_loader.task == 'pos':
            sents, true_labels = [x[0] for x in dataset], [x[1] for x in dataset]
            verbose = True
            pos = model.seg(sents, split_long=split_long)['pos']
            acc = accuracy(pos['labels'], true_labels)
            logging.info("{}, accuracy: {}".format(dev_loader.task, acc))
            return acc
        elif dev_loader.task == 'seg_pos':
            sents, true_labels = [x[0] for x in dataset], [x[1] for x in dataset]
            verbose = True
            seg_pos = model.seg(sents, split_long=split_long)['seg_pos']
            acc = accuracy(seg_pos['labels'], true_labels)
            logging.info("{}, accuracy: {}".format(dev_loader.task, acc))
            return acc



if __name__ == "__main__":
    from deepthulac.legacy import thulac
    import ltp
    import hanlp
    TEST_FILE = ''
    PRED_FILE = ".cache/temp_pred.txt"
    cuda_device = 2

    lac = thulac.load()  # 只进行分词，不进行词性标注
    deeplac = Seg.load(device=f'cuda:{cuda_device}', use_f16=False)
    ltp_tok = ltp.LTP("LTP/base1", cache_dir='.cache', local_files_only=True)
    ltp_tok.cuda(cuda_device)
    hanlp_tok_fine = hanlp.load('FINE_ELECTRA_SMALL_ZH')
    hanlp_tok_coarse = hanlp.load('COARSE_ELECTRA_SMALL_ZH')

    def test_thulac(lines):
        sents = [line.replace(' ', '') for line in lines]
        pred = lac.seg(sents, show_progress_bar=False)
        lines = [' '.join(p) for p in pred]
        store_lines(lines, PRED_FILE)
        f1, p, r = SegEvaluator.eval_files(PRED_FILE, TEST_FILE)
        print("THULAC:\nf1 score: {}, precision: {}, recall: {}".format(f1, p, r))
        print("{:.2f}/{:.2f}/{:.2f}".format(f1*100, p*100, r*100))

    def test_ltp(lines):
        sents = [line.replace(' ', '') for line in lines]
        pred = []
        batch_size = 128  # NOTE: 要根据显存来定，太大了CUDA内存会不够
        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i:min(i+batch_size, len(sents))]
            pred.extend(ltp_tok.pipeline(batch_sents, tasks=["cws"]).cws)
        lines = [' '.join(p) for p in pred]
        store_lines(lines, PRED_FILE)
        f1, p, r = SegEvaluator.eval_files(PRED_FILE, TEST_FILE)
        print("LTP:\nf1 score: {}, precision: {}, recall: {}".format(f1, p, r))
        print("{:.2f}/{:.2f}/{:.2f}".format(f1*100, p*100, r*100))

    def test_hanlp_fine(lines):
        sents = [line.replace(' ', '') for line in lines]
        pred = []
        batch_size = 128  # NOTE: 要根据显存来定，太大了CUDA内存会不够
        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i:min(i+batch_size, len(sents))]
            pred.extend(hanlp_tok_fine(batch_sents))
        lines = [' '.join(p) for p in pred]
        store_lines(lines, PRED_FILE)
        f1, p, r = SegEvaluator.eval_files(PRED_FILE, TEST_FILE)
        print("hanlp fine:\nf1 score: {}, precision: {}, recall: {}".format(f1, p, r))
        print("{:.2f}/{:.2f}/{:.2f}".format(f1*100, p*100, r*100))

    def test_hanlp_coarse(lines):
        sents = [line.replace(' ', '') for line in lines]
        pred = []
        batch_size = 128  # NOTE: 要根据显存来定，太大了CUDA内存会不够
        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i:min(i+batch_size, len(sents))]
            pred.extend(hanlp_tok_coarse(batch_sents))
        lines = [' '.join(p) for p in pred]
        store_lines(lines, PRED_FILE)
        f1, p, r = SegEvaluator.eval_files(PRED_FILE, TEST_FILE)
        print("hanlp coarse:\nf1 score: {}, precision: {}, recall: {}".format(f1, p, r))
        print("{:.2f}/{:.2f}/{:.2f}".format(f1*100, p*100, r*100))

    def test_deepthulac(lines):
        from deepthulac.seg.model import Seg
        sents = [line.replace(' ', '') for line in lines]
        pred = deeplac.seg(sents, show_progress_bar=False)
        lines = [' '.join(p) for p in pred]
        store_lines(lines, PRED_FILE)
        f1, p, r = SegEvaluator.eval_files(PRED_FILE, TEST_FILE)
        print("DeepTHULAC:\nf1 score: {}, precision: {}, recall: {}".format(f1, p, r))
        print("{:.2f}/{:.2f}/{:.2f}".format(f1*100, p*100, r*100))

    for file in [
        "rmrb2014_sample",
        "cbt_sample",
        "our_seg_test",
        "pku_test_gold.txt",
        "msr_test_gold.txt",
        "nlpcc_wordseg_weibo",
        "as_testing_gold",
        "cityu_test_gold",
    ]:
        TEST_FILE = "data/seg_eval/"+file
        test_lines = load_lines(TEST_FILE)
        print('\n'+'*'*20+f' {file} '+'*'*20)
        test_thulac(test_lines)
        test_ltp(test_lines)
        test_hanlp_coarse(test_lines)
        test_hanlp_fine(test_lines)
        test_deepthulac(test_lines)
