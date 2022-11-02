import os
import logging
import numpy as np
from deepthulac.seg.model import Seg
from deepthulac.utils import load_lines, store_json, store_lines
from deepthulac.seg.seg_utils import *


def f1_score(pred_spans, true_spans):
    """ 通过span列表计算指标 """
    # span是前闭后闭的闭区间
    # 每个span前加上句子编号，构成唯一标识
    y_pred, y_true = [], []
    for i, ys in enumerate(pred_spans):
        y_pred.extend([(i, y[0], y[1]) for y in ys])
    for i, ys in enumerate(true_spans):
        y_true.extend([(i, y[0], y[1]) for y in ys])
    y_pred, y_true = set(y_pred), set(y_true)
    n_pred, n_true, n_correct = len(y_pred), len(y_true), len(y_pred & y_true)
    p = n_correct / n_pred if n_pred > 0 else 0
    r = n_correct / n_true if n_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return score, p, r


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
    calc_mean_var()
    draw_top2_diff_distribution()


class SegEvaluator:
    @staticmethod
    def eval_files(pred_file, true_file):
        pred_segs = [line.split(' ') for line in load_lines(pred_file)]
        true_segs = [line.split(' ') for line in load_lines(true_file)]
        return SegEvaluator.eval_segs(pred_segs, true_segs)

    @staticmethod
    def eval_segs(pred_segs, true_segs):
        pred_spans = [segs2spans(segs) for segs in pred_segs]
        true_spans = [segs2spans(segs) for segs in true_segs]
        return f1_score(pred_spans, true_spans)

    @staticmethod
    def eval_model(dev_loader, model: Seg, save_result, saved_path, split_long, grain=0.5) -> float:
        dataset = dev_loader.dataset
        if type(dataset) == list and type(dataset[0]) == str:  # 直接用分词的句子进行评价
            lines = dataset
            sents = [line.replace(' ', '') for line in lines]
            true_segs = [line.split(' ') for line in lines]
            pred_segs = model.seg(sents, split_long=split_long)
            f1, p, r = SegEvaluator.eval_segs(pred_segs, true_segs)
        else:  # 用标签进行评价
            sents, true_labels = [x[0] for x in dataset], [x[1] for x in dataset]
            true_spans = [labels2spans(list(labels), model.mode) for labels in true_labels]

            verbose = True
            pred_labels, pred_spans, pred_results, pred_emissions = model.seg(sents, verbose=verbose, grain=grain, split_long=split_long)

            if save_result:
                save_bad_cases(sents, pred_labels, true_labels, saved_path + '/bad_case.txt')
                save_segs(pred_results, saved_path + '/res.txt')
                analyse_results(sents, pred_emissions, pred_labels, true_labels, saved_path)

            f1, p, r = f1_score(pred_spans, true_spans)
        logging.info("f1 score: {}, precision: {}, recall: {}".format(f1, p, r))
        return f1


if __name__ == "__main__":
    def test_thulac():
        import thulac
        thulab_tok = thulac.thulac(seg_only=True)  # 只进行分词，不进行词性标注
        lines = load_lines("data/msr_test.txt")
        store_lines([line.replace(' ', '') for line in lines], 'data/temp_sent.txt')
        thulab_tok.cut_f('data/temp_sent.txt', "data/temp_pred.txt")
        f1, p, r = SegEvaluator.eval_files('data/temp_pred.txt', 'data/msr_test.txt')
        print("f1 score: {}, precision: {}, recall: {}".format(f1, p, r))
    test_thulac()
