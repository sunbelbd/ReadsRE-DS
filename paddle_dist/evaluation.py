import paddle
import numpy as np
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from scipy.special import softmax
import json
from operator import attrgetter
from sklearn.metrics import auc
from os.path import join
import pickle
import os

def official_eval(pred_result, targets, rel2id):
    """
    Args:
        pred_result: a list of predicted label (id)
            Make sure that the `shuffle` param is set to `False` when getting the loader.
        use_name: if True, `pred_result` contains predicted relation names instead of ids
    Return:
        {'acc': xx}
    """
    correct = 0
    total = len(targets)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg_name = "NA"
    neg = rel2id[neg_name]

    for i in range(total):
        golden = targets[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1

    acc = float(correct) / float(total)
    result = {'acc': acc}
    return result

def precision_recall(pred, targets, num_class):
    """
    :param pred: [batch, num_class]
    :param targets: [batch]
    :param num_class:
    :return:
    """

    # convert multi-class into binary class
    y = np.zeros([len(targets), num_class])
    for i in range(len(targets)):
        y[i, targets[i]] = 1
    pred = np.exp(pred)

    y = y.flatten()
    pred = pred.flatten()

    precision, recall, _ = precision_recall_curve(y, pred)
    auc_score = sklearn.metrics.auc(x=recall, y=precision)
    p_at_r = [0, 0, 0]
    for p, r in zip(precision[::-1], recall[::-1]):
        if r >= 0.1 and p_at_r[0] == 0:
            p_at_r[0] = p
        if r >= 0.2 and p_at_r[1] == 0:
            p_at_r[1] = p
        if r >= 0.3 and p_at_r[2] == 0:
            p_at_r[2] = p
    # print("P@0.1: %.4f\nP@0.2: %.4f\nP@0.3:%.4f" % (p_at_r[0], p_at_r[1], p_at_r[2]))
    result = {"P@0.1": p_at_r[0], "P@0.2": p_at_r[1], "P@0.3": p_at_r[2], "auc": auc_score}
    return result


def instance_id(mention):
    head = mention['h']['name']
    tail = mention['t']['name']
    return (head.lower(), tail.lower())


def load_nyt_bags(file):
    test_bags = {}
    data = open(file, "r").readlines()
    for line in data:
        item = json.loads(line.strip())
        instance = instance_id(item)
        if not (instance in test_bags):
            test_bags[instance] = set([])
        test_bags[instance].add(item["relation"])
    return test_bags


def compute_pr_curve_by_pair(targets, logits_list, entity_pair_list,
                                     id2rel, entity_bag_labels, output_dir=None):
    predict_bag = {}
    for i in range(len(targets)):
        pair = entity_pair_list[i]
        logit = logits_list[i]
        if not pair in predict_bag:
            predict_bag[pair] = []
        predict_bag[pair].append(logit)

    for pair in predict_bag:
        logit_list = predict_bag[pair]
        logit_tensor = paddle.to_tensor(logit_list, dtype='float32')
        # assert paddle.shape(logit_tensor)[0] == len(logit_list)
        # assert paddle.shape(logit_tensor)[1] == len(logit_list[0])
        max_logit = paddle.mean(logit_tensor, axis=0)
        # assert paddle.shape(max_logit)[0] == len(logit_list[0])
        predict_bag[pair] = max_logit

    predictions = []
    num_relation_facts = 0
    n_pos = 0
    for entity_pair in predict_bag:
        bag_labels = entity_bag_labels[entity_pair]
        bag_labels.discard('NA')
        if bag_labels:
            n_pos += 1
        num_relation_facts += len(bag_labels)

        logits = predict_bag[entity_pair]
        for idx, logit in enumerate(logits):
            if id2rel[idx] == "NA":
                continue
            is_correct = id2rel[idx] in bag_labels
            predictions.append({"score":logit, "is_correct": is_correct, "pair": entity_pair})
    print(num_relation_facts)
    predictions = sorted(predictions, key=lambda item: item['score'], reverse=True)

    correct = 0
    precision_values = []
    recall_values = []
    for idx, prediction in enumerate(predictions):
        if prediction["is_correct"]:
            correct += 1
        precision_values.append(correct / (idx + 1))
        recall_values.append(correct / num_relation_facts)

    def precision_at(n):
        return (sum([prediction["is_correct"] for prediction in predictions[:n]]) / n) * 100

    p_at_r = [0, 0, 0]
    for p, r in zip(precision_values, recall_values):
        if r >= 0.1 and p_at_r[0] == 0:
            p_at_r[0] = p
        if r >= 0.2 and p_at_r[1] == 0:
            p_at_r[1] = p
        if r >= 0.3 and p_at_r[2] == 0:
            p_at_r[2] = p

    pr_metrics = {
        'P/R AUC': auc(x=recall_values, y=precision_values),
        'Precision@0.1': p_at_r[0],
        'Precision@0.2': p_at_r[1],
        'Precision@0.3': p_at_r[2],
        'Mean': np.mean(p_at_r)
    }

    if output_dir != None:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        with open(join(output_dir, 'pr_metrics.json'), 'w') as pr_metrics_f:
            json.dump(pr_metrics, pr_metrics_f)

        # with open(join(output_dir, 'predictions.pkl'), 'wb') as predictions_f:
        #     pickle.dump(predictions, predictions_f)

        np.save(join(output_dir, 'precision.npy'), precision_values)
        np.save(join(output_dir, 'recall.npy'), recall_values)

    return pr_metrics


