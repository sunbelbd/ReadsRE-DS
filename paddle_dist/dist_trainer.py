import paddle
import paddle.nn as nn
import pickle
import numpy as np
import math
from paddle.io import SequenceSampler, RandomSampler, DataLoader, BatchSampler
from evaluation import load_nyt_bags
import logging
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
import paddle.distributed as dist
from collections import OrderedDict
from os.path import join
import json
import os

from dist_retrieval_module import DensePassageRetriever
from data_loader import BertRetrievalDataset
from utils import get_optimizer
from evaluation import compute_pr_curve_by_pair

logging.getLogger().setLevel(logging.INFO)

dataDir = "../nyt10"
rel_file = join(dataDir, "nyt10_rel2id.json")
train_file = join(dataDir, "nyt10_train.txt")
test_file = join(dataDir, "nyt10_test.txt")
dev_file = join(dataDir, "nyt10_val.txt")


def train(args):
    # def __init__(self, args):
    shared_doc_embed = args.shared_doc_embed
    train_data = open(train_file, "r").readlines()
    rel2id = json.load(open(rel_file, "r"))
    id2rel = {}
    for rel in rel2id:
        id = rel2id[rel]
        id2rel[id] = rel
    args.num_class = len(rel2id)
    train_dataset = BertRetrievalDataset(args, args.max_seq_len, train_data, rel2id)

    # dist.init_parallel_env()
    fleet.init(is_collective=True)
    group = dist.new_group(list(range(args.gpus)))
    gpu = dist.get_rank()
    model = DensePassageRetriever(args, shared_doc_embed, train_dataset.weight, gpu=gpu)
    # model = paddle.DataParallel(model)
    model = fleet.distributed_model(model)

    parameters = set_parameters(model)

    # set optimizers
    optimizers = set_optimizers(parameters)

    # if args.multi_gpu and paddle.distributed.get_world_size() > 1:
    logging.info("%d: Using multi-gpu ..." % gpu)

    batch_sampler = DistributedBatchSampler(train_dataset,
                                            batch_size=args.train_batch_size,
                                            shuffle=True)
    dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        num_workers=0,
        batch_sampler=batch_sampler
    )
    t_total = len(dataloader) * args.num_train_epochs

    logging.info("***** Running training *****")
    logging.info("%d: Num examples = %d" % (gpu, len(train_dataset)))
    logging.info("%d: Num Epochs = %d" % (gpu, args.num_train_epochs))
    logging.info("%d: Total train batch size = %d" % (gpu, args.train_batch_size))
    logging.info("%d: Total optimization steps = %d" % (gpu, t_total))
    logging.info("%d: Logging steps = %d" % (gpu, args.logging_steps))

    global_step = 0
    tr_loss = 0.0
    # self.model.train()
    model.clear_gradients()
    best_f1 = 0.0
    for _ in range(int(args.num_train_epochs)):
        for step, batch in enumerate(dataloader):
            model.train()
            if global_step % args.update_MIPS_steps == 0:
                logging.info("## Updating Passage Embeddings ##")
                model._layers.update_embeddings()
                logging.info("## Finish Updating Passage Embeddings ##")
                dist.barrier(group)

            # with paddle.amp.auto_cast(enable=self.args.fp16):
            idx, query_tokens, query_att_mask, query_type_ids, entity_list, label = batch
            query_tokens = query_tokens.cuda()
            query_att_mask = query_att_mask.cuda()
            query_type_ids = query_type_ids.cuda()
            label = paddle.reshape(label, [-1]).cuda() #label.view(-1).cuda()
            outputs = model(query_tokens,
                                 query_att_mask,
                                 query_type_ids,
                                 entity_list,
                                 label)
            loss = outputs[0]
            # print(loss)
            tr_loss += loss.cpu().numpy()[0]

            optimize(args, optimizers, model, loss)
            global_step += 1

            if global_step % 100 == 0 or global_step==1:
                logging.info("***** train results *****")
                logging.info("Train Loss: [%d, %f]" % (global_step, tr_loss / global_step))
                candidate_probs = outputs[2].detach().cpu().numpy()
                topK_doc_id = outputs[3].cpu().numpy()
                for i in range(len(idx)):
                    logging.info("[%d, %d]: %s" % (i, idx[i], ' '.join(map(str, candidate_probs[i, :]))))
                    logging.info(list(topK_doc_id[i * args.topK: (i + 1) * args.topK]))

            if global_step > 0 and global_step % args.logging_steps == 0:
                if gpu == 0:
                    eval_results = evaluate(args, model, rel2id, "test", output_dir="save_NYT10", gpu=gpu)
                    micro_f1 = eval_results["P/R AUC"]
                    if micro_f1 > best_f1:
                        best_f1 = micro_f1
                        save_model(args, model)
                dist.barrier(group)
    return global_step, tr_loss / global_step


def set_parameters(model):
    """
    Set parameters.
    """
    parameters = {}
    named_params = [(k, p) for k, p in model.named_parameters() if not p.stop_gradient]

    # model (excluding memory values)
    parameters['model'] = [p for k, p in named_params]
    # log
    for k, v in parameters.items():
        logging.info("Found %i parameters in %s." % (len(v), k))
        assert len(v) >= 1
    return parameters


def set_optimizers(parameters):
    """
    Set optimizers.
    """
    optimizers = {}

    # model optimizer (excluding memory values)
    optimizers['model'] = get_optimizer(parameters['model'], "adamW")
    optimizers['model'] = fleet.distributed_optimizer(optimizers['model'])
    # log
    logging.info("Optimizers: %s" % ", ".join(optimizers.keys()))
    return optimizers


def optimize(args, optimizers, model, loss, mode='model'):
    """
    Optimize. Not ready for AMP paddle yet
    """

    params = args
    # optimizers
    # optimizers = [v for k, v in self.optimizers.items()]
    optimizer = optimizers[mode]
    # for optimizer in optimizers:
    #    optimizer.clear_grad()

    # regular optimization: already updated for paddle
    # if not params.fp16:
    # print("Loss before backward:", loss)
    loss.backward()
    # print("Loss after backward:", loss)
    # check unused parameters
    for name, param in model.named_parameters():
        if param.grad is None:
            print("grad is none:", name, ", param.stop_gradient:", param.stop_gradient)
    optimizer.step()
    optimizer.clear_grad()


def evaluate(args, model, rel2id, mode, output_dir, gpu):
    model.eval()
    id2rel = {}
    for rel in rel2id:
        id = rel2id[rel]
        id2rel[id] = rel
    if mode == "test":
        test_data = open(test_file, "r").readlines()
        test_dataset = BertRetrievalDataset(args,
                                            args.max_seq_len, test_data,
                                            rel2id)
        dataset = test_dataset
        test_bag = load_nyt_bags(test_file)
    elif mode == "dev":
        dev_data = open(dev_file, "r").readlines()
        dev_dataset = BertRetrievalDataset(args, args.max_seq_len, dev_data, rel2id)
        dataset = dev_dataset
        test_bag = load_nyt_bags(dev_file)
    else:
        raise Exception("Only dev and test dataset available")

    batch_sampler = DistributedBatchSampler(dataset,
                                            batch_size=args.eval_batch_size,
                                            shuffle=False)
    eval_dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)

    # Eval!
    logging.info("***** Running evaluation on "+mode+" dataset *****")
    logging.info("  Num examples = %d" % len(dataset))
    logging.info("  Batch size = %d" % args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    model.eval()
    pair_list = []
    for iter, batch in enumerate(eval_dataloader):
        with paddle.no_grad():
            sent_id, query_tokens, query_att_mask, query_type_ids, entity_list, label = batch
            query_tokens = query_tokens.cuda()
            query_att_mask = query_att_mask.cuda()
            query_type_ids = query_type_ids.cuda()
            label = paddle.reshape(label, [-1]).cuda()
            outputs = model(query_tokens, query_att_mask, query_type_ids, entity_list, label)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.cpu().numpy()[0]
        nb_eval_steps += 1
        head_list = entity_list[0]
        tail_list = entity_list[1]
        pairs = []

        for i in range(len(head_list)):
            pair = (head_list[i], tail_list[i])
            pairs.append(pair)
        if preds is None:
            preds = paddle.exp(logits).detach().cpu().numpy()
            out_label_ids = label.detach().cpu().numpy()
            pair_list += pairs
        else:
            preds = np.append(preds, paddle.exp(logits).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)
            pair_list += pairs
        # break
    eval_loss = eval_loss / nb_eval_steps
    results = {"loss": eval_loss}

    result = compute_pr_curve_by_pair(out_label_ids, preds, pair_list,
                                      id2rel, test_bag, output_dir=output_dir)
    results.update(result)
    logging.info("***** "+ mode+" results *****")
    for key in sorted(results.keys()):
        logging.info("{} = {:.4f}".format(key, results[key]))
    return results



def save_model(args, model):
    paddle.save(model.state_dict(), './bert_result/model.pdparams')
    logging.info("Save Model")

    paddle.save(args, "./bert_result/args.pth")


def load_model():
    args = paddle.load("./bert_result/args.pth")
    model = DensePassageRetriever(args, args.shared_doc_embed, None, gpu=0)
    state_dict = paddle.load('./bert_result/model.pdparams')
    model.set_state_dict(state_dict)
    logging.info("Load Model")
