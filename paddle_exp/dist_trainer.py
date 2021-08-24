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


class Trainer(object):
    def __init__(self, args):
        self.args = args
        passage_data = open(args.passage_path, "r").readlines()
        if "base" in args.model_name_or_path:
            doc_embed = paddle.zeros([len(passage_data), 768], dtype="float32")
        else:  # large
            doc_embed = paddle.zeros([len(passage_data), 1024], dtype="float32")
        shared_doc_embed = doc_embed

        train_data = open(train_file, "r").readlines()
        self.rel2id = json.load(open(rel_file, "r"))
        self.id2rel = {}
        for rel in self.rel2id:
            id = self.rel2id[rel]
            self.id2rel[id] = rel
        self.args.num_class = len(self.rel2id)
        self.train_dataset = BertRetrievalDataset(args, args.max_seq_len, train_data, self.rel2id)
        self.model = DensePassageRetriever(args, shared_doc_embed, self.train_dataset.weight)
        # self.model.train()
        # set parameters
        self.set_parameters()

        # set optimizers
        self.set_optimizers()

        if args.multi_gpu and paddle.distributed.get_world_size() > 1:
            logging.info("Using paddle.distributed.fleet ...")
            self.model = fleet.distributed_model(self.model)
            self.dist_strategy = fleet.DistributedStrategy()
            for opt_name, optimizer in self.optimizers.items():
                self.optimizers[opt_name] = fleet.distributed_optimizer(optimizer, strategy=self.dist_strategy)

        if args.fp16:
            self.scaler = paddle.amp.GradScaler()

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.gpu = int(os.getenv("FLAGS_selected_gpus", "-1"))

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = [(k, p) for k, p in self.model.named_parameters() if not p.stop_gradient]

        # model (excluding memory values)
        self.parameters['model'] = [p for k, p in named_params]
        # log
        for k, v in self.parameters.items():
            logging.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1


    def set_optimizers(self):
        """
        Set optimizers.
        """
        self.optimizers = {}

        # model optimizer (excluding memory values)
        self.optimizers['model'] = get_optimizer(self.parameters['model'], "adamW")
        # log
        logging.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))


    def optimize(self, loss, mode='model'):
        """
        Optimize. Not ready for AMP paddle yet
        """
        # check NaN
        if (loss != loss).detach().numpy().any():
            logging.warning("NaN detected")
            exit(1)

        params = self.args
        # optimizers
        # optimizers = [v for k, v in self.optimizers.items()]
        optimizer = self.optimizers[mode]
        # for optimizer in optimizers:
        #    optimizer.clear_grad()

        # regular optimization: already updated for paddle
        if not params.fp16:
            # print("Loss before backward:", loss)
            loss.backward()
            # print("Loss after backward:", loss)
            # check unused parameters
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print("grad is none:", name, ", param.stop_gradient:", param.stop_gradient)
            optimizer.step()
        # AMP/FP16 optimization
        else:
            # with paddle.amp.auto_cast():
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()  # do backward
            self.scaler.minimize(optimizer, scaled_loss)
        optimizer.clear_grad()


    def train(self):# Train!
        batch_sampler = DistributedBatchSampler(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        dataloader = DataLoader(
            self.train_dataset,
            shuffle=False,
            num_workers=0,
            batch_sampler=batch_sampler
        )
        t_total = len(dataloader) * self.args.num_train_epochs

        logging.info("***** Running training *****")
        logging.info("%d: Num examples = %d" % (self.gpu, len(self.train_dataset)))
        logging.info("%d: Num Epochs = %d" % (self.gpu, self.args.num_train_epochs))
        logging.info("%d: Total train batch size = %d" % (self.gpu, self.args.train_batch_size))
        logging.info("%d: Total optimization steps = %d" % (self.gpu, t_total))
        logging.info("%d: Logging steps = %d" % (self.gpu, self.args.logging_steps))

        global_step = 0
        tr_loss = 0.0
        # self.model.train()
        self.model.clear_gradients()
        best_f1 = 0.0
        for _ in range(int(self.args.num_train_epochs)):
            for step, batch in enumerate(dataloader):
                self.model.train()
                if global_step % self.args.update_MIPS_steps == 0:
                    logging.info("## Updating Passage Embeddings ##")
                    self.model.update_embeddings()
                    logging.info("## Finish Updating Passage Embeddings ##")

                # with paddle.amp.auto_cast(enable=self.args.fp16):
                idx, query_tokens, query_att_mask, query_type_ids, entity_list, label = batch
                query_tokens = query_tokens.cuda()
                query_att_mask = query_att_mask.cuda()
                query_type_ids = query_type_ids.cuda()
                label = paddle.reshape(label, [-1]).cuda() #label.view(-1).cuda()
                outputs = self.model(query_tokens, query_att_mask, query_type_ids, entity_list, label)
                loss = outputs[0]
                # print(loss)
                # loss.backward()
                tr_loss += loss.numpy().squeeze()

                self.optimize(loss)
                global_step += 1

                if global_step % 100 == 0 or global_step==1:
                    logging.info("***** train results *****")
                    logging.info("Train Loss: [%d, %f]" % (global_step, tr_loss / global_step))
                    candidate_probs = outputs[2].detach().cpu().numpy()
                    topK_doc_id = outputs[3].cpu().numpy()
                    for i in range(len(idx)):
                        logging.info("[%d, %d]: %s" % (i, idx[i], ' '.join(map(str, candidate_probs[i, :]))))
                        logging.info(list(topK_doc_id[i * self.args.topK: (i + 1) * self.args.topK]))

                # eval_results = self.evaluate(mode="test", output_dir="save_NYT10")

                if global_step > 0 and global_step % self.args.logging_steps == 0:
                    eval_results = self.evaluate("test", output_dir="save_NYT10")
                    micro_f1 = eval_results["P/R AUC"]
                    if micro_f1 > best_f1:
                        best_f1 = micro_f1
                        self.save_model()
        return global_step, tr_loss / global_step


    def evaluate(self, mode, output_dir):
        self.model.eval()
        if mode == "test":
            test_data = open(test_file, "r").readlines()
            test_dataset = BertRetrievalDataset(self.args,
                                                self.args.max_seq_len, test_data,
                                                self.rel2id)
            dataset = test_dataset
            test_bag = load_nyt_bags(test_file)
        elif mode == "dev":
            dev_data = open(dev_file, "r").readlines()
            dev_dataset = BertRetrievalDataset(self.args, self.args.max_seq_len, dev_data, self.rel2id)
            dataset = dev_dataset
            test_bag = load_nyt_bags(dev_file)
        else:
            raise Exception("Only dev and test dataset available")

        batch_sampler = DistributedBatchSampler(dataset, batch_size=self.args.eval_batch_size, shuffle=False)
        eval_dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)

        # Eval!
        logging.info("***** Running evaluation on "+mode+" dataset *****")
        logging.info("  Num examples = %d" % len(dataset))
        logging.info("  Batch size = %d" % self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()
        pair_list = []
        for iter, batch in enumerate(eval_dataloader):
            with paddle.no_grad():
                sent_id, query_tokens, query_att_mask, query_type_ids, entity_list, label = batch
                query_tokens = query_tokens.cuda()
                query_att_mask = query_att_mask.cuda()
                query_type_ids = query_type_ids.cuda()
                label = paddle.reshape(label, [-1]).cuda()
                outputs = self.model(query_tokens, query_att_mask, query_type_ids, entity_list, label)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.numpy().squeeze()
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

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        result = compute_pr_curve_by_pair(out_label_ids, preds, pair_list,
                                          self.id2rel, test_bag, output_dir=output_dir)
        results.update(result)
        logging.info("***** "+ mode+" results *****")
        for key in sorted(results.keys()):
            logging.info("  {} = {:.4f}".format(key, results[key]))
        return results


    def save_model(self):
        paddle.save(self.model.state_dict(), './bert_result/model.pdparams')
        logging.info("Save Model")

    def load_model(self):
        self.model = paddle.load('./bert_result/model.pdparams')
        logging.info("Load Model")