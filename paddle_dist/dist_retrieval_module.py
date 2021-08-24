import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddlenlp.transformers import BertModel, BertPretrainedModel, BertTokenizer
from paddle.nn import TransformerEncoder, Linear, Layer, Embedding, LayerNorm, Tanh
import pickle
import numpy as np
import math
from data_loader import DocumentDataset, REDataset
from paddle.io import SequenceSampler, RandomSampler, DataLoader, BatchSampler


class BertLMHead(Layer):
    def __init__(self, hidden_size, vocab_size, embedding_weights=None):
        super(BertLMHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        # self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        # hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        # print("hidden_states shape:", hidden_states.shape, "decoder_weight shape:", self.decoder_weight.shape)
        hidden_states = paddle.matmul(
            hidden_states, self.decoder_weight) + self.decoder_bias
        # hidden_states = self.decoder(hidden_states)
        return hidden_states


class DensePassageRetriever(BertPretrainedModel):
    def __init__(self, args, shared_doc_embed, weight, gpu):
        super(DensePassageRetriever, self).__init__()
        self.max_length = args.max_seq_len
        self.num_class = args.num_class
        self.batch_size = args.train_batch_size
        self.vector_dim = 768
        self.args = args
        self.num_gpu = args.gpus
        self.gpu = gpu

        self.passage_data = open(args.passage_path, "r").readlines()
        self._doc_embed = shared_doc_embed # None #torch.FloatTensor([len(self.passage_data), config.hidden_size]).cpu()
        self.topK = args.topK
        self.train_bag = pickle.load(open(args.bag_path, "rb"))
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        self.query_encoder = BertModel.from_pretrained(args.model_name_or_path)
        self.passage_encoder = BertModel.from_pretrained(args.model_name_or_path)
        self.RE_encoder = BertModel.from_pretrained(args.model_name_or_path)
        self.label_classifier = nn.Linear(self.vector_dim, self.num_class) #BertLMHead(hidden_size=self.vector_dim, vocab_size=self.num_class)
        self.seqcls_dropout = nn.Dropout(args.dropout_rate)
        self.criterion = nn.NLLLoss(weight=weight)

    def get_topK_BruteForce(self, query_embed, entity_list):
        query_embed_cpu = query_embed.detach().cpu()
        h_list = entity_list[0]
        t_list = entity_list[1]
        vector_id_matrix = np.zeros([len(h_list), self.topK]) #paddle.zeros([len(h_list), self.topK], dtype='int64').cpu()
        for i in range(len(h_list)):
            pair = (h_list[i], t_list[i])
            sent_idx_list = list((self.train_bag[pair]))
            tmp_doc_embed = paddle.to_tensor(self._doc_embed[sent_idx_list, :])
            # tmp_doc_embed = paddle.index_select(self._doc_embed, paddle.to_tensor(sent_idx_list, dtype='int64'), axis=0)
            cand_scores = paddle.sum(query_embed_cpu[i,:] * tmp_doc_embed, axis=-1)
            if len(sent_idx_list) < self.topK:
                relative_idx_list = np.random.choice(list(range(len(sent_idx_list))), self.topK)
            else:
                relative_idx_list = np.array(list(range(len(sent_idx_list))))
            #cand_scores[relative_idx_list]
            tmp_cand_scores = paddle.index_select(cand_scores, paddle.to_tensor(relative_idx_list, dtype="int64"))
            score_vector, top_idxs = paddle.topk(tmp_cand_scores, k=self.topK, axis=-1)
            tmp_idx = relative_idx_list[top_idxs.numpy()]
            vector_id_matrix[i, :] = np.array(sent_idx_list)[tmp_idx]
        vector_id_matrix = paddle.to_tensor(vector_id_matrix)
        return vector_id_matrix

    def update_embeddings(self):
        section_len = math.ceil(len(self.passage_data) / self.num_gpu)
        dataset = DocumentDataset(self.args,
                                  self.max_length,
                                  self.passage_data[self.gpu*section_len: (self.gpu+1)*section_len])
        sampler = SequenceSampler(dataset)
        bs = BatchSampler(sampler=sampler, batch_size=500)
        dataloader = DataLoader(dataset=dataset, batch_sampler=bs)
        self.passage_encoder.eval()
        for step, batch in enumerate(dataloader):
            with paddle.no_grad():
                idx, indexed_tokens, att_mask, token_type_ids = batch
                # print(paddle.shape(indexed_tokens))
                # print(paddle.shape(att_mask))
                # print(paddle.shape(token_type_ids))
                # print("###############")
                indexed_tokens = indexed_tokens.cuda()
                att_mask = att_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                outputs = self.passage_encoder(indexed_tokens, attention_mask=att_mask, token_type_ids=token_type_ids)
                passage_embed = outputs[1]  # [CLS]
                p_l2 = paddle.norm(passage_embed, p=2, axis=1).detach()
                passage_embed = paddle.divide(passage_embed, p_l2.unsqueeze(-1).expand_as(passage_embed)) #passage_embed.div(p_l2.unsqueeze(-1).expand_as(passage_embed))
                self._doc_embed[self.gpu*section_len+idx,:] = passage_embed.cpu().numpy()
                # logging.info("Update Embeddings: %d" % step)
            # self.passage_encoder.train()
            # return
        self.passage_encoder.train()

    def forward(self, query_tokens, query_att_mask, query_type_ids, entity_list, label):
        query_outputs = self.query_encoder(query_tokens, attention_mask=query_att_mask, token_type_ids=query_type_ids)
        query_embed = query_outputs[1] # [CLS]

        topK_doc_id = self.get_topK_BruteForce(query_embed, entity_list)  # [batch, topK]
        topK_doc_id = topK_doc_id.reshape((-1,))
        docs = []
        for k, p_idx in enumerate(topK_doc_id):
            doc = self.passage_data[p_idx]
            docs.append(doc)

        # P(z|x)
        dataset = DocumentDataset(self.args, self.max_length, docs)
        sampler = SequenceSampler(dataset)
        bs = BatchSampler(sampler=sampler, batch_size=self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_sampler=bs)
        passage_embeddings = paddle.zeros([len(docs), 768])
        for step, batch in enumerate(dataloader):
            doc_idx, doc_tokens, doc_att_mask, doc_type_ids = batch
            doc_tokens = doc_tokens.cuda()
            doc_att_mask = doc_att_mask.cuda()
            doc_type_ids = doc_type_ids.cuda()
            doc_outputs = self.passage_encoder(doc_tokens, attention_mask=doc_att_mask, token_type_ids=doc_type_ids)
            passage_embed = doc_outputs[1]  # [CLS]
            p_l2 = paddle.norm(passage_embed, p=2, axis=1).detach()
            # passage_embed = passage_embed.div(p_l2.unsqueeze(-1).expand_as(passage_embed))
            passage_embed = paddle.divide(passage_embed, p_l2.unsqueeze(-1).expand_as(passage_embed))
            passage_embeddings[doc_idx,:] = passage_embed
        #     passage_embeddings.append(passage_embed)
        # passage_embeddings = paddle.stack(passage_embeddings, axis=0)
        passage_embeddings = paddle.reshape(passage_embeddings,
                                            [paddle.shape(query_embed)[0], self.topK, self.vector_dim])

        cand_scores = (passage_embeddings * query_embed.unsqueeze(1)).sum(-1)
        logSoftmax = nn.LogSoftmax(-1)
        candidate_log_probs = logSoftmax(cand_scores) # [batch_size, num_candidates]

        # P(y|x, z)
        dataset = REDataset(self.args, docs, entity_list, self.topK, self.max_length)
        sampler = SequenceSampler(dataset)
        bs = BatchSampler(sampler=sampler, batch_size=self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_sampler=bs)
        pair_embeddings = paddle.zeros([len(docs), 768])
        for step, batch in enumerate(dataloader):
            doc_idx, indexed_tokens, att_mask, token_type = batch
            indexed_tokens = indexed_tokens.cuda()
            att_mask = att_mask.cuda()
            token_type = token_type.cuda()
            RE_outputs = self.RE_encoder(indexed_tokens, attention_mask=att_mask, token_type_ids=token_type)
            RE_pooled_output = RE_outputs[1]  # [CLS]
            pair_embeddings[doc_idx, :] = RE_pooled_output
        #     pair_embeddings.append(RE_pooled_output)
        # pair_embeddings = paddle.stack(pair_embeddings, axis=0)
        logits = self.label_classifier(pair_embeddings)  # [batch*topK, num_class]
        logits = paddle.reshape(logits, [paddle.shape(query_tokens)[0], self.topK, self.num_class])

        # Marginal Probability: P(y|x)
        logit_log_probs = logSoftmax(logits)
        marginal_prob = paddle.matmul(paddle.exp(candidate_log_probs).unsqueeze(1),
                                      paddle.exp(logit_log_probs)).squeeze(1)
        marginal_log_prob = paddle.log(marginal_prob)
        outputs = (marginal_log_prob, cand_scores, topK_doc_id)

        if not (label is None):
            marginal_log_prob = paddle.reshape(marginal_log_prob, [-1, self.num_class])
            label = paddle.reshape(label, [-1])
            loss = self.criterion(marginal_log_prob, label)
            outputs = (loss, ) + outputs
        return outputs
