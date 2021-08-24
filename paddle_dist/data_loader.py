from paddle.io import Dataset
import paddle
# from ernie.tokenizing_ernie import ErnieTokenizer
from paddlenlp.transformers import BertModel, BertPretrainedModel, BertTokenizer
import paddle as P
import numpy as np
from os import path
import json

class BertRetrievalDataset(Dataset):
    def __init__(self, args, max_seq_len, doc_data, rel2id, mask_entity=False):
        super(BertRetrievalDataset, self).__init__()
        # self.tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        self.max_length = max_seq_len
        self.data = doc_data
        self.rel2id = rel2id
        self.mask_entity = mask_entity

        self.count = np.ones((len(self.rel2id)), dtype=np.float32)
        for line in self.data:
            item = json.loads(line.strip())
            self.count[self.rel2id[item['relation']]] += 1.0
        self.weight = 1.0 / self.count ** 0.05  # (self.weight ** 0.05)
        self.weight = P.to_tensor(self.weight)
        self.weight = self.weight / P.sum(self.weight)

        self.rel_dict = {}
        self.rel_list = []
        for idx in range(len(self.data)):
            item = json.loads(self.data[idx].strip())
            relation = item["relation"]
            if not (relation in self.rel_dict):
                self.rel_dict[relation] = []
            self.rel_dict[relation].append(idx)
        self.rel_list = list(self.rel_dict)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent_idx = idx
        line = self.data[sent_idx].strip()
        item = json.loads(line)
        text = item["text"]
        h_pos = item["h"]["pos"]
        h_name = item["h"]["name"]
        t_pos = item["t"]["pos"]
        t_name = item["t"]["name"]

        pos_min = h_pos
        pos_max = t_pos
        if h_pos[0] > t_pos[0]:
            pos_min = t_pos
            pos_max = h_pos
            rev = True
        else:
            rev = False
        sent0 = self.tokenizer.tokenize(text[:pos_min[0]])
        ent0 = self.tokenizer.tokenize(text[pos_min[0]:pos_min[1]])
        sent1 = self.tokenizer.tokenize(text[pos_min[1]:pos_max[0]])
        ent1 = self.tokenizer.tokenize(text[pos_max[0]:pos_max[1]])
        sent2 = self.tokenizer.tokenize(text[pos_max[1]:])

        if self.mask_entity:
            query_ent0 = ['[unused4]'] if not rev else ['[unused5]']
            query_ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            query_ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            query_ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
        query_tokens = sent0 + query_ent0 + sent1 + query_ent1 + sent2
        if len(query_tokens) > self.max_length - 2:
            query_tokens = query_tokens[:self.max_length - 2]
        query_tokens = ['[CLS]'] + query_tokens + ['[SEP]']
        query_indexed_tokens = self.tokenizer.convert_tokens_to_ids(query_tokens)
        query_type_ids = [0] * len(query_indexed_tokens)
        query_att_mask = [1] * len(query_indexed_tokens)
        # padding
        padding = [0] * (self.max_length - len(query_indexed_tokens))
        query_indexed_tokens += padding
        query_att_mask += padding
        query_type_ids += padding
        query_indexed_tokens = P.to_tensor(query_indexed_tokens, dtype="int64")
        query_att_mask = P.to_tensor(query_att_mask, dtype="int64")
        query_type_ids = P.to_tensor(query_type_ids, dtype="int64")

        relation = item["relation"]
        label = self.rel2id[relation]
        label = P.to_tensor([label], "int64")
        pair = [h_name.lower(), t_name.lower()]
        return sent_idx, query_indexed_tokens, query_att_mask, query_type_ids, pair, label


class DocumentDataset(Dataset):
    def __init__(self, args, max_seq_len, docs):
        super(DocumentDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        self.max_length = max_seq_len
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        text = self.docs[idx]
        query_token = self.tokenizer.tokenize(text)
        if len(query_token) > self.max_length- 2:
            query_token = query_token[:self.max_length - 2]
        query_token = ['[CLS]'] + query_token + ['[SEP]']
        query_indexed_tokens = self.tokenizer.convert_tokens_to_ids(query_token)
        query_type_ids = [0] * len(query_indexed_tokens)
        query_att_mask = [1] * len(query_indexed_tokens)
        # padding
        padding = [0] * (self.max_length - len(query_token))
        query_indexed_tokens += padding
        query_att_mask += padding
        query_type_ids += padding
        query_indexed_tokens = P.to_tensor(query_indexed_tokens, dtype='int64')
        query_att_mask = P.to_tensor(query_att_mask, dtype='int64')
        query_type_ids = P.to_tensor(query_type_ids, dtype='int64')

        return idx, query_indexed_tokens, query_att_mask, query_type_ids


class REDataset(Dataset):
    def __init__(self, args, chosen_doc, entity_list, topK, max_seq_len, mask_entity=False):
        super(REDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        self.max_length = max_seq_len
        self.chosen_doc = chosen_doc
        self.topK = topK
        self.head_list = entity_list[0]
        self.tail_list = entity_list[1]
        self.mask_entity = mask_entity
        self.entity_length = 10

    def __len__(self):
        return len(self.chosen_doc)

    def __getitem__(self, idx):
        doc = self.chosen_doc[idx]
        query_idx = idx // self.topK
        h_name = self.head_list[query_idx]
        t_name = self.tail_list[query_idx]

        head_tokens = self.tokenizer.tokenize(h_name)
        if len(head_tokens) > self.entity_length - 2:
            head_tokens = head_tokens[:self.entity_length - 2]
        tail_tokens = self.tokenizer.tokenize(t_name)
        if len(tail_tokens) > self.entity_length - 1:
            tail_tokens = head_tokens[:self.entity_length - 1]
        title_tokens = ['[CLS]'] + head_tokens + ['[SEP]'] + tail_tokens + ['[SEP]']
        title_indexed_tokens = self.tokenizer.convert_tokens_to_ids(title_tokens)
        title_type_ids = [0] * len(title_indexed_tokens)
        title_att_mask = [1] * len(title_indexed_tokens)

        query_tokens = self.tokenizer.tokenize(doc)
        if len(query_tokens) > self.max_length - 1:
            query_tokens = query_tokens[:self.max_length - 1]
        query_tokens = query_tokens + ['[SEP]']
        query_indexed_tokens = self.tokenizer.convert_tokens_to_ids(query_tokens)
        query_type_ids = [1] * len(query_indexed_tokens)
        query_att_mask = [1] * len(query_indexed_tokens)
        query_indexed_tokens = title_indexed_tokens + query_indexed_tokens
        query_type_ids = title_type_ids + query_type_ids
        query_att_mask = title_att_mask + query_att_mask
        # padding
        padding = [0] * (self.max_length + self.entity_length * 2 - len(query_indexed_tokens))
        query_indexed_tokens += padding
        query_att_mask += padding
        query_type_ids += padding
        query_indexed_tokens = paddle.to_tensor(query_indexed_tokens, dtype="int64")
        query_att_mask = paddle.to_tensor(query_att_mask, dtype="int64")
        query_type_ids = paddle.to_tensor(query_type_ids, dtype="int64")
        return idx, query_indexed_tokens, query_att_mask, query_type_ids



