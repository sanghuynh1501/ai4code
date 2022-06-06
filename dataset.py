import random

import torch
from torch.utils.data import Dataset
from config import BERT_PATH
from transformers import RobertaTokenizer


class PairWiseDataset(Dataset):

    def __init__(self, df, data_dic, dict_cellid_source, max_len, mode='train'):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.data_dic = data_dic
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = RobertaTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True)
        self.mode = mode

    def __getitem__(self, index):
        row = self.df[index]

        note_id = row[0]
        note = self.data_dic[note_id]

        cell_id = row[1]
        r_cell_id = cell_id

        rank = row[2]
        r_rank = rank

        while r_cell_id == cell_id:
            r_idx = random.randint(0, note['len'] - 1)
            r_cell_id = note['source'][r_idx]
            r_rank = note['rank'][r_idx]

        label = 1 if rank > r_rank else 0

        ids = []
        masks = []
        for id in [cell_id, r_cell_id]:
            txt = self.dict_cellid_source[id]

            inputs = self.tokenizer.encode_plus(
                txt,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            )
            id = torch.LongTensor(inputs['input_ids'])
            mask = torch.LongTensor(inputs['attention_mask'])

            id = torch.unsqueeze(id, 0)
            mask = torch.unsqueeze(mask, 0)

            ids.append(id)
            masks.append(mask)

        return torch.cat(ids, 0), torch.cat(masks, 0), torch.FloatTensor([label])

    def __len__(self):
        return len(self.df)


class PointWiseDataset(Dataset):

    def __init__(self, df, dict_cellid_source, max_len):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = RobertaTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True)

    def __getitem__(self, index):
        row = self.df[index]
        text_id, label = row

        txt = self.dict_cellid_source[text_id]

        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        id = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return id, mask, torch.FloatTensor([label])

    def __len__(self):
        return len(self.df)
