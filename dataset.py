import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import BERT_PATH


class PairWiseRandomDataset(Dataset):

    def __init__(self, df, dict_code, dict_cellid_source, max_len):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_code = dict_code
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True)

    def __getitem__(self, index):
        row = self.df[index]
        note_id = row[0]

        mark_id = row[1]
        mark_rank = row[2]

        note = self.dict_code[note_id]

        idx = random.randint(0, len(note['codes']) - 1)
        code_id = note['codes'][idx]
        code_rank = note['ranks'][idx]

        label = 0
        if code_rank == mark_rank + 1:
            label = 1
        else:
            if random.random() > 0.7:
                idx = -1
                for i in range(len(note['ranks'])):
                    if note['ranks'][i] == mark_rank + 1:
                        idx = i
                if idx >= 0:
                    code_id = note['codes'][idx]
                    code_rank = note['ranks'][idx]
                    label = 1
                else:
                    label = 0
            else:
                label = 0

        ids = []
        masks = []

        for id in [mark_id, code_id]:
            txt = self.dict_cellid_source[id]

            inputs = self.tokenizer.encode_plus(
                txt,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True
            )
            id = torch.LongTensor(inputs['input_ids'])
            mask = torch.LongTensor(inputs['attention_mask'])

            ids.append(id)
            masks.append(mask)

        return ids[0], masks[0], ids[1], masks[1], torch.FloatTensor([label])

    def __len__(self):
        return len(self.df)


class PointWiseDataset(Dataset):

    def __init__(self, df, dict_cellid_source, max_len):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = AutoTokenizer.from_pretrained(
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
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        id = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return id, mask, torch.FloatTensor([label])

    def __len__(self):
        return len(self.df)


class TestDataset(Dataset):

    def __init__(self, df, dict_cellid_source, max_len):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True)

    def __getitem__(self, index):
        text_id = self.df[index]

        txt = self.dict_cellid_source[text_id]

        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        id = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return id, mask

    def __len__(self):
        return len(self.df)
