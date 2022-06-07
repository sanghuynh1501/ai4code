import random

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from config import BERT_PATH


class PairWiseRandomDataset(Dataset):

    def __init__(self, df, dict_code, dict_cellid_source, max_len, mode='train'):
        super().__init__()
        self.df = df
        self.random_rate = 0.4
        self.max_len = max_len
        self.dict_code = dict_code
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = RobertaTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True)
        self.mode = mode

    def __getitem__(self, index):
        row = self.df[index]
        note_id = row[0]

        mark_id = row[1]
        mark_rank = row[2]
        label = 0

        note = self.dict_code[note_id]

        idx = 0

        if random.random() >= self.random_rate:
            for i in range(len(note['codes'])):
                if note['ranks'][i] == mark_rank + 1:
                    idx = i
                    break
        else:
            rank = mark_rank
            if len(note['ranks']) > 1:
                while mark_rank == rank or mark_rank + 1 == rank:
                    idx = random.randint(0, note['len'] - 1)
                    rank = note['ranks'][idx]

        code_id = note['codes'][idx]
        code_rank = note['ranks'][idx]

        if code_rank == mark_rank + 1:
            label = 1

        ids = []
        masks = []

        for id in [mark_id, code_id]:
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

    def get_random_item(self):
        index = random.randint(0, len(self.df))
        row = self.df[index]
        note_id = row[0]

        mark_id = row[1]
        mark_rank = row[2]
        label = 0

        note = self.dict_code[note_id]

        idx = 0

        if random.random() >= self.random_rate:
            for i in range(len(note['codes'])):
                if note['ranks'][i] == mark_rank + 1:
                    idx = i
                    break
        else:
            rank = mark_rank
            if len(note['ranks']) > 1:
                while mark_rank == rank or mark_rank + 1 == rank:
                    idx = random.randint(0, note['len'] - 1)
                    rank = note['ranks'][idx]

        code_id = note['codes'][idx]
        code_rank = note['ranks'][idx]

        if code_rank == mark_rank + 1:
            label = 1

        ids = []
        masks = []

        for id in [mark_id, code_id]:
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

        return torch.unsqueeze(torch.cat(ids, 0), 0), torch.unsqueeze(torch.cat(masks, 0), 0), torch.unsqueeze(torch.FloatTensor([label]), 0)

    def __len__(self):
        return len(self.df)


class PairWiseDataset(Dataset):

    def __init__(self, df, dict_cellid_source, max_len, mode='train'):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = RobertaTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True)
        self.mode = mode

    def __getitem__(self, index):
        row = self.df[index]

        ids = []
        masks = []
        label = row[-1]

        for id in row[:2]:
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


class TestDataset(Dataset):

    def __init__(self, df, dict_cellid_source, max_len):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = RobertaTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True)

    def __getitem__(self, index):
        text_id = self.df[index]

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

        return id, mask

    def __len__(self):
        return len(self.df)
