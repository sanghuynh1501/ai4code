import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import alphabet


class MarkdownDataset(Dataset):

    def __init__(self, df, dict_cellid_source, max_len, mode='train'):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased', do_lower_case=True)
        self.mode = mode

    def __getitem__(self, index):
        row = self.df[index]

        label = row[-1]

        ids = []
        masks = []
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

        idx_numbers = []
        for idx in row[:2]:
            idx_numbers.append([alphabet.index(i) for i in idx])

        return torch.cat(ids, 0), torch.cat(masks, 0), torch.FloatTensor([label]), torch.LongTensor(idx_numbers)

    def __len__(self):
        return len(self.df)


class DatasetTest(Dataset):

    def __init__(self, df, dict_cellid_source, max_len):
        super().__init__()
        self.df = df
        self.max_len = max_len
        self.dict_cellid_source = dict_cellid_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased', do_lower_case=True)

    def __getitem__(self, index):
        row = self.df[index]

        ids = []
        masks = []
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

        idx_numbers = []
        for idx in row[:2]:
            idx_numbers.append([alphabet.index(i) for i in idx])

        return torch.cat(ids, 0), torch.cat(masks, 0), torch.LongTensor(idx_numbers)

    def __len__(self):
        return len(self.df)
