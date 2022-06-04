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

        txt = self.dict_cellid_source[row[0]] + \
            '[SEP]' + self.dict_cellid_source[row[1]]

        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        idx_numbers = []
        for idx in row[:2]:
            idx_numbers.append([alphabet.index(i) for i in idx])

        return ids, mask, torch.FloatTensor([label]), torch.LongTensor(idx_numbers)

    def __len__(self):
        return len(self.df)
