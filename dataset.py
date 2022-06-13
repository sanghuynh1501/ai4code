import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import BERT_MODEL_PATH, RANKS


class MarkdownDataset(Dataset):

    def __init__(self, dict_cellid_source, code_dict, total_max_len, md_max_len, fts):
        super().__init__()
        self.dict_cellid_source = dict_cellid_source
        self.md_max_len = md_max_len
        self.code_dict = code_dict
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.fts = fts.reset_index(drop=True)

    def __getitem__(self, index):
        row = self.fts.iloc[index]

        inputs = self.tokenizer.encode_plus(
            self.dict_cellid_source[row['mark']],
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        codes = self.code_dict[row['id']]
        codes = codes[row['code_start']: row['code_end']]
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(self.dict_cellid_source[x]) for x in codes],
            add_special_tokens=True,
            max_length=23,
            padding='max_length',
            truncation=True
        )
        n_md = row['total_md']
        n_code = row['total_md']
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * \
                (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * \
                (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        label = RANKS.index(row['rank'])

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.LongTensor([label])

    def __len__(self):
        return len(self.fts)
