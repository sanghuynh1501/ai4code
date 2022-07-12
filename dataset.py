import torch
from config import BERT_MODEL_PATH, CODE_MAX_LEN, RANK_COUNT, RANKS
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from helper import get_cosine_features, id_to_label


class MarkdownDataset(Dataset):

    def __init__(self, dict_cellid_source, total_max_len, md_max_len, fts):
        super().__init__()
        self.dict_cellid_source = dict_cellid_source
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.fts = fts

    def __getitem__(self, index):
        row = self.fts[index]

        inputs = self.tokenizer.encode_plus(
            self.dict_cellid_source[row['mark']],
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        codes = row['codes']
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(self.dict_cellid_source[x]) for x in codes],
            add_special_tokens=True,
            max_length=CODE_MAX_LEN,
            padding='max_length',
            truncation=True
        )
        n_md = row['total_md']
        n_code = row['total_code']
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

        label = row['pct_rank']

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([len(codes) / RANK_COUNT]), torch.FloatTensor([label])

    def __len__(self):
        return len(self.fts)


class MarkdownRankDataset(Dataset):

    def __init__(self, dict_cellid_source, total_max_len, md_max_len, fts):
        super().__init__()
        self.dict_cellid_source = dict_cellid_source
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.fts = fts

    def __getitem__(self, index):
        row = self.fts[index]

        inputs = self.tokenizer.encode_plus(
            self.dict_cellid_source[row['mark']],
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        codes = row['codes']
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(self.dict_cellid_source[x]) for x in codes],
            add_special_tokens=True,
            max_length=CODE_MAX_LEN,
            padding='max_length',
            truncation=True
        )
        n_md = row['total_md']
        n_code = row['total_code']
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

        label = row['code_rank']

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([len(codes) / RANK_COUNT]), torch.LongTensor([label])

    def __len__(self):
        return len(self.fts)


class MarkdownOnlyDataset(Dataset):

    def __init__(self, fts, dict_cellid_source, max_len):
        super().__init__()
        self.dict_cellid_source = dict_cellid_source
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.fts = fts

    def __getitem__(self, index):
        row = self.fts[index]

        inputs = self.tokenizer.encode_plus(
            self.dict_cellid_source[row['mark']],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([row['pct_rank']]), torch.LongTensor(id_to_label(row['mark']))

    def __len__(self):
        return len(self.fts)


class SigMoidDataset(Dataset):

    def __init__(self, dict_cellid_source, total_max_len, md_max_len, fts):
        super().__init__()
        self.dict_cellid_source = dict_cellid_source
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.fts = fts

    def __getitem__(self, index):
        row = self.fts[index]

        inputs = self.tokenizer.encode_plus(
            self.dict_cellid_source[row['mark']],
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        codes = row['codes']
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(self.dict_cellid_source[x]) for x in codes],
            add_special_tokens=True,
            max_length=CODE_MAX_LEN,
            padding='max_length',
            truncation=True
        )
        n_md = row['total_md']
        n_code = row['total_code']
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

        label = row['pct_rank']
        relative = row['relative']
        total_code_len = row['total_code_len']

        loss_mask = torch.ones(RANK_COUNT + 1,)
        loss_mask[:len(codes) + 1] = 0
        loss_mask = loss_mask.type(torch.ByteTensor)

        assert len(ids) == self.total_max_len

        return ids, mask, fts, loss_mask, torch.FloatTensor([len(codes) / RANK_COUNT]), torch.FloatTensor([label]), torch.FloatTensor([relative]), torch.FloatTensor([total_code_len])

    def __len__(self):
        return len(self.fts)


class SigMoidDatasetNew(Dataset):

    def __init__(self, dict_cellid_source, vectorizer, fts):
        super().__init__()
        self.dict_cellid_source = dict_cellid_source
        self.vectorizer = vectorizer
        self.fts = fts

    def __getitem__(self, index):
        row = self.fts[index]

        codes = row['codes']
        mark = row['mark']
        similarity_scores = get_cosine_features(
            mark, codes, self.dict_cellid_source, self.vectorizer)
        label = row['relative']
        total_code_len = row['total_code_len']

        return torch.FloatTensor(similarity_scores),  torch.FloatTensor([label]), torch.FloatTensor([total_code_len])

    def __len__(self):
        return len(self.fts)


class MarkdownRankNewDataset(Dataset):

    def __init__(self, dict_cellid_source, total_max_len, md_max_len, fts):
        super().__init__()
        self.dict_cellid_source = dict_cellid_source
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.fts = fts

    def __getitem__(self, index):
        row = self.fts[index]

        inputs = self.tokenizer.encode_plus(
            self.dict_cellid_source[row['mark']],
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        codes = row['codes']
        ranks = row['ranks']
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(self.dict_cellid_source[x]) for x in codes],
            add_special_tokens=True,
            max_length=CODE_MAX_LEN,
            padding='max_length',
            truncation=True
        )
        n_md = row['total_md']
        n_code = row['total_code']
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

        label = row['code_rank']

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([len(codes) / RANK_COUNT]), torch.LongTensor([label]), torch.LongTensor(id_to_label(row['mark'])), torch.LongTensor(ranks)

    def __len__(self):
        return len(self.fts)
