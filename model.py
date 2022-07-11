import torch
import torch.nn as nn
from transformers import AutoModel

from config import BERT_MODEL_PATH, RANKS


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(770, 1)

    def forward(self, ids, mask, fts, code_lens):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts, code_lens), 1)

        x = self.top(x)
        # x = torch.sigmoid(x)

        return x


class MarkdownRankModel(nn.Module):
    def __init__(self):
        super(MarkdownRankModel, self).__init__()
        self.model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(770, len(RANKS))
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, ids, mask, fts, code_lens):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts, code_lens), 1)

        x = self.top(x)
        x = self.activation(x)

        return x


class MarkdownOnlyModel(nn.Module):
    def __init__(self):
        super(MarkdownOnlyModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x)

        return x


class SigMoidModel(nn.Module):
    def __init__(self):
        super(SigMoidModel, self).__init__()
        self.model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(770, 1)

    def forward(self, ids, mask, fts, code_lens):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts, code_lens), 1)
        x = self.top(x)
        return x
