import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import BERT_MODEL_PATH, RANKS


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(769, len(RANKS))

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        x = self.top(x)
        return x
