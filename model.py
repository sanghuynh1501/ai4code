import torch.nn.functional as F
import torch.nn as nn
import torch
from config import BERT_MODEL_PATH, RANK_COUNT
from transformers import AutoModel


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(769, RANK_COUNT)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        x = self.softmax(self.top(x))
        return x
