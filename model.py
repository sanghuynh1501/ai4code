import torch.nn.functional as F
import torch.nn as nn
import torch
from config import BERT_MODEL_PATH
from transformers import AutoModel


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        x = self.top(x)
        return x