import numpy as np
import torch
from torch import nn
from transformers import RobertaModel

from config import BERT_PATH


class ScoreModel(nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()
        self.bert = RobertaModel.from_pretrained(BERT_PATH)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(0.1)
        self.top = nn.Linear(768, 32)
        self.relu = nn.ReLU()

    def forward(self, ids, mask):
        x = self.bert(ids, mask)[0]
        # x = self.drop(x)
        x = self.relu(self.top(x[:, 0, :]))
        return x


class PairWiseModel(nn.Module):
    def __init__(self):
        super(PairWiseModel, self).__init__()
        self.code_model = ScoreModel()
        self.top = nn.Linear(64, 1)

    def forward(self, mark, mark_mask, code, code_mask):

        mark = self.code_model(mark, mark_mask)
        code = self.code_model(code, code_mask)

        distance = torch.cat([mark, code], -1)
        distance = self.top(distance)

        return torch.sigmoid(distance)

    def get_score(self, mark, code):
        distance = torch.cat([mark, code], -1)
        distance = self.top(distance)

        return torch.sigmoid(distance)

    def get_feature(self, ids, mask):
        feature = self.code_model(ids, mask)

        return feature
