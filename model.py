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
        self.top = nn.Linear(768, 1)
        self.relu = nn.ReLU()

    def forward(self, ids, mask):
        x = self.bert(ids, mask)[0]
        # x = self.drop(x)
        x = self.top(x[:, 0, :])
        return x


class PairWiseModel(nn.Module):
    def __init__(self):
        super(PairWiseModel, self).__init__()
        self.code_model = ScoreModel()

    def forward(self, mark, mark_mask, code, code_mask):

        mark = self.code_model(mark, mark_mask)
        code = self.code_model(code, code_mask)

        distance = torch.subtract(mark, code)

        return torch.sigmoid(distance)

    def get_score(self, mark, mark_mask):
        score = self.code_model(mark, mark_mask)

        return score
