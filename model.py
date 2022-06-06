import torch
from torch import nn
from config import BERT_PATH
from transformers import RobertaModel


def linear(x):
    return x


def sigmoid(x):
    return torch.sigmoid(x)


class ScoreModel(nn.Module):
    def __init__(self, pair_mode=False):
        super(ScoreModel, self).__init__()
        self.bert = RobertaModel.from_pretrained(BERT_PATH)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(0.2)
        self.top = nn.Linear(768, 1)
        if pair_mode:
            self.activation = linear
        else:
            self.activation = sigmoid

    def forward(self, ids, mask):
        x = self.bert(ids, mask)[0]
        x = self.drop(x)
        x = self.top(x[:, 0, :])
        return self.activation(x)


class PairWiseModel(nn.Module):
    def __init__(self):
        super(PairWiseModel, self).__init__()
        self.score_model = ScoreModel(True)

    def forward(self, ids, mask):
        left_ids = ids[:, 0, :]
        right_ids = ids[:, 1, :]

        left_mask = mask[:, 0, :]
        right_mask = mask[:, 1, :]

        left = self.score_model(left_ids, left_mask)
        right = self.score_model(right_ids, right_mask)

        distance = torch.subtract(left, right, alpha=1)

        return torch.sigmoid(distance)

    def score(self, ids, mask):
        score = self.score_model(ids, mask)

        return score
