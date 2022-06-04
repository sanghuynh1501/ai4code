import torch
from torch import nn
from transformers import AutoModel


class ScoreModel(nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained(
            'distilbert-base-uncased')
        self.top = nn.Linear(768, 32)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.dropout(x)
        x = self.relu(self.top(x[:, 0, :]))
        return x


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.score_model = ScoreModel()
        self.top = nn.Linear(32, 1)

    def forward(self, ids, mask):
        left_ids = ids[:, 0, :]
        right_ids = ids[:, 1, :]

        left_mask = mask[:, 0, :]
        right_mask = mask[:, 1, :]

        left = self.score_model(left_ids, left_mask)
        right = self.score_model(right_ids, right_mask)
        distance = torch.subtract(left, right, alpha=1)

        return torch.sigmoid(self.top(distance))
