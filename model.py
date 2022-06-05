import torch
from torch import nn
from transformers import AutoModel


class ScoreModel(nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained(
            'distilbert-base-uncased')

        for param in self.distill_bert.parameters():
            param.requires_grad = False

        self.out = nn.Linear(768, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        # x = self.dropout(x)
        x = self.out(x[:, 0, :])
        return x


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.score_model = ScoreModel()

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
