import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from config import BERT_MODEL_PATH, RANKS


class MarkdownTwoStageModel(nn.Module):
    def __init__(self):
        super(MarkdownTwoStageModel, self).__init__()
        self.model = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.sigmoid_top = nn.Linear(770, 1)
        self.class_top = nn.Linear(771, len(RANKS))
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, ids, mask, fts, code_lens):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts, code_lens), 1)
        sigmoid_x = self.sigmoid_top(x)
        x = torch.cat((x, sigmoid_x), 1)

        class_x = self.class_top(x)
        class_x = self.activation(class_x)

        return sigmoid_x, class_x


class MarkdownOnlyModel(nn.Module):
    def __init__(self):
        super(MarkdownOnlyModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained(BERT_MODEL_PATH)
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.top(x[:, 0, :])

        return x
