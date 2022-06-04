import torch
from torch import nn
from transformers import AutoModel


class RankingModel(nn.Module):
    def __init__(self):
        super(RankingModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained('weights/mymodelpairbertsmallpretrained/models/checkpoint-120000')
        # self.distill_bert = AutoModel.from_pretrained('weights/mymodelpairbertsmallpretrained/models/checkpoint-120000')
        self.top = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
        
    def forward(self, ids, mask):
        batch_size = len(ids)
        
        ids = ids.view(batch_size * 2, -1)
        mask = mask.view(batch_size * 2, -1)
        
        x = self.distill_bert(ids, mask)[0]
        x = x.view(batch_size, 2, 128, -1)
        x = self.dropout(x)
        x = self.top(x[:, :, 0, :])
        x = torch.squeeze(x)
        return x


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.distill_bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.top = nn.Linear(768, 1)

        self.dropout = nn.Dropout(0.25)
        
    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x) 
        return x


if __name__ == '__main__':
    ids = torch.ones(64, 2, 128).long()
    masks = torch.ones(64, 2, 128).long()

    model = RankingModel()
    out = model(ids, masks)

    print(out.shape)