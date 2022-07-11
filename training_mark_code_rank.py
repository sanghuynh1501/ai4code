import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from config import (BS, CODE_MARK_RANK_PATH, DATA_DIR, EPOCH, MARK_PATH, MD_MAX_LEN, NW, SIGMOID_PATH, TOTAL_MAX_LEN,
                    accumulation_steps)
from dataset import MarkdownOnlyDataset, MarkdownRankDataset, SigMoidDataset
from helper import cal_kendall_tau_rank, get_features_mark, get_features_rank, markdown_validate, sigmoid_validate, validate_rank
from model import MarkdownOnlyModel, MarkdownRankModel, SigMoidModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = MarkdownRankModel()
# model.load_state_dict(torch.load(CODE_MARK_PATH))
model = model.cuda()

model_sigmoid = SigMoidModel().to(device)
model_sigmoid.load_state_dict(torch.load(SIGMOID_PATH))
model_sigmoid = model_sigmoid.cuda()

model_mark_only = MarkdownOnlyModel()
model_mark_only.load_state_dict(torch.load(MARK_PATH))
model_mark_only = model_mark_only.cuda()

df_orders = pd.read_csv(
    DATA_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_df = pd.read_csv('data_dump/train_df.csv')
val_df = pd.read_csv('data_dump/val_df.csv')
with open('data_dump/dict_cellid_source.pkl', 'rb') as f:
    dict_cellid_source = pickle.load(f)
f.close()

# unique_ids = pd.unique(train_df['id'])
# ids = unique_ids[:100]
# train_df = train_df[train_df['id'].isin(ids)]
train_df["pct_rank"] = train_df["rank"] / \
    train_df.groupby("id")["cell_id"].transform("count")

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:1000]
val_df = val_df[val_df['id'].isin(ids)]
val_df["pct_rank"] = val_df["rank"] / \
    val_df.groupby("id")["cell_id"].transform("count")

train_fts, all_labels = get_features_rank(train_df, 'classification')
val_fts, _ = get_features_rank(val_df, 'test')
val_fts_only = get_features_mark(val_df, 'test')

class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = class_weights.astype(np.float32)

train_ds = MarkdownRankDataset(
    dict_cellid_source, md_max_len=MD_MAX_LEN, total_max_len=TOTAL_MAX_LEN, fts=train_fts)
val_ds = SigMoidDataset(dict_cellid_source, md_max_len=MD_MAX_LEN,
                        total_max_len=512, fts=val_fts)
val_ds_only = MarkdownOnlyDataset(val_fts_only, dict_cellid_source, 128)

train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS * 8, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

val_loader_only = DataLoader(val_ds_only, batch_size=BS, shuffle=False, num_workers=NW,
                             pin_memory=False, drop_last=False)

acc, true, false, relative, _, _ = sigmoid_validate(
    model_sigmoid, val_loader, device, 0.397705)
print('accurancy ', acc, true, false)
mark_dict = markdown_validate(model_mark_only, val_loader_only, device)


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(
        epochs * len(train_loader) / accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.NLLLoss(
        weight=torch.tensor(class_weights).to(device),
        reduction='mean',
    )
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):
        model.train()
        total_loss = 0
        total_step = 0
        tbar = tqdm(train_loader, file=sys.stdout)

        y_pred = []
        for idx, (ids, mask, fts, code_lens, target) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device),
                             fts.to(device), code_lens.to(device))
                loss = criterion(pred, torch.squeeze(target).to(device))
            scaler.scale(loss).backward()
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.detach().cpu().item()
            total_step += 1

            avg_loss = np.round(total_loss / total_step, 4)

            tbar.set_description(
                f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

            if (idx + 1) % 10000 == 0 or idx == len(tbar) - 1:
                y_pred, _, _ = validate_rank(model, val_loader, device)
                cal_kendall_tau_rank(val_df, y_pred, mark_dict, relative, df_orders)
                torch.save(model.state_dict(), CODE_MARK_RANK_PATH)
                model.train()


train(model, train_loader, val_loader, epochs=EPOCH)
