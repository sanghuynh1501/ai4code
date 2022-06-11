import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from config import DATA_DIR, MARK_PATH
from dataset import MarkdownDataset
from helper import (get_features, get_ranks, kendall_tau, preprocess_text,
                    read_notebook)
from model import MarkdownModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = MarkdownModel().to(device)


def train_step(ids, mask, fts, labels, idx, max_len):
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        pred = model(ids, mask, fts)
        loss = criterion(pred, labels)

    scaler.scale(loss).backward()
    if idx % accumulation_steps == 0 or idx == max_len - 1:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    return loss.item()


def test_step(ids, mask, fts, labels):
    outputs = model(ids, mask, fts)
    loss = criterion(outputs, labels)

    return loss.item()


def predict(ids, mask, fts):
    predictions = model(ids, mask, fts)

    return predictions


paths_train = list((DATA_DIR / 'train').glob('*.json'))
notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
]

df = (
    pd.concat(notebooks_train)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
)

df_orders = pd.read_csv(
    DATA_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

df_ranks = (
    pd.DataFrame
    .from_dict(ranks, orient='index')
    .rename_axis('id')
    .apply(pd.Series.explode)
    .set_index('cell_id', append=True)
)

df_ancestors = pd.read_csv(DATA_DIR / 'train_ancestors.csv', index_col='id')

df = df.reset_index().merge(
    df_ranks, on=['id', 'cell_id']).merge(df_ancestors, on=['id'])
df['pct_rank'] = df['rank'] / df.groupby('id')['cell_id'].transform('count')

NVALID = 0.1

splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df, groups=df['ancestor_id']))
train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)

train_df_mark = train_df[train_df['cell_type']
                         == 'markdown'].reset_index(drop=True)
train_df_mark.source = train_df_mark.source.apply(preprocess_text)

val_df_mark = val_df[val_df['cell_type'] == 'markdown'].reset_index(drop=True)
val_df_mark.source = val_df_mark.source.apply(preprocess_text)

train_fts = get_features(train_df)
val_fts = get_features(val_df)

train_ds = MarkdownDataset(train_df_mark, md_max_len=64,
                           total_max_len=512, fts=train_fts)
val_ds = MarkdownDataset(val_df_mark, md_max_len=64,
                         total_max_len=512, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

EPOCHS = 5
accumulation_steps = 4

num_train_optimization_steps = int(
    EPOCHS * len(train_loader) / accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                            num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

criterion = torch.nn.L1Loss()
scaler = torch.cuda.amp.GradScaler()

max_test_tau = 0

for epoch in range(EPOCHS):

    train_loss = 0
    total_train = 0
    idx = 0

    with tqdm(total=len(train_ds)) as pbar:
        for (ids, mask, fts, labels) in train_loader:
            model.train()
            loss = train_step(ids.to(device), mask.to(
                device), fts.to(device), labels.to(device), idx, int(len(train_ds) / BS))

            train_loss += loss * len(ids)
            total_train += len(ids)

            if idx % 200 == 0:

                test_loss = 0
                total_test = 0
                test_accuracy = 0
                test_tau = 0
                test_preds = []

                model.eval()
                with torch.no_grad():
                    with tqdm(total=len(val_ds)) as tpbar:
                        for tids, tmask, tfts, tlabels in val_loader:
                            loss = test_step(tids.to(device), tmask.to(
                                device), tfts.to(device), tlabels.to(device))

                            test_loss += loss * len(tids)
                            total_test += len(tids)

                            predicts = predict(tids.to(device), tmask.to(
                                device), tfts.to(device)).detach().cpu().numpy().ravel()
                            test_preds += predicts.tolist()

                            tpbar.update(len(tids))

                val_df['rank'] = val_df.groupby(['id', 'cell_type']).cumcount()
                val_df['pred'] = val_df.groupby(['id', 'cell_type'])[
                    'rank'].rank(pct=True)
                val_df.loc[val_df['cell_type'] ==
                           'markdown', 'pred'] = test_preds
                y_dummy = val_df.sort_values('pred').groupby('id')[
                    'cell_id'].apply(list)
                test_tau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)

                if test_tau > max_test_tau:
                    torch.save(model.state_dict(), MARK_PATH)
                    max_test_tau = test_tau

                print(
                    f'Epoch {epoch + 1}, Step {idx}:\n'
                    f'Loss: {train_loss / total_train}, '
                    f'Test Loss: {test_loss / total_test}, '
                    f'Test Tau: {test_tau} '
                )

                train_loss = 0
                total_train = 0

            idx += 1
            pbar.update(len(ids))
