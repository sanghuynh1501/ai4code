import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import BS, CODE_MARK_RANK_PATH, DATA_DIR, MARK_PATH, MD_MAX_LEN, NW, SIGMOID_PATH
from dataset import MarkdownOnlyDataset, MarkdownRankNewDataset, SigMoidDataset
from helper import cal_kendall_tau_inference, get_features_mark, get_features_rank, validate_markdown, preprocess_code, preprocess_text, read_notebook, validate_sigmoid, validate_rank_inference

from model import MarkdownOnlyModel, MarkdownRankModel, SigMoidModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = MarkdownRankModel()
model.load_state_dict(torch.load(CODE_MARK_RANK_PATH))
model = model.cuda()

model_sigmoid = SigMoidModel().to(device)
model_sigmoid.load_state_dict(torch.load(SIGMOID_PATH))
model_sigmoid = model_sigmoid.cuda()

model_mark_only = MarkdownOnlyModel()
model_mark_only.load_state_dict(torch.load(MARK_PATH))
model_mark_only = model_mark_only.cuda()

paths_test = list((DATA_DIR / 'train').glob('*.json'))[-1000:]
notebooks_test = [
    read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
]

df = (
    pd.concat(notebooks_test)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
)

df.reset_index(inplace=True)

df_orders = pd.read_csv(
    DATA_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

df.loc[df['cell_type'] == 'markdown', 'source'] = df[df['cell_type']
                                                     == 'markdown'].source.apply(preprocess_text)

df.loc[df['cell_type'] == 'code', 'source'] = df[df['cell_type']
                                                 == 'code'].source.apply(preprocess_code)

dict_cellid_source = dict(
    zip(df['cell_id'].values, df['source'].values))

df["rank"] = df.groupby(["id", "cell_type"]).cumcount()
df = df.sort_values('rank').reset_index(drop=True)
df["pct_rank"] = df["rank"] / \
    df.groupby("id")["cell_id"].transform("count")

val_fts, _, _ = get_features_rank(df, 'test')
val_fts_only = get_features_mark(df, 'test')

val_ds = SigMoidDataset(dict_cellid_source, md_max_len=MD_MAX_LEN,
                        total_max_len=512, fts=val_fts)
val_ds_only = MarkdownOnlyDataset(val_fts_only, dict_cellid_source, 128)

val_loader = DataLoader(val_ds, batch_size=BS * 8, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)
val_loader_only = DataLoader(val_ds_only, batch_size=BS, shuffle=False, num_workers=NW,
                             pin_memory=False, drop_last=False)

acc, true, false, relative, _, _ = validate_sigmoid(
    model_sigmoid, val_loader, device, 0.397705)
print(acc, true, false)
mark_dict = validate_markdown(model_mark_only, val_loader_only, device)

class_fts = []
one_object = {}
mark_id_dict = {}
for i in range(len(relative)):
    if relative[i] == 1:
        class_fts.append(val_fts[i])
        if val_fts[i]['mark'] not in one_object:
            one_object[val_fts[i]['mark']] = 1
        else:
            del one_object[val_fts[i]['mark']]

for ft in val_fts:
    if ft['mark'] not in one_object:
        mark_id_dict[ft['mark']] = mark_dict[ft['mark']] * \
            (ft['total_code'] + ft['total_md'])

val_ds = MarkdownRankNewDataset(dict_cellid_source, md_max_len=MD_MAX_LEN,
                                total_max_len=512, fts=class_fts)
val_loader = DataLoader(val_ds, batch_size=BS * 8, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)


y_pred, _, acc, mark_dict = validate_rank_inference(model, val_loader, device)
cal_kendall_tau_inference(df, mark_id_dict, mark_dict, df_orders)
