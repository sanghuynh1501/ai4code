import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import BERT_PATH, BS, CODE_MARK_PATH, CODE_PATH, MARK_PATH, MAX_LEN, NW, DATA_DIR
from dataset import TestDataset
from helper import generate_data_test, generate_mark_code_dict, get_ranks, get_token, kendall_tau, preprocess_text, read_notebook
from model import PairWiseModel, ScoreModel

device = 'cpu'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)


def cal_distance(a, b):
    d = a - b
    return 1/(1 + np.exp(-d))


def get_code(mar, codes, pairs_hash):
    max_codes = []
    for code in codes:
        score = cal_distance(pairs_hash[mar], pairs_hash[code])
        if score > 0.5:
            max_codes.append(code)
            codes.remove(code)
    return max_codes, codes


# model_code = ScoreModel().to(device)
# model_code.load_state_dict(torch.load(CODE_PATH))
# model_code.eval()

# model_markdown = ScoreModel().to(device)
# model_markdown.load_state_dict(torch.load(MARK_PATH))
# model_markdown.eval()

model_pairse = PairWiseModel().to(device)
model_pairse.load_state_dict(torch.load(CODE_PATH))
model_pairse.eval()

paths_train = list((DATA_DIR / 'train').glob('*.json'))[20000:20010]
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
df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

df_code = df[df['cell_type'] == 'code'].reset_index(drop=True)
df_mark = df[df['cell_type'] == 'markdown'].reset_index(drop=True)
df_mark.source = df_mark.source.apply(preprocess_text)

data_code = generate_data_test(df_code)
data_mark = generate_data_test(df_mark)


dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))

val_code = TestDataset(data_code, dict_cellid_source, MAX_LEN)
val_mark = TestDataset(data_mark, dict_cellid_source, MAX_LEN)

loader_code = DataLoader(val_code, batch_size=BS * 2, shuffle=False, num_workers=NW,
                         pin_memory=False, drop_last=False)

loader_mark = DataLoader(val_mark, batch_size=BS * 2, shuffle=False, num_workers=NW,
                         pin_memory=False, drop_last=False)

with torch.no_grad():
    code_orders = []
    with tqdm(total=len(val_code)) as pbar:
        for ids, mask in loader_code:

            predicts = model_pairse.get_score(ids.to(device), mask.to(
                device)).detach().cpu().numpy().ravel()
            code_orders += predicts.tolist()

            pbar.update(len(ids))

    # mark_orders = []
    # with tqdm(total=len(val_mark)) as pbar:
    #     for ids, mask in loader_mark:

    #         predicts = model_pairse.get_mark_score(ids.to(device), mask.to(
    #             device)).detach().cpu().numpy().ravel()
    #         mark_orders += predicts.tolist()

    #         pbar.update(len(ids))

# =======================================================================
df_code['pred'] = code_orders
y_dummy = df_code.groupby('id')['cell_id'].apply(list)
test_tau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
print('test tau code ', test_tau)

# =======================================================================
# df_mark['pred'] = mark_orders
# y_dummy = df_mark.sort_values('pred').groupby('id')['cell_id'].apply(list)
# test_tau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
# print('test tau mark ', test_tau)

# # df_mark = y_dummy.to_frame()
# # df_mark.reset_index(inplace=True)
# # marks = df_mark['cell_id'].tolist()

# # results = []
# # for mark, code in zip(marks, codes):
# #     result = []
# #     for id, ma in enumerate(mark):
# #         result.append(ma)
# #         if ma in markcode:
# #             print(ma, markcode[ma])
# #             result.append(markcode[ma])
# #     results.append(result)

# # test_tau = kendall_tau(df_orders.loc[ids].tolist(), results)
# # print('test tau ', test_tau)

df.loc[df["cell_type"] == "code", "pred"] = code_orders
# # df.loc[df["cell_type"] == "markdown", "pred"] = mark_orders

for id, df_tmp in tqdm(df.groupby('id')):
    df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']
    df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']

    df_tmp_code_pred = df_tmp_code['pred'].values
    df_tmp_code_cell_id = df_tmp_code['cell_id'].values

    df_tmp_mark_pred = df_tmp_markdown['pred'].values
    df_tmp_mark_cell_id = df_tmp_markdown['cell_id'].values

    result_preds = df_tmp_code_pred
    results = df_tmp_code_cell_id

    print('==============================================================')
    print('result_preds ', result_preds)

#     for mark_pred, mark_id in zip(df_tmp_mark_pred, df_tmp_mark_cell_id):
#         print([(cal_distance(mark_pred, pred)) for pred in result_preds])


# df.loc[df["cell_type"] == "markdown", "pred"] = mark_orders
# # df.loc[df["cell_type"] == "code", "pred"] = code_orders
# # df.loc[df["cell_type"] == "code", "pred"] = code_orders
# sub_df = df.sort_values("pred").groupby("id")["cell_id"].apply(list)
# test_tau = kendall_tau(df_orders.loc[sub_df.index], sub_df)

# print('test tau ', test_tau)
