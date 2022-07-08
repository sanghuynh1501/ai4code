
import pickle

import pandas as pd
from torch.utils.data import DataLoader

from config import NW
from dataset import SigMoidDatasetNew
from helper import get_features_new, sigmoid_validate_detail

val_df = pd.read_csv('data_dump/val_df.csv')
with open('data_dump/dict_cellid_source.pkl', 'rb') as f:
    dict_cellid_source = pickle.load(f)
f.close()

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:1000]
val_df = val_df[val_df['id'].isin(ids)]
val_df["pct_rank"] = val_df["rank"] / \
    val_df.groupby("id")["cell_id"].transform("count")

val_features = get_features_new(val_df, 'test')

filename = 'weights/tfidfvectorizer.sav'
grid = pickle.load(open(filename, 'rb'))

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
val_ds = SigMoidDatasetNew(dict_cellid_source, vectorizer, val_features)

val_loader = DataLoader(val_ds, batch_size=20000, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

filename = 'weights/sigmoid_model.sav'
model = pickle.load(open(filename, 'rb'))

acc, true, false, percent = sigmoid_validate_detail(model, val_loader)
print(acc, true, false, percent)
