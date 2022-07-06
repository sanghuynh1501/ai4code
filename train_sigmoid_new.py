import pickle
import sys

import fasttext
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (DATA_DIR, EARLY_STOP, FASTTEST_MODEL, GAMMA,
                    LEARNING_RATE, MAX_TREE_DEPTH, NW, POS_WEIGHT,
                    REGULARIZATION, SUBSAMPLE, TREE_METHOD)
from dataset import SigMoidDatasetNew
from helper import get_features_new, get_features_sigmoid_text_all

np.random.seed(0)
fasttext_model = fasttext.load_model(FASTTEST_MODEL)

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

unique_ids = pd.unique(train_df['id'])
ids = unique_ids[:100000]
train_df = train_df[train_df['id'].isin(ids)]
train_df["pct_rank"] = train_df["rank"] / \
    train_df.groupby("id")["cell_id"].transform("count")

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:1000]
val_df = val_df[val_df['id'].isin(ids)]
val_df["pct_rank"] = val_df["rank"] / \
    val_df.groupby("id")["cell_id"].transform("count")

xTrain, yTrain = get_features_sigmoid_text_all(
    train_df, dict_cellid_source, 'train')

print(len(xTrain))
print(len(yTrain))

pipe = make_pipeline(TfidfVectorizer(min_df=5, max_features=5000, norm=None),
                     LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(xTrain, yTrain)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))

vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]

full_df = pd.concat([train_df, val_df], ignore_index=True)
vectors_batch = full_df['source'].astype(str).to_list()
cell_ids = full_df['cell_id']

train_features = get_features_new(train_df, 'sigmoid')
val_features = get_features_new(val_df, 'test')

model = xgb.XGBClassifier(tree_method=TREE_METHOD,
                          max_depth=MAX_TREE_DEPTH,
                          alpha=REGULARIZATION,
                          gamma=GAMMA,
                          subsample=SUBSAMPLE,
                          scale_pos_weight=POS_WEIGHT,
                          learning_rate=LEARNING_RATE,
                          objective='binary:logistic'
                          )

train_ds = SigMoidDatasetNew(dict_cellid_source, vectorizer, train_features)
val_ds = SigMoidDatasetNew(dict_cellid_source, vectorizer, val_features)

train_loader = DataLoader(train_ds, batch_size=20000, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=20000, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

train_bar = tqdm(train_loader, file=sys.stdout)
val_bar = tqdm(val_loader, file=sys.stdout)

xTrain = None
yTrain = None
xVal = None
yVal = None

for idx, (features, targets, _) in enumerate(train_bar):
    features, targets = features.cpu().detach().numpy(), targets.cpu().detach().numpy()
    targets = np.reshape(targets, (-1,))
    if xTrain is None:
        xTrain = features
        yTrain = targets
    else:
        xTrain = np.concatenate([xTrain, features], 0)
        yTrain = np.concatenate([yTrain, targets], 0)

for idx, (features, targets, _) in enumerate(val_bar):
    features, targets = features.cpu().detach().numpy(), targets.cpu().detach().numpy()
    targets = np.reshape(targets, (-1,))
    if xVal is None:
        xVal = features
        yVal = targets
    else:
        xVal = np.concatenate([xVal, features], 0)
        yVal = np.concatenate([yVal, targets], 0)

print(xTrain.shape, yTrain.shape, xVal.shape, yVal.shape)
model.fit(xTrain, yTrain, eval_set=[
          (xTrain, yTrain), (xVal, yVal)], early_stopping_rounds=EARLY_STOP, verbose=20)

yPred = model.predict_proba(xVal, ntree_limit=model.best_ntree_limit)
score = log_loss(yVal, yPred)
print(f'logloss:{score:.4f}')
yPred = np.argmax(yPred, -1)
print(yPred.shape, yVal.shape)
total_true = np.sum((yVal == yPred).astype(np.int8))
print('accurancy ', total_true / len(yVal))
