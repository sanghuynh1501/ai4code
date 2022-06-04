import pandas as pd
from tqdm import tqdm

from config import data_dir
from helper import check_english, get_ranks, preprocess_text, read_notebook

TYPE = 'code'

paths_train = list((data_dir / 'train').glob('*.json'))
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
    data_dir / 'train_orders.csv',
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

df.source = df.source.apply(preprocess_text)
df = df.reset_index().merge(df_ranks, on=["id", "cell_id"])

if TYPE != 'code':
    df['lang'] = df.source.apply(check_english)
    df = df.loc[(df['lang'] == 'en')].reset_index(drop=True)

df = df.loc[(df['cell_type'] == TYPE) & (df['source'].notnull()) & (
    df['source'] != '')].reset_index(drop=True)

print(len(df.index))
df.to_csv(f'{TYPE}_dataset.csv')
