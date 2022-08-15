import pickle
import secrets
import string

import pandas as pd

from helper import preprocess_code, preprocess_text


def gen_code():
    N = 8
    res = ''.join(secrets.choice(string.ascii_uppercase + string.digits)
                  for i in range(N))
    return res

df = pd.read_csv('data_dump/val_df.csv')
unique_ids = pd.unique(df['id'])
ids = unique_ids[:100]
df = df[df['id'].isin(ids)]
df[['id', 'cell_id', 'cell_type', 'rank']].to_csv('val_df.csv')
# df_length = df.shape[0]
# codes = [gen_code() for _ in range(df_length)]

# df = df.rename(columns={'notebook_id': 'id'})
# df['cell_id'] = codes

# print(df.head())

# df.loc[df['cell_type'] == 'markdown', 'source'] = df[df['cell_type']
#                                                      == 'markdown'].source.apply(preprocess_text)

# df.loc[df['cell_type'] == 'code', 'source'] = df[df['cell_type']
#                                                  == 'code'].source.apply(preprocess_code)

# dict_cellid_source = dict(
#     zip(df['cell_id'].values, df['source'].values))

# df.to_csv('data_dump/train_extra_df.csv')
# with open('data_dump/dict_cellid_source_extra.pkl', 'wb') as handle:
#     pickle.dump(dict_cellid_source, handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)
# handle.close()
