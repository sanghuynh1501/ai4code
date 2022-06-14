from pathlib import Path

BERT_MODEL_PATH = 'microsoft/codebert-base'
MARK_PATH = 'weights/model_markdown.pth'
CODE_PATH = 'weights/model_code.pth'
CODE_MARK_PATH = 'weights/model_code_mark.pth'

DATA_DIR = Path('AI4Code')

NUM_TRAIN = 200
RANK_COUNT = 20
MAX_LEN = 128
NVALID = 0.1
EPOCH = 1
BS = 4
NW = 1
RANKS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
