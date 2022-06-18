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
EPOCH = 5
BS = 4
NW = 1
RANKS = [i for i in range(-1, RANK_COUNT + 2, 1)]
accumulation_steps = 4
