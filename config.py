from pathlib import Path

BERT_MODEL_PATH = 'microsoft/codebert-base'
MARK_PATH = 'model_markdown.pth'
CODE_PATH = 'model_code.pth'
CODE_MARK_PATH = 'model_code_mark.pth'

DATA_DIR = Path('data')

NUM_TRAIN = 200
MAX_LEN = 128
NVALID = 0.1
EPOCH = 1
BS = 16
NW = 1