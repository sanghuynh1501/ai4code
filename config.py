from pathlib import Path

BERT_PATH = "weights/codebert-base"
MARK_PATH = './weights/model_markdown.pth'
CODE_PATH = './weights/model_code.pth'

DATA_DIR = Path('data')

NUM_TRAIN = 10000
NVALID = 0.1
MAX_LEN = 128
BS = 128
NW = 2
