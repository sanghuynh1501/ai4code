from pathlib import Path

BERT_MODEL_PATH = 'microsoft/codebert-base'
MARK_PATH = 'weights/model_markdown.pth'
CODE_PATH = 'weights/model_code.pth'
CODE_MARK_PATH = 'weights/model_code_mark_07575.pth'
SIGMOID_PATH = 'weights/model_sigmoid_40_mae.pth'
DATA_DIR = Path('AI4Code')
LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'k', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

NUM_TRAIN = 200
RANK_COUNT = 20
MD_MAX_LEN = 64
CODE_MAX_LEN = 23
TOTAL_MAX_LEN = 512
MAX_LEN = 128
NVALID = 0.1
EPOCH = 5
BS = 2
NW = 1
RANKS = [i for i in range(0, RANK_COUNT + 1, 1)]
accumulation_steps = 32
