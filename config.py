from pathlib import Path

BERT_MODEL_PATH = 'microsoft/codebert-base'
MARK_PATH = 'weights/model_markdown.pth'
CODE_PATH = 'weights/model_code.pth'
CODE_MARK_PATH = 'weights/model_code_mark.pth'
SIGMOID_PATH = 'weights/model_sigmoid_40_mae.pth'
DATA_DIR = Path('AI4Code')

NUM_TRAIN = 200
RANK_COUNT = 50
MAX_LEN = 128
NVALID = 0.1
EPOCH = 5
BS = 2
NW = 1
RANKS = [i for i in range(0, RANK_COUNT + 1, 1)]
accumulation_steps = 16
