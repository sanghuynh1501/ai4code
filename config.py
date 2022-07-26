from pathlib import Path

BERT_MODEL_PATH = 'microsoft/codebert-base'
MARK_PATH = 'weights/model_markdown_07840.pth'
CODE_PATH = 'weights/model_code.pth'
CODE_MARK_PATH = 'weights/model_code_mark_07575.pth'
CODE_MARK_RANK_PATH = 'weights/model_code_mark_rank.pth'
SIGMOID_PATH = 'weights/model_sigmoid_40_mae.pth'
SIGMOID_NEW_PATH = 'weights/model_sigmoid.pth'
FASTTEST_MODEL = 'weights/model140000.bin'
DATA_DIR = Path('AI4Code')
LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'k',
          'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
MAX_TREE_DEPTH = 8
TREE_METHOD = 'gpu_hist'
SUBSAMPLE = 0.6
REGULARIZATION = 0.1
GAMMA = 0.3
POS_WEIGHT = 1
EARLY_STOP = 50
LEARNING_RATE = 0.01
NUM_TRAIN = 200
RANK_COUNT = 20
SIGMOID_RANK_COUNT = 10
MD_MAX_LEN = 64
CODE_MAX_LEN = 23
TOTAL_MAX_LEN = 512
MAX_LEN = 128
NVALID = 0.1
EPOCH = 5
BS = 2
NW = 1
RANKS = [i for i in range(0, RANK_COUNT + 2, 1)]
ACCUMULATION_STEPS = 16
