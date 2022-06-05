from pathlib import Path


fasttext_link = f'weights/fasttext-model-for-google-ai4code/model140000.bin'
checkpoint_path = f'weights/bert_model'
data_dir = Path('data')
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

NUM_TRAIN = 200
NVALID = 0.1
MAX_LEN = 128
MAX_LEN_FAST_TEXT = 100
BS = 128
NW = 2