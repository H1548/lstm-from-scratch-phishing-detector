import numpy as np
import pandas as pd
import sentencepiece as spm
import random
import torch
import torch.nn.functional as F
from utils.py import tokenize_data

def load_data(dataset, tokenizer_path):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    vocab_size = sp.get_piece_size()
    Pad_token = sp.pad_id()
    bos_token = sp.bos_id()
    eos_token = sp.eos_id()

    df= pd.read_csv(dataset, delimiter = ';', names=['email', 'label'])

    data = np.array(df['email'])
    targets = np.array(df['label'])

    label_map = {
        'Phishing Email': 1,
        'Safe Email': 0
    }

    numerical_labels = [label_map[target] for target in targets]
    numerical_labels = np.array(numerical_labels)

    new_data = tokenize_data(data)

    paired_data = list(zip(new_data,numerical_labels))
    copy_data = paired_data.copy()
    random.shuffle(copy_data)
    n = int(0.8 * len(paired_data))
    train_data = copy_data[:n]
    val_data = copy_data[n:133000]
    test_data = copy_data[133000:135000]

    return train_data, val_data, test_data