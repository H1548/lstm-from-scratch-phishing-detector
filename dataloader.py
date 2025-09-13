import numpy as np
import pandas as pd
import sentencepiece as spm
import random
import torch
import torch.nn.functional as F
from utils.py import tokenize_data

def load_data(dataset):

    

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
    val_data = copy_data[n:]
    

    return train_data, val_data