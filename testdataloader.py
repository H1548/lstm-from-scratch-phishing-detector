import numpy as np
import pandas as pd
import sentencepiece as spm
import random
import torch
import torch.nn.functional as F
from utils import tokenize_data

def load_testdata(dataset):

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
    test_data = copy_data
    

    return test_data