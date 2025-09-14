from model import LSTM
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from utils import pad_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor()
sp.load('m.model')

bos_token = sp.bos_id()
eos_token = sp.eos_id()
vocab_size = sp.get_piece_size()
num_classes = 1
batch_size = 16
block_size=80
embd_dim = 16
hidden_dims = 32

model = LSTM(embd_dim, hidden_dims, num_classes, vocab_size, batch_size)
model.to(device)

prompt = input('Enter Your Email: ')
output = model.sample(prompt, bos_token, eos_token, pad_sequence, block_size)
print(output)