import sentencepiece as spm
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataloader.py import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor()
sp.load('tokenizer\m.model')

batch_size = 16
block_size=80
eval_iters = 50
threshold = 0.5
vocab_size = sp.get_piece_size()
Pad_token = sp.pad_id()
bos_token = sp.bos_id()
eos_token = sp.eos_id()
dataset_path = 'Dataset\Phishing emails - Classification\PhishingEmails3.csv'


train_data, val_data, test_data = load_data(dataset_path)


def tokenize_data(data):
    for i in range(data.shape[0]):
        data[i] = [bos_token] + sp.encode_as_ids(data[i]) + [eos_token]
    return data

def pad_sequence(sequence, max_length, pad_token_id):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_token_id] * (max_length - len(sequence))


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, size=(batch_size,))

    x, y = [], []

    for i in ix:
        input_i, output_i = data[i]
        
        pad_input_i = pad_sequence(input_i, block_size,Pad_token)
        x.append(torch.tensor(pad_input_i, dtype=torch.long))
        y.append(torch.tensor(output_i, dtype = torch.long))

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)
    y = y.unsqueeze(1)
    return x, y

def sigmoid(x):
    
        pos_mask = x >= 0
        neg_mask = x < 0
        z = torch.zeros_like(x).to(device)
        z[pos_mask] = torch.exp(-x[pos_mask])
        z[neg_mask] = torch.exp(x[neg_mask])
        top = torch.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            loss, _ = model.loss(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

def get_accuracy(model):
    
    x,y_true = get_batch('val')
    pred,_   = model.loss(x)
    for i in range(pred.shape[0]):
        pred[i] = 1 if pred[i] >= threshold else 0
    pred = pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    accuracy = (pred == y_true).mean()
    f1 = f1_score(y_true, pred)
    recall = recall_score(y_true, pred)
    precision = precision_score(y_true, pred)
    return accuracy, f1, recall, precision