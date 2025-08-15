import torch
import sentencepiece as spm
import torch.nn.functional as F
from model.py import LSTM
from utils.py import estimate_loss, get_accuracy, get_batch

sp = spm.SentencePieceProcessor()
sp.load('m.model')

vocab_size = sp.get_piece_size()
batch_size = 16
block_size=80
embd_dim = 16
hidden_dims = 32
num_iters = 6000
num_classes = 1
learning_rate = 1e-3
eval_iters = 50
model_checkpoint = 10

model = LSTM(embd_dim, hidden_dims, num_classes, vocab_size, batch_size)

configs = {}
for p,w in model.params.items():
    configs[p] = model.create_config(learning_rate,w)
    
lossi = []
best_val = 100
for i in range(num_iters):
    if i % eval_iters == 0 or i == num_iters-1:
        losses = estimate_loss()
        accuracy, f1, recall, precision = get_accuracy(model)
        if losses['val'] < best_val: 
            torch.save({'params': model.params},'BestLSTM/BestModel.pth' )
            best_val = losses['val']
        print(f"step{i}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}, accuracy: {accuracy}, f1: {f1}, recall: {recall}, precision: {precision}")
    if i % model_checkpoint == 0:
        torch.save({
            "params": model.params,
            "optimizer": configs
        }, f"LSTM-Checkpoint/checkpoint_epoch_{i}.pth")
    xb, yb = get_batch('train')
    loss,grads = model.loss(xb, yb)
    max_norm = 1.0
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads.values()))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in grads:
            grads[p] = grads[p] * clip_coef
    for p,w in model.params.items():
        dw = grads[p]
        config = configs[p]
        next_w, next_config = model.update_params(w,dw,config)
        model.params[p] = next_w
        configs[p] = next_config
    grads = model.reset_grads(grads)