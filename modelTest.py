from model import LSTM
import sentencepiece as spm
from model import LSTM
import torch
from utils import tokenize_data, pad_sequence, get_testbatch
import sentencepiece as spm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from testdataloader import load_testdata
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sp = spm.SentencePieceProcessor()
sp.load('m.model')

vocab_size = sp.get_piece_size()
batch_size = 16
block_size=80
embd_dim = 16
hidden_dims = 32
num_classes = 1

test_data = load_testdata('DataSet\Phishing emails - Classification\Test - Binary.csv')

model = LSTM(embd_dim,hidden_dims,num_classes, vocab_size, batch_size)
model.to(device)

target_names  = ['Phishing Email', 'Safe Email']
labels = [1, 0]

all_preds, all_labels = [], []
loss_sum = 0.0
n = 0.0
for start in range(0, len(test_data), batch_size):
        idx = list(range(start, min(start + batch_size, len(test_data))))
        x, y = get_testbatch(test_data, idx)
        loss,_, preds  = model.loss(x,y)
        loss_sum += loss.item() * y.size(0)
        n += y.size(0)

        
        all_preds.append(preds)
        all_labels.append(y.detach().cpu())

test_loss = loss_sum / n
y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_labels).numpy()
acc = accuracy_score(y_true, y_pred)
f1w = f1_score(y_true, y_pred, average="weighted")
print(f"Test loss: {test_loss:.4f} | Acc: {acc:.4f} | F1(w): {f1w:.4f}")

report = classification_report(y_true, y_pred,labels=labels,target_names=target_names, digits=4,zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels= labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in target_names],
    columns=[f"pred_{c}" for c in target_names]
)

plt.figure(figsize=(12, 10))  # Make the whole figure bigger
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix", fontsize=30)  # Bigger title
plt.colorbar()

tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, ha="right", fontsize=25)  # Bigger font
plt.yticks(tick_marks, target_names, fontsize=25)  # Bigger font

# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="red", fontsize=35)  # Bigger font for cell values

plt.ylabel("True label", fontsize=28)
plt.xlabel("Predicted label", fontsize=28)
plt.tight_layout()
plt.savefig("Results/confusion_matrix.png")
report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
df = pd.DataFrame(report_dict).transpose()
df.to_csv("Results/classification_report.csv", index=True)

