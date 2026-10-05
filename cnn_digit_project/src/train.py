import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'; RESULTS.mkdir(exist_ok=True)

digits = load_digits()
X = digits.images.astype('float32') / 16.0
y = digits.target.astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

train_ds = TensorDataset(torch.tensor(X_train[:,None,:,:]), torch.tensor(y_train))
val_ds = TensorDataset(torch.tensor(X_val[:,None,:,:]), torch.tensor(y_val))
test_ds = TensorDataset(torch.tensor(X_test[:,None,:,:]), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*2*2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,10)
        )
    def forward(self,x): return self.net(x)

model = SmallCNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

def evaluate(loader):
    model.eval(); total=0; correct=0; losses=[]
    with torch.no_grad():
        for xb,yb in loader:
            out=model(xb); losses.append(loss_fn(out,yb).item()); correct += (out.argmax(1)==yb).sum().item(); total += len(yb)
    return float(np.mean(losses)), correct/total

best_val=0; best_state=None; patience=6; bad=0
for epoch in range(1,31):
    model.train()
    for xb,yb in train_loader:
        opt.zero_grad(); out=model(xb); loss=loss_fn(out,yb); loss.backward(); opt.step()
    tl,ta=evaluate(train_loader); vl,va=evaluate(val_loader)
    history['train_loss'].append(tl); history['val_loss'].append(vl); history['train_acc'].append(ta); history['val_acc'].append(va)
    if va>best_val: best_val=va; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
    else: bad+=1
    if bad>=patience: break

model.load_state_dict(best_state)
model.eval(); preds=[]; probs=[]
with torch.no_grad():
    for xb,yb in test_loader:
        p=torch.softmax(model(xb),dim=1); probs.extend(p.max(1).values.numpy()); preds.extend(p.argmax(1).numpy())
preds=np.array(preds)
acc=accuracy_score(y_test,preds)
prec,rec,f1,_=precision_recall_fscore_support(y_test,preds,average='weighted',zero_division=0)
cm=confusion_matrix(y_test,preds)

np.savez(RESULTS/'test_samples.npz', X=X_test[:25], y=y_test[:25], pred=preds[:25], conf=np.array(probs[:25]))
torch.save(model.state_dict(), RESULTS/'cnn_digit_model.pt')
with open(RESULTS/'metrics.json','w') as f: json.dump({'test_accuracy':acc,'weighted_precision':prec,'weighted_recall':rec,'weighted_f1':f1,'epochs_trained':len(history['train_loss']),'train_samples':len(y_train),'validation_samples':len(y_val),'test_samples':len(y_test)},f,indent=2)
with open(RESULTS/'history.json','w') as f: json.dump(history,f,indent=2)

plt.figure(figsize=(7,4)); plt.plot(history['train_acc'],label='Train'); plt.plot(history['val_acc'],label='Validation'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('CNN Training History'); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS/'training_accuracy.png',dpi=180); plt.close()
plt.figure(figsize=(6,5)); plt.imshow(cm, cmap='Blues'); plt.colorbar(); plt.xticks(range(10)); plt.yticks(range(10)); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Test Confusion Matrix');
for i in range(10):
    for j in range(10): plt.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8)
plt.tight_layout(); plt.savefig(RESULTS/'confusion_matrix.png',dpi=180); plt.close()
print(json.dumps({'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'epochs':len(history['train_loss'])},indent=2))
