from pathlib import Path
import numpy as np, torch
from torch import nn
from sklearn.datasets import load_digits
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,10))
    def forward(self,x): return self.net(x)
root=Path(__file__).resolve().parents[1]; model=SmallCNN(); model.load_state_dict(torch.load(root/'results/cnn_digit_model.pt',map_location='cpu')); model.eval()
d=load_digits(); idx=int(input(f'Enter test image index (0-{len(d.images)-1}): ')); x=torch.tensor((d.images[idx]/16.0)[None,None,:,:],dtype=torch.float32)
with torch.no_grad(): p=torch.softmax(model(x),1)[0]; pred=int(p.argmax())
print(f'True digit: {d.target[idx]}'); print(f'Predicted digit: {pred}'); print(f'Confidence: {float(p[pred])*100:.2f}%')
