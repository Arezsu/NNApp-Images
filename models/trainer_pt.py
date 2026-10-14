# app/models/trainer_pt.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .pytorch_arch import TabularNet
from pathlib import Path

def train_tabular(X_train, y_train, X_test, y_test, epochs=20):
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TabularNet(input_dim=X_train.shape[1])
    # loss_fn = nn.BCELoss() # for binary classification with sigmoid
    loss_fn = nn.CrossEntropyLoss() # for multi-class classification with softmax
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    Path("models/saved").mkdir(exist_ok=True)
    save_path = Path("models/saved/pt_tabular.pt")
    torch.save(model.state_dict(), save_path)
    return save_path
