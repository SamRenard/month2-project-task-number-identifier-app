"""
Day 59 — Train the digit model and save it for the API to load.
NIZAM AI · 150-Day AI Engineering Protocol
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DigitMLP(nn.Module):
    def __init__(self, in_features=64, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

    model = DigitMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    print(f"Test accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "digit_model.pt")
    print("Saved model weights to digit_model.pt")


if __name__ == "__main__":
    main()
