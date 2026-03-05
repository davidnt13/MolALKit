#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 256), dropout=0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]  # binary logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


class david_mlp:
    """
    Chemprop-like wrapper that matches BaseModel interface:
      - fit_molalkit(train_data)
      - predict_value(pred_data): P(y=1)
      - predict_uncertainty(pred_data): heuristic based on prob variance
    """
    def __init__(
        self,
        hidden_dims=(256, 256),
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=128,
        epochs=20,
        device=None,
        seed=0,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = None

    def _check_xy(self, data):
        X = np.asarray(data.X)
        y = np.asarray(data.y)
        assert y.ndim == 2 and y.shape[1] == 1, "Expected y shape (N, 1)"
        return X, y

    def fit_molalkit(self, train_data):
        torch.manual_seed(self.seed)

        X, y = self._check_xy(train_data)
        # Ensure float32 for X, float32 for y (BCEWithLogits)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.ravel(), dtype=torch.float32)

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = MLP(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            n = 0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)              # (batch,)
                loss = loss_fn(logits, yb)           # scalar
                loss.backward()
                opt.step()

                total += loss.item() * xb.size(0)
                n += xb.size(0)

            # optional: print(f"epoch {epoch+1}/{self.epochs} loss={total/n:.4f}")

        return self

    @torch.no_grad()
    def _predict_proba(self, X: np.ndarray, batch_size=1024):
        assert self.model is not None, "Call fit_molalkit first."
        self.model.eval()

        X_t = torch.tensor(X, dtype=torch.float32)
        dl = DataLoader(X_t, batch_size=batch_size, shuffle=False)

        probs = []
        for xb in dl:
            xb = xb.to(self.device)
            logits = self.model(xb)
            p1 = torch.sigmoid(logits)  # P(class=1)
            probs.append(p1.detach().cpu().numpy())

        p1 = np.concatenate(probs, axis=0)  # (N,)
        # match sklearn predict_proba shape (N, 2): [P(0), P(1)]
        p = np.stack([1.0 - p1, p1], axis=1)
        return p

    def predict_value(self, pred_data):
        X = np.asarray(pred_data.X)
        p = self._predict_proba(X)
        return p[:, 1]  # like your BaseSklearnModel.predict_value_c

    def predict_uncertainty(self, pred_data):
        X = np.asarray(pred_data.X)
        p = self._predict_proba(X)
        return (0.25 - np.var(p, axis=1)) * 4  # same as your sklearn helper
