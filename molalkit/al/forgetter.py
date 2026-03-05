#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from molalkit.models.random_forest.RandomForestClassifier import RFClassifier
from molalkit.models.gaussian_process.GaussianProcessRegressor import GPRegressor
from molalkit.al.selection_method import get_topn_idx

class BaseForgetter(ABC):
    def __init__(self, batch_size: int = 1, forget_size: int = 1, forget_cutoff: float = None):
        self.batch_size = batch_size
        self.forget_size = forget_size
        self.forget_cutoff = forget_cutoff

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[List[int], List[float]]:
        pass

    @property
    @abstractmethod
    def info(self) -> str:
        pass


class BaseRandomForgetter(BaseForgetter, ABC):
    """Base Forgetter that uses random seed."""
    def __init__(self, batch_size: int = 1, forget_size: int = 1, forget_cutoff: float = None, seed: int = 0):
        super().__init__(batch_size=batch_size, forget_size=forget_size, forget_cutoff=forget_cutoff)
        np.random.seed(seed)


class RandomForgetter(BaseRandomForgetter):
    def __call__(self, data, batch_size: int = 1, **kwargs) -> Tuple[List[int], None]:
        assert batch_size < len(data)
        return np.random.choice(list(range(len(data))), batch_size, replace=False).tolist(), None

    @property
    def info(self) -> str:
        return 'RandomForgetter'


class FirstForgetter(BaseForgetter):
    def __call__(self, data, batch_size: int = 1, **kwargs) -> Tuple[List[int], None]:
        assert batch_size < len(data)
        return list(range(batch_size)), None

    @property
    def info(self) -> str:
        return 'FirstForgetter'


class MinOOBUncertaintyForgetter(BaseRandomForgetter):
    def __call__(self, model: RFClassifier, data, batch_size: int = 1, **kwargs) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        assert model.oob_score is True
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size, target='min')
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinOOBUncertaintyForgetter'


class MaxOOBUncertaintyForgetter(BaseRandomForgetter):
    def __call__(self, model: RFClassifier, data, batch_size: int = 1, **kwargs) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the highest out-of-bag (OOB) uncertainty.

        Parameters
        ----------
        model: Only random forest classifier is supported due to efficient OOB uncertainty calculation.
        data: The dataset to forget.
        batch_size: The number of samples to forget.

        Returns
        -------
        The index of samples to forget.
        """
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size)
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MaxOOBUncertaintyForgetter'


class MinOOBErrorForgetter(BaseRandomForgetter):
    def __call__(self, model: RFClassifier, data, batch_size: int = 1, cutoff: float = None, **kwargs
                 ) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        assert data.y.ndim == 2
        assert data.y.shape[1] == 1
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        oob_error = np.absolute(y_oob_proba[:, 1] - data.y.ravel())
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(oob_error, n=batch_size, target='min', cutoff=cutoff)
        acquisition = oob_error[np.array(forgotten_idx)].tolist() if forgotten_idx else []
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinOOBErrorForgetter'


class MinLOOErrorForgetter(BaseRandomForgetter):
    def __call__(self, model: GPRegressor, data, batch_size: int = 1, cutoff: float = None, **kwargs
                 ) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the lowest Leave-one-out cross-validation (LOOCV) error.
        Parameters
        ----------
        model: Only Gaussian process regressor is supported due to efficient LOOCV of GPR.
        data: The dataset to forget.
        batch_size: The number of samples to forget.
        cutoff: The cutoff value of LOOCV error. Only samples with LOOCV error lower than cutoff will be forgot.

        Returns
        -------
        The index and the acquisition value of samples to forget.
        """
        assert batch_size < len(data)
        assert isinstance(model, GPRegressor)
        assert data.y.ndim == 2
        assert data.y.shape[1] == 1
        y_loocv = model.predict_loocv(data.X, data.y.ravel(), return_std=False)
        # uncertainty calculation, normalized into 0 to 1
        loo_error = np.absolute(y_loocv - data.y.ravel())
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(loo_error, n=batch_size, target='min', cutoff=cutoff)
        acquisition = loo_error[np.array(forgotten_idx)].tolist() if forgotten_idx else []
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinLOOErrorForgetter'

# Updates with MC/Dropout
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class XWithIdDataset(Dataset):
    """
    Adapter that yields (x_tensor, idx) for forgetter evaluation.

    Supports:
      - XWithIdDataset(X, ids)
      - XWithIdDataset(molalkit_dataset=subset)  where subset has .X and .data
    """
    def __init__(self, X=None, ids=None, molalkit_dataset=None, dtype=torch.float32):
        self.dtype = dtype

        if molalkit_dataset is not None:
            assert hasattr(molalkit_dataset, "X"), "molalkit_dataset must have .X"
            assert hasattr(molalkit_dataset, "data"), "molalkit_dataset must have .data list"
            self.X = np.asarray(molalkit_dataset.X, dtype=np.float32)

            extracted = []
            for i, dp in enumerate(molalkit_dataset.data):
                if hasattr(dp, "data_idx") and dp.data_idx is not None:
                    extracted.append(int(dp.data_idx))
                elif hasattr(dp, "id") and dp.id is not None:
                    extracted.append(int(dp.id))
                else:
                    # last-resort: subset-local index (only if your pipeline supports it)
                    extracted.append(i)

            self.ids = extracted

        else:
            self.X = np.asarray(X, dtype=np.float32)
            self.ids = [int(i) for i in ids]
            assert self.X.shape[0] == len(self.ids), "Length mismatch between X and ids"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=self.dtype), self.ids[i]

class MCDropoutForgetter(BaseForgetter):
    def __init__(
        self,
        device: str = "cpu",
        mc_samples: int = 20,
        score_method: str = "bald",     # bald|entropy|variance|variation_ratio
        batch_size_eval: int = 256,
        target: str = "max",
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.device = torch.device(device)
        self.mc_samples = int(mc_samples)
        self.score_method = score_method.lower()
        assert self.score_method in ("bald", "entropy", "variance", "variation_ratio")
        self.batch_size_eval = int(batch_size_eval)
        assert target in ("max", "min")
        self.target = target
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    @property
    def info(self) -> str:
        return "MCDropoutForgetter"

    @staticmethod
    def _unwrap_nn(model) -> nn.Module:
        if isinstance(model, nn.Module):
            return model
        m = getattr(model, "model", None)
        if isinstance(m, nn.Module):
            return m
        raise TypeError("Expected nn.Module or wrapper with `.model` as nn.Module")

    @staticmethod
    def _enable_dropout_only(m: nn.Module):
        m.eval()  # keep BN stable
        for layer in m.modules():
            if isinstance(layer, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                layer.train()

    @staticmethod
    def _to_probs(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            p1 = torch.sigmoid(logits)
            return torch.stack([1.0 - p1, p1], dim=1)
        if logits.dim() == 2 and logits.size(1) == 1:
            p1 = torch.sigmoid(logits.squeeze(1))
            return torch.stack([1.0 - p1, p1], dim=1)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def _mc_probs(self, m: nn.Module, x: torch.Tensor) -> np.ndarray:
        self._enable_dropout_only(m)
        x = x.to(self.device, non_blocking=True)

        probs = []
        for _ in range(self.mc_samples):
            logits = m(x)
            p = self._to_probs(logits).detach().cpu().numpy()  # (B,C)
            probs.append(p)
        return np.stack(probs, axis=0)  # (S,B,C)

    @staticmethod
    def _predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return -np.sum(mean_probs * np.log2(mean_probs + eps), axis=1)

    @staticmethod
    def _expected_entropy(mc_probs: np.ndarray) -> np.ndarray:
        eps = 1e-12
        ent = -np.sum(mc_probs * np.log2(mc_probs + eps), axis=2)  # (S,B)
        return ent.mean(axis=0)

    @staticmethod
    def _variation_ratio(mc_probs: np.ndarray) -> np.ndarray:
        preds = np.argmax(mc_probs, axis=2)  # (S,B)
        S, B = preds.shape
        out = np.empty(B, dtype=float)
        for i in range(B):
            _, counts = np.unique(preds[:, i], return_counts=True)
            out[i] = 1.0 - (counts.max() / float(S))
        return out

    @staticmethod
    def _variance(mc_probs: np.ndarray) -> np.ndarray:
        return np.var(mc_probs, axis=0).sum(axis=1)

    def _score(self, mc_probs: np.ndarray) -> np.ndarray:
        mean_probs = mc_probs.mean(axis=0)  # (B,C)
        if self.score_method == "entropy":
            return self._predictive_entropy(mean_probs)
        if self.score_method == "variance":
            return self._variance(mc_probs)
        if self.score_method == "variation_ratio":
            return self._variation_ratio(mc_probs)
        # BALD
        H = self._predictive_entropy(mean_probs)
        EH = self._expected_entropy(mc_probs)
        return H - EH

    @staticmethod
    def _collate_x_idx(batch):
        xs, idxs = zip(*batch)
        if isinstance(xs[0], torch.Tensor):
            x = torch.stack(xs, dim=0).float()
        else:
            x = torch.tensor(np.stack([np.asarray(t) for t in xs], axis=0), dtype=torch.float32)
        idx = torch.tensor([int(i) for i in idxs], dtype=torch.long)
        return x, idx

    def __call__(self, model, data: Dataset, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        m = self._unwrap_nn(model).to(self.device)

        dl = DataLoader(
            data,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.pin_memory and self.device.type == "cuda"),
            collate_fn=self._collate_x_idx,
        )

        all_ids = []
        all_scores = []

        for x, ids in dl:
            mc = self._mc_probs(m, x)     # (S,B,C)
            s = self._score(mc)           # (B,)
            all_ids.append(ids.cpu().numpy())
            all_scores.append(s)

        all_ids = np.concatenate(all_ids, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)

        picked = get_topn_idx(all_scores, n=batch_size, target=self.target)
        return all_ids[picked].tolist(), all_scores[picked].tolist()
