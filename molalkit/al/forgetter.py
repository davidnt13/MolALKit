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
import numpy as np
from typing import List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def get_topn_idx(scores: np.ndarray, n: int, target: str = "max") -> List[int]:
    assert target in ("max", "min")
    if n <= 0:
        return []
    if target == "max":
        return list(np.argsort(-scores)[:n])
    else:
        return list(np.argsort(scores)[:n])


class MCDropoutForgetter(BaseForgetter):
    """
    Monte Carlo Dropout forgetter aligned with your other forgetter examples.

    Usage:
      forgetter = MCDropoutForgetter(device='cuda', mc_samples=20, score_method='bald', batch_size_eval=64, target='max')
      idxs, scores = forgetter(model, data, batch_size=10)
    """

    def __init__(
        self,
        device: str = "cpu",
        mc_samples: int = 20,
        score_method: str = "bald",   # 'bald'|'entropy'|'variance'|'variation_ratio'
        batch_size_eval: int = 64,
        target: str = "max"           # whether to pick top 'max' uncertainty or 'min' uncertainty
    ):
        self.device = torch.device(device)
        self.mc_samples = int(mc_samples)
        self.score_method = score_method.lower()
        assert self.score_method in ("bald", "entropy", "variance", "variation_ratio")
        assert target in ("max", "min")
        self.batch_size_eval = int(batch_size_eval)
        self.target = target

    @property
    def info(self) -> str:
        # return f"MCDropoutForgetter(mc={self.mc_samples}, score={self.score_method}, target={self.target})"
        return f"MCDropoutForgetter"

    # def _enable_dropout(self, model: nn.Module):
    #     # enable only Dropout layers (keep BatchNorm in eval)
    #     for m in model.modules():
    #         if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
    #             m.train()

    # def _disable_dropout(self, model: nn.Module):
    #     for m in model.modules():
    #         if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
    #             m.eval()

    def _to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # logits -> probs. handle binary logits too
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            p_pos = torch.sigmoid(logits.view(-1))  # [B]
            p = torch.stack([1.0 - p_pos, p_pos], dim=1)  # [B, 2]
        else:
            p = F.softmax(logits, dim=1)
        return p

    # def _mc_forward(self, model: nn.Module, batch: torch.Tensor) -> np.ndarray:
    #     """
    #     Run mc_samples forward passes. Return shape (mc, B, C) numpy array of probabilities.
    #     """
    #     model.to(self.device)
    #     model.eval()               # keep BN, etc. in eval
    #     self._enable_dropout(model)  # enable dropout layers

    #     batch = batch.to(self.device)
    #     mc_probs = []
    #     with torch.no_grad():
    #         for _ in range(self.mc_samples):
    #             out = model(batch)
    #             probs = self._to_probs(out)            # torch tensor [B, C]
    #             mc_probs.append(probs.cpu().numpy())  # append [B, C]
    #     mc_probs = np.stack(mc_probs, axis=0)  # [mc, B, C]

    #     self._disable_dropout(model)
    #     return mc_probs
    def _find_nn_module(self, model) -> Optional[nn.Module]:
        """
        Try to find an underlying torch.nn.Module inside `model`.
        Returns:
          - a torch.nn.Module object if found, else None.
        Looks for attributes commonly used by wrapper classes:
          - model, _model, network, net, module
        """
        if isinstance(model, nn.Module):
            return model
        # common wrapper attributes
        for attr in ("model", "_model", "network", "net", "module"):
            m = getattr(model, attr, None)
            if isinstance(m, nn.Module):
                return m
        # perhaps the wrapper has an attribute that *is* nn.Module but nested in dict
        # or the wrapper subclassed nn.Module but didn't expose isinstance: unlikely
        return None

    def _enable_dropout(self, model):
        """
        Enable dropout layers on the underlying nn.Module if we can find one.
        This avoids touching non-nn objects.
        """
        nn_mod = self._find_nn_module(model)
        if nn_mod is None:
            # nothing to enable; silently skip (model may not use dropout)
            return
        for m in nn_mod.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def _disable_dropout(self, model):
        nn_mod = self._find_nn_module(model)
        if nn_mod is None:
            return
        for m in nn_mod.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()

    def _maybe_move_model(self, model):
        """
        Try to move model to device if possible. Return the model actually moved (or the original).
        Attempts:
          - model.to(device) if exists
          - model.cuda() if device is cuda and cuda exists
          - find internal nn.Module and call .to() on that submodule
        If nothing works, return the original model (assume it is already on the right device).
        """
        # If model has to(), use it safely
        try:
            if hasattr(model, "to") and callable(getattr(model, "to")):
                # some wrappers implement to() but have a different signature; wrap in try
                try:
                    model.to(self.device)
                    return model
                except Exception:
                    # fall through to try moving internal module
                    pass

            # maybe a plain method cuda()
            if str(self.device).startswith("cuda") and hasattr(model, "cuda") and callable(getattr(model, "cuda")):
                try:
                    model.cuda()
                    return model
                except Exception:
                    pass

            # fallback: move underlying nn.Module if present
            nn_mod = self._find_nn_module(model)
            if nn_mod is not None:
                try:
                    nn_mod.to(self.device)
                    return model
                except Exception:
                    pass
        except Exception:
            # be conservative: don't crash moving model
            pass

        # last resort: do nothing (model may already be device-aware)
        return model

    def _mc_forward(self, model, batch: torch.Tensor) -> np.ndarray:
        """
        Robust MC forward that doesn't assume model.to() exists.
        - Moves input batch to self.device.
        - Attempts to move model or its internal nn.Module if possible.
        - Enables dropout only on found nn.Module.
        - Runs mc_samples forward passes and returns mc_probs np array [mc, B, C].
        """
        # Move inputs to device always
        batch = batch.to(self.device)

        # Try to move model safely; don't fail if not possible
        model = self._maybe_move_model(model)

        # Enable dropout on underlying module (if any). If not possible, skip.
        self._enable_dropout(model)

        mc_probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                # forward: try calling model(batch); if wrapper expects a dict or different args,
                # this may fail — handle with informative errors.
                try:
                    logits = model(batch)
                except TypeError:
                    # some wrappers expect a dict or multiple args; try common alternatives:
                    # - model.forward(batch)
                    # - model.forward(input=batch)
                    try:
                        logits = model.forward(batch)
                    except Exception:
                        # final fallback: try calling model with batch.cpu() (in case model expects CPU tensors)
                        try:
                            logits = model(batch.cpu())
                        except Exception as e:
                            # we cannot safely call the model; raise a helpful error
                            raise RuntimeError(
                                "Failed to call the model on a tensor. The model object does not accept a single "
                                "tensor argument. You may need to wrap your model with a callable that accepts the "
                                "input tensor and returns logits. Underlying error: " + repr(e)
                            ) from e

                # convert logits -> probs (reuse your existing _to_probs logic)
                probs = self._to_probs(logits)  # expects a torch.Tensor
                mc_probs.append(probs.cpu().numpy())

        mc_probs = np.stack(mc_probs, axis=0)  # [mc, B, C]

        # restore dropout state
        self._disable_dropout(model)

        return mc_probs

    @staticmethod
    def _predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return -np.sum(mean_probs * np.log2(mean_probs + eps), axis=1)  # [B,]

    @staticmethod
    def _expected_entropy(mc_probs: np.ndarray) -> np.ndarray:
        eps = 1e-12
        mc_ent = -np.sum(mc_probs * np.log2(mc_probs + eps), axis=2)  # [mc, B]
        return mc_ent.mean(axis=0)  # [B,]

    @staticmethod
    def _variation_ratio(mc_probs: np.ndarray) -> np.ndarray:
        # Variation ratio = 1 - mode_fraction
        mc_preds = np.argmax(mc_probs, axis=2)  # [mc, B]
        mc = mc_preds.shape[0]
        B = mc_preds.shape[1]
        vr = np.empty(B, dtype=float)
        for i in range(B):
            vals, counts = np.unique(mc_preds[:, i], return_counts=True)
            mode_count = counts.max()
            vr[i] = 1.0 - (mode_count / float(mc))
        return vr

    def _bald(self, mc_probs: np.ndarray) -> np.ndarray:
        mean_probs = mc_probs.mean(axis=0)  # [B, C]
        H_mean = self._predictive_entropy(mean_probs)
        E_H = self._expected_entropy(mc_probs)
        return H_mean - E_H

    def _variance(self, mc_probs: np.ndarray) -> np.ndarray:
        var = np.var(mc_probs, axis=0)  # [B, C]
        return var.sum(axis=1)

    def _entropy(self, mc_probs: np.ndarray) -> np.ndarray:
        mean_probs = mc_probs.mean(axis=0)
        return self._predictive_entropy(mean_probs)

    def _score_from_mc(self, mc_probs: np.ndarray) -> np.ndarray:
        if self.score_method == "bald":
            return self._bald(mc_probs)
        elif self.score_method == "variance":
            return self._variance(mc_probs)
        elif self.score_method == "entropy":
            return self._entropy(mc_probs)
        elif self.score_method == "variation_ratio":
            return self._variation_ratio(mc_probs)
        else:
            raise ValueError("Unknown score_method")

    # inside your forgetter class...

    def _default_collate_fn(self, batch: list) -> tuple:
        """
        Convert a batch of dataset items into (tensor_inputs, tensor_indices).
        The dataset item may be:
          - (x, idx)
          - x (where x is tensor/ndarray)  -> indices will be auto-generated
          - custom object (e.g. MoleculeDatapoint) -> we try to extract features
        Extraction heuristics (in order):
          - If item is tuple/list and len>=2: assume (x, idx)
          - If item is Tensor/ndarray/number: stack them
          - If item has attribute 'features' or 'feature_vector' or 'fp' -> use that
          - If item has attribute 'smiles' and you have a featurizer function, raise informative error (see below)
        """
        # If items are (x, idx) pairs, separate them
        first = batch[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            xs = [b[0] for b in batch]
            idxs = [b[1] for b in batch]
        else:
            xs = batch
            idxs = None

        # try stacking xs into a tensor
        # if elements are already tensors or arrays, convert/stack
        if isinstance(xs[0], torch.Tensor):
            x_tensor = torch.stack(xs)
        elif isinstance(xs[0], np.ndarray):
            x_tensor = torch.tensor(np.stack(xs))
        elif isinstance(xs[0], (int, float, np.number)):
            x_tensor = torch.tensor(xs, dtype=torch.float32)
        else:
            # fallback: custom objects (e.g. MoleculeDatapoint)
            extracted = []
            for obj in xs:
                # if object provides .features, .feature_vector, .fp, .fingerprint
                if hasattr(obj, "features"):
                    f = getattr(obj, "features")
                elif hasattr(obj, "feature_vector"):
                    f = getattr(obj, "feature_vector")
                elif hasattr(obj, "fp"):
                    f = getattr(obj, "fp")
                elif hasattr(obj, "fingerprint"):
                    f = getattr(obj, "fingerprint")
                # sometimes chemprop MoleculeDatapoint stores features in .smiles and needs featurizer
                elif hasattr(obj, "smiles") and hasattr(obj, "features"):
                    f = getattr(obj, "features")
                else:
                    # We cannot auto-extract — raise a helpful error telling the user what to do.
                    typename = type(obj).__name__
                    raise TypeError(
                        f"Cannot collate dataset element of type {typename}. "
                        "Please either (a) wrap your dataset so __getitem__ returns (input_tensor, index), "
                        "or (b) provide a custom collate_fn that converts a dataset element to a numeric array/tensor. "
                        "If this is a chemprop MoleculeDatapoint, extract the model input (fingerprint / features) "
                        "in your dataset or pass a small adapter dataset that returns (features_tensor, idx)."
                    )
                # ensure f is numeric / array-like
                if isinstance(f, torch.Tensor):
                    extracted.append(f.cpu().numpy())
                else:
                    extracted.append(np.asarray(f))
            x_tensor = torch.tensor(np.stack(extracted), dtype=torch.float32)

        # indices default if not present
        if idxs is None:
            idxs = list(range(len(x_tensor)))
        else:
            # try to coerce to list of ints
            idxs = [int(i) for i in idxs]

        return x_tensor, torch.tensor(idxs, dtype=torch.long)


    def _prepare_dataloader(self, data) -> DataLoader:
        """
        Accepts:
          - a PyTorch Dataset (will use our collate function)
          - an ndarray / tensor (we construct a simple Dataset that yields (x, idx))
        """
        if isinstance(data, Dataset):
            # use our collate_fn to convert MoleculeDatapoint -> (tensor, idx)
            return DataLoader(data,
                              batch_size=self.batch_size_eval,
                              shuffle=False,
                              collate_fn=self._default_collate_fn,
                              num_workers=0)  # try 0 first for debugging; you can increase
        else:
            # construct simple dataset from array/tensor
            if isinstance(data, torch.Tensor):
                X_np = data.cpu().numpy()
            else:
                X_np = np.asarray(data)

            class _ArrDataset(Dataset):
                def __len__(self_inner): return X_np.shape[0]
                def __getitem__(self_inner, idx):
                    return X_np[idx], idx

            return DataLoader(_ArrDataset(),
                              batch_size=self.batch_size_eval,
                              shuffle=False,
                              collate_fn=self._default_collate_fn,
                              num_workers=0)
    
    # def _prepare_dataloader(self, data) -> DataLoader:
    #     # Accept Dataset, numpy array, torch tensor, or list
    #     if isinstance(data, Dataset):
    #         return DataLoader(data, batch_size=self.batch_size_eval, shuffle=False)
    #     else:
    #         # assume array-like where len(data) gives N and indexing returns an input tensor/ndarray
    #         # build a simple dataset that returns (x, idx)
    #         X = data
    #         if isinstance(X, torch.Tensor):
    #             X_np = X.cpu().numpy()
    #         else:
    #             X_np = np.asarray(X)
    #         class _ArrDataset(Dataset):
    #             def __len__(self_inner):
    #                 return X_np.shape[0]
    #             def __getitem__(self_inner, idx):
    #                 return X_np[idx], idx
    #         return DataLoader(_ArrDataset(), batch_size=self.batch_size_eval, shuffle=False)

    def __call__(self, model: nn.Module, data, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        """
        model: PyTorch model
        data: Dataset or ndarray/tensor of inputs
        batch_size: number of indices to forget (like your other forgetters)
        returns: (list(indices), list(scores))
        """
        assert batch_size < (len(data) if not isinstance(data, Dataset) else len(data)), "batch_size must be < len(data)"
        dl = self._prepare_dataloader(data)

        all_indices = []
        all_scores = []

        for batch in dl:
            # dataloader yields either (x, idx) (if Dataset) or (x_np, idx) from our wrapper
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x_batch, idxs = batch[0], batch[1]
            else:
                # fallback: dataloader might yield raw x only
                x_batch = batch
                idxs = torch.arange(len(x_batch))

            # convert x_batch to torch tensor if needed
            if not isinstance(x_batch, torch.Tensor):
                x_batch = torch.tensor(np.asarray(x_batch), dtype=torch.float32)

            mc_probs = self._mc_forward(model, x_batch)  # [mc, B, C]
            scores = self._score_from_mc(mc_probs)       # [B,]

            all_indices.extend([int(i) for i in idxs])
            all_scores.extend([float(s) for s in scores])

        all_indices = np.array(all_indices, dtype=int)
        all_scores = np.array(all_scores, dtype=float)

        picked = get_topn_idx(all_scores, n=batch_size, target=self.target)
        picked_indices = all_indices[picked].tolist()
        picked_scores = all_scores[picked].tolist()

        return picked_indices, picked_scores
