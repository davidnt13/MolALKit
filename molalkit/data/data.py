#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import copy
import os
import json
import pickle
import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as Chem
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from mgktools.features_mol.features_generators import FeaturesGenerator
#from mgktools.data.data import tolist, to_numpy
def tolist(list_: Union[pd.Series, List]) -> List[str]:
    if list_ is None:
        return []
    else:
        result = []
        for string_ in list_:
            if ',' in string_:
                result.append(json.loads(string_))
            else:
                result.append(string_)
        return result
def to_numpy(list_: pd.Series) -> Optional[np.ndarray]:
    if list_ is None:
        return None
    else:
        ndarray = list_.to_numpy()
        if ndarray.size == 0:
            return None
        else:
            return ndarray

# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}


def remove_none(X: List):
    X_ = []
    for x in X:
        if x is not None and len(x) != 0:
            X_.append(x)
    if len(X_) == 0:
        return None
    else:
        return X_


def concatenate(X: List, axis: int = 0, dtype=None):
    X_ = remove_none(X)
    if X_:
        return np.concatenate(X_, axis=axis, dtype=dtype)
    else:
        return None


class DataPointPure:
    """
    This is the class that stores a data point containing one molecule. (pure substance)

    Parameters
    ----------
    smiles: str
        SMILES string of the molecule
    mol:
        molecule object in RDKit
    fingerprints:
        1-d numpy array
    """

    def __init__(self, smiles: str):
        self.smiles = smiles
        self.fingerprints = None

    def __repr__(self) -> str:
        return self.smiles

    @property
    def mol(self):
        if self.smiles in SMILES_TO_MOL:
            return SMILES_TO_MOL[self.smiles]
        else:
            mol = Chem.MolFromSmiles(self.smiles)
            SMILES_TO_MOL[self.smiles] = mol
            return mol

    @property
    def mols(self):
        return [self.mol]

    def set_fingerprints(self, features_generator: List[FeaturesGenerator]):
        fingerprints = []
        for fg in features_generator:
            fingerprints.append(self.calc_features_mol(self.mol, fg))
        fingerprints = np.concatenate(fingerprints)
        # Fix nans in features_mol
        replace_token = 0
        self.fingerprints = np.where(np.isnan(fingerprints), replace_token, fingerprints)

    @staticmethod
    def calc_features_mol(mol: Chem.Mol, features_generator: FeaturesGenerator):
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            features_mol = features_generator(mol)
        elif mol is not None and mol.GetNumHeavyAtoms() == 0:
            # Hydrogen
            # not all features_mol are equally long, so use methane as dummy molecule to determine length
            features_mol = np.zeros(len(features_generator(Chem.MolFromSmiles('C'))))
        else:
            raise ValueError('rdkit mol object cannot be None.')
        return np.asarray(features_mol)


class DataPointMix:
    """
    This is the class that stores a data point containing more than one molecule. (mixture substances)

    Parameters
    ----------
    data:
        list of DatapointPure object
    concentration: list of float
        the mole concentration of molecules
    """

    def __init__(self, data: List[DataPointPure],
                 concentration: List[float] = None):
        assert len(data) == len(concentration)
        assert len(data) > 1
        # read data point
        self.data = data
        # set concentration
        if concentration is None:
            self.concentration = [1.0 / len(data)] * len(data)
        else:
            self.concentration = concentration
        self.fingerprints = None

    def __repr__(self) -> str:
        repr = ';'.join(list(map(lambda x, y: x.__repr__() + ',%.3f' % y, self.data, self.concentration)))
        return repr

    @property
    def smiles(self) -> List[str]:
        return [d.smiles for d in self.data]

    @property
    def mols(self) -> List[Chem.Mol]:
        return [d.mol for d in self.data]

    @classmethod
    def from_smiles_list(cls,
                         smiles: List[str],
                         concentration: List[float] = None):
        return cls([DataPointPure(s) for s in smiles], concentration=concentration)

    def set_fingerprints(self, features_generator: List[FeaturesGenerator],
                         features_combination: Literal['concat', 'mean'] = None):
        if features_generator is None:
            return
        if len(self.data) != 1:
            self.fingerprints = []
            for i, d in enumerate(self.data):
                d.set_fingerprints(features_generator)
                self.fingerprints.append(d.fingerprints * self.concentration[i])
            if features_combination == 'mean':
                self.fingerprints = np.mean(self.fingerprints, axis=0)
            elif features_combination == 'concat':
                self.fingerprints = np.concatenate(self.fingerprints)
            else:
                raise ValueError(f'unknown feature combination: f{features_combination}')
        else:
            self.data[0].set_fingerprints(features_generator)
            self.fingerprints = self.data[0].fingerprints


class DataPoint:
    """
    This is the class that stores a data point containing more than one molecule. (mixture substances)

    Parameters
    ----------
    data:
        list of DatapointPure object
    targets: list of float
        target properties to be predicted
    concentration: list of float
        the mole concentration of molecules
    features_add: list of float
        additional features, such as temperature, pressure, etc.
    """
    def __init__(self,
                 data_p: List[DataPointPure],
                 data_m: List[DataPointMix],
                 targets: List[float],
                 features_add: List[float] = None):
        # read data point
        self.data_p = data_p or []
        self.data_m = data_m or []
        self.data = self.data_p + self.data_m
        self.targets = targets
        self.features_add = features_add
        self.fingerprints = None

    def __repr__(self) -> str:
        repr = ';'.join(list(map(lambda x : x.__repr__(), self.data)))
        if self.features_add is not None:
            repr += ';' + ','.join(list(map(lambda x: '%.5f' % x, self.features_add)))
        return repr

    @property
    def mols(self) -> List[List[Chem.Mol]]:
        return [d.mols for d in self.data]

    def set_fingerprints(self, features_generator: List[FeaturesGenerator],
                         features_combination: Literal['concat', 'mean'] = None):
        for d in self.data_p:
            d.set_fingerprints(features_generator)
        for d in self.data_m:
            d.set_fingerprints(features_generator, features_combination=features_combination)
        self.fingerprints = np.concatenate([d.fingerprints for d in self.data], axis=0)  # 1-d array

    @property
    def y(self) -> np.ndarray:  # List[List[float]]
        return np.array(self.targets, dtype=float)


class Dataset:
    def __init__(self, data: List[DataPoint] = None,
                 mgk_type: Literal['graph', 'pre-computed'] = None):
        self.data = data
        self.mgk_type = mgk_type
        self.fingerprints_scaler = None
        self.features_add_scaler = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> Union[DataPoint, List[DataPoint]]:
        return self.data[item]

    @property
    def fingerprints_raw(self) -> Optional[np.ndarray]:  # List[List[float]]
        return np.asarray([d.fingerprints for d in self.data])

    @property
    def fingerprints(self) -> np.ndarray:  # List[List[float]]
        fingerprints = self.fingerprints_raw
        if self.fingerprints_scaler is not None:
            fingerprints = self.fingerprints_scaler.transform(fingerprints)
        return fingerprints

    @property
    def features_add_raw(self) -> Optional[np.ndarray]:  # List[List[float]]
        if self.data[0].features_add is None:
            return None
        else:
            return np.asarray([d.features_add for d in self.data])

    @property
    def features_add(self) -> np.ndarray:  # List[List[float]]
        features_add = self.features_add_raw
        if self.features_add_scaler is not None:
            features_add = self.features_add_scaler.transform(features_add)
        return features_add

    @property
    def repr(self) -> np.ndarray:  # # List[List[str]]
        return np.asarray([d.__repr__() for d in self.data])

    @property
    def X(self) -> np.ndarray:
        return concatenate([self.fingerprints, self.features_add], axis=1)

    @property
    def y(self) -> np.ndarray:  # List[List[float]]
        return concatenate([d.targets for d in self.data], axis=0)

    @property
    def N_tasks(self) -> int:
        return len(self.data[0].targets)

    @property
    def N_fingerprints(self) -> int:
        if self.fingerprints_raw is None:
            return 0
        else:
            return self.fingerprints_raw.shape[1]

    @property
    def N_features_add(self) -> int:
        if self.features_add_raw is None:
            return 0
        else:
            return self.features_add_raw.shape[1]

    def copy(self):
        return copy.deepcopy(self)

    def normalize_features(self, normalize_fingerprints: bool = False,
                           normalize_features_add: bool = False):
        if normalize_fingerprints:
            if self.fingerprints_raw is None:
                raise ValueError('You need to set_fingerprints before normalization.')
            self.fingerprints_scaler = StandardScaler().fit(self.fingerprints_raw)
        if normalize_features_add:
            if self.features_add_raw is None:
                raise ValueError('You need to set_fingerprints before normalization.')
            self.features_add_scaler = StandardScaler().fit(self.features_add_raw)

    def set_fingerprints(self, features_generator: List[FeaturesGenerator]):
        for d in self.data:
            d.set_fingerprints(features_generator)

    def save(self, path, filename='data.pkl', overwrite=False):
        f_dataset = os.path.join(path, filename)
        if os.path.isfile(f_dataset) and not overwrite:
            raise RuntimeError(
                f'Path {f_dataset} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_dataset, 'wb'), protocol=4)

    @classmethod
    def load(cls, path, filename='data.pkl'):
        f_dataset = os.path.join(path, filename)
        store = pickle.load(open(f_dataset, 'rb'))
        dataset = cls()
        dataset.__dict__.update(**store)
        return dataset

    def features_size(self) -> int:
        return self.N_fingerprints + self.N_features_add

    @staticmethod
    def read_DataPoint(
            smiles: List[str],
            mixture: List[Union[str, float]],
            targets: List[float],
            features_add: List[float] = None,
            features_generator: List[FeaturesGenerator] = None,
            features_combination: Literal['concat', 'mean'] = None,
    ) -> DataPoint:
        data_p = [DataPointPure(s) for s in smiles] if smiles is not None else None
        data_m = [DataPointMix.from_smiles_list(m[0::2], m[1::2]) for m in mixture] if mixture is not None else None
        data = DataPoint(data_p=data_p,
                         data_m=data_m,
                         targets=targets,
                         features_add=features_add)
        data.set_fingerprints(features_generator, features_combination=features_combination)
        return data

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       pure_columns: List[str] = None,
                       mixture_columns: List[str] = None,
                       target_columns: List[str] = None,
                       feature_columns: List[str] = None,
                       features_generator: List[FeaturesGenerator] = None,
                       features_combination: Literal['concat', 'mean'] = None,
                       n_jobs: int = 8):
        data = Parallel(
            n_jobs=n_jobs, verbose=True, prefer='processes')(
            delayed(cls.read_DataPoint)(
                tolist(df.iloc[i].get(pure_columns)),
                tolist(df.iloc[i].get(mixture_columns)),
                to_numpy(df.iloc[i:i + 1][target_columns]),
                features_add=to_numpy(df.iloc[i:i + 1].get(feature_columns)),
                features_generator=features_generator,
                features_combination=features_combination,
            )
            for i in df.index)
        return cls(data)
