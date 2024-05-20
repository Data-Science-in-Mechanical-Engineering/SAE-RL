from collections.abc import MutableMapping
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def __filter_by(df: pd.DataFrame, constraints: Dict[Any, List[Any]]):
    """Filter pd.MultiIndex by sublevels.

    Use as member function of a pd.DataFrame, e.g. `df.filter_by({'experiment' : ['dsae', 'basic', 'keynet']})`
    """
    indexer = [constraints[name] if name in constraints else slice(None)
               for name in df.index.names]
    return df.loc[tuple(indexer)] if len(df.shape) == 1 else df.loc[tuple(indexer),]


pd.Series.filter_by = __filter_by
pd.DataFrame.filter_by = __filter_by


def __flatten_dict_gen(d: MutableMapping, parent_key: str = '', sep: str = '.') -> Tuple[Any, Any]:
    """Generator yielding flattened dictionary entries with nested keys combined into one.

    Args:
        d (MutableMapping): The dictionary to flatten.
        parent_key (str): Parent key for current nested dictionary.
        sep (str): Separator to add in-between nested keys.

    Yields:
        Tuple[Any, Any]: Flattened key and value pair.
    """

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from __flatten_dict_gen(v, new_key, sep=sep)
        else:
            yield (new_key, v)


def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    """Flattens dictionary entries with nested keys combined into one.

    E.g. input {'a': 1, 'c': {'a': 2, 'b': {'x': 3, 'y': 4, 'z': 5}}, 'd': [6, 7, 8]} gives output {'a': 1, 'c.a': 2, 'c.b.x': 3, 'c.b.y': 4, 'c.b.z': 5, 'd': [6, 7, 8]}

    Args:
        d (MutableMapping): The dictionary to flatten.
        sep (str, optional): Separator to add in-between nested keys. Defaults to '.'.

    Returns:
        MutableMapping: The flattened dictionary
    """

    return dict(__flatten_dict_gen(d, sep=sep))


def compute_iqm(values: np.array) -> float:
    """Computes the interquartile mean (IQM) value of a set of scalar values.

    Args:
        values (np.array): One-dimensional array of values to compute the IQM of.

    Returns:
        float: IQM value of the set.
    """

    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    iqm_values = values[(values >= q1) & (values <= q3)]
    if iqm_values.size == 0:
        iqm = np.nan
    else:
        iqm = iqm_values.mean()

    return iqm


def apply_iqm(row: pd.DataFrame, ci_level: float = 0.95, ci_n_resamples: int = 100) -> pd.Series:
    """Computes the interquartile mean (IQM) value and a bootstrapped IQM confidence interval (CI) for one row of a data frame.

    Args:
        row (pd.DataFrame): Row of values to compute the IQM and CI of.
        ci_level (float, optional): Significance level for the CI. Defaults to 0.95.
        ci_n_resamples (int, optional): Number of bootstrap resamples for CI computation. Defaults to 100.

    Returns:
        pd.Series: IQM, lower CI bound and upper CI bound.
    """

    values = row.values
    non_nans = ~np.isnan(values)
    if non_nans.sum() >= 2:
        values = values[non_nans]
        iqm = compute_iqm(values)
        iqm_ci = bootstrap(
            values.reshape([1, -1]), compute_iqm, method='percentile', random_state=0, confidence_level=ci_level, n_resamples=ci_n_resamples
        ).confidence_interval
        res = [iqm, iqm_ci.low, iqm_ci.high]
    else:
        res = [np.nan] * 3

    return pd.Series(res, index=['iqm', 'ci_low', 'ci_high'])


def mm2in(*args) -> List[float]:
    """Computes length values in inches given values in millimeters.

    Every arguments is treated as one input value, e.g. `mm2in(32, 49)` returns `[1.2598425197, 1.9291338583]`.

    Returns:
        List[float]: Length values in inches.
    """

    return [x / 25.4 for x in args]
