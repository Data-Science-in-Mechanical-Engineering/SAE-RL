from typing import List

import pandas as pd
from wandb.apis.public import Runs

import wandb

from .utils import apply_iqm, flatten_dict


def wandb_load_runs(entity: str, project: str) -> Runs:
    """Loads runs from Weights and Biases (WandB).

    Args:
        entity (str): WandB entity.
        project (str): WandB project.

    Returns:
        Runs: Runs logged to WandB.
    """

    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}')

    return runs


def wandb_load_overview(runs: Runs, info_fields: List[str] = ['id', 'name', 'state', 'createdAt', 'tags']) -> pd.DataFrame:
    """Loads an overview of metadata for Weights and Biases (WandB) runs.

    Args:
        runs (Runs): Runs to load metadata for.
        info_fields (List[str], optional): Info fields to load. Defaults to ['id', 'name', 'state', 'createdAt', 'tags'].

    Returns:
        pd.DataFrame: Data frame with run metadata.
    """

    overview = []

    for run in runs:
        row = {'run': run}  # link to wandb run
        for key in info_fields:  # some status information
            row[key] = getattr(run, key)
        for key, value in run.summary.items():  # all config items and metrics
            if type(value) in [float, int] and not key.startswith('_'):
                row[key] = value

        row.update(flatten_dict(run.config))
        overview.append(row)

    overview = pd.DataFrame(overview).sort_values('createdAt')

    return overview


def wandb_load_time_series(overview: pd.DataFrame, variable: str, x_axis: str = 'global_step') -> pd.DataFrame:
    """Loads time series for all runs for individual variables logged to Weights and Biases (WandB).

    Args:
        overview (pd.DataFrame): Runs to load time series for.
        variable (str): Name of variable to load time series of.
        x_axis (str, optional): What to use for the x-axis. Defaults to 'global_step'.

    Returns:
        pd.DataFrame: Data frame of each run's time series for the specified variable.
    """

    df_time = pd.DataFrame(columns=overview.index)

    for run in df_time.columns:
        time_series = overview.loc[run]['run'].history(keys=[variable], x_axis=x_axis).set_index(x_axis)

        df_time[run] = time_series[variable]

    return df_time


def compute_iqm_values(values: pd.Series, ci_level: float = 0.95, ci_n_resamples: int = 10000) -> pd.Series:
    """Computes static interquartile mean (IQM) values and confidence intervals (CIs) for multiple configurations.

    Args:
        values (pd.Series): Multiindexed series of raw values to compute IQMs and CIs for.
        ci_level (float, optional): Significance level for the CIs. Defaults to 0.95.
        ci_n_resamples (int, optional): Number of bootstrap resamples for CI computation. Defaults to 10000.

    Returns:
        pd.Series: Series with IQMs and CIs for all run configurations.
    """

    # build index with 'iqm', 'ci_low', and 'ci_high' instead of run names
    run_sets = values.index.droplevel(-1).unique()
    idx = [[*run_set, ext] for run_set in run_sets
           for ext in ['iqm', 'ci_low', 'ci_high']]

    iqm_values = pd.Series(index=pd.MultiIndex.from_tuples(idx), dtype=float)

    for run_set in run_sets:
        iqm_and_cis = apply_iqm(values[run_set], ci_level=ci_level, ci_n_resamples=ci_n_resamples)
        iqm_values[run_set] = iqm_and_cis

    return iqm_values


def compute_iqm_time_series(time_series: pd.DataFrame, ci_level: float = 0.95, ci_n_resamples: int = 100) -> pd.DataFrame:
    """Computes time series of interquartile mean (IQM) values and confidence intervals (CIs) for multiple configurations.

    Args:
        time_series (pd.DataFrame): Multiindexed data frame with time series of raw values to compute IQMs and CIs for.
        ci_level (float, optional): Significance level for the CIs. Defaults to 0.95.
        ci_n_resamples (int, optional): Number of bootstrap resamples for CI computation. Defaults to 100.

    Returns:
        pd.DataFrame: Data frame with time series of IQMs and CIs for all run configurations.
    """

    # build index with 'iqm', 'ci_low', and 'ci_high' instead of run names
    run_sets = time_series.columns.droplevel(-1).unique()
    idx = [[*run_set, ext] for run_set in run_sets
           for ext in ['iqm', 'ci_low', 'ci_high']]

    iqm_time_series = pd.DataFrame(columns=pd.MultiIndex.from_tuples(idx))

    for run_set in run_sets:
        func = lambda x: apply_iqm(x, ci_level=ci_level, ci_n_resamples=ci_n_resamples)
        iqm_and_cis = time_series[run_set].apply(func, axis=1)
        iqm_time_series[run_set] = iqm_and_cis

    return iqm_time_series


def smooth_time_series(time_series: pd.DataFrame, std: float = 2.5) -> pd.DataFrame:
    """Applies Gaussian smoothing to time series.

    Args:
        time_series (pd.DataFrame): Data frame with one time series in each column.
        std (float, optional): Standard deviation to use for smoothing. Defaults to 2.5.

    Returns:
        pd.DataFrame: Smoothed time series.
    """

    smooth_time_series = time_series.rolling(
        window=int(std * 20 + 1), min_periods=int(std * 10), center=True, win_type='gaussian'
    ).mean(std=std)

    return smooth_time_series
