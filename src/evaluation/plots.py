from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axis import Axis
from matplotlib.figure import Figure


def iqm_line_plot(iqm_time_series: pd.DataFrame, labels: Dict = None, colors: Dict = None) -> Tuple[Figure, Axis]:
    """Line plot of interquartile mean (IQM) values and their confidence intervals (CIs) over time.

    Args:
        iqm_time_series (pd.DataFrame): Data Frame with time series of IQM values and CIs for different configurations.
        labels (Dict, optional): Dictionary mapping configuration names to labels. Defaults to None.
        colors (Dict, optional): Colors to use for each IQM line. Defaults to None.

    Returns:
        Figure: Matplotlib figure and axis.
    """

    fig, ax = plt.subplots()

    x_axis = iqm_time_series.index
    run_sets = iqm_time_series.columns.droplevel(-1).unique()
    if labels is None:
        labels = {run_set: str(run_set) for run_set in run_sets}
    if colors is None:
        colors = {run_set: f'C{i}' for i, run_set in enumerate(run_sets)}

    for run_set in run_sets:
        # IQM line
        ax.plot(x_axis, iqm_time_series[run_set]['iqm'], label=labels[run_set], color=colors[run_set])

        # confidence intervals
        ax.fill_between(x_axis, iqm_time_series[run_set]['ci_low'], iqm_time_series[run_set]['ci_high'], color=colors[run_set], alpha=0.05)

    return fig, ax


def iqm_ci_plot(iqm_values: pd.Series, labels: Dict = None) -> Tuple[Figure, Axis]:
    """Plot of static interquartile mean (IQM) values and their confidence intervals (CIs).

    Args:
        iqm_values (pd.Series): Series with IQM values and CIs for different configurations.
        labels (Dict, optional): Dictionary mapping configuration names to labels. Defaults to None.

    Returns:
        Tuple[Figure, Axis]: Matplotlib figure and axis.
    """

    fig, ax = plt.subplots()

    run_sets = iqm_values.index.droplevel(-1).unique()
    if labels is None:
        labels = {run_set: str(run_set) for run_set in run_sets}

    for run_set in run_sets:
        iqm, ci_low, ci_high = iqm_values.loc[run_set]
        ax.scatter(iqm, labels[run_set], marker='|')
        ax.plot([ci_low, ci_high], [labels[run_set]]*2, alpha=0.25, lw=6)

    return fig, ax
