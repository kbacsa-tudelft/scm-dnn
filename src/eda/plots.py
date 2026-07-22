"""Figure generators. Every function saves a PNG to `out_path` and returns it,
so callers can embed the returned path directly into the markdown report."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.signal import welch

from .style import (
    ATTACK_COLOR,
    DIVERGING_BLUE_RED,
    GRIDLINE,
    INK_MUTED,
    NORMAL_COLOR,
    SEQUENTIAL_BLUE,
    apply_style,
    categorical_color,
)

apply_style()


def _grid_shape(n: int, ncols: int) -> tuple[int, int]:
    return math.ceil(n / ncols), ncols


def plot_histograms(df: pd.DataFrame, cols: list[str], out_path: str, ncols: int = 4) -> str:
    nrows, ncols = _grid_shape(len(cols), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=40, color=categorical_color(0), edgecolor="none")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_actuator_bars(df: pd.DataFrame, cols: list[str], out_path: str, ncols: int = 4) -> str:
    nrows, ncols = _grid_shape(len(cols), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, cols):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=categorical_color(4))
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_missingness(missing_pct: pd.Series, out_path: str, top_n: int = 25) -> str | None:
    nonzero = missing_pct[missing_pct > 0].head(top_n)
    if nonzero.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.28 * len(nonzero))))
    ax.barh(nonzero.index[::-1], nonzero.values[::-1], color=categorical_color(5))
    ax.set_xlabel("% missing")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_timeseries(
    df: pd.DataFrame, cols: list[str], out_path: str, attack_mask: np.ndarray | None = None
) -> str:
    """Small multiples: one subplot per sensor sharing a time axis, rather
    than overlaying differently-scaled sensors on one shared y-axis."""
    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 1.6 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(df))
    for ax, col in zip(axes, cols):
        ax.plot(x, df[col].to_numpy(), color=categorical_color(0), linewidth=1.0)
        if attack_mask is not None:
            for start, end in _mask_to_spans(attack_mask):
                ax.axvspan(start, end, color=ATTACK_COLOR, alpha=0.15, linewidth=0)
        ax.set_ylabel(col, fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("row index (time order)")
    if attack_mask is not None:
        fig.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=ATTACK_COLOR, alpha=0.15)],
            labels=["attack window"], loc="upper right", fontsize=8, frameon=False,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _mask_to_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask)
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: str) -> str:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * n), max(5, 0.22 * n)))
    im = ax.imshow(corr.values, cmap=DIVERGING_BLUE_RED, vmin=-1, vmax=1)
    show_labels = n <= 40
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if show_labels:
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr.columns, fontsize=6)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_class_balance(balance: dict, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.bar(["normal", "attack"], [balance["n_normal"], balance["n_attack"]],
           color=[NORMAL_COLOR, ATTACK_COLOR])
    ax.set_ylabel("row count")
    for i, v in enumerate([balance["n_normal"], balance["n_attack"]]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_segment_length_hist(lengths: list[int], out_path: str) -> str | None:
    if not lengths:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(lengths, bins=min(30, max(5, len(lengths))), color=categorical_color(5))
    ax.set_xlabel("attack segment length (rows)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pca_projection(
    coords: np.ndarray, labels: np.ndarray, explained_var: tuple[float, float], out_path: str
) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    is_attack = labels == 1
    ax.scatter(coords[~is_attack, 0], coords[~is_attack, 1], s=6, alpha=0.4,
               color=NORMAL_COLOR, label="normal", linewidths=0)
    ax.scatter(coords[is_attack, 0], coords[is_attack, 1], s=6, alpha=0.6,
               color=ATTACK_COLOR, label="attack", linewidths=0)
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_graph_topology(graph: nx.DiGraph, out_path: str, seed: int = 0) -> str:
    fig, ax = plt.subplots(figsize=(11, 11))
    pos = nx.spring_layout(graph, seed=seed, k=0.9)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=900, node_color=NORMAL_COLOR, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=GRIDLINE, arrows=True,
                            arrowsize=10, width=1.0, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7, font_color="white")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_psd_grid(
    signals: dict[str, np.ndarray], fs: float, out_path: str, ncols: int = 5, nperseg: int = 4096
) -> str:
    """Small multiples of power spectral density, one subplot per named
    signal (e.g. one per scenario) -- past ~4 categorical series a single
    overlaid multi-line plot stops being readable, so this facets instead."""
    names = list(signals.keys())
    nrows, ncols = _grid_shape(len(names), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, names):
        freqs, power = welch(signals[name], fs=fs, nperseg=nperseg)
        ax.semilogy(freqs, power, color=categorical_color(0), linewidth=1.0)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
