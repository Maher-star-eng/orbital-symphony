"""
plotter.py — Science-grade 2D visualization engine.

Part of the Orbital Symphony project.
This module renders orbital trajectories and divergence curves using
Matplotlib.  It is a pure presentation layer that accepts only NumPy
arrays and display parameters — no physics objects, no simulation logic.

Design Contract:
    - This module does NOT import Body, System, or any simulation class.
    - It only knows about NumPy arrays, strings, and Matplotlib.
    - All scientific formatting (equal aspect, log scales, grids) is
      handled internally so the caller just passes data.

Two primary visualizations:
    1. plot_orbits      — 2D orbit trail map for all bodies.
    2. plot_divergence  — Divergence-vs-time graph (the killer feature).
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Shared style constants
# ------------------------------------------------------------------

# Curated colour palette — distinct, colour-blind-friendly, ordered
# by visual weight so the most massive body gets the strongest colour.
_PALETTE = [
    "#FFD700",  # gold        (star / primary)
    "#1E90FF",  # dodger blue (planet 1)
    "#FF6347",  # tomato      (planet 2)
    "#32CD32",  # lime green  (planet 3)
    "#DA70D6",  # orchid      (planet 4)
    "#00CED1",  # dark cyan   (planet 5)
]

_BACKGROUND = "#0E1117"
_GRID_COLOR = "#1E2530"
_TEXT_COLOR = "#C8D0DA"
_ACCENT = "#4FC3F7"


def _apply_dark_theme(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply a consistent dark-sky theme to the figure and axes."""
    fig.patch.set_facecolor(_BACKGROUND)
    ax.set_facecolor(_BACKGROUND)
    ax.tick_params(colors=_TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.4, alpha=0.6)


# ==================================================================
# Function 1: Orbit Trails
# ==================================================================

def plot_orbits(
    history_dict: dict[str, np.ndarray],
    title: str = "Orbital Trajectories",
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D orbit trails for all bodies on a shared coordinate frame.

    Each body's full trajectory is drawn as a coloured line, with a
    solid marker at its final position.  The aspect ratio is forced to
    'equal' so circular orbits render as circles, not ellipses.

    Args:
        history_dict: Mapping of body name → position history.
            Each value must be a NumPy array of shape (N, 2) where
            columns are [x, y] coordinates.
        title: Plot title string.
        save_path: If provided, the figure is saved to this path
            (e.g., "output/orbits.png") and closed automatically.
            If None, the figure is displayed interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    _apply_dark_theme(ax, fig)

    for idx, (name, history) in enumerate(history_dict.items()):
        colour = _PALETTE[idx % len(_PALETTE)]

        # Full trajectory trail
        ax.plot(
            history[:, 0],
            history[:, 1],
            color=colour,
            linewidth=0.8,
            alpha=0.85,
            label=name,
        )

        # Starting position — small open circle
        ax.plot(
            history[0, 0],
            history[0, 1],
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=1.2,
        )

        # Final position — solid dot
        ax.plot(
            history[-1, 0],
            history[-1, 1],
            marker="o",
            markersize=8,
            color=colour,
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        loc="upper right",
        fontsize=9,
        facecolor=_BACKGROUND,
        edgecolor=_GRID_COLOR,
        labelcolor=_TEXT_COLOR,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


# ==================================================================
# Function 2: Divergence Graph (The Killer Feature)
# ==================================================================

def plot_divergence(
    time_array: np.ndarray,
    divergence_array: np.ndarray,
    title: str = "Trajectory Divergence",
    save_path: Optional[str] = None,
) -> None:
    """Plot trajectory divergence vs. time on a logarithmic y-scale.

    This is the visual centrepiece of the "Divergence Moment" feature.
    In a chaotic system, the curve will rise exponentially — a straight
    line on a log-y plot indicates a positive Lyapunov exponent.

    A horizontal reference line marks the initial divergence (t=0) so
    the viewer can immediately see the amplification factor.

    Args:
        time_array: 1D array of shape (N,) with simulation timestamps
            (in seconds or any consistent unit).
        divergence_array: 1D array of shape (N,) with Euclidean
            distances between the reference and perturbed trajectories,
            as produced by `analysis.divergence.calculate_divergence`.
        title: Plot title string.
        save_path: If provided, the figure is saved to this path and
            closed.  If None, displayed interactively.

    Raises:
        ValueError: If array shapes are incompatible.
    """
    if time_array.shape != divergence_array.shape:
        raise ValueError(
            f"Shape mismatch: time_array {time_array.shape} vs "
            f"divergence_array {divergence_array.shape}"
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_dark_theme(ax, fig)

    # Main divergence curve
    ax.plot(
        time_array,
        divergence_array,
        color=_ACCENT,
        linewidth=1.4,
        label="Divergence  Δd(t)",
    )

    # Reference line at initial divergence
    d0 = divergence_array[0]
    if d0 > 0:
        ax.axhline(
            y=d0,
            color="#FF6347",
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
            label=f"Initial offset  ({d0:.2e} m)",
        )

    # Logarithmic y-scale — the signature of chaos is a straight line here
    # Guard against zero/negative values that would break log scale
    if np.all(divergence_array > 0):
        ax.set_yscale("log")

    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Divergence  Δd  [m]")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        loc="upper left",
        fontsize=9,
        facecolor=_BACKGROUND,
        edgecolor=_GRID_COLOR,
        labelcolor=_TEXT_COLOR,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()
