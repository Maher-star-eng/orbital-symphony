"""
divergence.py — Trajectory divergence measurement engine.

Part of the Orbital Symphony project.
This module quantifies how two initially-similar trajectories diverge
over time by computing the Euclidean distance between corresponding
positions at each time step.

Design Contract:
    - This module is ABSOLUTELY DECOUPLED from the physics engine.
    - It does NOT import Body, System, or any simulation class.
    - It operates exclusively on NumPy arrays — a pure data-processing
      pipeline that knows nothing about gravity or orbits.
    - All computation is FULLY VECTORIZED with zero Python loops.

Usage:
    The orchestrator (main.py) records position histories as (N, 2)
    arrays during simulation, then passes them here for analysis.
    The output feeds directly into the divergence plot.
"""

import numpy as np


def calculate_divergence(
    history_a: np.ndarray,
    history_b: np.ndarray,
) -> np.ndarray:
    """Compute per-timestep Euclidean distance between two trajectories.

    Given two position-history arrays of shape (N, 2), calculates the
    Euclidean distance ||a_i - b_i|| at every time step i using fully
    vectorized NumPy operations (zero Python loops).

    This is the mathematical core of the "Divergence Moment" feature:
    in a chaotic system, even a sub-meter initial offset will cause
    this distance to grow exponentially over time.

    Args:
        history_a: Position history of a body in the reference system.
            Shape must be (N, 2) where N is the number of recorded
            time steps and columns are [x, y] coordinates in meters.
        history_b: Position history of the same body in the perturbed
            system.  Must have the exact same shape as `history_a`.

    Returns:
        A 1D NumPy array of shape (N,) containing the Euclidean
        distance (in meters) between the two trajectories at each
        time step.

    Raises:
        TypeError: If either input is not a NumPy ndarray.
        ValueError: If arrays are not 2D with 2 columns, or if their
            shapes do not match.

    Example:
        >>> a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> b = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        >>> calculate_divergence(a, b)
        array([1., 1., 1.])
    """
    # --- Type validation ---
    if not isinstance(history_a, np.ndarray):
        raise TypeError(
            f"history_a must be a numpy ndarray, got {type(history_a).__name__}"
        )
    if not isinstance(history_b, np.ndarray):
        raise TypeError(
            f"history_b must be a numpy ndarray, got {type(history_b).__name__}"
        )

    # --- Shape validation ---
    if history_a.ndim != 2 or history_a.shape[1] != 2:
        raise ValueError(
            f"history_a must have shape (N, 2), got {history_a.shape}"
        )
    if history_b.ndim != 2 or history_b.shape[1] != 2:
        raise ValueError(
            f"history_b must have shape (N, 2), got {history_b.shape}"
        )
    if history_a.shape != history_b.shape:
        raise ValueError(
            f"Shape mismatch: history_a {history_a.shape} vs "
            f"history_b {history_b.shape}. Both must have identical "
            f"dimensions (same number of time steps)."
        )

    # --- Vectorized Euclidean distance (zero loops) ---
    # displacement[i] = a[i] - b[i]  →  shape (N, 2)
    # norm along axis=1              →  shape (N,)
    displacement = history_a - history_b
    distances = np.linalg.norm(displacement, axis=1)

    return distances
