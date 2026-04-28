"""
integrator.py — Runge-Kutta 4th Order (RK4) time-stepping engine.

Part of the Orbital Symphony project.
This module implements the classical RK4 integration scheme adapted for
coupled N-body systems where the acceleration of each body depends on
the simultaneous positions of ALL other bodies.

Design Contract:
    - This module is a PURE MATHEMATICAL ENGINE.
    - It contains NO physics equations, gravitational constants, or
      force calculations.
    - All physics (gravity) is injected via the `compute_accelerations`
      callback, preserving strict separation of concerns.

RK4 Algorithm (adapted for N-body coupling):
    For state vector y = [position, velocity] and derivative f(y):
        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        k3 = f(y_n + dt/2 * k2)
        k4 = f(y_n + dt   * k3)
        y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Because accelerations are coupled (body A's acceleration depends on
    body B's position), ALL bodies must be advanced to each intermediate
    state simultaneously before evaluating the derivative there.
"""

from typing import Callable

import numpy as np

from core.body import Body


def step_rk4(
    bodies: list[Body],
    dt: float,
    compute_accelerations: Callable[[list[Body]], None],
) -> None:
    """Advance all bodies forward by one time step using RK4.

    This function mutates the `position` and `velocity` attributes of
    each body in-place. The original bodies are NOT touched until the
    final weighted sum is applied, ensuring numerical consistency.

    Args:
        bodies: List of Body instances representing the current system
            state. Modified in-place at the end of the step.
        dt: Time step size in seconds.
        compute_accelerations: A callback that takes a list of Body
            instances and updates each body's `.acceleration` attribute
            in-place based on their current positions.  This is the
            physics injection point — the integrator never computes
            forces itself.

    Algorithm Detail:
        For each RK4 stage (k1–k4), a temporary clone of the entire
        system is created at the appropriate intermediate state. The
        callback evaluates accelerations on this temporary system, and
        the resulting derivatives (velocity → position rate, acceleration
        → velocity rate) are stored as k-vectors. The original bodies
        remain untouched until the final weighted average is applied.
    """
    n = len(bodies)

    # --- Storage for the four sets of k-vectors ---
    # Each k_pos[i] / k_vel[i] is the derivative contribution for body i.
    k1_pos = [np.zeros(2) for _ in range(n)]
    k1_vel = [np.zeros(2) for _ in range(n)]

    k2_pos = [np.zeros(2) for _ in range(n)]
    k2_vel = [np.zeros(2) for _ in range(n)]

    k3_pos = [np.zeros(2) for _ in range(n)]
    k3_vel = [np.zeros(2) for _ in range(n)]

    k4_pos = [np.zeros(2) for _ in range(n)]
    k4_vel = [np.zeros(2) for _ in range(n)]

    # ================================================================
    # STAGE 1: Evaluate derivatives at the current state  (t_n)
    # ================================================================
    compute_accelerations(bodies)

    for i, body in enumerate(bodies):
        k1_pos[i] = body.velocity * dt          # dx/dt = v
        k1_vel[i] = body.acceleration * dt       # dv/dt = a(x)

    # ================================================================
    # STAGE 2: Evaluate derivatives at the midpoint  (t_n + dt/2)
    #          using k1 to estimate the midpoint state.
    # ================================================================
    temp = _build_temp_system(bodies, k1_pos, k1_vel, fraction=0.5)
    compute_accelerations(temp)

    for i in range(n):
        k2_pos[i] = temp[i].velocity * dt
        k2_vel[i] = temp[i].acceleration * dt

    # ================================================================
    # STAGE 3: Evaluate derivatives at the midpoint  (t_n + dt/2)
    #          using k2 to re-estimate the midpoint state.
    # ================================================================
    temp = _build_temp_system(bodies, k2_pos, k2_vel, fraction=0.5)
    compute_accelerations(temp)

    for i in range(n):
        k3_pos[i] = temp[i].velocity * dt
        k3_vel[i] = temp[i].acceleration * dt

    # ================================================================
    # STAGE 4: Evaluate derivatives at the endpoint  (t_n + dt)
    #          using k3 to estimate the full-step state.
    # ================================================================
    temp = _build_temp_system(bodies, k3_pos, k3_vel, fraction=1.0)
    compute_accelerations(temp)

    for i in range(n):
        k4_pos[i] = temp[i].velocity * dt
        k4_vel[i] = temp[i].acceleration * dt

    # ================================================================
    # FINAL: Apply the RK4 weighted average to the ORIGINAL bodies.
    #        y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
    # ================================================================
    for i, body in enumerate(bodies):
        body.position += (k1_pos[i] + 2*k2_pos[i] + 2*k3_pos[i] + k4_pos[i]) / 6.0
        body.velocity += (k1_vel[i] + 2*k2_vel[i] + 2*k3_vel[i] + k4_vel[i]) / 6.0


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _build_temp_system(
    bodies: list[Body],
    k_pos: list[np.ndarray],
    k_vel: list[np.ndarray],
    fraction: float,
) -> list[Body]:
    """Clone the system and offset each body by `fraction * k`.

    Creates an independent snapshot of all bodies with positions and
    velocities shifted to the intermediate RK4 evaluation point.

    Args:
        bodies: The original (unmodified) system state.
        k_pos: Position derivative contributions (one per body).
        k_vel: Velocity derivative contributions (one per body).
        fraction: Scaling factor (0.5 for midpoints, 1.0 for endpoint).

    Returns:
        A new list of cloned Body instances at the intermediate state.
    """
    temp = []
    for i, body in enumerate(bodies):
        clone = body.clone()
        clone.position += fraction * k_pos[i]
        clone.velocity += fraction * k_vel[i]
        temp.append(clone)
    return temp
