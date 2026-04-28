"""
system.py — Gravitational N-body system manager.

Part of the Orbital Symphony project.
This module is the physics engine: it owns a collection of Body instances,
computes Newtonian gravitational accelerations between all pairs, and
delegates time-stepping to the RK4 integrator.

Design Contract:
    - This is the ONLY module that contains gravitational physics.
    - It bridges the pure data containers (Body) with the pure math
      engine (integrator) by providing the `compute_accelerations`
      callback that the integrator requires.
    - It supports deep cloning of the entire system state, which is
      essential for the Divergence Moment (perturbation compare) feature.
"""

import numpy as np

from core.body import Body
from core.integrator import step_rk4


# Newtonian gravitational constant (SI units: m³ kg⁻¹ s⁻²)
G: float = 6.6743e-11


class System:
    """A self-contained N-body gravitational system.

    Manages a collection of bodies, computes pairwise Newtonian gravity,
    and advances the system through time via RK4 integration.

    Attributes:
        bodies (list[Body]): The point-mass bodies in this system.
        time (float): Accumulated simulation time in seconds.
    """

    def __init__(self, bodies: list[Body]) -> None:
        """Initialize a System with the given bodies.

        Args:
            bodies: List of Body instances to simulate. The system
                takes ownership of these objects and will mutate their
                state during time-stepping.

        Raises:
            ValueError: If fewer than 2 bodies are provided.
        """
        if len(bodies) < 2:
            raise ValueError(
                f"System requires at least 2 bodies, got {len(bodies)}"
            )

        self.bodies: list[Body] = bodies
        self.time: float = 0.0

    # ------------------------------------------------------------------
    # Physics: Newtonian Gravity
    # ------------------------------------------------------------------

    @staticmethod
    def compute_accelerations(bodies: list[Body]) -> None:
        """Compute gravitational accelerations for all bodies in-place.

        For every unique pair (i, j), calculates the gravitational
        force and applies Newton's Third Law to update both bodies
        simultaneously, avoiding redundant computation.

        This method signature matches the callback interface expected
        by `step_rk4`: it takes a list of bodies and updates their
        `.acceleration` attributes based on current positions.

        The gravitational acceleration on body i due to body j is:

            a_i += G * m_j / |r_ij|² * r̂_ij

        where r_ij = pos_j - pos_i and r̂_ij is the unit vector.

        Args:
            bodies: List of Body instances whose `.acceleration`
                attributes will be reset and recomputed.
        """
        n = len(bodies)

        # Reset all accelerations to zero before accumulating
        for body in bodies:
            body.acceleration = np.zeros(2, dtype=np.float64)

        # Pairwise loop — each pair computed once, applied to both bodies
        for i in range(n):
            for j in range(i + 1, n):
                # Displacement vector from body i to body j
                r_vec = bodies[j].position - bodies[i].position

                # Squared distance (avoids a sqrt we'd just square again)
                r_sq = np.dot(r_vec, r_vec)

                # Distance magnitude (needed for the unit vector)
                r_mag = np.sqrt(r_sq)

                # Unit direction vector from i toward j
                r_hat = r_vec / r_mag

                # Gravitational force magnitude: F = G * m_i * m_j / r²
                # Acceleration magnitude on i: a_i = G * m_j / r²
                # Acceleration magnitude on j: a_j = G * m_i / r²
                accel_i_mag = G * bodies[j].mass / r_sq
                accel_j_mag = G * bodies[i].mass / r_sq

                # Apply Newton's Third Law: equal and opposite
                bodies[i].acceleration += accel_i_mag * r_hat
                bodies[j].acceleration -= accel_j_mag * r_hat

    # ------------------------------------------------------------------
    # Time Evolution
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the system by one time step using RK4 integration.

        Delegates the actual integration to `step_rk4`, passing in
        `self.compute_accelerations` as the physics callback.

        Args:
            dt: Time step size in seconds.
        """
        step_rk4(self.bodies, dt, self.compute_accelerations)
        self.time += dt

    # ------------------------------------------------------------------
    # Cloning (for Perturbation Compare Mode)
    # ------------------------------------------------------------------

    def clone(self) -> "System":
        """Create an independent deep copy of the entire system.

        Returns a new System with cloned bodies so that the original
        and the copy can be evolved independently. This is the
        foundation of the Divergence Moment feature: clone a system,
        apply a tiny perturbation to the clone, run both forward, and
        measure how they diverge.

        Returns:
            A new System instance with identical but independent state.
        """
        cloned_bodies = [body.clone() for body in self.bodies]
        cloned_system = System(cloned_bodies)
        cloned_system.time = self.time
        return cloned_system

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        body_names = ", ".join(b.name for b in self.bodies)
        return (
            f"System(t={self.time:.2f}s, "
            f"bodies=[{body_names}])"
        )
