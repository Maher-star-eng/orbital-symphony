"""
body.py — Pure data container for a celestial body.

Part of the Orbital Symphony project.
This module defines the Body class, which stores the physical state
of a single point-mass object (mass, position, velocity, acceleration).

Design Contract:
    - This class is strictly a data container.
    - It contains NO physics equations, gravity logic, or references
      to other bodies.
    - All gravitational interactions and time-stepping are handled
      externally by the system and integrator modules.
"""

import numpy as np
from copy import deepcopy


class Body:
    """A point-mass celestial body with kinematic state vectors.

    Attributes:
        name (str): Human-readable label for identification and plotting.
        mass (float): Gravitational mass of the body (kg).
        position (np.ndarray): 2D position vector [x, y] (m).
        velocity (np.ndarray): 2D velocity vector [vx, vy] (m/s).
        acceleration (np.ndarray): 2D acceleration vector [ax, ay] (m/s²),
            initialized to zero and updated externally by the physics engine.
    """

    def __init__(
        self,
        name: str,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        """Initialize a Body with the given physical properties.

        Args:
            name: Human-readable identifier (e.g., "Sun", "Earth").
            mass: Gravitational mass in kilograms. Must be positive.
            position: Initial 2D position vector [x, y] in meters.
            velocity: Initial 2D velocity vector [vx, vy] in m/s.

        Raises:
            ValueError: If mass is not positive, or if position/velocity
                are not 2-element arrays.
        """
        # --- Validate mass ---
        if mass <= 0:
            raise ValueError(f"Mass must be positive, got {mass}")

        # --- Validate and cast state vectors ---
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)

        if position.shape != (2,):
            raise ValueError(
                f"Position must be a 2-element vector, got shape {position.shape}"
            )
        if velocity.shape != (2,):
            raise ValueError(
                f"Velocity must be a 2-element vector, got shape {velocity.shape}"
            )

        self.name: str = name
        self.mass: float = float(mass)
        self.position: np.ndarray = position
        self.velocity: np.ndarray = velocity
        self.acceleration: np.ndarray = np.zeros(2, dtype=np.float64)

    def clone(self) -> "Body":
        """Create an independent deep copy of this body.

        Used by the perturbation compare mode to duplicate an entire
        system before applying a tiny offset.

        Returns:
            A new Body instance with identical but independent state.
        """
        return Body(
            name=self.name,
            mass=self.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
        )

    def __repr__(self) -> str:
        return (
            f"Body(name='{self.name}', mass={self.mass:.4e}, "
            f"pos={self.position}, vel={self.velocity})"
        )
