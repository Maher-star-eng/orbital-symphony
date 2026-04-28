"""
animator.py — Live 3D visualization engine using VPython.

Part of the Orbital Symphony project.
This module creates a real-time, interactive 3D simulation of the chaotic
N-body system. It runs the original and perturbed systems side-by-side,
allowing the user to visually experience the Divergence Moment.

Design Contract:
    - This module only handles rendering and 3D updates.
    - It delegates all physics calculations to the `System` class.
    - It uses VPython to create an interactive browser-based view.
"""

import vpython as vp
from core.system import System


class Animator:
    """Live 3D interactive visualizer for Orbital Symphony.

    Takes two identical `System` instances (one pristine, one perturbed)
    and evolves them in real-time, rendering their bodies as spheres and
    their trajectories as glowing trails.
    """

    def __init__(self, sys_original: System, sys_perturbed: System) -> None:
        """Initialize the 3D scene and map physics bodies to 3D objects.

        Args:
            sys_original: The reference physics system.
            sys_perturbed: The perturbed physics system.
        """
        self.sys_original = sys_original
        self.sys_perturbed = sys_perturbed

        # Setup the 3D canvas (opens in the browser)
        self.scene = vp.canvas(
            title="<b>Orbital Symphony: The Divergence Moment</b>",
            width=1200,
            height=800,
            background=vp.color.black,
            align="center",
        )
        self.scene.append_to_caption(
            "\\n<b>Controls:</b> Right-click and drag to rotate. Scroll to zoom.\\n"
            "<b>Legend:</b> Colored trails = Original System | Ghostly white trails = Perturbed System"
        )

        # Scale factor for astronomical distances
        # Radii are exaggerated so the bodies are visible at AU scales.
        self.radius_scale = 8e9

        # Curated VPython color palette for the original system
        self.palette = [
            vp.color.yellow,
            vp.color.cyan,
            vp.color.red,
            vp.color.green,
            vp.color.magenta,
            vp.color.orange,
        ]

        # Initialize lists to hold the VPython sphere objects
        self.orig_spheres = []
        self.pert_spheres = []

        self._setup_objects()

    def _setup_objects(self) -> None:
        """Create the VPython spheres and trails for all bodies."""
        # Setup original system (Colored, thick trails)
        for i, body in enumerate(self.sys_original.bodies):
            color = self.palette[i % len(self.palette)]
            sphere = vp.sphere(
                pos=vp.vector(body.position[0], body.position[1], 0),
                radius=self.radius_scale,
                color=color,
                make_trail=True,
                trail_color=color,
                trail_radius=self.radius_scale * 0.2,
                retain=3000,  # Trail length
            )
            self.orig_spheres.append(sphere)

        # Setup perturbed system (Ghostly white, thin trails)
        for i, body in enumerate(self.sys_perturbed.bodies):
            sphere = vp.sphere(
                pos=vp.vector(body.position[0], body.position[1], 0),
                radius=self.radius_scale * 0.8,
                color=vp.vector(0.9, 0.9, 0.9),  # off-white
                opacity=0.6,                     # slightly transparent
                make_trail=True,
                trail_color=vp.vector(0.9, 0.9, 0.9),
                trail_radius=self.radius_scale * 0.1,
                retain=3000,
            )
            self.pert_spheres.append(sphere)

    def run_live(self, dt: float, rate_val: int = 200) -> None:
        """Evolve the systems and update the 3D rendering in an infinite loop.

        Args:
            dt: The physics time step in seconds.
            rate_val: Maximum number of loop iterations per second (controls
                playback speed without affecting physics accuracy).
        """
        print(f"\\n▸ Launching Live 3D Simulation...")
        print("    Physics Δt:   {:.0f} s".format(dt))
        print("    Target Rate:  {} frames/sec".format(rate_val))
        print("    (Please check your web browser for the VPython window)\\n")

        # The VPython animation loop
        while True:
            # Control the maximum frames per second
            vp.rate(rate_val)

            # 1. Step the physics
            self.sys_original.step(dt)
            self.sys_perturbed.step(dt)

            # 2. Update the visual positions
            for i, body in enumerate(self.sys_original.bodies):
                self.orig_spheres[i].pos = vp.vector(body.position[0], body.position[1], 0)

            for i, body in enumerate(self.sys_perturbed.bodies):
                self.pert_spheres[i].pos = vp.vector(body.position[0], body.position[1], 0)
