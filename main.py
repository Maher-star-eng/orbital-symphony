"""
main.py — Orbital Symphony: The Divergence Moment.

Grand orchestrator for the chaotic N-body perturbation experiment.
This script demonstrates the core thesis of chaos theory: in an
unstable gravitational system, an offset of just ONE MILLIMETER
grows exponentially until the two trajectories become completely
unrecognisable from each other.

Workflow:
    1. Initialize a chaotic 3-body system (three stars in an unstable
       triangular configuration).
    2. Clone the system and apply a 1 mm spatial perturbation to one body.
    3. Evolve both systems in parallel, recording position histories.
    4. Measure trajectory divergence via Euclidean distance.
    5. Render science-grade orbit trails and divergence plots.
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # headless backend — save to file, no GUI required

import numpy as np

from core.body import Body
from core.system import System, G
from analysis.divergence import calculate_divergence
from visualization.plotter import plot_orbits, plot_divergence
from audio.sonify import generate_orbit_sound


# ==================================================================
# Configuration
# ==================================================================

OUTPUT_DIR = "output"

# --- Chaotic 3-body initial conditions ---
# Three solar-mass-class stars arranged in a scalene triangle with
# carefully chosen velocities that prevent immediate escape but
# guarantee close encounters → chaos.
#
# Physical scale:
#   Mass   ~ 1×10³⁰ kg  (roughly half a solar mass each)
#   Length ~ 1×10¹¹ m    (sub-AU separations → strong interactions)
#   Velocity ~ √(GM/r) ≈ 26 km/s characteristic orbital speed
#
# This configuration is inspired by the Pythagorean 3-body problem
# (Burrau 1913) — one of the most celebrated chaotic systems in
# computational astrophysics.

M = 1.0e30  # kg — mass of each star

BODIES_CONFIG = [
    {
        "name": "Star A",
        "mass": 3.0 * M,
        "position": [1.0e11, 3.0e11],
        "velocity": [0.0, 0.0],
    },
    {
        "name": "Star B",
        "mass": 4.0 * M,
        "position": [-2.0e11, -1.0e11],
        "velocity": [0.0, 0.0],
    },
    {
        "name": "Star C",
        "mass": 5.0 * M,
        "position": [1.0e11, -1.0e11],
        "velocity": [0.0, 0.0],
    },
]

# --- Simulation parameters ---
DT = 500.0             # time step (seconds) — fine enough for close encounters
T_TOTAL = 3.0e7        # total simulation time (seconds) ≈ 347 days
N_STEPS = int(T_TOTAL / DT)

# --- Perturbation ---
PERTURB_BODY_INDEX = 0   # which body to perturb (Star A)
PERTURB_OFFSET_M = 1e-3  # 1 millimeter — the magic number


# ==================================================================
# Helpers
# ==================================================================

def _build_system() -> System:
    """Construct the 3-body system from configuration."""
    bodies = [
        Body(
            name=cfg["name"],
            mass=cfg["mass"],
            position=cfg["position"],
            velocity=cfg["velocity"],
        )
        for cfg in BODIES_CONFIG
    ]
    return System(bodies)


def _fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hr"


def _fmt_days(seconds: float) -> str:
    """Format simulation time in days."""
    return f"{seconds / 86400:.1f} days"


# ==================================================================
# Main Execution
# ==================================================================

def main() -> None:
    """Run the full Divergence Moment experiment."""
    parser = argparse.ArgumentParser(description="Orbital Symphony Orchestrator")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Launch live 3D VPython visualization instead of offline rendering"
    )
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("   ◈  O R B I T A L   S Y M P H O N Y  ◈")
    print("   The Divergence Moment — Chaos in Three Bodies")
    print("=" * 62)
    print()

    # --- Ensure output directory exists ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Initialize the chaotic system
    # ----------------------------------------------------------
    print("▸ Initializing chaotic 3-body system...")
    sys_original = _build_system()
    for body in sys_original.bodies:
        print(f"    {body}")
    print()

    # ----------------------------------------------------------
    # 2. Clone & perturb
    # ----------------------------------------------------------
    sys_perturbed = sys_original.clone()
    target = sys_perturbed.bodies[PERTURB_BODY_INDEX]
    target.position[0] += PERTURB_OFFSET_M

    print(f"▸ Perturbation applied:")
    print(f"    Body:    {target.name}")
    print(f"    Offset:  {PERTURB_OFFSET_M * 1e3:.1f} mm  in x-direction")
    print(f"    (That's {PERTURB_OFFSET_M:.0e} m on a {1e11:.0e} m scale)")
    print()

    # ----------------------------------------------------------
    # Live 3D Mode Branch
    # ----------------------------------------------------------
    if args.live:
        from visualization.animator import Animator
        animator = Animator(sys_original, sys_perturbed)
        animator.run_live(dt=DT, rate_val=200)
        return

    # ----------------------------------------------------------
    # 3. Simulation loop — evolve both systems in parallel
    # ----------------------------------------------------------
    n_bodies = len(sys_original.bodies)
    total_frames = N_STEPS + 1

    # Position histories: one (N, 2) array per body in the original system
    histories_orig = [np.zeros((total_frames, 2)) for _ in range(n_bodies)]

    # Velocity magnitude histories: one (N,) array per body (for sonification)
    vel_histories = [np.zeros(total_frames) for _ in range(n_bodies)]

    # We only need the perturbed body's history for divergence analysis
    history_perturbed = np.zeros((total_frames, 2))

    # Time array
    time_array = np.zeros(total_frames)

    # Record initial state (step 0)
    for i, body in enumerate(sys_original.bodies):
        histories_orig[i][0] = body.position.copy()
        vel_histories[i][0] = np.linalg.norm(body.velocity)
    history_perturbed[0] = sys_perturbed.bodies[PERTURB_BODY_INDEX].position.copy()

    print(f"▸ Running simulation...")
    print(f"    Steps:     {N_STEPS:,}")
    print(f"    Δt:        {DT:.0f} s")
    print(f"    Duration:  {_fmt_days(T_TOTAL)}")
    print()

    wall_start = time.perf_counter()
    report_interval = N_STEPS // 10  # progress every 10%

    for step in range(1, N_STEPS + 1):
        sys_original.step(DT)
        sys_perturbed.step(DT)

        # Record positions and velocities
        for i, body in enumerate(sys_original.bodies):
            histories_orig[i][step] = body.position.copy()
            vel_histories[i][step] = np.linalg.norm(body.velocity)
        history_perturbed[step] = (
            sys_perturbed.bodies[PERTURB_BODY_INDEX].position.copy()
        )
        time_array[step] = sys_original.time

        # Progress reporting
        if step % report_interval == 0:
            pct = step / N_STEPS * 100
            elapsed = time.perf_counter() - wall_start
            eta = elapsed / step * (N_STEPS - step)
            print(
                f"    [{pct:5.1f}%]  "
                f"t = {_fmt_days(sys_original.time):>12s}   "
                f"elapsed: {_fmt_time(elapsed)}   "
                f"ETA: {_fmt_time(eta)}"
            )

    wall_total = time.perf_counter() - wall_start
    print(f"\n    ✓ Simulation complete in {_fmt_time(wall_total)}")
    print()

    # ----------------------------------------------------------
    # 4. Analysis — compute divergence
    # ----------------------------------------------------------
    print("▸ Computing trajectory divergence...")
    history_ref = histories_orig[PERTURB_BODY_INDEX]
    divergence = calculate_divergence(history_ref, history_perturbed)

    d_initial = divergence[0]
    d_final = divergence[-1]
    d_max = divergence.max()
    amplification = d_max / d_initial if d_initial > 0 else float("inf")

    print(f"    Initial Δd:      {d_initial:.2e} m")
    print(f"    Final Δd:        {d_final:.2e} m")
    print(f"    Peak Δd:         {d_max:.2e} m")
    print(f"    Amplification:   {amplification:.2e}×")
    print()

    # ----------------------------------------------------------
    # 5. Visualization — generate plots
    # ----------------------------------------------------------
    print("▸ Generating plots...")

    # --- Orbit trail plot ---
    orbit_dict = {
        sys_original.bodies[i].name: histories_orig[i]
        for i in range(n_bodies)
    }
    orbit_path = os.path.join(OUTPUT_DIR, "chaotic_orbits.png")
    plot_orbits(
        orbit_dict,
        title="Chaotic 3-Body System — Orbital Trajectories",
        save_path=orbit_path,
    )
    print(f"    Saved: {orbit_path}")

    # --- Divergence plot ---
    div_path = os.path.join(OUTPUT_DIR, "chaotic_divergence.png")
    plot_divergence(
        time_array,
        divergence,
        title=(
            f"Divergence Moment — {PERTURB_OFFSET_M*1e3:.1f} mm offset "
            f"→ {d_max:.2e} m peak"
        ),
        save_path=div_path,
    )
    print(f"    Saved: {div_path}")

    # --- Sonification — hear the chaos ---
    print("\n▸ Generating sonification...")
    for i in range(n_bodies):
        body_name = sys_original.bodies[i].name.lower().replace(" ", "_")
        wav_path = os.path.join(OUTPUT_DIR, f"{body_name}_sound.wav")
        generate_orbit_sound(
            vel_histories[i],
            wav_path,
            duration=8.0,
        )
        v_min = vel_histories[i].min()
        v_max = vel_histories[i].max()
        print(f"    Saved: {wav_path}  (v: {v_min:.0f}–{v_max:.0f} m/s → 200–2000 Hz)")

    # ----------------------------------------------------------
    # Finale
    # ----------------------------------------------------------
    print()
    print("=" * 62)
    print("   ◈  THE DIVERGENCE MOMENT  ◈")
    print()
    print(f"   A {PERTURB_OFFSET_M*1e3:.1f} mm offset — smaller than a grain of sand —")
    print(f"   grew by a factor of {amplification:.2e}×")
    print(f"   into a {d_max:.2e} m divergence.")
    print()
    print("   This is chaos.")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
