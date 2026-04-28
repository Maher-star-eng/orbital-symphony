"""
sonify.py — Orbital data sonification engine.

Part of the Orbital Symphony project.
Transforms time-series velocity data from an N-body simulation into
audible .wav files, letting the user literally *hear* chaos unfold.

Physical → Auditory Mapping:
    ┌────────────────────┬──────────────────────────────────────────┐
    │  Physics quantity   │  Sound parameter                        │
    ├────────────────────┼──────────────────────────────────────────┤
    │  Velocity magnitude │  Pitch (instantaneous frequency)        │
    │                     │  200 Hz (slow) → 2000 Hz (fast)         │
    ├────────────────────┼──────────────────────────────────────────┤
    │  Velocity magnitude │  Amplitude (louder during fast passes)  │
    │                     │  0.3 (slow) → 1.0 (fast)               │
    └────────────────────┴──────────────────────────────────────────┘

Synthesis Technique:
    Phase-accumulation FM synthesis. Rather than generating a fixed-
    frequency sine wave, we compute the instantaneous phase at every
    sample as the running integral of the time-varying frequency:

        φ[n] = Σ  2π · f[k] / sample_rate    (k = 0..n)

    This produces smooth, click-free frequency sweeps even when the
    velocity (and thus pitch) changes rapidly during close encounters.

Design Contract:
    - This module is DECOUPLED from the physics engine.
    - It accepts only NumPy arrays and scalar parameters.
    - All synthesis is FULLY VECTORIZED — zero Python loops.
"""

import numpy as np
from scipy.io import wavfile


# ------------------------------------------------------------------
# Configuration defaults
# ------------------------------------------------------------------

_FREQ_MIN: float = 200.0    # Hz — lowest audible pitch (slow orbit)
_FREQ_MAX: float = 2000.0   # Hz — highest audible pitch (fast flyby)
_AMP_MIN: float = 0.3       # minimum amplitude (quiet baseline)
_AMP_MAX: float = 1.0       # maximum amplitude (close encounter)
_FADE_FRACTION: float = 0.02  # fraction of total duration for fade in/out


def generate_orbit_sound(
    velocity_history: np.ndarray,
    output_path: str,
    duration: float = 5.0,
    sample_rate: int = 44100,
    freq_range: tuple[float, float] = (_FREQ_MIN, _FREQ_MAX),
    amp_range: tuple[float, float] = (_AMP_MIN, _AMP_MAX),
) -> None:
    """Synthesize an audible .wav file from orbital velocity data.

    Converts a time-series of velocity magnitudes into a sine tone
    whose pitch and volume track the dynamics of the simulation.
    Fast close encounters become high-pitched, loud sweeps; slow
    coasting phases become low, quiet drones.

    The entire synthesis pipeline is vectorized — no Python loops.

    Args:
        velocity_history: 1D array of shape (N,) containing velocity
            magnitudes (m/s) at each simulation time step.
        output_path: File path for the output .wav file
            (e.g., "output/orbit_sound.wav").
        duration: Desired audio duration in seconds.
        sample_rate: Audio sample rate in Hz (default: 44100 = CD quality).
        freq_range: (min_freq, max_freq) in Hz for pitch mapping.
        amp_range: (min_amp, max_amp) in [0, 1] for volume mapping.

    Raises:
        TypeError: If velocity_history is not a NumPy ndarray.
        ValueError: If velocity_history is empty or not 1D.
    """
    # --- Input validation ---
    if not isinstance(velocity_history, np.ndarray):
        raise TypeError(
            f"velocity_history must be a numpy ndarray, "
            f"got {type(velocity_history).__name__}"
        )
    if velocity_history.ndim != 1 or velocity_history.size == 0:
        raise ValueError(
            f"velocity_history must be a non-empty 1D array, "
            f"got shape {velocity_history.shape}"
        )

    total_samples = int(duration * sample_rate)
    f_lo, f_hi = freq_range
    a_lo, a_hi = amp_range

    # ==============================================================
    # 1. Resample velocity history to match audio sample count
    # ==============================================================
    # The simulation may have N=60,000 steps but we need ~220,500
    # audio samples (5s × 44100).  Use linear interpolation to
    # create a smooth, sample-accurate velocity envelope.
    sim_indices = np.linspace(0, 1, velocity_history.size)
    audio_indices = np.linspace(0, 1, total_samples)
    velocity_resampled = np.interp(audio_indices, sim_indices, velocity_history)

    # ==============================================================
    # 2. Normalise velocity to [0, 1] range
    # ==============================================================
    v_min = velocity_resampled.min()
    v_max = velocity_resampled.max()

    if v_max - v_min < 1e-12:
        # Constant velocity — map to mid-range
        v_norm = np.full(total_samples, 0.5)
    else:
        v_norm = (velocity_resampled - v_min) / (v_max - v_min)

    # ==============================================================
    # 3. Map normalised velocity → instantaneous frequency
    # ==============================================================
    # Linear mapping: v_norm=0 → f_lo,  v_norm=1 → f_hi
    freq_envelope = f_lo + v_norm * (f_hi - f_lo)

    # ==============================================================
    # 4. Phase-accumulation synthesis (vectorized FM)
    # ==============================================================
    # The instantaneous phase at sample n is the running integral
    # of 2π·f(t)/sr.  Using cumulative sum avoids any loop and
    # produces perfectly smooth frequency transitions.
    phase_increment = 2.0 * np.pi * freq_envelope / sample_rate
    phase = np.cumsum(phase_increment)

    # Generate the carrier wave
    waveform = np.sin(phase)

    # ==============================================================
    # 5. Apply amplitude envelope (velocity → volume)
    # ==============================================================
    # Louder during fast passes, quieter during slow coasting.
    amp_envelope = a_lo + v_norm * (a_hi - a_lo)
    waveform *= amp_envelope

    # ==============================================================
    # 6. Fade-in / fade-out to avoid click artefacts
    # ==============================================================
    waveform = _apply_fades(waveform, sample_rate, _FADE_FRACTION)

    # ==============================================================
    # 7. Normalise to 16-bit PCM and write .wav
    # ==============================================================
    # Scale to int16 range with a small headroom margin (0.95)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak * 0.95

    pcm_data = (waveform * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, pcm_data)


def _apply_fades(
    waveform: np.ndarray,
    sample_rate: int,
    fade_fraction: float,
) -> np.ndarray:
    """Apply smooth cosine fade-in and fade-out to a waveform.

    Uses a raised-cosine (Hann) window shape for the fades, which
    produces a perceptually natural volume ramp without the abrupt
    linear-fade artefact.

    Args:
        waveform: 1D audio buffer (modified in-place and returned).
        sample_rate: Audio sample rate (used for minimum fade length).
        fade_fraction: Fraction of total waveform length for each fade.

    Returns:
        The same waveform array with fades applied.
    """
    n = waveform.size
    fade_samples = max(int(n * fade_fraction), int(sample_rate * 0.01))
    fade_samples = min(fade_samples, n // 2)  # guard against tiny buffers

    # Raised-cosine ramp: 0 → 1 (smooth, no discontinuity in derivative)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))

    waveform[:fade_samples] *= ramp             # fade-in
    waveform[-fade_samples:] *= ramp[::-1]      # fade-out (mirror)

    return waveform
