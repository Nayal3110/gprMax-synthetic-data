"""Calibrated spectra in, a homogeneous reference medium out.

``estimate_background`` takes ``u_cal [n_src, n_rx, n_freq]`` with the source
and receiver positions and the frequency axis, and returns a ``Background``
carrying ``eps_bg``, ``sigma_bg`` and the ``coherence`` those two achieved. It
stacks the measured phasors against analytic 2-D Green's functions over every
source-receiver pair, one frequency at a time, and maximises the normalised
stack on a coarse grid followed by a Nelder-Mead refine.

Stacking phasors is wrap-free: phase enters only through ``u * conj(G)``, so a
path many wavelengths long contributes a rotated unit vector rather than a
branch choice, and the estimate holds at any panel-to-wavelength ratio.
``phase_slope_permittivity`` fits unwrapped phase against offset instead, and
is a diagnostic.

``coherence`` is a quality signal a real survey can produce: it needs no truth
model, and a panel too heterogeneous for one reference permittivity drives it
down, which is what ``require_coherent`` gates on.

Risk: coherence rewards any medium that explains the *direct* arrival, so a
panel with a strong late event inside the band, or heterogeneity symmetric
about the array, can score well while the single-eps reference is still wrong.
The gate is necessary, not sufficient.

Conventions
-----------
``e^{+j w t}``, ``eps_c = eps' - 1j sigma / (w eps_0)``, outgoing ``e^{-j k r}``,
``G_2D = -(1j/4) H_0^(2)(k_b r)``. Fields ``complex128``, model reals ``float64``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c as C_LIGHT
from scipy.optimize import minimize
from scipy.special import hankel2

from . import EPS_MIN, EPS_WATER_MAX
from .operators import background_wavenumber

# Coherence below this means no single permittivity explains the panel, so an
# inversion started from that background would be fitting the wrong reference.
COHERENCE_GATE = 0.6

# Coarse-grid resolution before the Nelder-Mead refine. The eps axis is dense
# enough that the refine starts inside the correct lobe of the stack.
N_EPS_COARSE = 96
N_SIGMA_COARSE = 6


def green_2d(k, r) -> np.ndarray:
    """``G_2D = -(1j/4) H_0^(2)(k r)`` -- the field of a unit 2-D line source."""
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("green_2d: every source-receiver offset must be positive")
    return -0.25j * hankel2(0, np.asarray(k) * r)


@dataclass(frozen=True)
class Background:
    """The homogeneous reference an inversion is run against.

    ``eps_bg`` is relative permittivity, ``sigma_bg`` conductivity in S/m, and
    ``coherence`` the normalised phasor stack that pair achieved, in ``[0, 1]``.
    """

    eps_bg: float
    sigma_bg: float
    coherence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eps_bg", float(self.eps_bg))
        object.__setattr__(self, "sigma_bg", float(self.sigma_bg))
        object.__setattr__(self, "coherence", float(self.coherence))
        if self.eps_bg < EPS_MIN:
            raise ValueError(f"eps_bg must be >= {EPS_MIN}, got {self.eps_bg}")
        if self.sigma_bg < 0.0:
            raise ValueError(f"sigma_bg must be >= 0, got {self.sigma_bg}")

    def wavenumber(self, omega: float) -> complex:
        """``k_b`` of this background at angular frequency ``omega``."""
        return background_wavenumber(omega, self.eps_bg, self.sigma_bg)

    def require_coherent(self, threshold: float = COHERENCE_GATE) -> None:
        """Raise unless the stack cleared ``threshold``.

        The gate a run calls before inverting, so a bad reference stops the
        run instead of quietly shaping the answer.
        """
        if self.coherence < threshold:
            raise ValueError(
                f"background coherence {self.coherence:.3f} is below the gate "
                f"{threshold:.3f}: no single permittivity explains this panel, "
                "so a homogeneous reference would be wrong. Check the "
                "calibration, narrow the band, or split the array."
            )


def _values_and_freqs(u_cal, freqs):
    """Accept a ``Spectra`` or a bare array; return in-band values and freqs."""
    values = getattr(u_cal, "values", None)
    if values is not None:                       # a Spectra: honour its band mask
        band = u_cal.band
        values = np.asarray(values, dtype=np.complex128)[:, :, band]
        freqs = np.asarray(u_cal.freqs, dtype=float)[band]
    else:
        if freqs is None:
            raise ValueError("freqs is required when u_cal is a bare array")
        values = np.asarray(u_cal, dtype=np.complex128)
        freqs = np.asarray(freqs, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            f"u_cal must be [n_src, n_rx, n_freq], got shape {values.shape}")
    if freqs.shape != (values.shape[2],):
        raise ValueError(
            f"freqs must be [n_freq] = [{values.shape[2]}], got {freqs.shape}")
    keep = freqs > 0.0                           # DC carries no wavenumber
    if not np.any(keep):
        raise ValueError("no positive frequency to stack on")
    return values[:, :, keep], freqs[keep]


def offsets(src_pos, rx_pos) -> np.ndarray:
    """``[n_src, n_rx]`` source-receiver distances."""
    src_pos = np.asarray(src_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)
    if src_pos.ndim != 2 or src_pos.shape[1] != 2:
        raise ValueError(f"src_pos must be [n_src, 2], got {src_pos.shape}")
    if rx_pos.ndim != 2 or rx_pos.shape[1] != 2:
        raise ValueError(f"rx_pos must be [n_rx, 2], got {rx_pos.shape}")
    return np.hypot(src_pos[:, None, 0] - rx_pos[None, :, 0],
                    src_pos[:, None, 1] - rx_pos[None, :, 1])


def model_field(dists: np.ndarray, freqs: np.ndarray, eps: float,
                sigma: float) -> np.ndarray:
    """``G_2D`` for every pair and frequency in a homogeneous ``(eps, sigma)``."""
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    k = np.array([background_wavenumber(float(w), eps, sigma) for w in omega])
    return green_2d(k[None, None, :], np.asarray(dists, float)[:, :, None])


def coherence(values: np.ndarray, dists: np.ndarray, freqs: np.ndarray,
              eps: float, sigma: float) -> float:
    """Normalised phasor stack of ``values`` against ``G_2D(eps, sigma)``, in ``[0, 1]``.

    Pairs stack coherently within a frequency and the resulting magnitudes sum
    across frequencies, so an unknown complex scale per frequency -- the source
    wavelet, the instrument response -- leaves the score untouched and only the
    pair-to-pair phase pattern is scored.
    """
    if eps < EPS_MIN:
        raise ValueError(f"eps must be >= {EPS_MIN}, got {eps}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    values = np.asarray(values, dtype=np.complex128)
    model = model_field(dists, freqs, eps, sigma)

    num = np.abs(np.sum(values * np.conj(model), axis=(0, 1)))          # [n_freq]
    den = np.sqrt(np.sum(np.abs(values) ** 2, axis=(0, 1))
                  * np.sum(np.abs(model) ** 2, axis=(0, 1)))
    total = float(np.sum(den))
    if total <= 0.0:
        return 0.0
    return float(np.sum(num) / total)


def estimate_background(u_cal, src_pos, rx_pos, freqs=None,
                        eps_range: tuple[float, float] = (EPS_MIN, EPS_WATER_MAX),
                        sigma_range: tuple[float, float] = (0.0, 0.5),
                        n_eps: int = N_EPS_COARSE,
                        n_sigma: int = N_SIGMA_COARSE) -> Background:
    """Best homogeneous ``(eps, sigma)`` for ``u_cal``, by maximum coherence.

    ``u_cal`` is a ``Spectra`` -- whose band mask is honoured and whose
    ``freqs`` are used -- or a ``[n_src, n_rx, n_freq]`` array, in which case
    ``freqs`` must be given. Positions are ``[n_src, 2]`` and ``[n_rx, 2]``.

    A coarse grid over both parameters locates the lobe, then Nelder-Mead
    refines inside the bounds. The reported ``coherence`` is the score at the
    reported pair, ready for ``Background.require_coherent``.
    """
    values, freqs = _values_and_freqs(u_cal, freqs)
    dists = offsets(src_pos, rx_pos)
    if dists.shape != values.shape[:2]:
        raise ValueError(
            f"positions give {dists.shape} pairs, spectra have {values.shape[:2]}")

    eps_lo, eps_hi = float(eps_range[0]), float(eps_range[1])
    sig_lo, sig_hi = float(sigma_range[0]), float(sigma_range[1])
    if not eps_hi > eps_lo:
        raise ValueError(f"eps_range must increase, got {eps_range}")
    if eps_lo < EPS_MIN:
        raise ValueError(f"eps_range must start at >= {EPS_MIN}, got {eps_range}")
    if not sig_hi >= sig_lo >= 0.0:
        raise ValueError(
            f"sigma_range must be non-negative and increase, got {sigma_range}")

    eps_axis = np.linspace(eps_lo, eps_hi, max(int(n_eps), 2))
    sig_axis = (np.array([sig_lo]) if sig_hi == sig_lo
                else np.linspace(sig_lo, sig_hi, max(int(n_sigma), 2)))

    best_score, best_eps, best_sigma = -1.0, float(eps_axis[0]), float(sig_axis[0])
    for eps in eps_axis:
        for sig in sig_axis:
            score = coherence(values, dists, freqs, float(eps), float(sig))
            if score > best_score:
                best_score, best_eps, best_sigma = score, float(eps), float(sig)

    def negative(p):
        eps = float(np.clip(p[0], eps_lo, eps_hi))
        sig = float(np.clip(p[1], sig_lo, sig_hi))
        return -coherence(values, dists, freqs, eps, sig)

    result = minimize(negative, x0=[best_eps, best_sigma], method="Nelder-Mead",
                      bounds=[(eps_lo, eps_hi), (sig_lo, sig_hi)],
                      options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 2000})
    eps_ref = float(np.clip(result.x[0], eps_lo, eps_hi))
    sig_ref = float(np.clip(result.x[1], sig_lo, sig_hi))
    score_ref = coherence(values, dists, freqs, eps_ref, sig_ref)
    if score_ref < best_score:                   # the refine never makes it worse
        eps_ref, sig_ref, score_ref = best_eps, best_sigma, best_score
    return Background(eps_bg=eps_ref, sigma_bg=sig_ref, coherence=score_ref)


def phase_slope_permittivity(u_cal, src_pos, rx_pos, freq: float,
                             freqs=None) -> float:
    """Permittivity from the slope of the unwrapped phase against offset.

    Diagnostic. Pairs are sorted by offset, the phase at the bin nearest
    ``freq`` is unwrapped along that ordering, and ``eps = (beta c / w)**2``
    follows from the fitted ``beta``.

    One frequency and one straight-line fit make it a cheap independent
    cross-check on the stack. On ``test_B_array_soil_merged.out`` -- 81 pairs,
    truth ``eps = 6.0`` -- it returns 6.00 at 0.625 GHz and 6.00 at 0.812 GHz,
    then 2.32 at 1.0 GHz and 0.06 at 1.5 GHz.

    Risk: the unwrap needs ``beta * delta_d < pi`` between neighbouring pairs,
    which fails above roughly 0.9 GHz on that panel, and it fails silently --
    the fit still returns a smooth number. Never gate a run on it.
    """
    values, all_freqs = _values_and_freqs(u_cal, freqs)
    dists = offsets(src_pos, rx_pos)
    if dists.shape != values.shape[:2]:
        raise ValueError(
            f"positions give {dists.shape} pairs, spectra have {values.shape[:2]}")

    index = int(np.argmin(np.abs(all_freqs - float(freq))))
    flat_d = dists.ravel()
    order = np.argsort(flat_d)
    d_sorted = flat_d[order]
    phase = np.unwrap(np.angle(values[:, :, index].ravel()[order]))

    design = np.column_stack([d_sorted, np.ones_like(d_sorted)])
    slope = float(np.linalg.lstsq(design, phase, rcond=None)[0][0])
    beta = -slope                                  # outgoing e^{-j beta d}
    omega = 2.0 * np.pi * float(all_freqs[index])
    return float((beta * C_LIGHT / omega) ** 2)


__all__ = ["COHERENCE_GATE", "N_EPS_COARSE", "N_SIGMA_COARSE", "Background",
           "coherence", "estimate_background", "green_2d", "model_field",
           "offsets", "phase_slope_permittivity"]
