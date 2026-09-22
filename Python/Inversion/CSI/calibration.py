"""Measured spectra in, spectra in operator units out.

Two calibrators, one output contract: each takes a ``Spectra`` (or a bare
``[n_src, n_rx, n_freq]`` array with ``freqs``) and returns a ``Spectra`` whose
values are directly comparable with what a ``GreenOperator`` computes.

``calibrate_wavelet`` estimates the complex scale ``A(f)`` that best maps a
modelled background field onto the measurement and divides it out. This is
standard FWI source-wavelet estimation, and it is the mandatory path: it needs
only the data and a background model, both of which exist on every survey. It
also absorbs the absolute amplitude of a gprMax ``#hertzian_dipole``, which in
a 2-D TMz scene radiates as a line source of unstated strength.

``calibrate_air`` divides by a free-space reference and multiplies by the
analytic 2-D Green's function. The air ratio is dimensionless -- instrument
response, antenna pattern and 3-D spreading all divide out -- so multiplying by
the *2-D* Green's function performs the 3-D-to-2-D transformation exactly, per
pair, with no Bleistein filter and no far-field approximation.

Risk: that exactness covers the direct path only. Out-of-plane scattering in a
real 3-D earth has no 2-D counterpart, so whatever the air reference cannot
see stays uncorrected in the calibrated data.

Conventions
-----------
``e^{+j w t}``, outgoing ``e^{-j k r}``, ``G_2D = -(1j/4) H_0^(2)(k_b r)``.
Spectra are one-sided; fields ``complex128``, real quantities ``float64``.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import c as C_LIGHT

from .background import green_2d, offsets
from .spectra import Spectra

# How A(f) is shared across the array.
PER_MODES = ("global", "source", "pair")


def _unpack(data, freqs=None, label: str = "data"):
    """Return ``(values, freqs, band, component)`` from a ``Spectra`` or an array."""
    if isinstance(data, Spectra):
        return data.values, data.freqs, data.band, data.component
    values = np.asarray(data, dtype=np.complex128)
    if values.ndim != 3:
        raise ValueError(
            f"{label} must be [n_src, n_rx, n_freq], got shape {values.shape}")
    if freqs is None:
        raise ValueError(f"freqs is required when {label} is a bare array")
    freqs = np.asarray(freqs, dtype=float)
    if freqs.shape != (values.shape[2],):
        raise ValueError(
            f"freqs must be [n_freq] = [{values.shape[2]}], got {freqs.shape}")
    return values, freqs, None, None


def _axes_for(per: str) -> tuple[int, ...]:
    """Axes summed over when forming ``A``; the axes ``A`` is constant along."""
    if per == "global":
        return (0, 1)
    if per == "source":
        return (1,)
    if per == "pair":
        return ()
    raise ValueError(f"per must be one of {PER_MODES}, got {per!r}")


def wavelet_estimate(meas, bg, per: str = "global", freqs=None) -> np.ndarray:
    """Least-squares complex scale ``A`` mapping ``bg`` onto ``meas``.

    ``A = sum(u_meas * conj(u_bg)) / sum(|u_bg|**2)`` summed over the axes
    ``per`` shares, giving ``[n_freq]`` for ``'global'``, ``[n_src, 1, n_freq]``
    for ``'source'`` and ``[n_src, n_rx, n_freq]`` for ``'pair'``. It is the
    exact minimiser of ``||u_meas - A u_bg||**2``, so the round trip through
    ``calibrate_wavelet`` is exact when the data really is a scaled background.
    """
    axes = _axes_for(per)
    u_meas, meas_freqs, _, _ = _unpack(meas, freqs, "meas")
    u_bg, bg_freqs, _, _ = _unpack(bg, freqs, "bg")
    if u_meas.shape != u_bg.shape:
        raise ValueError(
            f"meas is {u_meas.shape}, bg is {u_bg.shape}; they must match")
    if not np.allclose(meas_freqs, bg_freqs):
        raise ValueError("meas and bg are on different frequency grids")

    numerator = np.sum(u_meas * np.conj(u_bg), axis=axes, keepdims=bool(axes))
    denominator = np.sum(np.abs(u_bg) ** 2, axis=axes, keepdims=bool(axes))
    if np.any(denominator == 0.0):
        raise ValueError(
            "wavelet_estimate: the background model is identically zero for at "
            "least one frequency, so no scale is defined there")
    scale = numerator / denominator
    return np.squeeze(scale, axis=axes) if per == "global" else scale


def calibrate_wavelet(meas, bg, per: str = "global", freqs=None) -> Spectra:
    """Divide out the estimated source wavelet: ``u_cal = u_meas / A``.

    ``per`` selects how much freedom ``A`` gets: ``'global'`` one scale per
    frequency for the whole array, ``'source'`` one per source, ``'pair'`` one
    per source-receiver pair. ``'global'`` is the honest default -- it fits a
    transmitter property with the whole array, so a per-pair anomaly stays in
    the data where the inversion can explain it.
    """
    scale = wavelet_estimate(meas, bg, per, freqs)
    u_meas, meas_freqs, band, component = _unpack(meas, freqs, "meas")
    values = u_meas / (scale[None, None, :] if per == "global" else scale)
    return Spectra(values=values, freqs=meas_freqs, band=band, component=component)


def calibrate_air(meas, air, src_pos, rx_pos, k0=None, freqs=None) -> Spectra:
    """Ratio against a free-space reference, restored to 2-D: ``(u_meas / u_air) * G_2D(k0, d)``.

    ``k0`` is the free-space wavenumber, given as a ``[n_freq]`` array, a
    callable of frequency, or left as ``None`` for the vacuum ``2 pi f / c``.
    ``src_pos [n_src, 2]`` and ``rx_pos [n_rx, 2]`` set the offset ``d`` of
    every pair.
    """
    u_meas, meas_freqs, band, component = _unpack(meas, freqs, "meas")
    u_air, air_freqs, _, _ = _unpack(air, freqs, "air")
    if u_meas.shape != u_air.shape:
        raise ValueError(
            f"meas is {u_meas.shape}, air is {u_air.shape}; they must match")
    if not np.allclose(meas_freqs, air_freqs):
        raise ValueError("meas and air are on different frequency grids")
    if np.any(u_air == 0.0):
        raise ValueError(
            "calibrate_air: the air reference has an exact zero, so the ratio "
            "is undefined there; mask the band before calibrating")

    dists = offsets(src_pos, rx_pos)
    if dists.shape != u_meas.shape[:2]:
        raise ValueError(
            f"positions give {dists.shape} pairs, spectra have {u_meas.shape[:2]}")

    if k0 is None:
        k_free = 2.0 * np.pi * meas_freqs / C_LIGHT
    elif callable(k0):
        k_free = np.asarray([k0(f) for f in meas_freqs])
    else:
        k_free = np.asarray(k0)
        if k_free.shape != meas_freqs.shape:
            raise ValueError(
                f"k0 must be [n_freq] = [{meas_freqs.size}], got {k_free.shape}")
    if np.any(np.asarray(k_free) == 0.0):
        raise ValueError("calibrate_air: k0 must be non-zero on every frequency")

    values = (u_meas / u_air) * green_2d(k_free[None, None, :], dists[:, :, None])
    return Spectra(values=values, freqs=meas_freqs, band=band, component=component)


__all__ = ["PER_MODES", "calibrate_air", "calibrate_wavelet", "wavelet_estimate"]
