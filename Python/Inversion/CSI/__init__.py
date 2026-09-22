"""Frequency-domain Contrast Source Inversion of GPR crosshole data for eps'.

Input is complex S21 ``[n_src, n_rx, n_freq]`` with ``freqs``, ``src_pos``
``[n_src, 2]`` and ``rx_pos [n_rx, 2]``. Output is a permittivity map on an
``InversionGrid``.

CSI solves for two unknowns at once -- the contrast ``chi`` and the contrast
source ``w = chi * u`` -- and alternates closed-form updates between them. The
total field ``u`` is never recomputed by a forward solve inside the loop, so
the scattering is carried exactly rather than linearized.

No linearization means high contrast is survivable. Water in soil is
``chi = 80/6 - 1 ~ 12``, where Born and Rytov both break down.

Risk: the cost is non-convex, and convexity comes from starting at long
wavelength. Frequency continuation climbs rungs defined by ``L/lambda``, so a
dataset whose band lacks the low rungs must cold-start at high contrast. That
is the dominant failure mode.

Conventions
-----------
Time convention ``e^{+j w t}``, matching ``np.fft``::

    eps_c(r, w) = eps'(r) - 1j * sigma(r) / (w * eps_0)
    k**2        = w**2 * mu_0 * eps_0 * eps_c
    outgoing    = e^{-j k r}
    G_2D        = -(1j / 4) * H_0^(2)(k_b * |r - r'|)      (TM, free space)

Pixels flatten row-major, ``flat = iy * nx + ix``, images ``reshape(ny, nx)``.
Spectra are one-sided (``np.fft.rfftfreq``). Fields are ``complex128``, real
model unknowns ``float64``.
"""
from __future__ import annotations

# Bounds the permittivity search; water at room temperature.
EPS_WATER_MAX = 90.0
EPS_MIN = 1.0

__all__ = ["EPS_WATER_MAX", "EPS_MIN"]
