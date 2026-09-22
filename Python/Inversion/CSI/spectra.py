"""Time traces in, one-sided complex spectra out.

``spectra_from_traces`` takes real traces ``[n_src, n_rx, n_time]`` -- or a
mapping of field component to such an array -- plus the sample interval, and
returns a ``Spectra``: ``[n_src, n_rx, n_freq]`` complex128 on
``np.fft.rfftfreq`` bins, carrying a boolean band mask that marks the bins the
source actually excited.

The co-polarised component is picked by energy rather than configured, so a
y-polarised model cannot be silently read on ``Ez``.

Risk: the band mask is a level threshold on the measured magnitude, so a strong
coherent artefact inside the record -- a wrap-around reflection, a DC offset --
reads as signal. It marks what stands above the noise floor, not what is
physical.

Conventions
-----------
Time convention ``e^{+j w t}``, matching ``np.fft``. Spectra are one-sided
(``np.fft.rfft`` / ``np.fft.rfftfreq``) and ``complex128``; frequencies are
``float64``. Amplitudes are the unnormalised ``np.fft.rfft``, so a scale factor
common to a measurement and its reference cancels in the ratio.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# Level, relative to the peak of the band profile, below which a frequency bin
# is treated as unexcited. Matches ``Tomograph.functions._bandwidth_mask``.
DEFAULT_BAND_DB = -40.0


@dataclass(frozen=True)
class Spectra:
    """One-sided spectra of one field component, with the usable band marked.

    ``values`` is ``[n_src, n_rx, n_freq]`` complex128 on ``freqs`` (float64,
    ascending, ``n_freq`` bins). ``band`` is a ``[n_freq]`` boolean; ``True``
    where the bin carries signal. ``component`` records which field component
    the traces came from, or ``None`` when the caller supplied spectra directly.
    """

    values: np.ndarray
    freqs: np.ndarray
    band: np.ndarray | None = None
    component: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.complex128)
        if values.ndim != 3:
            raise ValueError(
                f"values must be [n_src, n_rx, n_freq], got shape {values.shape}")
        freqs = np.asarray(self.freqs, dtype=float)
        if freqs.shape != (values.shape[2],):
            raise ValueError(
                f"freqs must be [n_freq] = [{values.shape[2]}], got {freqs.shape}")
        band = (np.ones(freqs.shape, dtype=bool) if self.band is None
                else np.asarray(self.band, dtype=bool))
        if band.shape != freqs.shape:
            raise ValueError(
                f"band must be [n_freq] = [{freqs.size}], got {band.shape}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "band", band)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape

    @property
    def n_src(self) -> int:
        return self.values.shape[0]

    @property
    def n_rx(self) -> int:
        return self.values.shape[1]

    @property
    def n_freq(self) -> int:
        return self.values.shape[2]

    @property
    def band_freqs(self) -> np.ndarray:
        """The frequencies the mask keeps."""
        return self.freqs[self.band]

    @property
    def band_edges(self) -> tuple[float, float]:
        """``(f_lo, f_hi)`` of the masked band; ``(nan, nan)`` if nothing passes."""
        kept = self.band_freqs
        if kept.size == 0:
            return (float("nan"), float("nan"))
        return (float(kept[0]), float(kept[-1]))

    def at(self, index) -> "Spectra":
        """Subset the frequency axis, keeping the band flags of the bins kept.

        ``index`` is anything that indexes a 1-D array -- a boolean mask, a list
        of bin numbers, a slice. Frequency continuation climbs rungs this way.
        """
        return Spectra(values=self.values[:, :, index], freqs=self.freqs[index],
                       band=self.band[index], component=self.component)


def band_profile(values: np.ndarray) -> np.ndarray:
    """``[n_freq]`` magnitude the band mask is measured on: RMS over src and rx.

    One profile for the whole array rather than one per pair, so every pair
    shares a band and the inversion never has a ragged frequency axis.
    """
    values = np.asarray(values)
    return np.sqrt(np.mean(np.abs(values) ** 2, axis=(0, 1)))


def band_mask(profile: np.ndarray, threshold_db: float = DEFAULT_BAND_DB) -> np.ndarray:
    """Bins within ``threshold_db`` of the profile peak.

    Same rule as ``Tomograph.functions._bandwidth_mask``, restated here so the
    inversion core stays on numpy alone; a test asserts the two agree.
    """
    mag = np.abs(np.asarray(profile))
    if mag.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.max(mag))
    if peak <= 0.0:
        return np.zeros(mag.shape, dtype=bool)
    return 20 * np.log10(np.maximum(mag, 1e-30)) - 20 * np.log10(peak) > threshold_db


def select_component(traces: Mapping[str, np.ndarray],
                     component: str | None = None) -> tuple[str, float]:
    """Pick the co-polarised component by RMS, returning ``(name, margin)``.

    ``margin`` is the winner's RMS over the runner-up's, ``inf`` when every
    other candidate is identically zero -- which is what a clean 2-D TM scene
    gives. Candidates must be in the same units for the comparison to mean
    anything, so pass E-components together, never E mixed with H.

    ``component``, when given, is returned as-is after checking it is present;
    the margin is still measured against the strongest rival.
    """
    if not traces:
        raise ValueError("select_component: no candidate components")
    rms = {name: float(np.sqrt(np.mean(np.asarray(arr, dtype=float) ** 2)))
           for name, arr in traces.items()}
    ranked = sorted(rms, key=lambda n: rms[n], reverse=True)

    if component is None:
        chosen = ranked[0]
        if rms[chosen] <= 0.0:
            raise ValueError(
                "select_component: every candidate is identically zero "
                f"({sorted(traces)}); nothing was recorded")
    else:
        if component not in traces:
            raise KeyError(
                f"select_component: component {component!r} not present; "
                f"available: {sorted(traces)}")
        chosen = component

    rivals = [rms[n] for n in ranked if n != chosen]
    best_rival = max(rivals) if rivals else 0.0
    margin = float("inf") if best_rival <= 0.0 else rms[chosen] / best_rival
    return chosen, margin


def spectra_from_traces(traces, dt: float, component: str | None = None,
                        threshold_db: float = DEFAULT_BAND_DB) -> Spectra:
    """One-sided spectra of ``[n_src, n_rx, n_time]`` traces sampled every ``dt``.

    ``traces`` is either that array or a ``{component: array}`` mapping, in
    which case the component is chosen by ``select_component`` unless
    ``component`` names one.
    """
    if isinstance(traces, Mapping):
        name, _ = select_component(traces, component)
        arr = np.asarray(traces[name], dtype=float)
    else:
        name = component
        arr = np.asarray(traces, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"traces must be [n_src, n_rx, n_time], got shape {arr.shape}")
    if not dt > 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    values = np.fft.rfft(arr, axis=-1)
    freqs = np.fft.rfftfreq(arr.shape[-1], d=dt)
    return Spectra(values=values, freqs=freqs,
                   band=band_mask(band_profile(values), threshold_db),
                   component=name)


__all__ = ["DEFAULT_BAND_DB", "Spectra", "band_mask", "band_profile",
           "select_component", "spectra_from_traces"]
