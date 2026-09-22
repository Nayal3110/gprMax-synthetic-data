"""gprMax ``.out`` files in, a ``CsiDataset`` out -- and the same door for bare arrays.

``from_out_files`` reads a merged B-scan or a set of per-run A-scans and returns
the measurement, an optional air reference, an optional simulated background
reference, and the acquisition geometry. ``from_arrays`` takes the same content
as plain arrays -- ``S21 [n_src, n_rx, n_freq]``, ``freqs``, ``src_pos``,
``rx_pos`` -- which is exactly what a real crosshole survey delivers.

This is the only module in the package that imports h5py, gprMax or Tomograph,
so everything downstream runs on numpy and scipy alone and both doors feed the
identical code path.

Risk: a merged B-scan carries no geometry whatsoever, so its positions are the
caller's word against the file's silence. Nothing here can check them, and a
transposed source line would invert to a plausible-looking wrong answer.

Requires ``REPO_ROOT/Python`` and ``REPO_ROOT/gprMax`` on sys.path (see conftest).
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import h5py
import numpy as np

from Inversion.traveltime.geometry import select_plane_axes
from Inversion.traveltime.loaders import Acquisition
from Tomograph.functions import (
    FieldTable, _natural_sort_key, get_field_data, get_rxs, get_src,
)

from .spectra import DEFAULT_BAND_DB, Spectra, band_mask, band_profile, spectra_from_traces

# Candidates for the co-polarised component. E only: the RMS comparison that
# chooses between them is meaningless across a unit change, and H trails E by
# the wave impedance.
E_COMPONENTS = ("Ex", "Ey", "Ez")

MERGED_NEEDS_ACQ = (
    "{path}: merged B-scan carries no geometry -- no 'srcs' group, no "
    "'dx_dy_dz' attribute, and empty receiver attrs -- so source and receiver "
    "positions cannot be read from the file. Pass acq=Acquisition(...), for "
    "example bscan_geometry((0.1, 0.1), (0.0, 0.1), 9, (0.9, 0.1), (0.0, 0.1), 9) "
    "from Inversion.traveltime.loaders, or load the per-run .out siblings, "
    "which do carry positions."
)


@dataclass(frozen=True)
class CsiDataset:
    """Everything an inversion is handed: spectra plus where they were measured.

    ``air`` (instrument reference) and ``background`` (simulated homogeneous
    reference, synthetic quality control only) are ``None`` on a real survey and
    the pipeline must run without them.
    """

    meas: Spectra
    air: Spectra | None = None
    background: Spectra | None = None
    acq: Acquisition | None = None      # required; defaulted only to keep the field order

    def __post_init__(self) -> None:
        if self.acq is None:
            raise ValueError("CsiDataset requires an Acquisition")
        n_src, n_rx = self.meas.n_src, self.meas.n_rx
        if self.acq.sources.shape != (n_src, 2):
            raise ValueError(
                f"acq.sources is {self.acq.sources.shape}, spectra have "
                f"n_src = {n_src}; expected [{n_src}, 2]")
        if self.acq.receivers.shape != (n_rx, 2):
            raise ValueError(
                f"acq.receivers is {self.acq.receivers.shape}, spectra have "
                f"n_rx = {n_rx}; expected [{n_rx}, 2]")
        for label in ("air", "background"):
            ref = getattr(self, label)
            if ref is None:
                continue
            if ref.shape != self.meas.shape:
                raise ValueError(
                    f"{label} reference is {ref.shape}, measurement is "
                    f"{self.meas.shape}; references must match the measurement")
            if not np.array_equal(ref.freqs, self.meas.freqs):
                raise ValueError(
                    f"{label} reference is on a different frequency grid than "
                    "the measurement")

    @property
    def s21(self) -> np.ndarray:
        """``[n_src, n_rx, n_freq]`` complex128 measurement."""
        return self.meas.values

    @property
    def freqs(self) -> np.ndarray:
        return self.meas.freqs

    @property
    def band(self) -> np.ndarray:
        return self.meas.band

    @property
    def src_pos(self) -> np.ndarray:
        return self.acq.sources

    @property
    def rx_pos(self) -> np.ndarray:
        return self.acq.receivers


def _is_merged(f: h5py.File) -> bool:
    """A merged B-scan has no ``srcs`` group and stores ``[n_time, n_src]``."""
    if "srcs" not in f:
        return True
    rxs = f["rxs"]
    first = sorted(rxs.keys(), key=_natural_sort_key)[0]
    for comp in E_COMPONENTS:
        if comp in rxs[first]:
            return rxs[first][comp].ndim == 2
    return False


def _read_merged(path, acq: Acquisition | None) -> tuple[dict, float, Acquisition]:
    with h5py.File(path, "r") as f:
        if acq is None:
            raise ValueError(MERGED_NEEDS_ACQ.format(path=path))
        dt = float(f.attrs["dt"])
        names = sorted(f["rxs"].keys(), key=_natural_sort_key)
        traces = {}
        for comp in E_COMPONENTS:
            if all(comp in f["rxs"][n] for n in names):
                # each [n_time, n_src] -> stack [n_rx, n_time, n_src]
                stack = np.stack([f["rxs"][n][comp][()] for n in names])
                traces[comp] = np.transpose(stack, (2, 0, 1)).astype(float)
    if not traces:
        raise ValueError(f"{path}: no E-field component present on every receiver")
    return traces, dt, acq


def _read_ascan_set(paths: Sequence, acq: Acquisition | None) -> tuple[dict, float, Acquisition]:
    """One file per source; positions come from the files unless ``acq`` overrides."""
    per_source: list[dict] = []
    sources3: list[list[float]] = []
    receivers3 = None
    rx_order = None
    dt = None
    for path in paths:
        with h5py.File(path, "r") as f:
            if _is_merged(f):
                raise ValueError(
                    f"{path}: this is a merged B-scan, not a per-run A-scan; "
                    "pass it on its own rather than in a list")
            dt = float(f.attrs["dt"])
            srcs = get_src(f)
            rxs = get_rxs(f)
            fields = get_field_data(f)
        if rx_order is None:
            rx_order = list(rxs.keys())
            receivers3 = [[rxs[n].position.x, rxs[n].position.y, rxs[n].position.z]
                          for n in rx_order]
        sources3.append([srcs[0].position.x, srcs[0].position.y, srcs[0].position.z])
        table = FieldTable.from_dict(fields, order=rx_order)
        per_source.append({c: getattr(table, c) for c in E_COMPONENTS
                           if getattr(table, c) is not None})

    shared = set.intersection(*(set(d) for d in per_source))
    if not shared:
        raise ValueError("no E-field component present in every A-scan file")
    traces = {c: np.stack([d[c] for d in per_source]).astype(float) for c in sorted(shared)}

    if acq is None:
        sources3 = np.asarray(sources3, float)
        receivers3 = np.asarray(receivers3, float)
        axes = select_plane_axes(sources3, receivers3)
        acq = Acquisition(sources=sources3[:, axes], receivers=receivers3[:, axes],
                          plane_axes=axes)
    return traces, dt, acq


def _read_traces(source, acq: Acquisition | None) -> tuple[dict, float, Acquisition]:
    """Dispatch on shape of the argument: one path is merged-or-A-scan, many are A-scans."""
    if isinstance(source, (str, bytes, os.PathLike)):
        with h5py.File(source, "r") as f:
            merged = _is_merged(f)
        return (_read_merged(source, acq) if merged
                else _read_ascan_set([source], acq))
    paths = list(source)
    if not paths:
        raise ValueError("no .out files given")
    return _read_ascan_set(paths, acq)


def from_out_files(meas, air=None, background=None, acq: Acquisition | None = None,
                   component: str | None = None,
                   threshold_db: float = DEFAULT_BAND_DB) -> CsiDataset:
    """Read gprMax ``.out`` files into a ``CsiDataset``.

    Each of ``meas``, ``air`` and ``background`` is one path (a merged B-scan or
    a single A-scan) or a sequence of per-run A-scan paths, one per source in
    order. ``acq`` is required for a merged file and optional otherwise, where
    it overrides the geometry read from the files.

    The component is chosen once, on the measurement, and the references are
    read on that same component so the three are comparable.
    """
    traces, dt, acq = _read_traces(meas, acq)
    meas_spectra = spectra_from_traces(traces, dt, component, threshold_db)
    chosen = meas_spectra.component

    refs = {}
    for label, ref in (("air", air), ("background", background)):
        if ref is None:
            refs[label] = None
            continue
        ref_traces, ref_dt, _ = _read_traces(ref, acq)
        refs[label] = spectra_from_traces(ref_traces, ref_dt, chosen, threshold_db)

    return CsiDataset(meas=meas_spectra, air=refs["air"],
                      background=refs["background"], acq=acq)


def _as_spectra(values, freqs, component, threshold_db) -> Spectra:
    if isinstance(values, Spectra):
        return values
    values = np.asarray(values, dtype=np.complex128)
    return Spectra(values=values, freqs=freqs,
                   band=band_mask(band_profile(values), threshold_db),
                   component=component)


def from_arrays(s21, freqs, src_pos, rx_pos, air=None, background=None,
                component: str | None = None, plane_axes: tuple[int, int] = (0, 1),
                threshold_db: float = DEFAULT_BAND_DB) -> CsiDataset:
    """Build a ``CsiDataset`` from the survey tuple, with no file involved.

    ``s21`` is ``[n_src, n_rx, n_freq]`` complex, ``freqs`` ``[n_freq]``,
    ``src_pos`` ``[n_src, 2]`` and ``rx_pos`` ``[n_rx, 2]``. ``air`` and
    ``background`` are arrays of the same shape, or ``Spectra``, or ``None``.

    This is the door real measured data comes through, and it reaches the same
    ``CsiDataset`` a gprMax read of the same scene produces.
    """
    src_pos = np.asarray(src_pos, dtype=float)
    rx_pos = np.asarray(rx_pos, dtype=float)
    acq = Acquisition(sources=src_pos, receivers=rx_pos, plane_axes=tuple(plane_axes))
    return CsiDataset(
        meas=_as_spectra(s21, freqs, component, threshold_db),
        air=None if air is None else _as_spectra(air, freqs, component, threshold_db),
        background=(None if background is None
                    else _as_spectra(background, freqs, component, threshold_db)),
        acq=acq,
    )


__all__ = ["CsiDataset", "E_COMPONENTS", "MERGED_NEEDS_ACQ", "from_arrays",
           "from_out_files"]
