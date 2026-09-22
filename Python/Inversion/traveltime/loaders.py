"""Load gprMax crosshole .out files into a geometry + traces container.

Reuses the existing Tomograph HDF5 loaders for individual A-scan files (which
carry source/receiver positions). Merged B-scan files drop positions and store
each receiver as [n_time, n_src]; for those the caller supplies an Acquisition
geometry (reconstruct with ``bscan_geometry`` from the .in directives).

Requires ``REPO_ROOT/Python`` and ``REPO_ROOT/gprMax`` on sys.path (see conftest).
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from Inversion.traveltime.geometry import select_plane_axes
from Tomograph.functions import _natural_sort_key, get_src, get_rxs, get_field_data


@dataclass
class Acquisition:
    sources: np.ndarray      # [n_src, 2] in the tomographic plane
    receivers: np.ndarray    # [n_rx, 2]
    plane_axes: tuple[int, int] = (0, 1)


@dataclass
class AcquisitionData:
    acq: Acquisition
    traces: np.ndarray       # [n_src, n_rx, n_time]
    dt: float
    component: str


def bscan_geometry(src_start, src_step, n_src, rx_start, rx_step, n_rx,
                   plane_axes=(0, 1)) -> Acquisition:
    """Reconstruct a stepped source line + fixed receiver line (2-D plane)."""
    src_start = np.asarray(src_start, float)
    src_step = np.asarray(src_step, float)
    rx_start = np.asarray(rx_start, float)
    rx_step = np.asarray(rx_step, float)
    sources = src_start[None, :] + np.arange(n_src)[:, None] * src_step[None, :]
    receivers = rx_start[None, :] + np.arange(n_rx)[:, None] * rx_step[None, :]
    return Acquisition(sources=sources[:, :2], receivers=receivers[:, :2],
                       plane_axes=plane_axes)


def load_merged_bscan(path: str, acq: Acquisition, component: str = "Ez") -> AcquisitionData:
    """Load a merged B-scan: rxs/<rx>/<comp> is [n_time, n_src]."""
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        rx_names = sorted(f["rxs"].keys(), key=_natural_sort_key)
        cols = [f["rxs"][name][component][()] for name in rx_names]  # each [n_time, n_src]
    stack = np.stack(cols, axis=0)                  # [n_rx, n_time, n_src]
    traces = np.transpose(stack, (2, 0, 1))         # [n_src, n_rx, n_time]
    return AcquisitionData(acq=acq, traces=traces, dt=dt, component=component)


def load_ascan_set(paths, component: str = "Ez", order=None) -> AcquisitionData:
    """Load a list of individual A-scan files (one source each) and stack them.

    Receiver positions/order come from the first file; ``order`` (list of rx
    names) pins the receiver axis across files if needed.
    """
    sources3, receivers3, dt = [], None, None
    rx_order = order
    traces = []
    for path in paths:
        with h5py.File(path, "r") as f:
            dt = float(f.attrs["dt"])
            srcs = get_src(f)
            rxs = get_rxs(f)
            fields = get_field_data(f)
        sources3.append([srcs[0].position.x, srcs[0].position.y, srcs[0].position.z])
        if rx_order is None:
            rx_order = list(rxs.keys())
        if receivers3 is None:
            receivers3 = [[rxs[n].position.x, rxs[n].position.y, rxs[n].position.z]
                          for n in rx_order]
        traces.append([getattr(fields[n], component) for n in rx_order])
    sources3 = np.asarray(sources3, float)
    receivers3 = np.asarray(receivers3, float)
    ax = select_plane_axes(sources3, receivers3)
    acq = Acquisition(sources=sources3[:, ax], receivers=receivers3[:, ax], plane_axes=ax)
    return AcquisitionData(acq=acq, traces=np.asarray(traces, float),
                           dt=dt, component=component)
