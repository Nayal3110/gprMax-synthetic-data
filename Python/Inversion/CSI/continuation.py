"""Frequency continuation: one CSI run per band group, climbing ``L/lambda`` rungs.

``frequency_ladder`` takes the panel length, the background permittivity and the
measured frequency axis with its band mask, and returns a ``Ladder``: the band
groups that survived, ascending, plus the rungs the band could not support.
``run_continuation`` runs ``run_csi`` once per group and carries ``xi`` forward.

Rungs sit at ``f_R = R * c / (L * sqrt(eps_bg))``, so one ladder places the same
``L/lambda`` sequence on a 0.5 m panel and on a 20 m one.

Risk: convexity comes from starting at long wavelength, so a band that begins
above the low rungs forces a cold start at high contrast. The dropped rungs are
logged as a warning and ride on the result -- a run that starts at
``L/lambda = 4`` is a different experiment from one that starts at 1, and
nothing downstream can tell the two apart unless the ladder says so.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import c as C_LIGHT

from . import EPS_MIN
from .contrast import ContrastModel
from .csi import CSIState, heldout_pairs, run_csi
from .operators import GreenOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandGroup:
    """The frequency bins one ``run_csi`` call sees, and the rung they stand for.

    ``indices`` point into the full frequency axis the ladder was built on;
    ``freqs`` are the bins themselves, ascending.
    """

    ratio: float
    f_target: float
    indices: np.ndarray
    freqs: np.ndarray

    @property
    def n_freq(self) -> int:
        return int(self.indices.size)


@dataclass(frozen=True)
class DroppedRung:
    """A rung the band cannot support, and the reason in words."""

    ratio: float
    f_target: float
    reason: str


@dataclass(frozen=True)
class Ladder:
    """Surviving band groups ascending, and every rung that was dropped."""

    groups: list[BandGroup]
    dropped: list[DroppedRung]
    panel_length: float
    eps_bg: float

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def ratios(self) -> tuple[float, ...]:
        """``L/lambda`` of the surviving rungs, ascending."""
        return tuple(g.ratio for g in self.groups)

    @property
    def dropped_ratios(self) -> tuple[float, ...]:
        """``L/lambda`` of the rungs the band could not support."""
        return tuple(d.ratio for d in self.dropped)

    @property
    def indices(self) -> np.ndarray:
        """Every frequency bin the ladder uses, ascending, without repeats."""
        if not self.groups:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate([g.indices for g in self.groups]))

    @property
    def starts_high(self) -> bool:
        """True when a rung below the first surviving one was dropped.

        The single diagnostic worth reading before the reconstruction: it says the
        run began at a contrast the cost is not convex in.
        """
        if not self.groups or not self.dropped:
            return False
        return min(self.dropped_ratios) < self.groups[0].ratio


def frequency_ladder(L: float, eps_bg: float, freqs, band_mask=None,
                     ratios: Sequence[float] = (1, 2, 4, 8, 16),
                     n_per_group: int = 3) -> Ladder:
    """Band groups at ``f_R = R * c / (L * sqrt(eps_bg))``, ascending.

    ``freqs`` is the full frequency axis and ``band_mask`` marks the bins that
    carry signal; a rung whose ``f_R`` falls outside the masked band is dropped
    and reported on ``Ladder.dropped`` rather than nudged inward, because a rung
    served by bins from the far side of the band is not that rung.

    Each surviving rung takes the ``n_per_group`` nearest masked bins in log
    frequency -- the rungs are geometric, so the distance that orders them is
    too. Bins are claimed lowest rung first and never shared, so a band too
    coarse to separate two rungs yields one group rather than two copies of it.
    """
    if not L > 0.0:
        raise ValueError(f"L must be positive, got {L}")
    if eps_bg < EPS_MIN:
        raise ValueError(f"eps_bg must be >= {EPS_MIN}, got {eps_bg}")
    if n_per_group < 1:
        raise ValueError(f"n_per_group must be >= 1, got {n_per_group}")

    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1:
        raise ValueError(f"freqs must be [n_freq], got shape {freqs.shape}")
    band = (np.ones(freqs.shape, dtype=bool) if band_mask is None
            else np.asarray(band_mask, dtype=bool))
    if band.shape != freqs.shape:
        raise ValueError(f"band_mask is {band.shape}, expected {freqs.shape}")

    # DC carries no rung and has no logarithm, so it never enters the ladder.
    usable = np.flatnonzero(band & (freqs > 0.0))
    f_lo = float(freqs[usable].min()) if usable.size else float("nan")
    f_hi = float(freqs[usable].max()) if usable.size else float("nan")
    log_f = np.log(np.where(freqs > 0.0, freqs, 1.0))

    groups: list[BandGroup] = []
    dropped: list[DroppedRung] = []
    claimed: set[int] = set()

    for ratio in sorted(float(r) for r in ratios):
        f_r = ratio * C_LIGHT / (L * math.sqrt(eps_bg))
        if usable.size == 0:
            reason = "no bin is in band"
        elif f_r < f_lo:
            reason = (f"{f_r / 1e6:.1f} MHz is below the band, which starts at "
                      f"{f_lo / 1e6:.1f} MHz")
        elif f_r > f_hi:
            reason = (f"{f_r / 1e6:.1f} MHz is above the band, which ends at "
                      f"{f_hi / 1e6:.1f} MHz")
        else:
            reason = ""

        free = np.array([i for i in usable if i not in claimed], dtype=int)
        if not reason and free.size == 0:
            reason = "every in-band bin is already claimed by a lower rung"
        if reason:
            dropped.append(DroppedRung(ratio=ratio, f_target=f_r, reason=reason))
            logger.warning("frequency_ladder: dropped rung L/lambda = %g -- %s",
                           ratio, reason)
            continue

        order = free[np.argsort(np.abs(log_f[free] - math.log(f_r)), kind="stable")]
        chosen = np.sort(order[:n_per_group])
        claimed.update(int(i) for i in chosen)
        groups.append(BandGroup(ratio=ratio, f_target=f_r, indices=chosen,
                                freqs=freqs[chosen]))

    ladder = Ladder(groups=groups, dropped=dropped, panel_length=float(L),
                    eps_bg=float(eps_bg))
    if ladder.starts_high:
        logger.warning(
            "frequency_ladder: the ladder starts at L/lambda = %g; the rungs below "
            "it are outside the band, so the inversion cold-starts at high "
            "contrast and the cost it starts on is not convex",
            ladder.groups[0].ratio)
    return ladder


@dataclass
class ContinuationResult:
    """One ``CSIState`` per band group, the ladder that produced them, and ``xi``."""

    ladder: Ladder
    states: list[CSIState] = field(default_factory=list)
    model: ContrastModel | None = None

    @property
    def xi(self) -> np.ndarray:
        return self.model.xi

    @property
    def zeta(self) -> np.ndarray | None:
        return self.model.zeta

    @property
    def eps(self) -> np.ndarray:
        """Absolute eps' per pixel; ``model.grid.reshape`` for an image."""
        return self.model.eps

    @property
    def dropped(self) -> list[DroppedRung]:
        return self.ladder.dropped

    @property
    def n_iter(self) -> int:
        """Outer iterations summed over the groups."""
        return sum(s.n_iter for s in self.states)


def run_continuation(ladder: Ladder, s21, make_operator: Callable[[float], GreenOperator],
                     eps_bg: float, sigma_bg: float = 0.0,
                     xi0: np.ndarray | None = None, zeta0: np.ndarray | None = None,
                     heldout_mask=None, heldout_frac: float = 0.15,
                     heldout_seed: int = 0, **csi_kwargs) -> ContinuationResult:
    """Run ``run_csi`` once per band group, ascending, warm-starting on ``xi``.

    ``s21`` is ``[n_src, n_rx, n_freq]`` on the frequency axis the ladder indexes,
    and ``make_operator(freq)`` builds the ``GreenOperator`` for one frequency.
    Remaining keywords go straight to ``run_csi``.

    Only ``xi`` and ``zeta`` cross a rung. ``w`` is the contrast source of one
    frequency and means nothing at the next, so every group rebuilds it from its
    own data, along with ``eta_D`` and the TV weights.

    The held-out pairs are drawn once and shared by every group: a withheld pair
    is a physical pair, and a pair the top rung never saw is the only honest test
    of the reconstruction the top rung produced.
    """
    if not ladder.groups:
        raise ValueError(
            "the ladder has no surviving rung; every one was dropped "
            f"({[(d.ratio, d.reason) for d in ladder.dropped]})")

    s21 = np.asarray(s21, dtype=np.complex128)
    if s21.ndim != 3:
        raise ValueError(f"s21 must be [n_src, n_rx, n_freq], got shape {s21.shape}")
    n_src, n_rx, n_freq = s21.shape
    top = int(ladder.indices.max())
    if top >= n_freq:
        raise ValueError(
            f"the ladder indexes bin {top} but s21 has {n_freq} frequency bins")

    mask = (heldout_pairs(n_src, n_rx, heldout_frac, heldout_seed)
            if heldout_mask is None else np.asarray(heldout_mask, dtype=bool))

    result = ContinuationResult(ladder=ladder)
    xi, zeta = xi0, zeta0
    for group in ladder.groups:
        ops = [make_operator(float(f)) for f in group.freqs]
        f_meas = [s21[:, :, int(i)] for i in group.indices]
        state = run_csi(ops, f_meas, eps_bg, sigma_bg=sigma_bg, xi0=xi, zeta0=zeta,
                        heldout_mask=mask, **csi_kwargs)
        logger.info("continuation: rung L/lambda = %g, %d bins, %d iterations, "
                    "F_S = %.3e, stop = %s", group.ratio, group.n_freq, state.n_iter,
                    state.f_s[-1] if state.f_s else float("nan"), state.stop_reason)
        xi, zeta = state.xi, state.zeta
        result.states.append(state)

    result.model = result.states[-1].model
    return result


__all__ = ["BandGroup", "ContinuationResult", "DroppedRung", "Ladder",
           "frequency_ladder", "run_continuation"]
