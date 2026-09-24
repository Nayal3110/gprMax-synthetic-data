# gprMax synthetic data — crosshole GPR permittivity inversion

Research tooling for **crosshole ground-penetrating radar**: generate synthetic
survey data with an FDTD simulator, then invert it into a map of the electrical
permittivity of the material between two boreholes.

Built during a research placement at **SUPSI — Istituto sistemi e elettronica
applicata (ISEA)**, Lugano, for a measurement setup acquiring S21 with an
antenna array over a VNA at 50–450 MHz in 50 kHz steps.

The simulator is a means, not the point. The target is a reconstruction that
survives **real measured data**, so every design decision here is made against
that constraint rather than against the synthetic case.

## Two inversion paths, kept as peers

**Travel-time (`Python/Inversion/traveltime/`).** Pick first breaks, build a
straight-ray Siddon system matrix, invert with SIRT/SART.

- `picking.py` — envelope → SNR gate → threshold → AIC refinement
- `geometry.py` — `PixelGrid`, Siddon cell lengths, sparse system matrix
- `sirt.py` — SIRT/SART iterations, slowness ↔ permittivity conversion
- `tomography.py` — `calibrate_t0`, `run_tomography`, tomogram plotting
- `loaders.py` — acquisition geometry and B-scan/A-scan loading

Its structural limit is documented rather than hidden: Fermat's principle routes
the first arrival *around* a slow body, so a water-filled cavity barely
perturbs the picked time, and at 50–450 MHz over panels of 0.5–20 m the first
Fresnel zone is comparable to the whole panel — rays are a useful fiction, not
a physical description.

**Contrast Source Inversion (`Python/Inversion/CSI/`).** Frequency-domain
inversion of complex S21 that never linearizes the scattering, so it survives
the contrast that breaks Born and Rytov: water in soil is `chi = 80/6 − 1 ≈ 12`.
CSI solves for the contrast and the contrast source at once, alternating
closed-form updates, with multiplicative TV regularization (no lambda to tune)
and frequency continuation over rungs defined by `L/lambda` rather than fixed
MHz.

Ten modules: `grid`, `io`, `background`, `contrast`, `csi`, `continuation`,
`regularization`, `spectra`, `calibration`, `operators`. Two operator backends
— finite-difference for production, integral-equation for verification — sit
behind one `GreenOperator` protocol, so the production path is checked against
an independent formulation rather than against itself.

## The input contract

An inversion is given exactly `S21 [n_src, n_rx, n_freq] complex128`, `freqs`,
`src_pos [n_src, 2]` and `rx_pos [n_rx, 2]`. No domain size, no material list,
no target geometry, no simulator input file.

That is what a real survey produces, and the synthetic case is deliberately not
allowed to be easier than the real one. Three rules follow, and are enforced by
tests:

1. One module knows the simulator exists (`CSI/io.py`); `from_arrays(...)`
   takes the tuple directly and exercises the identical downstream path.
2. Anything that cannot exist on a real survey defaults to `None` and the
   pipeline produces a result without it.
3. Anything derived from the ground-truth model is scoring-only, and no
   inversion module may import it.

The only quality signals allowed to gate a run are the ones available in the
field: background-estimate coherence, final data misfit, and a held-out
source–receiver-pair residual.

## Conventions that must not drift

- Time convention `e^{+jwt}`, matching `np.fft`
- `eps_c(r,w) = eps'(r) − j·sigma(r)/(w·eps_0)`; outgoing wave `e^{−jkr}`;
  2-D TM free-space Green's function `G = −(j/4)·H_0^(2)(k_b·|r−r'|)`
- Pixels flatten row-major: `flat = iy*nx + ix`, images `reshape(ny, nx)`
- **Invert in slowness**, never velocity or permittivity — travel time is
  linear in slowness, which is what makes the SIRT/SART iterations valid.
  Convert only at the boundaries: `eps_r = (c·m)²`
- Spectra are one-sided (`rfftfreq`); fields `complex128`, model unknowns
  `float64`

## Testing

167 test functions across the two suites: 33 for the travel-time path
(`test_geometry`, `test_picking`, `test_sirt`, `test_loaders`,
`test_end_to_end`) and 134 for CSI (`test_csi_*`). The collected count is much
higher — picking and end-to-end tests parametrise over every simulator waveform
type and over 50 MHz–3 GHz, so nothing may assume a source shape, duration or
timescale.

```bash
uv run python -m pytest Python/Inversion/tests/
uv run python -m pytest Python/Inversion/tests/ -m "not slow"   # skip end-to-end
uv run python -m pytest Python/Tomograph/tests/
```

The inversion core imports **only numpy and scipy**, so it tests on a bare
environment; HDF5, plotting and simulator-specific readers are quarantined
behind the loader modules.

## What is in this repository, and what is not

| In | Not in |
|---|---|
| `Python/Inversion/` — both inversion paths and their tests | The vendored FDTD simulator (gprMax v3.1.7, GPLv3) |
| `Python/Tomograph/` — HDF5 readers, `FieldTable`, S-parameters, Debye/Peplinski soil models | Simulation output (`.out`, `.vti`, `.vtp`) and scene definitions |
| `Python/View/` — PyVista `.vti` rendering, Yee-cell visualisation | Measured survey data |
| `Paraview/` — saved visualisation state | |

The simulator is upstream code and is not redistributed here. Install it
separately; `Python/Tomograph/functions.py` imports its `tools` package, so
that module needs the simulator on `PYTHONPATH`. The inversion packages do not.

**`pyproject.toml` lists `pycuda` and `mpi4py` as hard dependencies** — a
leftover from the simulation side. Without a CUDA toolkit and an MPI runtime,
`uv sync` will fail; drop them to work on the inversion code alone.

## Status

The travel-time path is complete and green. The CSI path is implemented and
tested against synthetic and analytic cases. **The end-to-end run on real
measured data has not been done** — the placement ended first.

One known gap is measured and documented rather than assumed: the low-frequency
rungs that convexify the CSI cost do not exist in the current dataset. The
−40 dB band of the available runs is 187.5 MHz–4249 MHz, while `L/lambda = 1`
for a 0.8 m panel at eps = 6 sits at 153 MHz — below the band. Closing it needs
three cheap 2-D simulations: a ~300 MHz source with a longer time window, a
free-space twin, and a conductivity-matched soil twin.