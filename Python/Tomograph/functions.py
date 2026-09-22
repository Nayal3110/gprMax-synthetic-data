import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from scipy import signal
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
# import matplotlib.gridspec as gridspec
from tools import plot_antenna_params
# import sys

MU_0 = np.pi*4e-7
EPS_0 = 8.854187817e-12
C = 1 / np.sqrt(MU_0 * EPS_0)

@dataclass
class FieldData:
    rx: str
    Ex: Optional[np.ndarray] = None
    Ey: Optional[np.ndarray] = None
    Ez: Optional[np.ndarray] = None
    Hx: Optional[np.ndarray] = None
    Hy: Optional[np.ndarray] = None
    Hz: Optional[np.ndarray] = None

    def find_max(self, field):
        trace = getattr(self, field, None)
        if trace is None:
            raise AttributeError(f"FieldData {self.rx!r} has no field {field!r}")
        abs_data = np.abs(signal.hilbert(trace))
        return int(np.argmax(abs_data))

@dataclass
class Voltage:
    Vx: Optional[np.ndarray]= None
    Vy: Optional[np.ndarray]= None
    Vz: Optional[np.ndarray]= None

@dataclass(frozen=True)
class Position:
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

@dataclass
class TransmissionLine:
    position: Position
    res: float
    dl: float
    Vinc: Optional[np.ndarray] = None
    Iinc: Optional[np.ndarray] = None
    Vtotal: Optional[np.ndarray] = None
    Itotal: Optional[np.ndarray] = None

@dataclass
class Receiver:
    name: str
    position: Position

@dataclass
class Source:
    type: str
    position: Position


KNOWN_FIELD_COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def _natural_sort_key(s: str) -> list:
    """Key for natural ordering: 'rx2' < 'rx10' (alphabetic order would put
    'rx10' before 'rx2' because '1' < '2'). h5py iterates groups
    alphabetically, so loaders must sort their keys with this helper to
    preserve the user's intuitive numeric ordering."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


@dataclass
class FieldTable:
    """Batched view of all receivers' field traces.

    Each component is a 2D array of shape `[n_rx, n_time]` aligned with
    `rx_names`. Built on demand from a `dict[str, FieldData]` via
    `FieldTable.from_dict`."""
    rx_names: list[str] = field(default_factory=list)
    Ex: Optional[np.ndarray] = None
    Ey: Optional[np.ndarray] = None
    Ez: Optional[np.ndarray] = None
    Hx: Optional[np.ndarray] = None
    Hy: Optional[np.ndarray] = None
    Hz: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, fields: dict[str, FieldData],
                  order: Optional[list[str]] = None) -> 'FieldTable':
        """Stack per-receiver traces into 2D arrays.

        `order` controls the row order of the output; defaults to dict
        insertion order. A component is included only if every receiver
        in `order` has it; mixed presence raises (the table is meant to
        be a clean rectangular tensor)."""
        names = list(order) if order is not None else list(fields.keys())
        missing = [n for n in names if n not in fields]
        if missing:
            raise KeyError(f"FieldTable.from_dict: receivers not in fields: {missing!r}")

        table = cls(rx_names=names)
        for comp in KNOWN_FIELD_COMPONENTS:
            traces = [getattr(fields[n], comp) for n in names]
            present = [t is not None for t in traces]
            if all(present):
                setattr(table, comp, np.stack(traces))
            elif any(present):
                gaps = [n for n, p in zip(names, present) if not p]
                raise ValueError(
                    f"FieldTable.from_dict: component {comp!r} present on "
                    f"some receivers but missing on {gaps!r}"
                )
        return table

    def index(self, rx_name: str) -> int:
        return self.rx_names.index(rx_name)


def find_transmitter(rxs: dict[str, Receiver], srcs: list[Source]) -> str:
    """Return the receiver name co-located with the (single) source.

    Replaces the old 'rx 0 is the transmitter' positional convention with a
    physical lookup: the source defines a Position, and the receiver at
    that same Position is the one used to recover Vtotal."""
    if not srcs:
        raise LookupError("find_transmitter: no sources in file")
    tx_pos = srcs[0].position
    for name, rx in rxs.items():
        if rx.position == tx_pos:
            return name
    raise LookupError(
        f"find_transmitter: no receiver co-located with source at {tx_pos}; "
        f"receiver positions: {[(n, r.position) for n, r in rxs.items()]}"
    )


def load_h5():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an output file",
        filetypes=[("Output Files", "*.out"), ("All files", "*.*")]
    )
    if not path:
        return None
    return h5py.File(path, 'r')

def get_h5_sizes(file: h5py.File) -> dict:
    def _recurse(group):
        info = {}
        for name, obj in group.items():
            if isinstance(obj, h5py.Dataset):
                info[name] = (obj.shape, str(obj.dtype))
            elif isinstance(obj, h5py.Group):
                info[name] = _recurse(obj)
        return info

    return _recurse(file)


def get_src(file: h5py.File) -> list[Source]:
    srcs: list[Source] = []
    for source in sorted(file['srcs'].keys(), key=_natural_sort_key):
        attrs = file['srcs'][source].attrs
        position_src = attrs.get('Position')
        pos_src = Position(
            x=float(position_src[0]),
            y=float(position_src[1]),
            z=float(position_src[2]),
        )
        srcs.append(Source(attrs.get('Type'), pos_src))
    return srcs


def get_rxs(file: h5py.File) -> dict[str, Receiver]:
    rxs: dict[str, Receiver] = {}
    for receiver in sorted(file['rxs'].keys(), key=_natural_sort_key):
        attrs = file['rxs'][receiver].attrs
        position_rx = attrs.get('Position')
        pos_rx = Position(
            x=float(position_rx[0]),
            y=float(position_rx[1]),
            z=float(position_rx[2]),
        )
        name = attrs.get('Name')
        rxs[name] = Receiver(name, pos_rx)
    return rxs


def get_field_data(file: h5py.File) -> dict[str, FieldData]:
    measurements: dict[str, FieldData] = {}
    for receiver in sorted(file['rxs'].keys(), key=_natural_sort_key):
        group = file['rxs'][receiver]
        name = group.attrs['Name']
        fd = FieldData(rx=name)
        for key in group.keys():
            if key in KNOWN_FIELD_COMPONENTS:
                setattr(fd, key, group[key][:])
        measurements[name] = fd
    return measurements


def get_tls(file: h5py.File) -> list[TransmissionLine]:
    tls: list[TransmissionLine] = []
    for name in sorted(file['tls'].keys(), key=_natural_sort_key):
        group = file['tls'][name]
        pos = Position(
            x=float(group.attrs['Position'][0]),
            y=float(group.attrs['Position'][1]),
            z=float(group.attrs['Position'][2]),
        )
        tl = TransmissionLine(
            position=pos,
            res=float(group.attrs['Resistance']),
            dl=float(group.attrs['dl']),
        )
        for key in group.keys():
            setattr(tl, key, group[key][:])
        tls.append(tl)
    return tls

def _to_db(arr: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        out = 20 * np.log10(np.abs(arr))
    return np.where(np.isfinite(out), out, 0.0)


def _bandwidth_mask(fft_vinc: np.ndarray, threshold_db: float = -40) -> np.ndarray:
    mag = np.abs(fft_vinc)
    if mag.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.max(mag))
    if peak <= 0:
        return np.zeros_like(mag, dtype=bool)
    return 20 * np.log10(np.maximum(mag, 1e-30)) - 20 * np.log10(peak) > threshold_db


_COMPONENT_AXIS = {'Ex': 0, 'Ey': 1, 'Ez': 2}


def _field_to_voltage(field: FieldData, component: str, dx_dy_dz: np.ndarray) -> np.ndarray:
    """V = -E·dl along one Yee edge, raising if the requested component is
    missing on this receiver and reporting which components ARE available."""
    if component not in _COMPONENT_AXIS:
        raise ValueError(
            f"component must be one of {list(_COMPONENT_AXIS)!r}, got {component!r}"
        )
    trace = getattr(field, component)
    if trace is None:
        available = [c for c in _COMPONENT_AXIS if getattr(field, c) is not None]
        raise ValueError(
            f"receiver {field.rx!r} has no {component!r} data; "
            f"available E-components: {available or 'none'}"
        )
    return trace * -1 * dx_dy_dz[_COMPONENT_AXIS[component]]


def gaussian (frequency: float, t: float):
    A = 1.000407
    tau = 2*pow(np.pi,2)*np.pow(frequency,2)
    x = 1/frequency
    waveform = A*np.exp(-tau*pow((t-x),2))
    return waveform

def gaussiandot (frequency: float, t:float):
    tau = 2*pow(np.pi,2)*np.pow(frequency,2)
    x =1/frequency
    waveform = -2*tau*(t-x)*np.exp(-tau*pow((t-x),2))
    return waveform

def LSM(f:list[float], s21_m: list[float], d:float, eps_g = 5, sigma_g=0.01,
        bw_mask: Optional[np.ndarray] = None):

    def S21_model(f, eps_r:float, sigma:float):
        omega = 2 * np.pi * f
        gamma = np.sqrt(1j * omega * MU_0 * (sigma + 1j * omega * eps_r * EPS_0))
        return np.exp(-gamma* d)

    def objective(f_in, eps_r: float, sigma: float):
        n = len(f_in) // 2
        s21_pred = S21_model(f_in[:n], eps_r, sigma)
        return np.concatenate([np.real(s21_pred), np.imag(s21_pred)])

    f_vals = np.array(f[1:])
    s21_m = np.array(s21_m[1:])

    if bw_mask is not None:
        m = np.asarray(bw_mask, dtype=bool)[1:]
        f_vals = f_vals[m]
        s21_m = s21_m[m]

    s21_r = np.real(s21_m)
    s21_i = np.imag(s21_m)

    s21_flat = np.concatenate([s21_r, s21_i])
    f_dup = np.concatenate([f_vals, f_vals])
    try:
        bounds_lower = [eps_g * 0.5, sigma_g * 0.1]
        bounds_upper = [eps_g * 1.5, sigma_g * 10]
        popt, pcov = curve_fit(objective, f_dup, s21_flat,
                              p0=[eps_g, sigma_g],
                              bounds=(bounds_lower, bounds_upper),
                              maxfev=20000,
                              method='trf',
                              ftol=1e-10,
                              xtol=1e-10)
        
        eps_r_exp, sigma_exp = popt
        print(f"ε_r = {eps_r_exp:.6f}")
        print(f"σ = {sigma_exp:.6f}")
        
    except Exception as e:
        print(f"Error en ajuste: {e}")
        return None, None

    S21_theo = S21_model(f_vals, eps_g, sigma_g)
    S21_exp = S21_model(f_vals, eps_r_exp, sigma_exp)

    return S21_theo, S21_exp

def cal_err(file: h5py.File, e_r: float, rx_name: str, component: str = 'Ez'):
    rxs = get_rxs(file)
    srcs = get_src(file)
    measures = get_field_data(file)
    tx_name = find_transmitter(rxs, srcs)

    value_rx = measures[rx_name].find_max(component)
    value_i = measures[tx_name].find_max(component)

    dt = file.attrs['dt']
    iterations = file.attrs['Iterations']
    time = np.linspace(0, (iterations - 1) * dt, num=iterations)

    t_rx = time[value_rx]
    t_i = time[value_i]

    d_sim = (C / np.sqrt(e_r)) * np.abs(t_rx - t_i)
    d_teo = np.sqrt(np.pow((rxs[rx_name].position.x - rxs[tx_name].position.x), 2)
                    + np.pow((rxs[rx_name].position.y - rxs[tx_name].position.y), 2))
    err = (abs(d_teo - d_sim) / d_teo) * 100
    print(f"Distancia: {d_sim:.4f} Distancia teorica: {d_teo:.4f} Error: {err:.2f}%")

def envolvente(file_ref: h5py.File, file_mat: h5py.File, rx_name: str, component: str = 'Ez'):
    """Compare Hilbert envelopes of `component` at receiver `rx_name` between
    a reference run and a material run, alongside the source-edge envelope
    from the reference run."""
    if component not in _COMPONENT_AXIS:
        raise ValueError(
            f"component must be one of {list(_COMPONENT_AXIS)!r}, got {component!r}"
        )

    m_ref = get_field_data(file_ref)
    m_mat = get_field_data(file_mat)
    tx_name = find_transmitter(get_rxs(file_ref), get_src(file_ref))

    dt = file_ref.attrs['dt']
    iterations = file_ref.attrs['Iterations']
    time = np.linspace(0, (iterations - 1) * dt, num=iterations)

    traces = {
        f'source (ref, {tx_name})': getattr(m_ref[tx_name], component),
        f'{rx_name} (ref)':         getattr(m_ref[rx_name], component),
        f'{rx_name} (mat)':         getattr(m_mat[rx_name], component),
    }
    for label, trace in traces.items():
        if trace is None:
            raise ValueError(f"envolvente: {label} has no {component!r} data")

    colors = {f'source (ref, {tx_name})': 'r',
              f'{rx_name} (ref)': 'b',
              f'{rx_name} (mat)': 'g'}
    for label, trace in traces.items():
        plt.plot(time, np.abs(signal.hilbert(trace)), colors[label],
                 label=f'{label} {component} envelope')
    plt.grid(True)
    plt.legend()
    plt.show()

def extract_vinc(file: h5py.File, component: str = 'Ez') -> np.ndarray:
    """Recover the empirical incident (forward-wave) voltage from a baseline
    (vacuum) run.

    `component` selects the E-axis at rx 0 to integrate; choose the axis the
    source is polarized along.

    A `#voltage_source` has Z_s = 0, so the full injected voltage appears
    at the source edge — that is the *open-circuit* source voltage, not the
    forward wave. The TL formalism splits voltage into forward (Vinc) and
    reflected components; on a matched line the forward wave is half the
    open-circuit source voltage. We divide by 2 here so the empirical Vinc
    matches what `plot_antenna_params.calculate_antenna_params` reads from a
    `#transmission_line` probe — without this, |S21| would be ~6 dB low."""
    measures = get_field_data(file)
    tx_name = find_transmitter(get_rxs(file), get_src(file))
    dx_dy_dz = file.attrs['dx_dy_dz']
    return _field_to_voltage(measures[tx_name], component, dx_dy_dz) / 2


def compute_s_params_1rx(
    file: h5py.File,
    frequency: float,
    vinc: Optional[np.ndarray] = None,
    delay_correct: bool = False,
    component: str = 'Ez',
    rx_name: Optional[str] = None,
) -> dict:
    """Compute S11 and S21 for a single non-transmitter receiver from an
    already-open gprMax .out file.

    Caller owns the file handle. The transmitter receiver (co-located with
    the source) supplies Vtotal; another receiver supplies Vrec.

    Args:
        vinc: optional empirical incident voltage array (length = iterations).
            If None, an analytical Gaussian at `frequency` is used as Vinc.
            Pass the result of `extract_vinc()` from a vacuum baseline run
            to match TL-style S-parameters.
        delay_correct: if True, multiply fft(Vinc) by exp(j*2π*f*dt/2) to
            compensate for the half-step E/H stagger when Vinc is analytical.
            No effect if `vinc` is empirical (already on the E-field clock).
        component: 'Ex'|'Ey'|'Ez' — the E-axis used on both the transmitter
            (Vtotal) and the receiver (Vrec). Caller is responsible for
            matching `vinc`'s axis to this when supplying an empirical baseline.
        rx_name: name of the receiver to use for Vrec. If None, the file
            must contain exactly one non-transmitter receiver.
    """
    measures = get_field_data(file)
    rxs = get_rxs(file)
    srcs = get_src(file)
    tx_name = find_transmitter(rxs, srcs)

    if rx_name is None:
        candidates = [n for n in measures if n != tx_name]
        if len(candidates) != 1:
            raise ValueError(
                f"compute_s_params_1rx: rx_name=None requires exactly one "
                f"non-transmitter receiver, found {candidates!r}"
            )
        rx_name = candidates[0]

    dx_dy_dz = file.attrs['dx_dy_dz']
    dt = file.attrs['dt']
    iterations = file.attrs['Iterations']

    time = np.linspace(0, (iterations - 1) * dt, num=iterations)

    d_teo = np.sqrt((rxs[rx_name].position.x - rxs[tx_name].position.x) ** 2
                    + (rxs[rx_name].position.y - rxs[tx_name].position.y) ** 2)

    Vtotal = _field_to_voltage(measures[tx_name], component, dx_dy_dz)
    Vrec = _field_to_voltage(measures[rx_name], component, dx_dy_dz)
    if vinc is None:
        Vinc = gaussian(frequency, time)
    else:
        Vinc = np.asarray(vinc)
    Vref = Vtotal - Vinc

    freqs = np.fft.fftfreq(Vinc.size, d=dt)

    fft_vinc = np.fft.fft(Vinc)
    if delay_correct and vinc is None:
        fft_vinc = fft_vinc * np.exp(1j * 2 * np.pi * freqs * (dt / 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        s11 = np.fft.fft(Vref) / fft_vinc
        s21 = np.fft.fft(Vrec) / fft_vinc

    bw_mask = _bandwidth_mask(fft_vinc)

    return {'time': time, 'freqs': freqs,
            's11': s11, 's21': s21,
            's11_db': _to_db(s11), 's21_db': _to_db(s21),
            'bw_mask': bw_mask, 'd': d_teo}


def compute_s_params(
    file: h5py.File,
    frequency: float,
    vinc: Optional[np.ndarray] = None,
    delay_correct: bool = False,
    component: str = 'Ez',
) -> dict:
    """Compute S21 per-receiver from an already-open gprMax .out file.

    Caller owns the file handle (this function does not close it). The
    transmitter receiver (co-located with the source) supplies Vtotal;
    every other receiver yields one row of S21.

    Args:
        vinc: optional empirical incident voltage array (length = iterations).
            If None, an analytical Gaussian at `frequency` is used as Vinc.
            Pass the result of `extract_vinc()` from a vacuum baseline run
            to match TL-style S-parameters.
        delay_correct: if True, multiply fft(Vinc) by exp(j*2π*f*dt/2) to
            compensate for the half-step E/H stagger when Vinc is analytical.
            No effect if `vinc` is empirical (already on the E-field clock).
        component: 'Ex'|'Ey'|'Ez' — the E-axis used on the transmitter (Vtotal)
            and on every non-transmitter receiver (Vrec). Caller is responsible
            for matching `vinc`'s axis to this when supplying an empirical
            baseline.

    Returns:
        Dict with arrays whose receiver axis is in `rx_names` order (the
        non-transmitter receivers in natural-sort order).
    """
    measures = get_field_data(file)
    rxs = get_rxs(file)
    srcs = get_src(file)
    tx_name = find_transmitter(rxs, srcs)
    non_tx_names = [n for n in measures if n != tx_name]

    dx_dy_dz = file.attrs['dx_dy_dz']
    dt = file.attrs['dt']
    iterations = file.attrs['Iterations']

    time = np.linspace(0, (iterations - 1) * dt, num=iterations)

    tx_pos = rxs[tx_name].position
    d_teo = np.array([
        np.sqrt((rxs[n].position.x - tx_pos.x) ** 2
                + (rxs[n].position.y - tx_pos.y) ** 2)
        for n in non_tx_names
    ])

    Vtotal = _field_to_voltage(measures[tx_name], component, dx_dy_dz)
    Vrec = np.stack([_field_to_voltage(measures[n], component, dx_dy_dz)
                     for n in non_tx_names])

    if vinc is None:
        Vinc = gaussian(frequency, time)
    else:
        Vinc = np.asarray(vinc)
    Vref = Vtotal - Vinc

    freqs = np.fft.fftfreq(Vinc.size, d=dt)

    fft_vinc = np.fft.fft(Vinc)
    if delay_correct and vinc is None:
        fft_vinc = fft_vinc * np.exp(1j * 2 * np.pi * freqs * (dt / 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        s21 = np.fft.fft(Vrec) / fft_vinc[np.newaxis, :]

    bw_mask = _bandwidth_mask(fft_vinc)

    return {'time': time, 'freqs': freqs,
            's21': s21, 's21_db': _to_db(s21),
            'bw_mask': bw_mask, 'd': d_teo,
            'rx_names': non_tx_names}


def s_param(frequency: float, use_baseline_vinc: bool = False, delay_correct: bool = False,
            component: str = 'Ez'):
    """Interactive wrapper: pop a file dialog, then call compute_s_params.

    `component` is passed to both `extract_vinc` and `compute_s_params` so
    the baseline and measurement always share the same E-axis.

    If `use_baseline_vinc` is True, a second dialog asks for a vacuum
    baseline `.out` whose rx-0 `component` is used as the empirical Vinc."""
    file = load_h5()
    if file is None:
        return None
    vinc = None
    baseline = None
    if use_baseline_vinc:
        baseline = load_h5()
        if baseline is None:
            file.close()
            return None
        vinc = extract_vinc(baseline, component=component)
    try:
        result = compute_s_params(file, frequency, vinc=vinc,
                                  delay_correct=delay_correct, component=component)
    finally:
        file.close()
        if baseline is not None:
            baseline.close()

    time = result['time']
    dt = time[1] - time[0]
    df = 1 / np.amax(time)
    print('Time window: {:g} s ({} iterations)'.format(np.amax(time), time.size))
    print('Time step: {:g} s'.format(dt))
    print('Frequency bin spacing: {:g} Hz'.format(df))
    return result

def plot_s_params(s_params:dict, s_params_tls: Optional[dict] = None, max_plots:Optional[int]=4, f_max: float = 5e8):

    s21 = s_params['s21_db']
    rx_names = s_params.get('rx_names')

    n_rx = s_params['s21'].shape[0]
    n_fig = int(np.ceil(n_rx/ max_plots))

    pltrangemin = 1
    above = np.where(s_params['freqs'] > f_max)[0]
    pltrangemax = above[0] if above.size else len(s_params['freqs'])
    pltrange = np.s_[pltrangemin:pltrangemax]
        
    rx_idx = 0
    for fig_idx in range(n_fig):
        fig_num = fig_idx + 1
        fig = plt.figure(num=f'S21 (Figura {fig_num})', figsize=(15, 10))
        plots_r = n_rx - rx_idx
        plots_fig = min(max_plots, plots_r)

        if plots_fig == 1:
            rows, cols = 1, 1
        elif plots_fig == 2:
            rows, cols = 1, 2
        elif plots_fig == 3 or plots_fig == 4:
            rows, cols = 2, 2
        else:
            cols = int(np.ceil(np.sqrt(plots_fig)))
            rows = int(np.ceil(plots_fig / cols))

        for plot_in_fig_idx in range(plots_fig):
            pos = plot_in_fig_idx + 1
            ax = fig.add_subplot(rows,cols,pos)
            s21_rx = s21[rx_idx,:]

            markerline, stemlines, baseline = ax.stem(s_params['freqs'][pltrange], s21_rx[pltrange], '-.')
            plt.setp(baseline, 'linewidth', 0)
            plt.setp(stemlines, 'color', 'g')
            plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
            ax.plot(s_params['freqs'][pltrange], s21_rx[pltrange], 'g', lw=2)
            if s_params_tls is not None and rx_idx != 7:
                markerline, stemlines, baseline = ax.stem(s_params_tls[rx_idx+1]['freqs'][pltrange], s_params_tls[rx_idx+1]['s21'][pltrange], '-.')
                plt.setp(baseline, 'linewidth', 0)
                plt.setp(stemlines, 'color', 'r')
                plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
                ax.plot(s_params_tls[rx_idx+1]['freqs'][pltrange], s_params_tls[rx_idx+1]['s21'][pltrange], 'r', lw=2)

            label = rx_names[rx_idx] if rx_names is not None else f'Receptor #{rx_idx + 1}'
            ax.set_title(label)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power (dB)')
            ax.grid(True, linestyle='--')

            rx_idx+=1
        plt.subplots_adjust(hspace=0.4)
    plt.show()

def eps(f: np.ndarray, f_c: float, eps_model: float, s21: np.ndarray, d: np.ndarray,
        sigma_model: float = 0, f_min: float = 50e6, f_max: float = 400e6,
        bw_mask: Optional[np.ndarray] = None,
        rx_names: Optional[list[str]] = None):
    d = d[:,np.newaxis]
    above_min = np.where(f > f_min)[0]
    above_max = np.where(f > f_max)[0]
    pltrangemin = above_min[0] if above_min.size else 0
    pltrangemax = above_max[0] if above_max.size else len(f)
    pltrange = np.s_[pltrangemin:pltrangemax]

    f_vals = np.asarray(f[pltrange])
    s21_vals = np.asarray(s21[:,pltrange]).astype(np.complex128)

    if bw_mask is not None:
        m = np.asarray(bw_mask, dtype=bool)[pltrange]
        f_vals = f_vals[m]
        s21_vals = s21_vals[:, m]

    f_vals = f_vals[np.newaxis,:]

    omega = 2*np.pi*f_vals

    mag = np.abs(s21_vals)
    phase_unwrap = np.unwrap(np.angle(s21_vals))

    eps_complex = eps_model * EPS_0 - 1j* sigma_model/omega
    
    gamma_model = 1j * omega * np.sqrt(MU_0 * eps_complex) 
    beta_model = np.imag(gamma_model)

    phase_teo = -beta_model * d

    N = 0
    min_error = float('inf')

    for N_test in range(-10, 10):
        temp = phase_unwrap + 2 * np.pi * N_test
        error = np.sum(np.pow((temp - phase_teo),2)) 
        if error < min_error:
            min_error = error
            N = N_test
    
    phase_abs = phase_unwrap + 2*np.pi* N

    alpha = -np.log(mag)/d
    beta = -phase_abs/d

    gamma = alpha + 1j*beta

    eps_r = -np.pow(gamma,2)/(np.pow(omega,2)*MU_0)/EPS_0

    
    idex = np.argmin(np.abs(f_vals - f_c)) 
    eps_r_f_c =np.mean(eps_r[:,idex])

    print(f"epsilon_r = {eps_r_f_c}")
    attconst = {'freqs': f_vals, 'phase_abs': phase_abs, 'phase_teo': phase_teo,
                'phase_unwrap': phase_unwrap, 'alpha': alpha, 'beta': beta,
                'gamma': gamma, 'eps_r': eps_r, 'rx_names': rx_names}
    return attconst

def plot_attconst(attconst: dict , max_plots = 4, attconst_tls : Optional[dict] = None):

    n_params = attconst['phase_abs'].shape[0]
    n_fig = int(np.ceil(n_params/ max_plots))
    rx_names = attconst.get('rx_names')

    f = attconst['freqs'][0,:]

    param_idx = 0
    for fig_idx in range(n_fig):
        fig_num = fig_idx + 1
        fig = plt.figure(num=f'Phase (Figure {fig_num})', figsize=(15, 10))
        plots_r = n_params - param_idx
        plots_fig = min(max_plots, plots_r)
        if plots_fig == 1:
            rows, cols = 1, 1
        elif plots_fig == 2:
            rows, cols = 1, 2
        elif plots_fig == 3 or plots_fig == 4:
            rows, cols = 2, 2
        else:
            cols = int(np.ceil(np.sqrt(plots_fig)))
            rows = int(np.ceil(plots_fig / cols))

        for plot_in_fig_idx in range(plots_fig):
            pos = plot_in_fig_idx + 1
            ax = fig.add_subplot(rows,cols,pos)
            param_rx = attconst['phase_abs'][param_idx,:]
            param_rx_teo = attconst['phase_teo'][param_idx,:]
            
            markerline, stemlines, baseline = ax.stem(f, param_rx, '-.')
            plt.setp(baseline, 'linewidth', 0)
            plt.setp(stemlines, 'color', 'g')
            plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
            ax.plot(f, param_rx, 'g', lw=2)

            markerline, stemlines, baseline = ax.stem(f, param_rx_teo, '-.')
            plt.setp(baseline, 'linewidth', 0)
            plt.setp(stemlines, 'color', 'r')
            plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
            ax.plot(f, param_rx_teo, 'r', lw=2)

            if attconst_tls is not None:
                param_rx_tls = attconst_tls['phase_abs'][param_idx,:]
                # param_rx_teo_tls = attconst_tls['phase_teo'][param_idx,:]

                markerline, stemlines, baseline = ax.stem(f, param_rx_tls, '-.')
                plt.setp(baseline, 'linewidth', 0)
                plt.setp(stemlines, 'color', 'b')
                plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
                ax.plot(f, param_rx_tls, 'b', lw=2)

                # markerline, stemlines, baseline = ax.stem(f, param_rx_teo_tls, '-.')
                # plt.setp(baseline, 'linewidth', 0)
                # plt.setp(stemlines, 'color', 'k')
                # plt.setp(markerline, 'markerfacecolor', 'k', 'markeredgecolor', 'k')
                # ax.plot(f, param_rx_teo_tls, 'k', lw=2)

            label = rx_names[param_idx] if rx_names is not None else f'Receptor #{param_idx + 1}'
            ax.set_title(label)
            ax.set_xlabel('Frequency [Hz]')
            ax.grid(True, linestyle='--')

            param_idx+=1
        plt.subplots_adjust(hspace=0.4)
    plt.show()


def compare_spectra(
    vs_file: h5py.File,
    tl_file: str,
    frequency: float,
    rx_index: int = 1,
    rxcomponent: str = 'Ez',
    f_max: float = 5e8,
    baseline_file: Optional[h5py.File] = None,
    delay_correct: bool = False,
):
    """Overlay |Vinc(f)|, |Vrec(f)|, and |S21(f)| in dB for a voltage-source
    run vs a transmission-line run. Use this to localize where any residual
    discrepancy lives — numerator (Vinc), denominator (Vrec), or the ratio.

    Args:
        vs_file: open .out from a #voltage_source simulation.
        tl_file: path to a .out from a #transmission_line simulation.
        rx_index: receiver index (>=1) to compare. Same convention as
            compute_s_params (rx 0 is the source probe; rx_index is offset
            into the per-receiver S21 returned in s_params['s21']).
        rxcomponent: 'Ex'|'Ey'|'Ez' — same E-axis used on BOTH the TL side
            (passed to calculate_antenna_params) and the VS side (passed to
            extract_vinc / compute_s_params). The two sides must match for
            the comparison to be physically meaningful.
        baseline_file: optional open .out from a vacuum baseline run; if
            provided, its rx-0 `rxcomponent` is used as the empirical Vinc
            for the voltage-source side via extract_vinc().
    """
    vinc = (extract_vinc(baseline_file, component=rxcomponent)
            if baseline_file is not None else None)
    vs = compute_s_params(vs_file, frequency, vinc=vinc,
                          delay_correct=delay_correct, component=rxcomponent)
    tl = plot_antenna_params.calculate_antenna_params(
        tl_file, rxnumber=rx_index, rxcomponent=rxcomponent,
    )

    freqs = vs['freqs']
    mask = (freqs > 0) & (freqs < f_max)

    dxdydz = vs_file.attrs['dx_dy_dz']
    Vinc_vs = np.asarray(vinc) if vinc is not None else gaussian(frequency, vs['time'])
    vs_measures = get_field_data(vs_file)
    vs_rx_name = list(vs_measures.keys())[rx_index]
    Vrec_vs = _field_to_voltage(vs_measures[vs_rx_name], rxcomponent, dxdydz)

    vinc_vs_db = _to_db(np.fft.fft(Vinc_vs))
    vrec_vs_db = _to_db(np.fft.fft(Vrec_vs))
    s21_vs_db = vs['s21_db'][rx_index - 1, :]
    vrec_tl_db = tl['Vincp'] + tl['s21']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vs_curve, tl_curve, title in [
        (axes[0], vinc_vs_db, tl['Vincp'], '|Vinc| (dB)'),
        (axes[1], vrec_vs_db, vrec_tl_db, '|Vrec| (dB)'),
        (axes[2], s21_vs_db, tl['s21'], '|S21| (dB)'),
    ]:
        ax.plot(freqs[mask], vs_curve[mask], 'g', lw=2, label='voltage-source')
        ax.plot(tl['freqs'][mask], tl_curve[mask], 'r', lw=2, label='TL')
        ax.set_title(title)
        ax.set_xlabel('Frequency [Hz]')
        ax.grid(True, linestyle='--')
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    f = 220e6
    COMPONENT = 'Ey'

    # file = load_h5()
    # info = get_h5_sizes(file)
    # print(info)

    REPO_ROOT = Path(__file__).resolve().parents[2]
    path = REPO_ROOT / 'gprMax' / 'user_models' / 'test' / 'Soil' / 'test_A (freq = 220 MHz)' / 'test_A_complex_soil+water_tls_freq_220MHz.out'
    s_params_tls = []
    for i in range (1,3):
         s_params_tls.append(plot_antenna_params.calculate_antenna_params(str(path),rxnumber=i,rxcomponent=COMPONENT))
    s_params = s_param(f, component=COMPONENT)
    plot_s_params(s_params=s_params,s_params_tls=s_params_tls)
    # print(s_params['s21'])
    # s21_tls = np.array([item['s21'] for item in s_params_tls[1:]])
    # print(s21_tls)
    # attconst = eps(f=s_params['freqs'],f_c=f,eps_model=1,s21=s_params['s21'],d=s_params['d'])
    # attconst_tls = eps(f = s_params_tls[0]['freqs'],f_c=f,eps_model=1,s21=s21_tls,d=s_params['d'])
    # plot_attconst(attconst=attconst,attconst_tls=attconst_tls)
    

