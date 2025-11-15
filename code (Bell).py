from __future__ import annotations
import os
import sys
import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import griddata, interp1d
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.circuit.library import UnitaryGate

@dataclass
class HParams:
    omega: float           
    lambda_max: float      
    drive_freq: float       
    phase: float = 0.0
    V_func: Optional[Callable[[float], np.ndarray]] = None

@dataclass
class NoiseMarkovParams:
    T1_us: float
    T2_us: float
    gate_time_us: float = 0.08
    cx_multiplier: float = 5.0
    jitter_T2_frac: float = 0.0  

@dataclass
class NoiseGaussianParams:
    T2_us: float

@dataclass
class ScanConfig:
    times_us: np.ndarray
    T2_list_us: List[float]
    lambda_list_rad_per_us: List[float]

@dataclass
class RunConfig:
    hparams: HParams
    noise_markov: NoiseMarkovParams
    noise_gauss: NoiseGaussianParams
    scan: ScanConfig
    dt_us: float = 0.2
    results_dir: str = "results"
    tag: str = "final"
    mc_realizations_gaussian: int = 40
    mc_realizations_markovian: int = 1
    compare_models: List[str] = None
    random_seed_base: int = 12345
    smooth_heatmap_with_scipy: bool = True

def default_runconfig() -> RunConfig:
    times = np.linspace(0.0, 120.0, 121)  # 1 μs resolution
    lam_list = [
        0.0,
        0.062832,   # ≈ 0.01 MHz
        0.125664,   # ≈ 0.02 MHz
        0.188496,   # ≈ 0.03 MHz
        0.251327,   # ≈ 0.04 MHz
        0.314159    # ≈ 0.05 MHz
    ]
    T2_list = [40.0, 80.0, 160.0, 320.0, 640.0, 1000.0]

    return RunConfig(
        hparams=HParams(
            omega=0.125664,
            lambda_max=0.314,
            drive_freq=0.062832
        ),
        noise_markov=NoiseMarkovParams(
            T1_us=500.0,
            T2_us=80.0,
            gate_time_us=0.08,
            cx_multiplier=5.0,
            jitter_T2_frac=0.0
        ),
        noise_gauss=NoiseGaussianParams(
            T2_us=80.0
        ),
        scan=ScanConfig(
            times_us=times,
            T2_list_us=T2_list,
            lambda_list_rad_per_us=lam_list
        ),
        dt_us=0.2,
        results_dir="results",
        tag="final",
        mc_realizations_gaussian=40,
        mc_realizations_markovian=1,
        compare_models=["markovian", "gaussian"],
        random_seed_base=12345,
        smooth_heatmap_with_scipy=True,
    )


I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PROJ0 = np.array([[1, 0], [0, 0]], dtype=complex)
PROJ1 = np.array([[0, 0], [0, 1]], dtype=complex)

def lambda_t(t: float, hp: HParams) -> float:
    return hp.lambda_max * math.cos(hp.drive_freq * t + hp.phase)

def decompose_H_to_pauli(H: np.ndarray) -> Tuple[complex, np.ndarray]:
    h0 = 0.5 * np.trace(H)
    c_x = np.trace(H @ X2)
    c_y = np.trace(H @ Y2)
    c_z = np.trace(H @ Z2)
    c = np.array([c_x.real, c_y.real, c_z.real], dtype=float)
    return h0, c

def unitary_from_H_general(dt_us: float, t_us: float, hp: HParams) -> np.ndarray:
    if hp.V_func is not None:
        H = hp.V_func(t_us)
    else:
        lam = lambda_t(t_us, hp)
        H = 0.5 * (hp.omega * Z2 + lam * X2)
    h0, c = decompose_H_to_pauli(H)
    norm_c = np.linalg.norm(c)
    if norm_c == 0:
        return np.exp(-1j * h0 * dt_us) * I2
    theta = (norm_c * dt_us) / 2.0
    nvec = c / norm_c
    rot = (math.cos(theta) * I2) - 1j * math.sin(theta) * (nvec[0]*X2 + nvec[1]*Y2 + nvec[2]*Z2)
    return np.exp(-1j * h0 * dt_us) * rot


def teleportation_circuit_pre_measure(psi: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.initialize(psi, 0)
    qc.h(1); qc.cx(1,2)
    qc.cx(0,1); qc.h(0)
    return qc

def projective_average_and_correct(rho_full: DensityMatrix) -> DensityMatrix:
    rho = rho_full.data
    out = np.zeros((8,8), dtype=complex)
    def U_on_q2(op: np.ndarray) -> np.ndarray:
        return np.kron(op, np.kron(I2, I2))
    Umap = {
        (0,0): np.eye(8, dtype=complex),
        (0,1): U_on_q2(X2),
        (1,0): U_on_q2(Z2),
        (1,1): U_on_q2(Z2 @ X2),
    }
    for m0 in (0,1):
        for m1 in (0,1):
            P0 = PROJ0 if m0 == 0 else PROJ1
            P1 = PROJ0 if m1 == 0 else PROJ1
            P = np.kron(I2, np.kron(P1, P0))
            piece = P @ rho @ P.conj().T
            out += Umap[(m0, m1)] @ piece @ Umap[(m0, m1)].conj().T
    return DensityMatrix(out)

def fidelity_from_rho_bob(rho_bob: DensityMatrix, psi: np.ndarray) -> float:
    psi_vec = psi.reshape(2,)
    return float(np.real(np.vdot(psi_vec, rho_bob.data @ psi_vec)))

def extract_density_from_result(res, label='rho') -> DensityMatrix:
    try:
        data = res.data(0)
        if label in data:
            return DensityMatrix(data[label])
    except Exception:
        pass
    try:
        dat0 = res.results[0].data
        if label in dat0:
            return DensityMatrix(dat0[label])
        for v in dat0.values():
            if isinstance(v, (list, np.ndarray)):
                return DensityMatrix(v)
    except Exception:
        pass
    raise RuntimeError("Could not extract density matrix from simulator result")


def build_markovian_noise_model(noise: NoiseMarkovParams) -> NoiseModel:
    nm = NoiseModel()
    u_err = thermal_relaxation_error(noise.T1_us, noise.T2_us, noise.gate_time_us)
    cx_time = noise.gate_time_us * noise.cx_multiplier
    cx_err = thermal_relaxation_error(noise.T1_us, noise.T2_us, cx_time).tensor(
        thermal_relaxation_error(noise.T1_us, noise.T2_us, cx_time)
    )
    nm.add_all_qubit_quantum_error(u_err, ["u1","u2","u3","id","h","delay"])
    nm.add_all_qubit_quantum_error(cx_err, ["cx"])
    return nm

def gaussian_phase_kick(dt_us: float, T2_us: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if T2_us <= 0 or math.isinf(T2_us):
        return I2
    if rng is None:
        rng = np.random.default_rng()
    sigma = math.sqrt(2.0 * dt_us / T2_us)
    phi = rng.normal(0.0, sigma)
    return np.array([[np.exp(-1j*phi/2), 0.0], [0.0, np.exp(1j*phi/2)]], dtype=complex)


def run_single_markovian(psi: np.ndarray, total_time_us: float, hp: HParams, noise: NoiseMarkovParams, dt_us: float,
                         sim: Optional[AerSimulator]=None, noise_model: Optional[NoiseModel]=None) -> float:
    qc = teleportation_circuit_pre_measure(psi)
    if total_time_us > 0:
        steps = max(1, int(round(total_time_us / dt_us)))
        dt = total_time_us / steps
        half_dt = dt / 2.0
        for k in range(steps):
            t_mid = (k + 0.5) * dt
            qc.delay(half_dt, 2, unit='us')
            U = unitary_from_H_general(dt, t_mid, hp)
            qc.append(UnitaryGate(U), [2])
            qc.delay(half_dt, 2, unit='us')
    qc.barrier(); qc.save_density_matrix(label='rho')
    if noise_model is None:
        noise_model = build_markovian_noise_model(noise)
    if sim is None:
        sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    tcirc = transpile(qc, sim)
    res = sim.run(tcirc).result()
    rho3 = extract_density_from_result(res, label='rho')
    rho_after = projective_average_and_correct(rho3)
    rho_bob = partial_trace(rho_after, [0,1])
    return fidelity_from_rho_bob(rho_bob, psi)

def run_single_gaussian(psi: np.ndarray, total_time_us: float, hp: HParams, noise: NoiseGaussianParams, dt_us: float,
                        rng: Optional[np.random.Generator]=None) -> float:
    qc = teleportation_circuit_pre_measure(psi)
    if rng is None:
        rng = np.random.default_rng()
    if total_time_us > 0:
        steps = max(1, int(round(total_time_us / dt_us)))
        dt = total_time_us / steps
        half_dt = dt / 2.0
        for k in range(steps):
            t_mid = (k + 0.5) * dt
            K1 = gaussian_phase_kick(half_dt, noise.T2_us, rng)
            qc.append(UnitaryGate(K1), [2])
            U = unitary_from_H_general(dt, t_mid, hp)
            qc.append(UnitaryGate(U), [2])
            K2 = gaussian_phase_kick(half_dt, noise.T2_us, rng)
            qc.append(UnitaryGate(K2), [2])
    qc.barrier(); qc.save_density_matrix(label='rho')
    sim = AerSimulator(method='density_matrix')
    tcirc = transpile(qc, sim)
    res = sim.run(tcirc).result()
    rho3 = extract_density_from_result(res, label='rho')
    rho_after = projective_average_and_correct(rho3)
    rho_bob = partial_trace(rho_after, [0,1])
    return fidelity_from_rho_bob(rho_bob, psi)


def noiseless_baseline_check() -> float:
    psi = Statevector.from_label('+').data
    qc = teleportation_circuit_pre_measure(psi)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    res = sim.run(transpile(qc, sim)).result()
    sv = res.get_statevector()
    dm = DensityMatrix(sv)
    rho_after = projective_average_and_correct(dm)
    rho_bob = partial_trace(rho_after, [0,1])
    F = fidelity_from_rho_bob(rho_bob, psi)
    logging.info("Noiseless baseline fidelity = %.12f", F)
    return F


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def run_full_comparison(cfg: RunConfig) -> Dict[str, pd.DataFrame]:
    model_folders = {}
    base = cfg.results_dir
    make_dir(base)
    for m in cfg.compare_models:
        model_folders[m] = make_dir(os.path.join(base, m))
    overlays_dir  = make_dir(os.path.join(base, "overlays"))
    heatmaps_dir  = make_dir(os.path.join(base, "heatmaps"))

    psi = Statevector.from_label('+').data
    results_by_model: Dict[str, List[Dict]] = {m: [] for m in cfg.compare_models}

    grid_points = len(cfg.scan.T2_list_us) * len(cfg.scan.lambda_list_rad_per_us) * len(cfg.scan.times_us)
    total_runs = len(cfg.compare_models) * grid_points
    run_idx = 0

    for model in cfg.compare_models:
        logging.info("Starting model '%s'...", model)
        for T2 in cfg.scan.T2_list_us:
            if model == "markovian":
                noise_mark = NoiseMarkovParams(T1_us=cfg.noise_markov.T1_us, T2_us=T2,
                                               gate_time_us=cfg.noise_markov.gate_time_us,
                                               cx_multiplier=cfg.noise_markov.cx_multiplier,
                                               jitter_T2_frac=cfg.noise_markov.jitter_T2_frac)
                nm = build_markovian_noise_model(noise_mark)
                sim_mark = AerSimulator(method='density_matrix', noise_model=nm)
            else:
                sim_mark = None

            for lam in cfg.scan.lambda_list_rad_per_us:
                hp = HParams(omega=cfg.hparams.omega, lambda_max=lam, drive_freq=cfg.hparams.drive_freq, phase=cfg.hparams.phase)
                fidelities = []
                stds = []
                for t in cfg.scan.times_us:
                    run_idx += 1
                    if model == "markovian":
                        vals = []
                        for r in range(cfg.mc_realizations_markovian):
                            if cfg.mc_realizations_markovian > 1 and cfg.noise_markov.jitter_T2_frac != 0.0:
                                rng = np.random.default_rng(cfg.random_seed_base + run_idx*1000 + r)
                                T2_this = T2 * (1.0 + cfg.noise_markov.jitter_T2_frac * rng.normal(0,1))
                                nm_r = build_markovian_noise_model(
                                    NoiseMarkovParams(T1_us=cfg.noise_markov.T1_us, T2_us=T2_this,
                                                      gate_time_us=cfg.noise_markov.gate_time_us,
                                                      cx_multiplier=cfg.noise_markov.cx_multiplier))
                                sim_r = AerSimulator(method='density_matrix', noise_model=nm_r)
                                vals.append(run_single_markovian(psi, t, hp, NoiseMarkovParams(T1_us=cfg.noise_markov.T1_us, T2_us=T2_this), cfg.dt_us, sim=sim_r, noise_model=nm_r))
                            else:
                                vals.append(run_single_markovian(psi, t, hp, noise_mark, cfg.dt_us, sim=sim_mark, noise_model=nm))
                        F = float(np.mean(vals))
                        S = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                    else:
                        vals = []
                        for r in range(cfg.mc_realizations_gaussian):
                            seed = cfg.random_seed_base + run_idx*cfg.mc_realizations_gaussian + r
                            rng = np.random.default_rng(int(seed))
                            vals.append(run_single_gaussian(psi, t, hp, NoiseGaussianParams(T2_us=cfg.noise_gauss.T2_us), cfg.dt_us, rng=rng))
                        F = float(np.mean(vals))
                        S = float(np.std(vals, ddof=1))
                    rec = {
                        'model': model,
                        'time_us': float(t),
                        'T2_us': float(T2),
                        'lambda_max_rad_per_us': float(lam),
                        'omega_rad_per_us': float(cfg.hparams.omega),
                        'drive_freq_rad_per_us': float(cfg.hparams.drive_freq),
                        'dt_us': float(cfg.dt_us),
                        'fidelity_mean': F,
                        'fidelity_std': S,
                        'n_realizations': cfg.mc_realizations_gaussian if model=="gaussian" else cfg.mc_realizations_markovian
                    }
                    results_by_model[model].append(rec)
                    fidelities.append(F)
                    stds.append(S)

                    
                    bar_len = 36
                    progress = run_idx / total_runs
                    filled_len = int(round(bar_len * progress))
                    bar = '█' * filled_len + '-' * (bar_len - filled_len)
                    sys.stdout.write(f'\r[{bar}] {run_idx}/{total_runs} ({model}) t={t:.1f}us F={F:.4f}')
                    sys.stdout.flush()
                    if run_idx == total_runs:
                        sys.stdout.write('\n')

                
                fname = os.path.join(model_folders[model], f"T2_{int(T2)}_lam_{lam:.5f}.png")
                plt.figure(figsize=(6.5,4.0))
                if SCIPY_AVAILABLE and len(cfg.scan.times_us) > 3:
                    interp_fn = interp1d(cfg.scan.times_us, fidelities, kind='cubic')
                    xu = np.linspace(cfg.scan.times_us[0], cfg.scan.times_us[-1], max(200, len(cfg.scan.times_us)*10))
                    yu = interp_fn(xu)
                    plt.plot(xu, yu, '-')
                else:
                    plt.plot(cfg.scan.times_us, fidelities, '-o', markersize=4)
                plt.xlabel(r'$t\ (\mu\mathrm{s})$'); plt.ylabel(r'$F(t)$')
                plt.ylim(0,1.02); plt.grid(True)
                plt.title(rf'$T_2={T2}\ \mu\mathrm{{s}},\ \lambda={lam/(2*math.pi):.3f}\ \mathrm{{MHz}}$')
                plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()

                
                pd.DataFrame(results_by_model[model]).to_csv(os.path.join(cfg.results_dir, f"results_{cfg.tag}_{model}.csv"), index=False)
        logging.info("Model '%s' complete.", model)

    
    df_by_model = {m: pd.DataFrame(results_by_model[m]) for m in results_by_model}
    combined = pd.concat([df_by_model[m].assign(model=m) for m in df_by_model], ignore_index=True)
    combined.to_csv(os.path.join(cfg.results_dir, f"combined_results_{cfg.tag}.csv"), index=False)

   
    overlays_dir = os.path.join(cfg.results_dir, "overlays")
    for T2 in cfg.scan.T2_list_us:
        for lam in cfg.scan.lambda_list_rad_per_us:
            series_dict = {}
            times = cfg.scan.times_us
            for model in cfg.compare_models:
                dfm = df_by_model[model]
                sel = dfm[(dfm['T2_us']==T2) & (dfm['lambda_max_rad_per_us']==lam)].sort_values('time_us')
                if sel.empty: continue
                series_dict[model] = sel['fidelity_mean'].values
            if series_dict:
                fname = os.path.join(overlays_dir, f"overlay_T2_{int(T2)}_lam_{lam:.5f}_{cfg.tag}.png")
                plot_overlay_time_series(times, series_dict, cfg, fname=fname)

    
    t_sel = float(cfg.scan.times_us[-1])
    for model in cfg.compare_models:
        dfm = df_by_model[model].rename(columns={'fidelity_mean':'fidelity'})
        plot_heatmap_basic(dfm, t_sel, cfg, model_label=model,
                           out_path=os.path.join(cfg.results_dir, "heatmaps", f"heatmap_{model}_t{int(t_sel)}_{cfg.tag}.png"))

    return df_by_model, combined

def plot_overlay_time_series(times, series_dict: Dict[str, np.ndarray], cfg: RunConfig, fname=None):
    plt.figure(figsize=(7,4.5))
    for model, arr in series_dict.items():
        if SCIPY_AVAILABLE and len(times) > 3:
            interp_fn = interp1d(times, arr, kind='cubic')
            xu = np.linspace(times[0], times[-1], max(200, len(times)*10))
            yu = interp_fn(xu)
            plt.plot(xu, yu, label=model)
        else:
            plt.plot(times, arr, '-', label=model)
    plt.xlabel(r'$t\ (\mu\mathrm{s})$')
    plt.ylabel(r'$F(t)$')
    plt.ylim(0,1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Fidelity: overlay Gaussian vs Markovian')
    if fname:
        plt.tight_layout(); plt.savefig(fname, dpi=300); plt.close()
    else:
        plt.show()

def plot_heatmap_basic(df: pd.DataFrame, t_sel: float, cfg: RunConfig, model_label: str, out_path: str):
    df_t = df[df['time_us'] == t_sel]
    if df_t.empty:
        logging.warning("No data at t=%.3f for %s", t_sel, model_label)
        return
    pivot = df_t.pivot(index='T2_us', columns='lambda_max_rad_per_us', values='fidelity')
    X = np.array([c/(2*math.pi) for c in pivot.columns]); Y = np.array(pivot.index); Z = pivot.values
    plt.figure(figsize=(7,5))
    im = plt.imshow(Z, origin='lower', aspect='auto', extent=(X.min(),X.max(),Y.min(),Y.max()),
                    vmin=0, vmax=1, cmap='viridis')
    cbar = plt.colorbar(im)
    cbar.set_label(fr'$F\ (t={t_sel:.1f}\ \mu\mathrm{{s}})$')
    try:
        if SCIPY_AVAILABLE:
            CS = plt.contour(np.repeat(X[np.newaxis,:], len(Y), axis=0).T,
                             np.repeat(Y[:,np.newaxis], len(X), axis=1),
                             Z, colors='w', linewidths=0.6, levels=[0.5,0.75,0.9], alpha=0.9)
            plt.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
    except Exception:
        pass
    plt.xlabel(r'$\lambda_{\max}\ (\mathrm{MHz})$')
    plt.ylabel(r'$T_2\ (\mu\mathrm{s})$')
    plt.title(f"{model_label} mean heatmap")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()


def make_plot_dirs(base: str) -> Dict[str,str]:
    paths = {
        "mean": make_dir(os.path.join(base, "Mean fidelity")),
        "difference": make_dir(os.path.join(base, "Difference")),
        "error_bands": make_dir(os.path.join(base, "Error bands")),
        "uncertainty": make_dir(os.path.join(base, "Uncertainty")),
        "sensitivity": make_dir(os.path.join(base, "Sensitivity")),
        "scatter": make_dir(os.path.join(base, "Scatter")),
        "bland_altman": make_dir(os.path.join(base, "BlandAltman")),
        "cv": make_dir(os.path.join(base, "CV")),
        "fft": make_dir(os.path.join(base, "FFT")),
        "phase": make_dir(os.path.join(base, "Phase diagrams")),
    }
    return paths


def plot_mean_heatmaps(df_by_model, cfg, outdir, times):
    for model, dfm in df_by_model.items():
        for t_sel in times:
            df_t = dfm[np.isclose(dfm['time_us'], t_sel)]
            if df_t.empty: continue
            pivot = df_t.pivot(index='T2_us', columns='lambda_max_rad_per_us', values='fidelity_mean')
            X = np.array([c/(2*math.pi) for c in pivot.columns])
            Y = np.array(pivot.index)
            Z = pivot.values
            plt.figure(figsize=(7,5))
            cs = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
            cbar = plt.colorbar(cs)
            cbar.set_label(fr'$F\ (t={t_sel:.1f}\ \mu\mathrm{{s}})$')
            CS = plt.contour(X, Y, Z, levels=[0.5,0.75,0.9], colors='white', linewidths=0.8)
            plt.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
            plt.xlabel(r'$\lambda_{\max}\ (\mathrm{MHz})$')
            plt.ylabel(r'$T_2\ (\mu\mathrm{s})$')
            plt.title(f"{model} mean fidelity")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"mean_contour_{model}_t{int(t_sel)}_{cfg.tag}.png"), dpi=300)
            plt.close()



def plot_difference_heatmap(combined, t_sel, cfg, outdir):
    dfg = combined[(combined['model']=='gaussian') & (np.isclose(combined['time_us'], t_sel))]
    dfm = combined[(combined['model']=='markovian') & (np.isclose(combined['time_us'], t_sel))]
    if dfg.empty or dfm.empty: return
    merged = pd.merge(
        dfg[['T2_us','lambda_max_rad_per_us','fidelity_mean']],
        dfm[['T2_us','lambda_max_rad_per_us','fidelity_mean']],
        on=['T2_us','lambda_max_rad_per_us'], suffixes=('_g','_m')
    )
    merged['delta'] = merged['fidelity_mean_g'] - merged['fidelity_mean_m']
    pivot = merged.pivot(index='T2_us', columns='lambda_max_rad_per_us', values='delta')
    X = np.array([c/(2*math.pi) for c in pivot.columns])
    Y = np.array(pivot.index)
    Z = pivot.values
    vmax = np.nanmax(np.abs(Z))
    plt.figure(figsize=(7,5))
    cs = plt.contourf(X, Y, Z, levels=50, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(cs)
    cbar.set_label(r'$\Delta F$')
    plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=1.2)
    plt.xlabel(r'$\lambda_{\max}\ (\mathrm{MHz})$')
    plt.ylabel(r'$T_2\ (\mu\mathrm{s})$')
    plt.title(f'Difference (Gauss − Markov) at t={t_sel:.1f} μs')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"diff_contour_t{int(t_sel)}_{cfg.tag}.png"), dpi=300)
    plt.close()


def plot_error_bands(df_by_model, cfg, outdir):
    for model, dfm in df_by_model.items():
        for T2 in cfg.scan.T2_list_us:
            for lam in cfg.scan.lambda_list_rad_per_us:
                sel = dfm[(dfm['T2_us']==T2) & (dfm['lambda_max_rad_per_us']==lam)].sort_values('time_us')
                if sel.empty: continue
                t = sel['time_us'].values
                y = sel['fidelity_mean'].values
                s = sel['fidelity_std'].values
                plt.figure(figsize=(7,4))
                plt.plot(t, y, '-o', label='mean', markersize=4)
                plt.fill_between(t, y-s, y+s, alpha=0.25, label='±1σ')
                plt.ylim(0,1.02)
                plt.xlabel(r'$t\ (\mu\mathrm{s})$')
                plt.ylabel(r'$F(t)$')
                plt.title(f'{model}  T2={T2} μs  λ={lam/(2*math.pi):.3f} MHz')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"bands_{model}_T2_{int(T2)}_lam_{lam:.5f}_{cfg.tag}.png"), dpi=300)
                plt.close()


from mpl_toolkits.mplot3d import Axes3D
def plot_uncertainty_heatmap(df_by_model, t_sel, cfg, outdir):
    if 'gaussian' not in df_by_model:
        return
    df_t = df_by_model['gaussian'][np.isclose(df_by_model['gaussian']['time_us'], t_sel)]
    if df_t.empty:
        return

    pivot = df_t.pivot(index='T2_us', columns='lambda_max_rad_per_us', values='fidelity_std')
    lam_vals = np.array(pivot.columns) / (2*np.pi)  # MHz
    T2_vals = np.array(pivot.index)
    X, Y = np.meshgrid(lam_vals, T2_vals)
    Z = pivot.values

    #3D surface plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='magma')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=r'$\sigma_F$')
    ax.set_xlabel(r'$\lambda_{\max}$ (MHz)')
    ax.set_ylabel(r'$T_2$ (μs)')
    ax.set_zlabel(r'$\sigma_F$')
    ax.set_title(f'Uncertainty surface (Gaussian) at t={t_sel:.1f} μs')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"uncertainty_surface_t{int(t_sel)}_{cfg.tag}.png"), dpi=300)
    plt.close()

    #2D contour
    plt.figure(figsize=(7, 5))
    im = plt.imshow(Z, origin='lower', aspect='auto',
                    extent=(lam_vals.min(), lam_vals.max(), T2_vals.min(), T2_vals.max()),
                    cmap='magma')
    cbar = plt.colorbar(im)
    cbar.set_label(r'$\sigma_F$')
    
    CS = plt.contour(X, Y, Z, colors='white', linewidths=0.8, levels=5, alpha=0.9)
    plt.clabel(CS, inline=True, fontsize=8, fmt="%.3f")
    plt.xlabel(r'$\lambda_{\max}$ (MHz)')
    plt.ylabel(r'$T_2$ (μs)')
    plt.title(f'Uncertainty heatmap (Gaussian) at t={t_sel:.1f} μs')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"uncertainty_heatmap_t{int(t_sel)}_{cfg.tag}.png"), dpi=300)
    plt.close()


def compute_sensitivity(df: pd.DataFrame, t_sel: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    df_t = df[np.isclose(df['time_us'], t_sel)]
    if df_t.empty: return None, None, None, None
    pivot = df_t.pivot(index='T2_us', columns='lambda_max_rad_per_us', values='fidelity_mean')
    T2_vals = np.array(pivot.index); lam_vals = np.array(pivot.columns)
    Z = pivot.values
    dT = np.zeros_like(Z); dL = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            dT[i,j] = (Z[i+1,j]-Z[i-1,j])/(T2_vals[i+1]-T2_vals[i-1]) if 0<i<Z.shape[0]-1 else 0.0
            dL[i,j] = (Z[i,j+1]-Z[i,j-1])/(lam_vals[j+1]-lam_vals[j-1]) if 0<j<Z.shape[1]-1 else 0.0
    return T2_vals, lam_vals, np.abs(dT), np.abs(dL)

def plot_sensitivity_heatmaps(df_by_model, t_sel, cfg, outdir):
    for model, dfm in df_by_model.items():
        df_t = dfm[np.isclose(dfm['time_us'], t_sel)]
        if df_t.empty: continue
        pivot = df_t.pivot(index='T2_us', columns='lambda_max_rad_per_us', values='fidelity_mean')
        T2_vals = np.array(pivot.index)
        lam_vals = np.array(pivot.columns)
        Z = pivot.values
        dT, dL = np.gradient(Z, T2_vals, lam_vals)   # returns (∂Z/∂T2, ∂Z/∂λ)
        X, Y = np.meshgrid(lam_vals/(2*math.pi), T2_vals)
        plt.figure(figsize=(7,5))
        plt.quiver(X, Y, dL, dT, np.sqrt(dL**2 + dT**2), cmap='plasma', scale=20)
        plt.colorbar(label=r'$|\nabla F|$')
        plt.xlabel(r'$\lambda_{\max}$ (MHz)')
        plt.ylabel(r'$T_2$ (μs)')
        plt.title(f'Sensitivity vectors — {model} at t={t_sel:.1f} μs')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"sens_quiver_{model}_t{int(t_sel)}_{cfg.tag}.png"), dpi=300)
        plt.close()

def plot_model_scatter(combined: pd.DataFrame, t_sel: float, cfg: RunConfig, outdir: str):
    dfg = combined[(combined['model']=='gaussian') & (np.isclose(combined['time_us'], t_sel))]
    dfm = combined[(combined['model']=='markovian') & (np.isclose(combined['time_us'], t_sel))]
    if dfg.empty or dfm.empty:
        logging.warning("No data for scatter at t=%.3f", t_sel); return
    merged = pd.merge(
        dfg[['T2_us','lambda_max_rad_per_us','fidelity_mean']],
        dfm[['T2_us','lambda_max_rad_per_us','fidelity_mean']],
        on=['T2_us','lambda_max_rad_per_us'], suffixes=('_g','_m')
    )
    plt.figure(figsize=(5.6,5.2))
    sc = plt.scatter(merged['fidelity_mean_m'], merged['fidelity_mean_g'], c=merged['T2_us'], cmap='viridis', s=55, edgecolor='k', linewidth=0.3)
    lims = [0,1]; plt.plot(lims, lims, 'k--', linewidth=1)
    cbar = plt.colorbar(sc); cbar.set_label(r'$T_2\ (\mu s)$')
    plt.xlabel('Markovian F'); plt.ylabel('Gaussian F'); plt.title(f'Agreement at t={t_sel:.1f} μs')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"scatter_agreement_t{int(round(t_sel))}_{cfg.tag}.png"), dpi=300); plt.close()

def plot_bland_altman(combined, t_sel, cfg, outdir):
    dfg = combined[(combined['model']=='gaussian') & (np.isclose(combined['time_us'], t_sel))]
    dfm = combined[(combined['model']=='markovian') & (np.isclose(combined['time_us'], t_sel))]
    if dfg.empty or dfm.empty: return
    merged = pd.merge(
        dfg[['T2_us','lambda_max_rad_per_us','fidelity_mean']],
        dfm[['T2_us','lambda_max_rad_per_us','fidelity_mean']],
        on=['T2_us','lambda_max_rad_per_us'], suffixes=('_g','_m')
    )
    mean = 0.5 * (merged['fidelity_mean_g'] + merged['fidelity_mean_m'])
    diff = merged['fidelity_mean_g'] - merged['fidelity_mean_m']
    mu = diff.mean()
    sd = diff.std(ddof=1)
    plt.figure(figsize=(6.4,4.6))
    plt.scatter(mean, diff, c=merged['lambda_max_rad_per_us']/(2*np.pi), cmap='plasma', s=50, edgecolor='k', linewidth=0.3)
    plt.axhline(mu, color='k', linestyle='-', label='Mean diff')
    plt.axhline(mu + 1.96*sd, color='k', linestyle='--', label='+1.96σ')
    plt.axhline(mu - 1.96*sd, color='k', linestyle='--', label='−1.96σ')
    plt.fill_between(mean, mu - 1.96*sd, mu + 1.96*sd, color='gray', alpha=0.2)
    plt.xlabel('Mean fidelity')
    plt.ylabel('Difference (Gauss − Markov)')
    plt.title(f'Bland–Altman at t={t_sel:.1f} μs')
    plt.colorbar(label=r'$\lambda_{\max}$ (MHz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bland_altman_shaded_t{int(t_sel)}_{cfg.tag}.png"), dpi=300)
    plt.close()

def plot_cv_heatmap(df_by_model, t_sel, cfg, outdir):
    if 'gaussian' not in df_by_model: return
    df_t = df_by_model['gaussian'][np.isclose(df_by_model['gaussian']['time_us'], t_sel)].copy()
    if df_t.empty: return
    df_t['CV'] = df_t['fidelity_std'] / df_t['fidelity_mean'].replace(0, np.nan)
    df_t['lam_MHz'] = df_t['lambda_max_rad_per_us'] / (2*np.pi)
    plt.figure(figsize=(7,5))
    sizes = 800 * df_t['CV'].fillna(0)**2
    scatter = plt.scatter(df_t['lam_MHz'], df_t['T2_us'], s=sizes, c=df_t['CV'], cmap='inferno', alpha=0.7, edgecolor='k')
    plt.colorbar(scatter, label='CV = σF / meanF')
    plt.xlabel(r'$\lambda_{\max}$ (MHz)')
    plt.ylabel(r'$T_2$ (μs)')
    plt.title(f'Coefficient of variation (bubble chart) at t={t_sel:.1f} μs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"cv_bubble_t{int(t_sel)}_{cfg.tag}.png"), dpi=300)
    plt.close()

def plot_fft_panel(df_by_model, cfg, outdir, T2_val, lambdas):
    for model, dfm in df_by_model.items():
        plt.figure(figsize=(7,4))
        for lam in lambdas:
            sel = dfm[(dfm['T2_us']==T2_val) & (dfm['lambda_max_rad_per_us']==lam)].sort_values('time_us')
            if sel.shape[0] < 4: continue
            t = sel['time_us'].values
            y = sel['fidelity_mean'].values
            y0 = y - np.mean(y)
            dt = np.median(np.diff(t))
            freqs = np.fft.rfftfreq(len(y0), d=dt)
            amp = np.abs(np.fft.rfft(y0))
            plt.plot(freqs, amp/amp.max(), label=f'λ={lam/(2*math.pi):.3f} MHz')
        # Add shaded bands
        for f in [0.05, 0.1, 0.2]:
            plt.axvspan(f-0.01, f+0.01, color='gray', alpha=0.1)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Normalized amplitude')
        plt.title(f'FFT of F(t), {model}, T₂={T2_val:.0f} μs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'fft_shaded_{model}_T2_{int(T2_val)}_{cfg.tag}.png'), dpi=300)
        plt.close()

def plot_phase_diagrams(df_by_model, cfg, outdir, times):
    for model, dfm in df_by_model.items():
        for t_sel in times:
            df_t = dfm[np.isclose(dfm['time_us'], t_sel)]
            if df_t.empty: continue
            pivot = df_t.pivot(index='T2_us', columns='lambda_max_rad_per_us', values='fidelity_mean')
            X = np.array([c/(2*np.pi) for c in pivot.columns])
            Y = np.array(pivot.index)
            Z = pivot.values
            plt.figure(figsize=(7,5))

            plt.contourf(X, Y, Z, levels=50, cmap='Greys', alpha=0.15)

            levels = [0.3, 0.5, 0.7, 0.9]
            CS = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=1.2)
            plt.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
            plt.xlabel(r'$\lambda_{\max}$ (MHz)')
            plt.ylabel(r'$T_2$ (μs)')
            plt.title(f'Phase boundaries — {model} at t={t_sel:.1f} μs')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'phase_contour_{model}_t{int(t_sel)}_{cfg.tag}.png'), dpi=300)
            plt.close()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    cfg = default_runconfig()
    logging.info("Running with config: %s", cfg)

    # Validation
    F0 = noiseless_baseline_check()
    if abs(F0 - 1.0) > 1e-9:
        logging.warning("Noiseless baseline deviates from 1.0: F0=%.12f", F0)

    df_by_model, combined = run_full_comparison(cfg)

    P = make_plot_dirs(cfg.results_dir)

    snapshots = [cfg.scan.times_us[-1]]  
    plot_mean_heatmaps(df_by_model, cfg, P["mean"], snapshots)

    t_sel = float(cfg.scan.times_us[-1])
    plot_difference_heatmap(combined, t_sel, cfg, P["difference"])

    plot_error_bands(df_by_model, cfg, P["error_bands"])

    plot_uncertainty_heatmap(df_by_model, t_sel, cfg, P["uncertainty"])

    plot_sensitivity_heatmaps(df_by_model, t_sel, cfg, P["sensitivity"])

    plot_model_scatter(combined, t_sel, cfg, P["scatter"])

    plot_bland_altman(combined, t_sel, cfg, P["bland_altman"])

    plot_cv_heatmap(df_by_model, t_sel, cfg, P["cv"])

    chosen_T2 = cfg.scan.T2_list_us[len(cfg.scan.T2_list_us)//2]
    lambdas_for_fft = cfg.scan.lambda_list_rad_per_us
    plot_fft_panel(df_by_model, cfg, P["fft"], chosen_T2, lambdas_for_fft)

    multi_times = [cfg.scan.times_us[i] for i in [0, max(1,len(cfg.scan.times_us)//3), max(2,2*len(cfg.scan.times_us)//3), len(cfg.scan.times_us)-1]]
    plot_phase_diagrams(df_by_model, cfg, P["phase"], multi_times)

    logging.info("All figures generated. See: %s", os.path.abspath(cfg.results_dir))

if __name__ == "__main__":
    main()