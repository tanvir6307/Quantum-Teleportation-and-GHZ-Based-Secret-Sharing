from __future__ import annotations
import os, math, logging
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy.interpolate import griddata, interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs): return iterator

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit_aer import AerSimulator


@dataclass
class HParams:
    omega: float            # rad/us
    lambda_max: float       # rad/us
    drive_freq: float       # rad/us
    phase: float = 0.0
    V_func: Optional[Callable[[float], np.ndarray]] = None

@dataclass
class NoiseCollectiveParams:
    sigma_phi: float
    tau_us: float
    phase_qubits: List[int] = None

@dataclass
class AmpDampParams:
    T1_us: float
    damp_qubits: List[int] = None 

@dataclass
class ScanConfig:
    times_us: np.ndarray
    T2_list_us: list
    lambda_list_rad_per_us: np.ndarray

@dataclass
class MetricsConfig:
    compute_ghz_fidelity: bool = False
    use_ghz_target: bool = False

@dataclass
class RunConfig:
    hparams: HParams
    noise_collective: NoiseCollectiveParams
    noise_amp: AmpDampParams
    scan: ScanConfig
    metrics: MetricsConfig
    dt_us: float = 0.2
    results_dir: str = "results_GHZ"
    tag: str = "ampdamp_vs_collective"
    mc_realizations_collective: int = 40
    compare_models: list = None
    random_seed_base: int = 12345
    smooth_heatmap_with_scipy: bool = True


I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
PROJ0 = np.array([[1, 0], [0, 0]], dtype=complex)
PROJ1 = np.array([[0, 0], [0, 1]], dtype=complex)
PROJ_PLUS  = 0.5 * np.array([[1,  1],[ 1, 1]], dtype=complex)  
PROJ_MINUS = 0.5 * np.array([[1, -1],[-1, 1]], dtype=complex)  

def fidelity_from_rho_bob(rho_bob: DensityMatrix, psi_initial: np.ndarray) -> float:
    """F = <psi| rho_bob |psi> for pure |psi>."""
    if not isinstance(rho_bob, DensityMatrix):
        rho_bob = DensityMatrix(rho_bob)
    psi_vec = psi_initial.reshape(-1, 1).astype(complex)
    val = psi_vec.conj().T @ rho_bob.data @ psi_vec
    return float(np.real(val[0, 0]))

def embed_single_qubit_op_reversed(op: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    
    mats = []
    for q in reversed(range(n_qubits)):
        mats.append(op if q == target else I2)
    M = mats[0]
    for k in range(1, len(mats)):
        M = np.kron(M, mats[k])
    return M


# Amplitude damping (Kraus)


def amplitude_damping_kraus(gamma: float):
    gamma = min(max(float(gamma), 0.0), 1.0)
    K0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]

def apply_kraus_single_qubit(rho: np.ndarray, kraus_ops: list, target: int, n_qubits: int):
    d = 2**n_qubits
    out = np.zeros((d, d), dtype=complex)
    for K in kraus_ops:
        K_full = embed_single_qubit_op_reversed(K, target, n_qubits)
        out += K_full @ rho @ K_full.conj().T
    return out


# Collective dephasing (OU correlated Z rotations)


def generate_collective_phases_OU(total_time: float, dt: float, sigma_phi: float, tau: float) -> np.ndarray:
    """Discrete OU: phi_k = a*phi_{k-1} + N(0, sigma_innov^2), with a = exp(-dt/tau)."""
    if dt <= 0 or total_time <= 0:
        return np.array([])
    n_steps = max(1, int(round(total_time / dt)))
    phases = np.zeros(n_steps, dtype=float)
    a = math.exp(-dt / max(tau, 1e-12))
    innov_std = sigma_phi * math.sqrt(max(1e-12, 1.0 - a * a))
    phi = 0.0
    for k in range(n_steps):
        phi = a * phi + np.random.normal(0.0, innov_std)
        phases[k] = phi
    return phases

def apply_collective_phase(rho: np.ndarray, phi: float, qubits: list, n_qubits: int):
    """Apply exp(-i phi Z/2) on specified qubits, correlated (same phi)."""
    U_single = np.array([[np.exp(-1j*phi/2), 0],[0, np.exp(1j*phi/2)]], dtype=complex)
    mats = []
    for q in reversed(range(n_qubits)):
        mats.append(U_single if q in qubits else I2)
    U = mats[0]
    for m in mats[1:]:
        U = np.kron(U, m)
    return U @ rho @ U.conj().T

def sigma_phi_from_T2(dt_us: float, T2_us: float) -> float:
    """Map T2 to per-step phase std (Gaussian small-angle analogy)."""
    if T2_us <= 0:
        return 0.0
    return math.sqrt(max(1e-12, 2.0 * dt_us / T2_us))


# GHZ circuit builder (pre-measure)

def ghz_secret_sharing_circuit_pre_measure(psi: np.ndarray) -> QuantumCircuit:
    """
    Qubits:
      q0: Alice's secret |ψ>
      q1: Alice's GHZ share
      q2: Bob's GHZ share
      q3: Charlie's GHZ share
    """
    qc = QuantumCircuit(4)
    qc.initialize(psi, 0)
    # GHZ on q1, q2, q3
    qc.h(1); qc.cx(1, 2); qc.cx(1, 3)
    # Alice's Bell pre-ops on (q0,q1)
    qc.cx(0, 1); qc.h(0)
    # Charlie to X basis
    qc.h(3)
    return qc


# GHZ correction (projective average + Pauli corrections)

def ghz_projective_average_and_correct(rho_full: DensityMatrix) -> DensityMatrix:
    """
    Project q0, q1 in Z; q3 in Z (because qc.h(3) was applied earlier),
    apply Bob's correction X^{m1} Z^{m0 ⊕ c}.
    Embedding matches MSB→LSB convention (embed_single_qubit_op_reversed).
    """
    rho = rho_full.data
    nq = 4
    out = np.zeros_like(rho, dtype=complex)

    Pz = {0: PROJ0, 1: PROJ1}
    for m0 in (0, 1):
        for m1 in (0, 1):
            for c in (0, 1):
                P0 = embed_single_qubit_op_reversed(Pz[m0], target=0, n_qubits=nq)
                P1 = embed_single_qubit_op_reversed(Pz[m1], target=1, n_qubits=nq)
                Pc = embed_single_qubit_op_reversed(Pz[c], target=3, n_qubits=nq)  # computational basis on saved rho
                P = Pc @ P1 @ P0
                piece = P @ rho @ P.conj().T

                #apply X^{m1} then Z^{m0 ⊕ c}
                x_pow = m1
                z_pow = (m0 ^ c)
                Ucorr = (X2 if x_pow else I2) @ (Z2 if z_pow else I2)

                U = embed_single_qubit_op_reversed(Ucorr, target=2, n_qubits=nq)
                out += U @ piece @ U.conj().T

    return DensityMatrix(out)



def run_single_ampdamp(psi: np.ndarray, total_time_us: float, hp: HParams, T1_us: float,
                       cfg: RunConfig, dt_us: float, damp_qubits: Optional[List[int]] = None) -> tuple[float, float]:
    
    qc = ghz_secret_sharing_circuit_pre_measure(psi)
    qc.barrier()
    qc.save_density_matrix(label='rho')
    sim = AerSimulator(method='density_matrix')
    res = sim.run(transpile(qc, sim), shots=1).result()
    rho4 = DensityMatrix(res.data(qc)['rho'])

    if damp_qubits is None:
        damp_qubits = cfg.noise_amp.damp_qubits if cfg.noise_amp.damp_qubits is not None else [1, 2, 3]

    steps = max(1, int(round(total_time_us / dt_us)))
    gamma_step = 1.0 - math.exp(-dt_us / max(T1_us, 1e-12))
    kraus = amplitude_damping_kraus(gamma_step)

    rho = rho4.data
    for _ in range(steps):
        for q in damp_qubits:
            rho = apply_kraus_single_qubit(rho, kraus, q, 4)

    rho_corrected = ghz_projective_average_and_correct(DensityMatrix(rho))
    rho_bob = partial_trace(rho_corrected, [0, 1, 3])
    F_bob = fidelity_from_rho_bob(rho_bob, psi)
    return F_bob, None

def run_single_collective(psi: np.ndarray, total_time_us: float, hp: HParams,
                          sigma_phi: float, tau_us: float, cfg: RunConfig, dt_us: float,
                          phase_qubits: Optional[List[int]] = None) -> tuple[float, float]:
    
    qc = ghz_secret_sharing_circuit_pre_measure(psi)
    qc.barrier()
    qc.save_density_matrix(label='rho')
    sim = AerSimulator(method='density_matrix')
    res = sim.run(transpile(qc, sim), shots=1).result()
    rho4 = DensityMatrix(res.data(qc)['rho'])

    if phase_qubits is None:
        phase_qubits = cfg.noise_collective.phase_qubits if cfg.noise_collective.phase_qubits is not None else [1, 2, 3]

    phases = generate_collective_phases_OU(total_time_us, dt_us, sigma_phi, tau_us)
    rho = rho4.data
    for phi in phases:
        rho = apply_collective_phase(rho, phi, phase_qubits, 4)

    rho_corrected = ghz_projective_average_and_correct(DensityMatrix(rho))
    rho_bob = partial_trace(rho_corrected, [0, 1, 3])
    F_bob = fidelity_from_rho_bob(rho_bob, psi)
    return F_bob, None


def noiseless_baseline_check() -> float:
    psi = Statevector.from_label('+').data
    qc = ghz_secret_sharing_circuit_pre_measure(psi)
    qc.barrier(); qc.save_density_matrix(label='rho_full')
    sim = AerSimulator(method='density_matrix')
    res = sim.run(transpile(qc, sim), shots=1).result()
    rho_full = DensityMatrix(res.data(qc)['rho_full'])
    rho_after = ghz_projective_average_and_correct(rho_full)
    rho_bob = partial_trace(rho_after, [0,1,3])
    F_bob = fidelity_from_rho_bob(rho_bob, psi)
    return F_bob


def run_full_comparison(cfg: RunConfig) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    psi_initial = np.array([1, 1])/math.sqrt(2)
    all_models_to_run = cfg.compare_models
    results = {model: [] for model in all_models_to_run}

    param_scan = [(T2, lam) for T2 in cfg.scan.T2_list_us for lam in cfg.scan.lambda_list_rad_per_us]

    for (T2, lam) in tqdm(param_scan, desc="Running Simulation Scan"):
        i_T2 = cfg.scan.T2_list_us.index(T2)
        i_lam = np.where(cfg.scan.lambda_list_rad_per_us == lam)[0][0]

        for model in all_models_to_run:
            realizations_bob = []

            if model == 'ampdamp':
                n_realizations = 1
                T1_us = T2 / 2.0
                runner = lambda psi, t, hp, cfg, dt: run_single_ampdamp(psi, t, hp, T1_us, cfg, dt)
            elif model == 'collective':
                n_realizations = cfg.mc_realizations_collective
                sigma_phi_eff = sigma_phi_from_T2(cfg.dt_us, T2)
                tau_us = cfg.noise_collective.tau_us
                runner = lambda psi, t, hp, cfg, dt: run_single_collective(psi, t, hp, sigma_phi_eff, tau_us, cfg, dt)
            else:
                raise ValueError(f"Unknown model type: {model}")

            for i_real in range(n_realizations):
                np.random.seed(cfg.random_seed_base + i_T2 * 1000 + i_lam * 100 + i_real)
                fidelities_t_bob = []
                for t in cfg.scan.times_us:
                    F_bob, _ = runner(psi_initial, t, cfg.hparams, cfg, cfg.dt_us)
                    fidelities_t_bob.append(F_bob)
                realizations_bob.append(fidelities_t_bob)

            results[model].append({'T2': T2, 'lambda': lam, 'fidelities': realizations_bob})

    df_by_model = {}
    for model in all_models_to_run:
        df = pd.DataFrame(results[model])
        df['mean_fid'] = df['fidelities'].apply(lambda x: np.mean(x, axis=0))
        df['std_fid'] = df['fidelities'].apply(lambda x: np.std(x, axis=0) if len(x) > 1 else np.zeros_like(np.mean(x, axis=0)))
        df_by_model[model] = df

    combined_list = []
    if len(all_models_to_run) > 1:
        for i in range(len(df_by_model[all_models_to_run[0]])):
            base_row = df_by_model[all_models_to_run[0]].iloc[i]
            T2, lam = base_row['T2'], base_row['lambda']
            entry = {'T2': T2, 'lambda': lam}
            for model in all_models_to_run:
                model_row = df_by_model[model].iloc[i]
                entry[f'mean_fid_{model}'] = model_row['mean_fid']
                entry[f'std_fid_{model}'] = model_row['std_fid']
            combined_list.append(entry)
    combined_df = pd.DataFrame(combined_list)
    return df_by_model, combined_df, None


def make_plot_dirs(base_dir: str) -> dict:
    os.makedirs(base_dir, exist_ok=True)
    plot_dirs = {
        "ampdamp_curves": f"{base_dir}/ampdamp_curves",
        "collective_curves": f"{base_dir}/collective_curves",
        "error_bands": f"{base_dir}/error_bands",
        "overlays": f"{base_dir}/overlays",
        "ampdamp_sensitivity": f"{base_dir}/ampdamp_sensitivity",
        "collective_tau_sensitivity": f"{base_dir}/collective_tau_sensitivity",
    }
    for d in plot_dirs.values():
        os.makedirs(d, exist_ok=True)
    return plot_dirs

def plot_ampdamp_curves(df_by_model: dict, cfg: RunConfig, savedir: str):
    logging.info(f"Plotting amplitude damping curves to {savedir}")
    if 'ampdamp' not in df_by_model:
        return
        
    df = df_by_model['ampdamp']
    for T2 in cfg.scan.T2_list_us:
        for lam in cfg.scan.lambda_list_rad_per_us:
            row = df[(df['T2'] == T2) & (df['lambda'] == lam)].iloc[0]
            mean_fid = row['mean_fid']
            
            plt.figure(figsize=(10, 8))
            plt.plot(cfg.scan.times_us, mean_fid, linewidth=2)
            
            # Convert lambda to MHz for display
            lam_MHz = lam / (2 * math.pi)
            
            plt.xlabel(r'Time ($\mu$s)', fontsize=16)
            plt.ylabel('Fidelity', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(rf'Amplitude Damping: $T_2$={T2} $\mu$s, $\lambda$={lam_MHz:.3f} MHz', fontsize=18)
            plt.ylim(0.0, 1.05)
            plt.grid(True, linestyle=':')
            plt.tight_layout()
            
            filename = f"ampdamp_T2_{int(T2)}_lam_{lam_MHz:.3f}MHz.png"
            plt.savefig(f"{savedir}/{filename}", dpi=300)
            plt.close()

def plot_collective_curves(df_by_model: dict, cfg: RunConfig, savedir: str):
    logging.info(f"Plotting collective dephasing curves to {savedir}")
    if 'collective' not in df_by_model:
        return
        
    df = df_by_model['collective']
    for T2 in cfg.scan.T2_list_us:
        for lam in cfg.scan.lambda_list_rad_per_us:
            row = df[(df['T2'] == T2) & (df['lambda'] == lam)].iloc[0]
            mean_fid = row['mean_fid']
            
            plt.figure(figsize=(10, 8))
            plt.plot(cfg.scan.times_us, mean_fid, linewidth=2)
            
            # Convert lambda to MHz for display
            lam_MHz = lam / (2 * math.pi)
            
            plt.xlabel(r'Time ($\mu$s)', fontsize=14)
            plt.ylabel('Fidelity', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(rf'Collective Dephasing: $T_2$={T2} $\mu$s, $\lambda$={lam_MHz:.3f} MHz', fontsize=16)
            plt.ylim(0.0, 1.05)
            plt.grid(True, linestyle=':')
            plt.tight_layout()
            
            filename = f"collective_T2_{int(T2)}_lam_{lam_MHz:.3f}MHz.png"
            plt.savefig(f"{savedir}/{filename}", dpi=300)
            plt.close()

def plot_error_bands(df_by_model: dict, cfg: RunConfig, savedir: str):
    logging.info(f"Plotting error bands to {savedir}")
    for T2 in cfg.scan.T2_list_us:
        for lam in cfg.scan.lambda_list_rad_per_us:
            plt.figure(figsize=(10, 8))
            
            # Convert lambda to MHz for display
            lam_MHz = lam / (2 * math.pi)
            
            for model, df in df_by_model.items():
                row = df[(df['T2'] == T2) & (df['lambda'] == lam)].iloc[0]
                mean_fid = row['mean_fid']
                std_fid = row['std_fid']
                plt.plot(cfg.scan.times_us, mean_fid, label=model, linewidth=2)
                if np.any(std_fid > 0):
                    plt.fill_between(cfg.scan.times_us, mean_fid - std_fid, mean_fid + std_fid, alpha=0.2)

            plt.xlabel(r'Time ($\mu$s)', fontsize=16)
            plt.ylabel('Fidelity', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(rf'Fidelity vs Time: $T_2$={T2} $\mu$s, $\lambda$={lam_MHz:.3f} MHz', fontsize=18)
            plt.ylim(0.0, 1.05)
            plt.grid(True, linestyle=':')
            plt.legend(fontsize=14)
            plt.tight_layout()
            
            filename = f"error_bands_T2_{int(T2)}_lam_{lam_MHz:.3f}MHz.png"
            plt.savefig(f"{savedir}/{filename}", dpi=300)
            plt.close()

def plot_overlay_time_series(df_by_model: dict, cfg: RunConfig, savedir: str):
    logging.info(f"Plotting overlay time series to {savedir}")
    for T2 in cfg.scan.T2_list_us:
        for lam in cfg.scan.lambda_list_rad_per_us:
            plt.figure(figsize=(10, 8))
            
            lam_MHz = lam / (2 * math.pi)
            
            for model, df in df_by_model.items():
                row = df[(df['T2'] == T2) & (df['lambda'] == lam)].iloc[0]
                mean_fid = row['mean_fid']
                
                if SCIPY_AVAILABLE and len(cfg.scan.times_us) > 3:
                    interp_fn = interp1d(cfg.scan.times_us, mean_fid, kind='cubic')
                    xu = np.linspace(cfg.scan.times_us[0], cfg.scan.times_us[-1], max(200, len(cfg.scan.times_us)*10))
                    yu = interp_fn(xu)
                    plt.plot(xu, yu, label=f"{model}", linewidth=2)
                else:
                    plt.plot(cfg.scan.times_us, mean_fid, label=f"{model}", linewidth=2)

            plt.xlabel(r'Time ($\mu$s)', fontsize=16)
            plt.ylabel('Fidelity', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.title(rf'Fidelity vs Time: $T_2$={T2} $\mu$s, $\lambda$={lam_MHz:.3f} MHz', fontsize=18)
            plt.ylim(0.0, 1.05)
            plt.grid(True, linestyle=':')
            plt.legend(fontsize=14)
            plt.tight_layout()
            
            filename = f"overlay_T2_{int(T2)}_lam_{lam_MHz:.3f}MHz.png"
            plt.savefig(f"{savedir}/{filename}", dpi=300)
            plt.close()
            
def plot_ampdamp_sensitivity(cfg: RunConfig, savedir: str):
    logging.info(f"Plotting amplitude damping sensitivity to {savedir}")
    ratios = [0.25, 0.5, 1.0, 2.0]  # T1/T2 ratios
    psi = np.array([1,1])/np.sqrt(2)

    for T2 in cfg.scan.T2_list_us:
        for lam in cfg.scan.lambda_list_rad_per_us:
            plt.figure(figsize=(10,8))
            lam_MHz = lam/(2*math.pi)
            for r in ratios:
                T1_us = T2 * r
                fids = []
                for t in cfg.scan.times_us:
                    F_bob,_ = run_single_ampdamp(psi, t, cfg.hparams, T1_us, cfg, cfg.dt_us)
                    fids.append(F_bob)
                plt.plot(cfg.scan.times_us, fids, label=f"T1={r:.2f}×T2")
            plt.xlabel(r'Time ($\mu$s)', fontsize=16)
            plt.ylabel('Fidelity', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(rf'AmpDamp Sensitivity: $T_2={T2}$ μs, λ={lam_MHz:.3f} MHz', fontsize=18)
            plt.ylim(0,1.05); plt.grid(True, linestyle=':')
            plt.legend(fontsize=14)
            plt.tight_layout()
            fname = f"ampdamp_sens_T2_{int(T2)}_lam_{lam_MHz:.3f}MHz.png"
            plt.savefig(os.path.join(savedir,fname), dpi=300)
            plt.close()

def plot_collective_tau_sensitivity(cfg: RunConfig, savedir: str):
    logging.info(f"Plotting collective τ sensitivity to {savedir}")
    tau_list = [10.0, 50.0, 100.0]  # μs
    psi = np.array([1,1])/np.sqrt(2)

    for T2 in cfg.scan.T2_list_us:
        for lam in cfg.scan.lambda_list_rad_per_us:
            plt.figure(figsize=(10,8))
            lam_MHz = lam/(2*math.pi)
            sigma_phi_eff = sigma_phi_from_T2(cfg.dt_us, T2)
            for tau_us in tau_list:
                fids = []
                for t in cfg.scan.times_us:
                    F_bob,_ = run_single_collective(psi, t, cfg.hparams, sigma_phi_eff, tau_us, cfg, cfg.dt_us)
                    fids.append(F_bob)
                plt.plot(cfg.scan.times_us, fids, label=f"τ={tau_us} μs")
            plt.xlabel(r'Time ($\mu$s)', fontsize=16)
            plt.ylabel('Fidelity', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(rf'Collective τ Sensitivity: $T_2={T2}$ μs, λ={lam_MHz:.3f} MHz', fontsize=18)
            plt.ylim(0,1.05); plt.grid(True, linestyle=':')
            plt.legend(fontsize=14)
            plt.tight_layout()
            fname = f"collective_tau_T2_{int(T2)}_lam_{lam_MHz:.3f}MHz.png"
            plt.savefig(os.path.join(savedir,fname), dpi=300)
            plt.close()            


def default_runconfig():
    times = np.linspace(0.0, 120.0, 121)
    lam_list = np.array([
        0.0,
        0.062832,   # ≈ 0.01 MHz
        0.125664,   # ≈ 0.02 MHz
        0.188496,   # ≈ 0.03 MHz
        0.251327,   # ≈ 0.04 MHz
        0.314159    # ≈ 0.05 MHz
        ])
    T2_list = [40.0, 80.0, 160.0, 320.0, 640.0, 1000.0]
    return RunConfig(
        hparams=HParams(omega=0.125664, lambda_max=0.3, drive_freq=0.062),
        noise_collective=NoiseCollectiveParams(sigma_phi=0.05, tau_us=50.0, phase_qubits=[1,2,3]),
        noise_amp=AmpDampParams(T1_us=100.0, damp_qubits=[1,2,3]),
        scan=ScanConfig(times_us=times, T2_list_us=T2_list, lambda_list_rad_per_us=lam_list),
        metrics=MetricsConfig(compute_ghz_fidelity=False, use_ghz_target=False),
        dt_us=0.2,
        results_dir="results_GHZ_final",
        tag="ampdamp_vs_collective",
        mc_realizations_collective=40,
        compare_models=["ampdamp", "collective"],
        random_seed_base=12345,
        smooth_heatmap_with_scipy=SCIPY_AVAILABLE
    )


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    F_bob0 = noiseless_baseline_check()
    if F_bob0 < 0.90:
        logging.warning(f"Noiseless baseline fidelity is low: F_bob0={F_bob0:.6f}. Check Charlie X-basis and correction.")
    else:
        logging.info(f"Noiseless baseline fidelity OK: F_bob0={F_bob0:.6f}")

    cfg = default_runconfig()
    df_by_model, combined, _ = run_full_comparison(cfg)

    P = make_plot_dirs(cfg.results_dir)

    plot_ampdamp_curves(df_by_model, cfg, P["ampdamp_curves"])
    plot_collective_curves(df_by_model, cfg, P["collective_curves"])
    plot_error_bands(df_by_model, cfg, P["error_bands"])
    plot_overlay_time_series(df_by_model, cfg, P["overlays"])
    plot_ampdamp_sensitivity(cfg, P["ampdamp_sensitivity"])
    plot_collective_tau_sensitivity(cfg, P["collective_tau_sensitivity"])

    logging.info(f"Simulation and plotting complete. Results are in '{cfg.results_dir}'")

if __name__ == "__main__":
    main()
