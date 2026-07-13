"""
run_disorder_avg.py
====================
Orchestrates the disorder-averaged OTOC experiment.

Four partition schemes are available for parallel execution:

  Scheme 1  — partition by N (system size)
              --scheme 1 --job-id {0..len(N_VALUES)-1}
              5 jobs for the default config.

  Scheme 2  — partition by c_idx (angle realization)
              --scheme 2 --job-id {0..N_CIRCUITS-1}
              20 jobs; each handles all N, all p, all init-states for one seed.

  Scheme 3  — partition by (c_idx, i_idx) (single disorder realization)
              --scheme 3 --job-id {c_idx} --secondary-id {i_idx}
              400 jobs; finest grain, best fault tolerance.

  Scheme 4  — partition by (N, p) grid point
              --scheme 4 --job-id {n_idx} --secondary-id {p_idx}
              25 jobs; maps 1-to-1 to grid cells in the C(p,T) figure.

Sequential (no flags)  — runs everything in one process (original behaviour).

Merge and plot after parallel jobs finish:
  python run_disorder_avg.py --scheme {1|2|3|4} --merge

Nohup launch helpers are printed when --help-nohup is passed.
"""

import argparse
import glob
import os
import pathlib
import pickle
import importlib.util
import datetime
from collections import defaultdict

import numpy as np

# ── Load src/__main__.py ───────────────────────────────────────────────────────
_src = pathlib.Path(__file__).parent / "__main__.py"
_spec = importlib.util.spec_from_file_location("schmetterling", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_brickwall_otoc_circuit = _mod.build_brickwall_otoc_circuit
build_bowtie_otoc_circuit   = _mod.build_bowtie_otoc_circuit
_run_circuits = _mod._run_circuits
compute_C_t = _mod.compute_C_t
plot_results = _mod.plot_results
sweep_over_all_disorder_axes = _mod.sweep_over_all_disorder_axes
sweep_single_shot_disorder = _mod.sweep_single_shot_disorder
QuantinuumBackend = _mod.QuantinuumBackend
build_aer_noise_model = _mod.build_aer_noise_model

# ── Load transfer_matrix_otoc.py ──────────────────────────────────────────────
_tm_src = pathlib.Path(__file__).parent / "transfer_matrix_otoc.py"
_tm_spec = importlib.util.spec_from_file_location("transfer_matrix_otoc", _tm_src)
_tm_mod = importlib.util.module_from_spec(_tm_spec)
_tm_spec.loader.exec_module(_tm_mod)
ButterflyTransferMatrix = _tm_mod.ButterflyTransferMatrix

# ══════════════════════════════════════════════════════════════════════════════
# Parameters  (edit here; all scheme functions read from these globals)
# ══════════════════════════════════════════════════════════════════════════════
#
#   Three independent disorder axes per realization (c_idx, i_idx):
#     Axis 1 — gate angles      : circuit_seed = BASE_SEED * 10_000 + c_idx
#     Axis 2 — initial Pauli word: init_seed   = BASE_SEED * 10_000_000 + c_idx*1_000 + i_idx
#     Axis 3 — measurement sites : meas_seed   = (MEAS_SEED+1)*10_000_000 + c_idx*1_000 + i_idx
#
DATE = f"{datetime.datetime.now():%Y_%m_%d_%H:%M:%S}"
P_VALUES = [0.0, 0.01]  # measurement rate threshold
L_MAX = 25  # max layers
N_VALUES = [6]  # system sizes (even, 8–16)
N_REALIZATIONS = 1000  # single-shot mode: independent circuits per (N, p, L)
N_SHOTS = 100  # multi-shot mode: shots per fixed circuit
N_CIRCUITS = 5  # multi-shot mode: gate-angle realizations
N_INIT_STATES = 5  # multi-shot mode: initial-state realizations
BASE_SEED = 1917  # root seed for parallel schemes (single seed)
BASE_SEEDS: list[int] = [
    1,
    10,
    25,
]  # seeds for sequential sweeps — add more to seed-average
MEAS_SEED = 7  # multi-shot mode: root seed for measurement stream
RECORD_EVERY = 4  # record at L = 5, 10, 15, 20
L_VALUES: list[int] | None = [
    2,
    4,
    8,
    16,
    25,
]  # explicit L list; overrides L_MAX + RECORD_EVERY when set
TOPOLOGY = "brickwall"  # "brickwall" or "bowtie"
PERT_OP = "X"  # perturbation gate
PROBE_ANGLE = 0.0  # Ry(π/2) on probe qubit
OPTIMISATION_LEVEL = 0  # pytket compilation level: 0 = minimal, 1 = light, 2 = full
NEXUS_BATCH = (
    False  # True = attempt_batching (requires org feature); False = standard queue
)
# DEVICE_NAME = "H2-1LE"  # uncomment for Quantinuum hardware
DEVICE_NAME = None  # None → local AerBackend

# ── Noise model (local AerBackend only; ignored when DEVICE_NAME is set) ──────
# All None → ideal (noiseless) simulation.  Mix and match as needed.
# Typical H-series-class values: p1q=1e-4, p2q=1e-3, pm=5e-3
#                                 T1=1e-3, T2=1e-3 (seconds)
NOISE_P1Q = None  # single-qubit gate depolarizing error rate
NOISE_P2Q = None  # two-qubit gate depolarizing error rate
NOISE_PM = None  # symmetric readout bitflip probability
NOISE_T1 = None  # longitudinal relaxation time T1 (seconds)
NOISE_T2 = None  # transverse relaxation time  T2 (seconds; must be <= 2*T1)
NOISE_T_1Q = 50e-9  # single-qubit gate duration (seconds)
NOISE_T_2Q = 300e-9  # two-qubit gate duration    (seconds)
FIGURE_DIR = pathlib.Path(__file__).parent.parent / "figs"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH_C = str(FIGURE_DIR / f"C_vs_t_{DATE}.png")
FIGURE_PATH_CV = str(FIGURE_DIR / f"cv_vs_t_{DATE}.png")
FIGURE_PATH_TM = str(FIGURE_DIR / f"qc_vs_tm_{DATE}.png")
PARTIALS_DIR = "partials"  # directory for intermediate files

# ── Transfer-matrix overlay  (set TM_NUM_QUBITS to None to skip entirely) ─────
TM_NUM_QUBITS: int | None = 6  # None → TM disabled; set to an int to enable
TM_TF: int = 25  # number of time steps to evolve
TM_PERT_SITE: int = 3  # perturbation qubit index
TM_PROBE_SITE: int = 0  # probe qubit index
TM_P_VALUES: list[float] | None = (
    None  # None → use P_VALUES; or set explicitly e.g. [0.0, 0.02]
)
TM_BC_TYPE: str = "open"  # "open" or "periodic"

# OTOC sweep parameters (used in sequential mode and OTOC-specific runs)
OTOC_W = N_VALUES[-1]
OTOC_P_VALUES = [0.02, 1.0]

# ══════════════════════════════════════════════════════════════════════════════
# Shared infrastructure
# ══════════════════════════════════════════════════════════════════════════════


def _L_steps():
    if L_VALUES is not None:
        return sorted(L_VALUES)
    return list(range(RECORD_EVERY, L_MAX + 1, RECORD_EVERY))


def _circuit_builder():
    if TOPOLOGY == "bowtie":
        return build_bowtie_otoc_circuit
    if TOPOLOGY == "brickwall":
        return build_brickwall_otoc_circuit
    raise ValueError(f"Unknown TOPOLOGY {TOPOLOGY!r}. Choose 'brickwall' or 'bowtie'.")


def _build_backend():
    """
    Construct and return the backend from DEVICE_NAME / noise config.

    Quantinuum hardware : DEVICE_NAME set → QuantinuumBackend (noise config ignored)
    Local noiseless     : DEVICE_NAME = None, NOISE_* = None → AerBackend()
    Local noisy         : DEVICE_NAME = None, any NOISE_* set → AerBackend(noise_model=...)
    """
    if DEVICE_NAME is not None:
        return QuantinuumBackend(DEVICE_NAME)

    from pytket.extensions.qiskit import AerBackend

    if any(v is not None for v in (NOISE_P1Q, NOISE_P2Q, NOISE_PM, NOISE_T1, NOISE_T2)):
        noise_model = build_aer_noise_model(
            p1q=NOISE_P1Q,
            p2q=NOISE_P2Q,
            pm=NOISE_PM,
            T1=NOISE_T1,
            T2=NOISE_T2,
            t_gate_1q=NOISE_T_1Q,
            t_gate_2q=NOISE_T_2Q,
        )
        print(
            f"Noisy AerBackend  p1q={NOISE_P1Q}  p2q={NOISE_P2Q}  pm={NOISE_PM}  "
            f"T1={NOISE_T1}  T2={NOISE_T2}  t_1q={NOISE_T_1Q}  t_2q={NOISE_T_2Q}"
        )
        return AerBackend(noise_model=noise_model)

    return AerBackend()


def estimate_hqc_cost(mode: str = "single_shot", verbose: bool = True) -> float:
    """
    Estimates the total HQC cost for a full experiment without running it.

    Formula per circuit (Quantinuum billing):
        HQC = 5 + (N1q + 10*N2q + 5*Nm) / 5000 * Ns

    where N1q = single-qubit gates, N2q = two-qubit gates, Nm = measurements,
    Ns = shots per circuit, all counted after compilation to native gates.

    Compilation estimates for a Haar SU(4) gate (KAK decomposition):
        N2q per gate  ≈ 3  (three ZZPhase/CNOT-equivalent native gates)
        N1q per gate  ≈ 6  (surrounding single-qubit rotations)

    Parameters
    ----------
    mode    : "single_shot" or "multi_shot" — must match intended run mode.
    verbose : Print a breakdown table when True.

    Returns
    -------
    Total estimated HQC as a float.
    """
    N2Q_PER_SU4 = 3
    N1Q_PER_SU4 = 6

    L_values = _L_steps()
    total_hqc = 0.0
    rows = []

    for W in N_VALUES:
        for p in P_VALUES:
            for L in L_values:
                # SU(4) gate count per half-circuit (U or U†)
                L_even = (L + 1) // 2
                L_odd = L // 2
                n_su4_half = L_even * (W // 2) + L_odd * (W // 2 - 1)
                n_su4 = 2 * n_su4_half  # U + U†

                N2q = N2Q_PER_SU4 * n_su4
                N1q = N1Q_PER_SU4 * n_su4 + (W - 1)  # W-1 state-prep Paulis

                # Measurements: U mid-circuit + U† mid-circuit + pert (only if
                # pert_op == "measure") + final probe.  A Pauli perturbation adds
                # no measurement, so perturbed and unperturbed share the same Nm.
                n_meas = min(round(p * W * L), W * max(L - 1, 0))
                n_pert_meas = 1 if PERT_OP == "measure" else 0
                Nm = 2 * n_meas + n_pert_meas + 1  # same for both variants

                if mode == "single_shot":
                    n_circuits_point = 2 * N_REALIZATIONS
                    Ns = 1
                else:
                    n_circuits_point = 2 * N_CIRCUITS * N_INIT_STATES
                    Ns = N_SHOTS

                hqc_point = (
                    5 + (N1q + 10 * N2q + 5 * Nm) / 5000 * Ns
                ) * n_circuits_point
                total_hqc += hqc_point

                rows.append((W, p, L, N1q, N2q, Nm, n_circuits_point, Ns, hqc_point))

    if verbose:
        print(f"\nHQC estimate  (mode={mode!r})")
        print(
            f"{'N':>4} {'p':>5} {'L':>4} {'N1q':>6} {'N2q':>6} {'Nm':>5} {'#circs':>7} {'Ns':>4} {'HQC':>10}"
        )
        print("-" * 62)
        for W, p, L, N1q, N2q, Nm, n_c, Ns, hqc in rows:
            print(
                f"{W:>4} {p:>5.2f} {L:>4} {N1q:>6} {N2q:>6} {Nm:>5} {n_c:>7} {Ns:>4} {hqc:>10.1f}"
            )
        print("-" * 62)
        print(f"{'TOTAL':>50}  {total_hqc:>10.1f}")
        print()

    return total_hqc


def _compute_partial(
    N_values: list[int],
    p_values: list[float],
    L_steps: list[int],
    realizations: list[tuple[int, int]],
) -> dict[tuple, np.ndarray]:
    """
    Core compute kernel shared by all four partition schemes.

    Runs the perturbed and unperturbed echo circuits for every combination of
    (N, p, L) in the given lists, but only for the specified disorder
    realizations (c_idx, i_idx pairs).

    Parameters
    ----------
    N_values      : system sizes to process in this partition
    p_values      : measurement rates to process in this partition
    L_steps       : layer counts to process in this partition
    realizations  : list of (c_idx, i_idx) pairs to iterate over

    Returns
    -------
    dict keyed (N, p, L) → 1-D np.ndarray of per-realization F_vals,
    where F_val = pert_mean - unpert_mean (one entry per realization).
    """
    backend = _build_backend()
    add_bar = isinstance(backend, QuantinuumBackend)
    meas_base = MEAS_SEED + 1
    partial = defaultdict(list)

    for W in N_values:
        pert_site = W // 2
        probe_site = 0

        for c_idx, i_idx in realizations:
            circuit_seed = BASE_SEED * 10_000 + c_idx
            init_seed_val = BASE_SEED * 10_000_000 + c_idx * 1_000 + i_idx
            meas_seed_val = meas_base * 10_000_000 + c_idx * 1_000 + i_idx

            for p in p_values:
                for L in L_steps:
                    shared = dict(
                        L=L,
                        W=W,
                        pert_site=pert_site,
                        pert_op=PERT_OP,
                        probe_site=probe_site,
                        probe_angle=PROBE_ANGLE,
                        p=p,
                        seed=circuit_seed,
                        init_seed=init_seed_val,
                        meas_seed=meas_seed_val,
                        add_barrier=add_bar,
                    )
                    _build = _circuit_builder()
                    qc_pert = _build(**shared, unperturbed=False)
                    qc_unpert = _build(**shared, unperturbed=True)
                    probe_bit = qc_pert.n_bits - 1

                    sp = _run_circuit_backend(qc_pert, backend, N_SHOTS)
                    su = _run_circuit_backend(qc_unpert, backend, N_SHOTS)

                    out_p = 1 - 2 * sp[:, probe_bit].astype(int)
                    out_u = 1 - 2 * su[:, probe_bit].astype(int)
                    F_val = float(out_p.mean() - out_u.mean())

                    partial[(W, p, L)].append(F_val)
                    print(
                        f"N={W:2d}  p={p:.2f}  L={L:3d}  "
                        f"c={c_idx:2d}  i={i_idx:2d}  dF={F_val:.4f}"
                    )

    return {k: np.array(v) for k, v in partial.items()}


# ── File I/O helpers ──────────────────────────────────────────────────────────


def _save_partial(data: dict, path: str) -> None:
    """Pickle a partial result dict to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"Saved → {path}  ({len(data)} (N,p,L) keys)")


def _load_partial(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _merge_partials(pattern: str) -> dict[tuple, np.ndarray]:
    """
    Load all pickle files matching glob pattern and concatenate their F_val
    arrays along the realization axis.

    Returns dict keyed (N, p, L) → full concatenated np.ndarray.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No partial files matching {pattern!r}")
    print(f"Merging {len(files)} partial files …")
    merged = defaultdict(list)
    for fname in files:
        partial = _load_partial(fname)
        for key, vals in partial.items():
            merged[key].extend(vals.tolist())
    return {k: np.array(v) for k, v in merged.items()}


def _merged_to_all_stats(merged: dict) -> tuple:
    """
    Convert the flat (N, p, L)-keyed merged dict into the nested
    all_stats[N] = compute_F({p: {L: array}}) structure used by plot_results.

    Returns (all_stats, N_values, p_values, L_values) sorted.
    """
    raw_by_N: dict[int, dict] = defaultdict(lambda: defaultdict(dict))
    for (N, p, L), vals in merged.items():
        raw_by_N[N][p][L] = vals

    N_values = sorted(raw_by_N)
    p_values = sorted({p for (N, p, L) in merged})
    L_values = sorted({L for (N, p, L) in merged})

    all_stats = {N: compute_C_t(dict(raw_by_N[N])) for N in N_values}
    return all_stats, N_values, p_values, L_values


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 1  — Partition by N
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_1(n_idx: int) -> None:
    """
    Run the full disorder average for ONE system size N = N_VALUES[n_idx].
    Covers all p_values, all L_steps, all N_CIRCUITS × N_INIT_STATES realizations.

    Parallel usage:  one nohup process per n_idx in range(len(N_VALUES)).
    Degree of parallelism: len(N_VALUES) = 5.

    Launch:
      for i in 0 1 2 3 4; do
        nohup .butterfly/bin/python run_disorder_avg.py --scheme 1 --job-id $i
              > partials/s1_$i.log 2>&1 &
      done
    """
    N = N_VALUES[n_idx]
    realizations = [(c, i) for c in range(N_CIRCUITS) for i in range(N_INIT_STATES)]

    print(f"\n=== Scheme 1 | job {n_idx} | N={N} ===")
    print(f"  realizations = {len(realizations)}")
    print(f"  L_steps = {_L_steps()}")

    partial = _compute_partial([N], P_VALUES, _L_steps(), realizations)
    _save_partial(partial, f"{PARTIALS_DIR}/s1_N{N}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 2  — Partition by c_idx (angle realization)
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_2(c_idx: int) -> None:
    """
    Run ALL N_values and ALL p_values, but only for ONE angle seed (c_idx),
    iterating over all N_INIT_STATES init-state seeds.

    Parallel usage: one process per c_idx in range(N_CIRCUITS).
    Degree of parallelism: N_CIRCUITS = 20.

    Launch:
      for i in $(seq 0 19); do
        nohup .butterfly/bin/python run_disorder_avg.py --scheme 2 --job-id $i
              > partials/s2_c$i.log 2>&1 &
      done
    """
    realizations = [(c_idx, i) for i in range(N_INIT_STATES)]

    print(f"\n=== Scheme 2 | c_idx={c_idx} ===")
    print(
        f"  realizations (c_idx={c_idx}, i_idx=0..{N_INIT_STATES-1}): {len(realizations)}"
    )

    partial = _compute_partial(N_VALUES, P_VALUES, _L_steps(), realizations)
    _save_partial(partial, f"{PARTIALS_DIR}/s2_c{c_idx}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 3  — Partition by (c_idx, i_idx)  — finest grain
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_3(c_idx: int, i_idx: int) -> None:
    """
    Run ALL N_values and ALL p_values for exactly ONE disorder realization
    (c_idx, i_idx).  Each job submits len(N_VALUES)*len(P_VALUES)*len(L_steps)*2
    circuits.

    Parallel usage: one process per (c_idx, i_idx) pair.
    Degree of parallelism: N_CIRCUITS × N_INIT_STATES = 400.

    Launch:
      for c in $(seq 0 19); do
        for i in $(seq 0 19); do
          nohup .butterfly/bin/python run_disorder_avg.py \\
                --scheme 3 --job-id $c --secondary-id $i \\
                > partials/s3_c${c}_i${i}.log 2>&1 &
        done
      done
    """
    n_circuits_per_job = len(N_VALUES) * len(P_VALUES) * len(_L_steps()) * 2
    print(f"\n=== Scheme 3 | c_idx={c_idx}  i_idx={i_idx} ===")
    print(f"  circuits in this job = {n_circuits_per_job}")

    partial = _compute_partial(N_VALUES, P_VALUES, _L_steps(), [(c_idx, i_idx)])
    _save_partial(partial, f"{PARTIALS_DIR}/s3_c{c_idx}_i{i_idx}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Scheme 4  — Partition by (N, p)  — grid point
# ══════════════════════════════════════════════════════════════════════════════


def run_scheme_4(n_idx: int, p_idx: int) -> None:
    """
    Run ALL realizations (N_CIRCUITS × N_INIT_STATES) for exactly ONE (N, p)
    grid point.  Each job independently produces a full disorder average for
    one cell of the C(N, p, T) output matrix.

    Parallel usage: one process per (n_idx, p_idx) pair.
    Degree of parallelism: len(N_VALUES) × len(P_VALUES) = 25.

    Launch:
      for n in $(seq 0 4); do
        for p in $(seq 0 4); do
          nohup .butterfly/bin/python run_disorder_avg.py \\
                --scheme 4 --job-id $n --secondary-id $p \\
                > partials/s4_n${n}_p${p}.log 2>&1 &
        done
      done
    """
    N = N_VALUES[n_idx]
    p = P_VALUES[p_idx]
    realizations = [(c, i) for c in range(N_CIRCUITS) for i in range(N_INIT_STATES)]

    print(f"\n=== Scheme 4 | N={N}  p={p:.2f} ===")
    print(f"  realizations = {len(realizations)}")

    partial = _compute_partial([N], [p], _L_steps(), realizations)
    _save_partial(partial, f"{PARTIALS_DIR}/s4_N{N}_p{p_idx}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Merge  — combine all partial files for a given scheme and plot
# ══════════════════════════════════════════════════════════════════════════════


def merge_and_plot(scheme: int) -> None:
    """
    Load all partial result files for the given scheme, concatenate their
    F_val arrays, compute F / var / CV, and produce the output figures.

    Run after all parallel jobs have completed:
      python run_disorder_avg.py --scheme 2 --merge
    """
    pattern = f"{PARTIALS_DIR}/s{scheme}_*.pkl"
    merged = _merge_partials(pattern)
    all_stats, N_values, p_values, L_values = _merged_to_all_stats(merged)

    print(f"\nMerged shape:")
    print(f"  N_values = {N_values}")
    print(f"  p_values = {p_values}")
    print(f"  L_values = {L_values}")
    total_realizations = len(next(iter(merged.values())))
    print(f"  realizations per (N, p, L) = {total_realizations}")

    plot_results(
        all_stats,
        L_values,
        p_values,
        N_values,
        figure_path_C=FIGURE_PATH_C,
        figure_path_cv=FIGURE_PATH_CV,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Transfer-matrix overlay plot
# ══════════════════════════════════════════════════════════════════════════════


def plot_tm_overlay(
    all_stats: dict,
    L_values: list[int],
    p_values: list[float],
    tm_results: dict[float, tuple[np.ndarray, np.ndarray]],
    tm_N: int,
    figure_path: str,
) -> None:
    """
    Overlay QC disorder-average results with exact transfer-matrix OTOC curves.

    Produces one subplot per p value in p_values.  Each subplot shows QC
    C(T) errorbars for all N, with the series matching tm_N drawn thicker.
    If a TM result exists for that p in tm_results, its exact C(t) curve is
    drawn as a solid crimson line on the same axes.

    Parameters
    ----------
    all_stats : dict
        Output of compute_C_t, keyed all_stats[N]["C" | "se"][p][L].
    L_values : list[int]
        Recorded layer counts (x-axis positions for QC errorbars).
    p_values : list[float]
        Measurement rates swept in the QC experiment.
    tm_results : dict[float, (t_arr, C_arr)]
        Keyed by p; each value is the (t, C) pair from ButterflyTransferMatrix.
    tm_N : int
        System size (num_qubits) used for the TM runs.
    figure_path : str
        Output file path for the saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N_values = sorted(all_stats)
    n_panels = len(p_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    N_colors = plt.cm.viridis(np.linspace(0, 1, max(len(N_values), 1)))

    for ax, p in zip(axes, p_values):
        for n_idx, N in enumerate(N_values):
            if N not in all_stats or p not in all_stats[N]["C"]:
                continue
            qc_C = [all_stats[N]["C"][p][L] for L in L_values]
            qc_se = [all_stats[N]["se"][p][L] for L in L_values]
            lw = 2.0 if N == tm_N else 0.8
            ax.errorbar(
                [L - 1 for L in L_values],
                qc_C,
                yerr=qc_se,
                fmt="o",
                capsize=3,
                capthick=lw,
                elinewidth=lw,
                color=N_colors[n_idx],
                label=f"QC  N={N}",
            )

        if p in tm_results:
            t_tm, C_tm = tm_results[p]
            ax.plot(
                t_tm,
                C_tm,
                "-",
                lw=2,
                color="crimson",
                label=f"TM  N={tm_N}",
                zorder=5,
            )

        ax.set_xlabel("t  (layers)")
        ax.set_ylabel(r"$C_t$")
        ax.set_title(f"p = {p:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"QC (errorbars) vs exact transfer matrix (line)  —  N$_{{TM}}$={tm_N}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {figure_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Sequential  — original single-process run (no partitioning)
# ══════════════════════════════════════════════════════════════════════════════


def run_sequential(mode: str = "single_shot") -> None:
    """
    Run the full disorder average in one process.

    Parameters
    ----------
    mode : "single_shot" (default) or "multi_shot"
        single_shot — sweep_single_shot_disorder: each of N_REALIZATIONS
            samples is a fully independent circuit (fresh angles, initial
            state, U and U† measurement sites) executed for 1 shot.
        multi_shot  — sweep_over_all_disorder_axes: N_CIRCUITS x N_INIT_STATES
            realizations, each circuit run for N_SHOTS shots.
    """
    L_steps = _L_steps()
    backend_label = "local AerBackend" if DEVICE_NAME is None else DEVICE_NAME

    print("=" * 64)
    print(f"Sequential run  —  mode={mode!r}")
    print("=" * 64)
    print(f"  p_values       = {P_VALUES}")
    print(f"  L_max          = {L_MAX}   →  recorded at L = {L_steps}")
    print(f"  N_values       = {N_VALUES}")
    print(f"  base_seeds     = {BASE_SEEDS}")
    print(f"  pert_op        = {PERT_OP!r}")
    print(f"  backend        = {backend_label!r}")

    if mode == "single_shot":
        n_total = (
            len(N_VALUES)
            * len(P_VALUES)
            * len(L_steps)
            * N_REALIZATIONS
            * len(BASE_SEEDS)
            * 2
        )
        print(
            f"  n_realizations = {N_REALIZATIONS} × {len(BASE_SEEDS)} seed(s) = {N_REALIZATIONS * len(BASE_SEEDS)}"
        )
        print(f"  Total circuits = {n_total}  (1 shot each)")
        print("=" * 64)
        all_stats = sweep_single_shot_disorder(
            p_values=P_VALUES,
            L_max=L_MAX,
            N_values=N_VALUES,
            n_realizations=N_REALIZATIONS,
            base_seeds=BASE_SEEDS,
            record_every=RECORD_EVERY,
            L_values=L_steps if L_VALUES is not None else None,
            pert_op=PERT_OP,
            probe_angle=PROBE_ANGLE,
            device_name=DEVICE_NAME,
            optimisation_level=OPTIMISATION_LEVEL,
            use_batch=NEXUS_BATCH,
            circuit_builder=_circuit_builder(),
        )
    elif mode == "multi_shot":
        n_real = N_CIRCUITS * N_INIT_STATES * len(BASE_SEEDS)
        n_total = len(N_VALUES) * len(P_VALUES) * n_real * len(L_steps) * 2
        print(f"  n_circuits     = {N_CIRCUITS}")
        print(f"  n_init_states  = {N_INIT_STATES}")
        print(f"  n_shots        = {N_SHOTS}")
        print(f"  meas_seed      = {MEAS_SEED}")
        print(
            f"  Total circuits = {n_total}  ({N_SHOTS} shots each, {len(BASE_SEEDS)} seed(s))"
        )
        print("=" * 64)
        all_stats = sweep_over_all_disorder_axes(
            p_values=P_VALUES,
            L_max=L_MAX,
            N_values=N_VALUES,
            n_shots=N_SHOTS,
            n_circuits=N_CIRCUITS,
            n_init_states=N_INIT_STATES,
            base_seeds=BASE_SEEDS,
            meas_seed=MEAS_SEED,
            record_every=RECORD_EVERY,
            L_values=L_steps if L_VALUES is not None else None,
            pert_op=PERT_OP,
            probe_angle=PROBE_ANGLE,
            device_name=DEVICE_NAME,
            optimisation_level=OPTIMISATION_LEVEL,
            use_batch=NEXUS_BATCH,
            circuit_builder=_circuit_builder(),
        )
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Choose 'single_shot' or 'multi_shot'."
        )

    plot_results(
        all_stats=all_stats,
        L_values=L_steps,
        p_values=P_VALUES,
        N_values=N_VALUES,
        figure_path_C=FIGURE_PATH_C,
        figure_path_cv=FIGURE_PATH_CV,
    )

    if TM_NUM_QUBITS is not None:
        _tm_p_values = TM_P_VALUES if TM_P_VALUES is not None else P_VALUES
        print(
            f"\nRunning transfer matrix: N={TM_NUM_QUBITS}  p_values={_tm_p_values}  "
            f"tf={TM_TF}  pert={TM_PERT_SITE}  probe={TM_PROBE_SITE}  "
            f"bc={TM_BC_TYPE!r}"
        )
        tm = ButterflyTransferMatrix(TM_NUM_QUBITS, bc_type=TM_BC_TYPE)
        tm_results = {
            p: (t, C / 2)
            for p in _tm_p_values
            for t, C in [tm.compute_otoc(p, TM_TF, TM_PERT_SITE, TM_PROBE_SITE)]
        }
        plot_tm_overlay(
            all_stats=all_stats,
            L_values=L_steps,
            p_values=P_VALUES,
            tm_results=tm_results,
            tm_N=TM_NUM_QUBITS,
            figure_path=FIGURE_PATH_TM,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def _print_nohup_commands() -> None:
    venv = ".butterfly/bin/python"
    script = "run_disorder_avg.py"
    d = PARTIALS_DIR

    print(f"""
Nohup launch commands
─────────────────────
mkdir -p {d}

Scheme 1 — 5 jobs (one per N):
  for i in 0 1 2 3 4; do
    nohup {venv} {script} --scheme 1 --job-id $i > {d}/s1_$i.log 2>&1 &
  done

Scheme 2 — {N_CIRCUITS} jobs (one per c_idx):
  for i in $(seq 0 {N_CIRCUITS-1}); do
    nohup {venv} {script} --scheme 2 --job-id $i > {d}/s2_c$i.log 2>&1 &
  done

Scheme 3 — {N_CIRCUITS*N_INIT_STATES} jobs (one per realization):
  for c in $(seq 0 {N_CIRCUITS-1}); do
    for i in $(seq 0 {N_INIT_STATES-1}); do
      nohup {venv} {script} --scheme 3 --job-id $c --secondary-id $i \\
            > {d}/s3_c${{c}}_i${{i}}.log 2>&1 &
    done
  done

Scheme 4 — {len(N_VALUES)*len(P_VALUES)} jobs (one per (N,p) grid point):
  for n in $(seq 0 {len(N_VALUES)-1}); do
    for p in $(seq 0 {len(P_VALUES)-1}); do
      nohup {venv} {script} --scheme 4 --job-id $n --secondary-id $p \\
            > {d}/s4_n${{n}}_p${{p}}.log 2>&1 &
    done
  done

Merge + plot after jobs complete:
  {venv} {script} --scheme 1 --merge
  {venv} {script} --scheme 2 --merge   # etc.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disorder-averaged OTOC experiment runner.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scheme",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help=(
            "Partition scheme:\n"
            "  1 = by N          (--job-id = n_idx)\n"
            "  2 = by c_idx      (--job-id = c_idx)\n"
            "  3 = by (c,i)      (--job-id = c_idx, --secondary-id = i_idx)\n"
            "  4 = by (N,p)      (--job-id = n_idx, --secondary-id = p_idx)\n"
            "  omit = sequential full run"
        ),
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=0,
        help="Primary partition index (meaning depends on --scheme).",
    )
    parser.add_argument(
        "--secondary-id",
        type=int,
        default=0,
        help="Secondary partition index for schemes 3 and 4.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Load all partial files for --scheme and produce final figures.",
    )
    parser.add_argument(
        "--mode",
        choices=["single_shot", "multi_shot"],
        default="single_shot",
        help=(
            "Sequential sweep mode (ignored when --scheme is set):\n"
            "  single_shot (default) — sweep_single_shot_disorder:\n"
            "      N_REALIZATIONS fully independent circuits, 1 shot each.\n"
            "  multi_shot            — sweep_over_all_disorder_axes:\n"
            "      N_CIRCUITS x N_INIT_STATES realizations, N_SHOTS shots each."
        ),
    )
    parser.add_argument(
        "--L-values",
        type=str,
        default=None,
        metavar="L1,L2,...",
        help=(
            "Comma-separated explicit list of layer counts to evaluate "
            "(e.g. 1,2,5,10,25). Overrides L_MAX and RECORD_EVERY."
        ),
    )
    parser.add_argument(
        "--estimate-hqc",
        action="store_true",
        help="Print an HQC cost estimate for the current config and exit (no circuits run).",
    )
    parser.add_argument(
        "--help-nohup",
        action="store_true",
        help="Print ready-to-paste nohup launch commands and exit.",
    )
    args = parser.parse_args()

    if args.L_values:
        L_VALUES = [int(x) for x in args.L_values.split(",")]

    if args.estimate_hqc:
        estimate_hqc_cost(mode=args.mode)

    elif args.help_nohup:
        _print_nohup_commands()

    elif args.merge:
        if args.scheme is None:
            parser.error("--merge requires --scheme {1|2|3|4}")
        merge_and_plot(args.scheme)

    elif args.scheme == 1:
        run_scheme_1(args.job_id)

    elif args.scheme == 2:
        run_scheme_2(args.job_id)

    elif args.scheme == 3:
        run_scheme_3(args.job_id, args.secondary_id)

    elif args.scheme == 4:
        run_scheme_4(args.job_id, args.secondary_id)

    else:
        run_sequential(mode=args.mode)
