"""
run_h2_hardware_test.py
=======================
Minimal end-to-end test: build OTOC circuits, batch-submit to Quantinuum
H2-1, retrieve results, compute C(t), and plot.
"""

import pathlib
import importlib.util
import datetime
import numpy as np

# ── Load library ──────────────────────────────────────────────────────────────
_src = pathlib.Path(__file__).parent / "__main__.py"
_spec = importlib.util.spec_from_file_location("schmetterling", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_build_brickwall = _mod.build_brickwall_otoc_circuit
_build_bowtie    = _mod.build_bowtie_otoc_circuit
_run_circuits = _mod._run_circuits
compute_C_t = _mod.compute_C_t
QuantinuumBackend = _mod.QuantinuumBackend

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TOPOLOGY = "brickwall"  # "brickwall" or "bowtie"
DEVICE_NAME = "H2-2"  # "H2-1" (hardware) or "H2-1E" (emulator)
NEXUS_PROJECT = "schmetterling-effect"  # Nexus project (created if absent)
USE_BATCH = (
    False  # True = attempt_batching (requires org feature); False = standard queue
)
N_QUBITS = 4
P = 0.0
L_VALUES = [1, 2, 3]
N_SHOTS = 1
BASE_SEED = 42
PERT_OP = "X"
PERT_SITE = N_QUBITS // 2
PROBE_SITE = 0
PROBE_ANGLE = 0.0
OPT_LEVEL = 0

FIGURE_DIR = pathlib.Path(__file__).parent.parent / "figs"
DATE = f"{datetime.datetime.now():%Y_%m_%d_%H:%M:%S}"
FIGURE_PATH = str(FIGURE_DIR / f"h2_test_{DATE}.png")

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if TOPOLOGY == "bowtie":
        build_otoc_circuit = _build_bowtie
    elif TOPOLOGY == "brickwall":
        build_otoc_circuit = _build_brickwall
    else:
        raise ValueError(f"Unknown TOPOLOGY {TOPOLOGY!r}. Choose 'brickwall' or 'bowtie'.")

    # QuantinuumBackend is used as a type token; _run_circuits detects it and
    # routes through the Nexus API (upload → compile job → batched execute job).
    backend = QuantinuumBackend(DEVICE_NAME)

    # Build one perturbed + one unperturbed circuit per depth.
    # Interleaved: [pert_L0, unpert_L0, pert_L1, unpert_L1, ...]
    circuits, tags = [], []
    for L in L_VALUES:
        shared = dict(
            L=L,
            W=N_QUBITS,
            pert_site=PERT_SITE,
            pert_op=PERT_OP,
            probe_site=PROBE_SITE,
            probe_angle=PROBE_ANGLE,
            p=P,
            seed=BASE_SEED,
            init_seed=BASE_SEED + 1,
            meas_seed=BASE_SEED + 2,
            meas_seed_ud=BASE_SEED + 3,
            add_barrier=True,
        )
        circuits += [
            build_otoc_circuit(**shared, unperturbed=False),
            build_otoc_circuit(**shared, unperturbed=True),
        ]
        tags += [
            f"N{N_QUBITS}_p{P:.2f}_L{L:02d}_pert",
            f"N{N_QUBITS}_p{P:.2f}_L{L:02d}_unpert",
        ]

    # _run_circuits detects QuantinuumBackend and automatically:
    #   1. Logs into Nexus
    #   2. Uploads circuits to NEXUS_PROJECT
    #   3. Runs a single compile job (opt level OPT_LEVEL)
    #   4. Runs a single batched execute job (attempt_batching=True)
    #   5. Downloads and returns shot arrays in input order
    shot_arrays = _run_circuits(
        circuits,
        backend,
        N_SHOTS,
        tags=tags,
        optimisation_level=OPT_LEVEL,
        nexus_project=NEXUS_PROJECT,
        use_batch=USE_BATCH,
    )

    # Pack shots into the raw dict that compute_C_t expects:
    #   raw[p][L] = 1-D array of per-realization (out_u - out_p) values.
    # Here we have one realization per L (one circuit pair), so each array
    # has length 1 — the mean over N_SHOTS shots for that single circuit.
    probe_bit = shot_arrays[0].shape[1] - 1
    raw = {P: {}}
    for idx, L in enumerate(L_VALUES):
        out_p = 1 - 2 * shot_arrays[2 * idx][:, probe_bit].astype(int)
        out_u = 1 - 2 * shot_arrays[2 * idx + 1][:, probe_bit].astype(int)
        raw[P][L] = np.array([out_u.mean() - out_p.mean()])

    stats = compute_C_t(raw)
    for L in L_VALUES:
        print(f"L={L}  C={stats['C'][P][L]:.4f}  SE={stats['se'][P][L]:.4f}")

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_vals = [stats["C"][P][L] for L in L_VALUES]
    se_vals = [stats["se"][P][L] for L in L_VALUES]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        L_VALUES,
        C_vals,
        yerr=se_vals,
        fmt="o-",
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        label=f"{DEVICE_NAME}  N={N_QUBITS}  p={P}",
    )
    ax.set_xlabel("T (layers)")
    ax.set_ylabel(r"$C(t)$")
    ax.set_title(f"OTOC — {DEVICE_NAME}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {FIGURE_PATH}")
