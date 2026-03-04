import datetime
from collections import Counter
import numpy as np
import qnexus as qnx
from pytket import Circuit


def calculate_hamming_distance_pdf(data):
    """
    Docstring for calculate_hamming_distance_pdf

    :param data: Description
    """
    distances = Counter()
    for key, item in data.items():
        dist = np.sum(key)
        distances.update({dist: item})

    xs = sorted(distances.keys())
    total = sum(distances.values())
    pdf = [distances[x] / total for x in xs]
    return xs, pdf


def validate_nexus_connection(*, nexus_hosted=True) -> bool:
    """
    Returns True if we can auth and reach Nexus (and therefore can submit jobs).
    """
    try:
        # Use whichever login mode fits your environment:
        # qnx.login()  # browser flow  [oai_citation:1‡docs.quantinuum.com](https://docs.quantinuum.com/nexus/trainings/notebooks/basics/getting_started.html?utm_source=chatgpt.com)
        qnx.login_with_credentials()  # prompt-based  [oai_citation:2‡docs.quantinuum.com](https://docs.quantinuum.com/nexus/nexus_api/auth.html?utm_source=chatgpt.com)

        # Smoke test: hit an endpoint that requires a working session.
        # This will fail if tokens are invalid, network is blocked, etc.
        df = qnx.devices.get_all(
            nexus_hosted=nexus_hosted
        ).df()  #  [oai_citation:3‡docs.olcf.ornl.gov](https://docs.olcf.ornl.gov/quantum/hello_qcup.html)
        return (df is not None) and (len(df) > 0)

    except Exception as e:
        # You can log e for debugging (HTTPStatus, timeout, auth, etc.)
        return False


ok = validate_nexus_connection(nexus_hosted=True)
print("Nexus connection OK?", ok)


def generate_random_brick_wall_echo_puncture_after_U(
    L, W, coherent=False, meas_q=None, do_reset=False
):
    """
    Docstring for generate_random_brick_wall_echo_puncture_after_U

    :param L: Description
    :param W: Description
    :param coherent: Description
    :param meas_q: Description
    :param do_reset: Description
    """

    if meas_q is None:
        meas_q = W // 2

    qc = Circuit(W, 1)
    forward = []

    def layer_pairs(l):
        if l % 2 == 0:
            return [(2 * w, 2 * w + 1) for w in range(W // 2)]
        else:
            return [(2 * w + 1, 2 * w + 2) for w in range(W // 2 - 1)]

    # Build U
    for l in range(L):
        for i, j in layer_pairs(l):
            a, b, g = np.random.normal(size=3)
            forward.append((a, b, g, i, j))
            qc.TK2(a, b, g, i, j)

    # Puncture at "time L+1" (human), i.e. after the loop (code)
    if not coherent:
        qc.Measure(meas_q, 0)
        if do_reset:
            qc.Reset(meas_q)

    # Apply unconditional U^\dagger
    for a, b, g, i, j in reversed(forward):
        qc.TK2(-a, -b, -g, i, j)

    return qc


def debug_after_measurement(qc, n=20):
    """
    Docstring for debug_after_measurement

    :param qc: Description
    :param n: Description
    """

    cmds = qc.get_commands()

    # find the (first) Measure command
    m_idx = None
    for idx, cmd in enumerate(cmds):
        if cmd.op.type.name == "Measure":
            m_idx = idx
            break
    if m_idx is None:
        raise RuntimeError("No Measure found in circuit")

    print("MEASURE at command index:", m_idx)
    print("Next commands:")
    for k in range(m_idx + 1, min(m_idx + 1 + n, len(cmds))):
        cmd = cmds[k]
        optype = cmd.op.type.name
        qbs = [q.index[0] for q in cmd.qubits]  # qubit indices
        print(f"{k:4d}  {optype:10s}  qubits={qbs}")


def count_consecutive_tk2_after_measure(qc):
    """
    Docstring for count_consecutive_tk2_after_measure

    :param qc: Description
    """

    cmds = qc.get_commands()
    m_idx = next(i for i, c in enumerate(cmds) if c.op.type.name == "Measure")
    cnt = 0
    pairs = []
    for cmd in cmds[m_idx + 1 :]:
        if cmd.op.type.name != "TK2":
            break
        cnt += 1
        qbs = [q.index[0] for q in cmd.qubits]
        pairs.append(tuple(qbs))
    return cnt, pairs


def main():

    ok = validate_nexus_connection(nexus_hosted=True)
    print("Nexus connection OK?", ok)

    # submit job part
    incoherent_qc = generate_random_brick_wall_echo_puncture_after_U(
        10, 10, coherent=False
    )

    incoherent_qc.measure_all()

    my_project_ref = qnx.projects.get_or_create(
        name="measurement-induced butterfly effect"
    )

    # upload circuit
    circuit_ref = qnx.circuits.upload(
        circuit=incoherent_qc,
        name=f"{datetime.datetime.now()}_butterfly_incoherent_test_circuit_ref",
        project=my_project_ref,
    )

    # compile circuit
    compile_job_ref = qnx.start_compile_job(
        programs=[circuit_ref],
        name=f"{datetime.datetime.now()}_butterfly_incoherent_test_compiled_circuits",
        optimisation_level=1,
        backend_config=qnx.QuantinuumConfig(device_name="H2-1E"),
        project=my_project_ref,
    )

    qnx.jobs.wait_for(compile_job_ref)

    compiled_circuits = [
        item.get_output() for item in qnx.jobs.results(compile_job_ref)
    ]

    # execute circuit
    collapsed_circuit = qnx.execute(
        programs=compiled_circuits,
        name=f"{datetime.datetime.now()}_butterfly_incoherent_test_execution",
        n_shots=100,
        backend_config=qnx.QuantinuumConfig(device_name="H2-1E"),
        project=my_project_ref,
        timeout=10000,
    )

    incoherent_data = dict(collapsed_circuit[0].get_counts())

    return incoherent_data


if __name__ == "__main__":

    result = main()
