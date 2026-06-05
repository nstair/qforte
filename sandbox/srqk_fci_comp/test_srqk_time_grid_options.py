import numpy as np
import qforte as qf


# ============================================================
# User-editable options
# ============================================================
S = 30

# DT = 0.2
# DT_MANUAL = 'lambda_inv'
DT = 'lambda_inv'

TARGET_ROOT = 0
USE_EXACT_EVOLUTION = False
LOW_MEMORY = False
DIAGONALIZE_EACH_STEP = True
TROTTER_NUMBER = 1
TROTTER_ORDER = 2

QK_TMAX_TYPE = "manual"        # "manual", "variance", "auto"
QK_TMAX = None
QK_TIME_GRID = "linear"        # "linear", "quadratic", "power"
QK_TIME_POWER = 2.0

QK_TROTTER_CONTROL = "fixed"   # "fixed", "auto"
QK_TARGET_TROTTER_ERROR = 1.0e-3
QK_TROTTER_BOUND_SCALE = 1.0e-6

QK_VARIANCE_BETA = 2.0
GEV_STABILIZATION_THRESH = 1.0e-10

rHH = 1.5

# geom = [
#     ("H", (0.0, 0.0, 1.00 * rHH)),
#     ("H", (0.0, 0.0, 2.00 * rHH)),
#     ("H", (0.0, 0.0, 3.00 * rHH)),
#     ("H", (0.0, 0.0, 4.00 * rHH)),
#     ("H", (0.0, 0.0, 5.00 * rHH)),
#     ("H", (0.0, 0.0, 6.00 * rHH)),
#     ("H", (0.0, 0.0, 7.00 * rHH)),
#     ("H", (0.0, 0.0, 8.00 * rHH)),
# ]

geom = [
    ("N", (0.0, 0.0, 1.00)),
    ("N", (0.0, 0.0, 2.00)),
]

# geom = [
#     ("H", (0.0, 0.0, 1.00)),
#     ("Be", (0.0, 0.0, 2.00)),
#     ("H", (0.0, 0.0, 3.00)),
# ]

mol = qf.system_factory(
    build_type="psi4",
    mol_geometry=geom,
    basis="sto-3g",
    run_fci=1,
    store_mo_ints_np=True,
)


def compact(values):
    if values is None:
        return "None"
    return np.array2string(np.asarray(values), precision=8, separator=", ")


def build_alg():
    return qf.SRQK(
        mol,
        computer_type="fci",
        trotter_number=TROTTER_NUMBER,
        trotter_order=TROTTER_ORDER,
    )


def run_srqk(**kwargs):
    alg = build_alg()
    alg.run(
        s=S,
        dt=DT,
        target_root=TARGET_ROOT,
        use_exact_evolution=USE_EXACT_EVOLUTION,
        diagonalize_each_step=DIAGONALIZE_EACH_STEP,
        low_memory_mat_formation=LOW_MEMORY,
        **kwargs,
    )
    return alg


def common_kwargs():
    return {
        "qk_tmax": QK_TMAX,
        "qk_target_trotter_error": QK_TARGET_TROTTER_ERROR,
        "qk_trotter_bound_scale": QK_TROTTER_BOUND_SCALE,
        "qk_variance_beta": QK_VARIANCE_BETA,
        "gev_stabilization_thresh": GEV_STABILIZATION_THRESH,
    }


def print_case_result(name, alg):
    final_energy = alg.get_ts_energy()
    print(f"\n==> {name} <==")
    print(f"  final target energy:       {final_energy:+16.10f}")
    print(f"  error vs FCI:              {abs(final_energy - mol.fci_energy):.6e}")
    print(f"  S condition number:        {alg._Scond:.6e}")
    print(f"  selected qk_tmax:          {alg._qk_tmax:+16.10f}")
    print(f"  qk_time_points:            {compact(alg._qk_time_points)}")
    print(f"  qk_macro_dt_list:          {compact(alg._qk_macro_dt_list)}")
    print(f"  qk_trotter_control:        {alg._qk_trotter_control}")
    print(f"  macro trotter nums:        {compact(alg._qk_macro_trotter_number_list)}")
    print(f"  effective micro dt:        {compact(alg._qk_effective_micro_dt_list)}")

    if getattr(alg, "_qk_trotter_lambda_int", None) is not None:
        print(f"  Lambda_int:                {alg._qk_trotter_lambda_int:.6e}")
        print(f"  micro_dt_allowed:          {alg._qk_trotter_micro_dt_allowed:.6e}")
        print(f"  proxy estimate:            {alg._qk_trotter_proxy_estimate:.6e}")

    if getattr(alg, "_qk_hf_variance", None) is not None:
        print(f"  HF variance:               {alg._qk_hf_variance:.6e}")
        print(f"  sigma_H:                   {alg._qk_hf_sigma_h:.6e}")
        print(f"  T_variance:                {alg._qk_tmax_variance:.6e}")


def summarize_case(name, alg):
    macro_dt = np.asarray(alg._qk_macro_dt_list, dtype=float)
    macro_trotter = np.asarray(alg._qk_macro_trotter_number_list, dtype=int)
    longest_idx = int(np.argmax(np.abs(macro_dt)))
    final_rank = alg._qk_geig_reduced_rank(alg._S)

    return {
        "name": name,
        "final_energy": alg.get_ts_energy(),
        "final_abs_error": abs(alg.get_ts_energy() - mol.fci_energy),
        "tmax": float(alg._qk_tmax),
        "longest_macro_dt": float(abs(macro_dt[longest_idx])),
        "longest_macro_trotter_number": int(macro_trotter[longest_idx]),
        "largest_cnot": int(alg._n_cnot),
        "final_reduced_rank": int(final_rank),
    }


def print_bottom_summary(records, skipped):
    print("\n\n==> Case summary <==")
    print(f"  FCI reference energy:          {mol.fci_energy:+16.10f}")

    header = (
        f"{'case':34s} {'final E':>16s} {'|E-FCI|':>12s} "
        f"{'Tmax':>12s} {'max macro dt':>14s} {'m(max dt)':>10s} "
        f"{'max CNOT':>12s} {'final RR':>9s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for record in records:
        print(
            f"{record['name']:34s} "
            f"{record['final_energy']:+16.10f} "
            f"{record['final_abs_error']:12.4e} "
            f"{record['tmax']:12.4e} "
            f"{record['longest_macro_dt']:14.4e} "
            f"{record['longest_macro_trotter_number']:10d} "
            f"{record['largest_cnot']:12d} "
            f"{record['final_reduced_rank']:9d}"
        )

    if skipped:
        print("\nSkipped/error cases:")
        for name, exc in skipped:
            print(f"  {name}: {exc}")


print("\n\n==> SRQK QK time-grid option checks <==")
print(f"  FCI reference: {mol.fci_energy:+16.10f}")
print(f"  S:             {S}")
print(f"  DT:            {DT}")
print(f"  Trotter:       order={TROTTER_ORDER}, number={TROTTER_NUMBER}")

print("\n\n==> Default/manual behavior comparison <==")
alg_default = run_srqk()
alg_manual = run_srqk(
    qk_tmax_type="manual",
    qk_time_grid="linear",
    qk_trotter_control="fixed",
)

# expected_time_points = DT * np.arange(S + 1, dtype=float)
# expected_macro_dt = np.full(S, DT, dtype=float)
# expected_trotter_numbers = np.full(S, alg_default._trotter_number, dtype=int)

# print(f"  default time points:      {compact(alg_default._qk_time_points)}")
# print(f"  explicit time points:     {compact(alg_manual._qk_time_points)}")
# print(f"  default macro dt:         {compact(alg_default._qk_macro_dt_list)}")
# print(f"  explicit macro dt:        {compact(alg_manual._qk_macro_dt_list)}")
# print(f"  default trotter nums:     {compact(alg_default._qk_macro_trotter_number_list)}")
# print(f"  explicit trotter nums:    {compact(alg_manual._qk_macro_trotter_number_list)}")
# print(f"  max |dS|:                 {np.max(np.abs(alg_default._S - alg_manual._S)):.6e}")
# print(f"  max |dH|:                 {np.max(np.abs(alg_default._Hbar - alg_manual._Hbar)):.6e}")
# print(f"  |dE target|:              {abs(alg_default.get_ts_energy() - alg_manual.get_ts_energy()):.6e}")

# np.testing.assert_allclose(
#     alg_default._qk_time_points,
#     expected_time_points,
#     atol=0.0,
#     rtol=0.0,
# )
# np.testing.assert_allclose(
#     alg_default._qk_macro_dt_list,
#     expected_macro_dt,
#     atol=0.0,
#     rtol=0.0,
# )
# np.testing.assert_array_equal(
#     np.asarray(alg_default._qk_macro_trotter_number_list, dtype=int),
#     expected_trotter_numbers,
# )
# np.testing.assert_allclose(alg_default._qk_time_points, alg_manual._qk_time_points)
# np.testing.assert_allclose(alg_default._qk_macro_dt_list, alg_manual._qk_macro_dt_list)
# np.testing.assert_array_equal(
#     np.asarray(alg_default._qk_macro_trotter_number_list, dtype=int),
#     np.asarray(alg_manual._qk_macro_trotter_number_list, dtype=int),
# )
# np.testing.assert_allclose(alg_default._S, alg_manual._S, atol=1.0e-12, rtol=1.0e-12)
# np.testing.assert_allclose(alg_default._Hbar, alg_manual._Hbar, atol=1.0e-12, rtol=1.0e-12)
# np.testing.assert_allclose(
#     alg_default.get_ts_energy(),
#     alg_manual.get_ts_energy(),
#     atol=1.0e-12,
#     rtol=1.0e-12,
# )

cases = [
    # {
    #     "name": "user_editable_case",
    #     "qk_tmax_type": QK_TMAX_TYPE,
    #     "qk_time_grid": QK_TIME_GRID,
    #     "qk_time_power": QK_TIME_POWER,
    #     "qk_trotter_control": QK_TROTTER_CONTROL,
    # },
    {
        "name": "manual_linear_old_behavior",
        "qk_tmax_type": "manual",
        "qk_time_grid": "linear",
        "qk_trotter_control": "fixed",
    },
    # {
    #     "name": "manual_linear_auto_trotter",
    #     "qk_tmax_type": "manual",
    #     "qk_time_grid": "linear",
    #     "qk_trotter_control": "auto",
    # },
    {
        "name": "variance_linear_fixed_trotter",
        "qk_tmax_type": "variance",
        "qk_time_grid": "linear",
        "qk_trotter_control": "fixed",
    },
    {
        "name": "variance_linear_auto_trotter",
        "qk_tmax_type": "variance",
        "qk_time_grid": "linear",
        "qk_trotter_control": "auto",
    },
    {
        "name": "variance_power_auto_trotter",
        "qk_tmax_type": "variance",
        "qk_time_grid": "power",
        "qk_time_power": 2.0,
        "qk_trotter_control": "auto",
    },
    # {
    #     "name": "auto_power_auto_trotter",
    #     "qk_tmax_type": "auto",
    #     "qk_time_grid": "power",
    #     "qk_time_power": 2.0,
    #     "qk_trotter_control": "auto",
    # },
]

case_records = []
skipped_cases = []

for case in cases:
    case = dict(case)
    name = case.pop("name")
    kwargs = common_kwargs()
    kwargs.update(case)

    # if name == "manual_linear_old_behavior":
    #     kwargs["dt"] = "inv_lambda"

    try:
        alg = run_srqk(**kwargs)
    except Exception as exc:
        print(f"\n==> {name} <==")
        print(f"  skipped/error: {exc}")
        skipped_cases.append((name, exc))
        continue

    print_case_result(name, alg)
    case_records.append(summarize_case(name, alg))

print_bottom_summary(case_records, skipped_cases)
