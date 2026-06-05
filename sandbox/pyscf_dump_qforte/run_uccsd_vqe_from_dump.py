#!/usr/bin/env python3
"""Run qforte UCCSD-VQE from a saved PySCF dump."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import qforte as qf


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DUMPS = {
    "n2": THIS_DIR / "dumps" / "n2_sto3g_pyscf_dump.npz",
    "naphthalene": THIS_DIR / "dumps" / "naphthalene_avas_ccpvdz_pyscf_dump.npz",
    "napthalene": THIS_DIR / "dumps" / "naphthalene_avas_ccpvdz_pyscf_dump.npz",
}

# ---------------------------------------------------------------------------
# Edit-and-run controls.
# ---------------------------------------------------------------------------
SYSTEM = "n2"  # "n2", "naphthalene", or "napthalene"
DUMP_FILE = None  # Set to a Path/string to override DEFAULT_DUMPS[SYSTEM].

COMPUTER_TYPE = "fci"
APPLY_HAM_AS_TENSOR = True
BUILD_QB_HAM = False
VERBOSE = False
PRINT_SUMMARY_FILE = False

USE_HESSIAN_DIAG = True

INIT_AMPS = "mp2"  # "zero" or "mp2"

RUN_OPTIONS = {
    "pool_type": "SD",
    "optimizer": "bfgs_qf",
    "opt_maxiter": 20,
    "opt_thresh": 1.0e-4,
    "opt_ftol": 1.0e-12,
    "use_analytic_grad": True,
    "init_amps": INIT_AMPS, # mp2 # zero
    "primary_pool_order": "none",
    "secondary_pool_order": "shell",
    "general_ex_pool_order": "default",
    "bfgs_qf_maxiter": 20,
    "bfgs_qf_max_ls": 10,
    "bfgs_qf_alpha0": 1.0,
    "bfgs_qf_max_step_norm": 0.5,
    "bfgs_qf_use_gradient_energy": True,
    "lbfgs_qf_max_ls": 10,
    "lbfgs_qf_alpha0": 1.0,
    "lbfgs_qf_max_step_norm": 0.5,
    "lbfgs_qf_use_gradient_energy": True,

    # Exact Hessian diagonal initialization/preconditioning.
    "bfgs_qf_use_hessian_diag": USE_HESSIAN_DIAG,
    "bfgs_qf_hdiag_start": 1,
    "bfgs_qf_hdiag_stop": 3, # WOAH!!
    "bfgs_qf_hdiag_update_freq": 1,
    "bfgs_qf_hdiag_floor": 1.0e-3,
    "bfgs_qf_hdiag_mode": "abs", # abs     # "positive", "abs", or "none"
    # "analytic"/"recursive", "finite_difference"/"fd", or "mp2".
    "bfgs_qf_hdiag_method": "mp2", # analytic
    "bfgs_qf_hdiag_fd_step": 1.0e-4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(DEFAULT_DUMPS))
    parser.add_argument("--dump-file", type=Path)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--init-amps", choices=["zero", "mp2"])
    parser.add_argument("--opt-thresh", type=float)
    parser.add_argument("--build-qb-ham", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    return parser


def runtime_settings(args: argparse.Namespace) -> dict:
    """Merge top-of-file controls with any explicit command-line overrides."""
    system = args.system or SYSTEM
    if system not in DEFAULT_DUMPS:
        raise ValueError(
            f"Unknown SYSTEM={system!r}; choose one of {sorted(DEFAULT_DUMPS)}."
        )

    dump_file = args.dump_file or DUMP_FILE or DEFAULT_DUMPS[system]
    run_options = dict(RUN_OPTIONS)
    if args.maxiter is not None:
        run_options["opt_maxiter"] = args.maxiter
        run_options["bfgs_qf_maxiter"] = args.maxiter
    if args.optimizer is not None:
        run_options["optimizer"] = args.optimizer
    if args.init_amps is not None:
        run_options["init_amps"] = args.init_amps
    if args.opt_thresh is not None:
        run_options["opt_thresh"] = args.opt_thresh

    build_qb_ham = BUILD_QB_HAM if args.build_qb_ham is None else args.build_qb_ham
    verbose = VERBOSE if args.verbose is None else args.verbose

    return {
        "system": system,
        "dump_file": Path(dump_file),
        "build_qb_ham": build_qb_ham,
        "verbose": verbose,
        "run_options": run_options,
    }


def main() -> None:
    settings = runtime_settings(build_parser().parse_args())
    dump_file = settings["dump_file"]
    if not dump_file.exists():
        raise FileNotFoundError(
            f"Missing dump file {dump_file}. Run the matching dump generator first."
        )

    mol = qf.system_factory(
        system_type="molecule",
        build_type="pyscf_dump",
        dump_file=str(dump_file),
        build_qb_ham=settings["build_qb_ham"],
        store_mo_ints=True,
        store_mo_ints_np=True,
    )

    metadata = getattr(mol, "pyscf_dump_metadata", {})
    print("\n==> Loaded PySCF dump <==")
    print(f"  file:      {dump_file}")
    print(f"  label:     {metadata.get('system_label', settings['system'])}")
    print(f"  active:    ({sum(mol.hf_reference)}e, {len(mol.hf_reference) // 2}o)")
    print(f"  RHF:       {getattr(mol, 'hf_energy', None)}")
    if getattr(mol, "fci_energy_list", []):
        print(f"  FCI/CASCI: {mol.fci_energy_list[0]:+.12f}")

    alg = qf.UCCNVQE(
        mol,
        computer_type=COMPUTER_TYPE,
        apply_ham_as_tensor=APPLY_HAM_AS_TENSOR,
        verbose=settings["verbose"],
        print_summary_file=PRINT_SUMMARY_FILE,
    )

    alg.run(**settings["run_options"])

    final_energy = float(np.real(alg.get_gs_energy()))
    print("\n==> qforte UCCSD-VQE from dump <==")
    print(f"  final energy: {final_energy:+.12f}")
    if getattr(mol, "fci_energy_list", []):
        print(f"  error vs FCI: {final_energy - mol.fci_energy_list[0]:+.6e}")


if __name__ == "__main__":
    main()
