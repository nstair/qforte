#!/usr/bin/env python3
"""Run unrelaxed AVAS-DMRG-CASCI calculations for chromium dimer.

This follows the same workflow as the acene AVAS-DMRG sandbox script:

    PySCF mean field -> AVAS active-space orbitals -> DMRG-CASCI

It intentionally does not run DMRG-SCF/CASSCF.  The AVAS orbitals are held fixed
for the correlated calculation, so the final energy is an unrelaxed CASCI-style
energy in the selected active space.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Atom = Tuple[str, Tuple[float, float, float]]

THIS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# User options for the simple edit-and-run workflow.
#
# With these defaults, this file can be run directly as:
#
#     python run_avas_cr2_drmg.py
#
# The filename intentionally matches the requested name, including "drmg".
# ---------------------------------------------------------------------------

# Geometry: neutral Cr2 centered on the origin, bond axis along z.
CR_CR_DISTANCE_ANGSTROM = 1.68

# Electronic state.  For Cr2 singlet, spin = N_alpha - N_beta = 0.
CHARGE = 0
SPIN = 0

# A modest all-electron basis to start.  For production Cr2 studies, you may
# want a larger basis and/or scalar-relativistic treatment, but this keeps the
# sandbox calculation approachable.
BASIS = "def2-svp"
SYMMETRY = False
MEMORY_MB = 8000
VERBOSE = 3

SCF_CONV_TOL = 1.0e-10
SCF_MAX_CYCLE = 200
SCF_LEVEL_SHIFT = 0.0
SCF_INIT_GUESS = "minao"
DENSITY_FIT = False
USE_NEWTON = False

# AVAS presets.  These sizes were checked in AVAS-only mode for the default
# Cr2/def2-svp/RHF setup, AVAS_THRESHOLD = 0.3, and AVAS_MINAO = BASIS:
#
#   valence_3d4s         -> (12e, 12o)
#   valence_3d4s4p       -> (12e, 18o)
#   valence_3d4s4p5s     -> (12e, 20o)
#   double_shell_3d4s4d  -> (12e, 22o)
#
# If AVAS_MINAO is changed back to "minao", the larger labels below do not
# expand the active space in this def2-svp test: at threshold 0.3 all four
# presets select (12e, 12o), and at threshold <= 0.2 they select (14e, 13o).
AVAS_LABEL_PRESETS: Dict[str, List[str]] = {
    # Usual Cr2 valence active space: 3d^5 4s^1 on each Cr -> (12e, 12o).
    "valence_3d4s": ["Cr 3d", "Cr 4s"],

    # Adds Cr 4p correlating/polarization orbitals -> (12e, 18o).
    "valence_3d4s4p": ["Cr 3d", "Cr 4s", "Cr 4p"],

    # Adds the next s-like correlating pair in this basis -> (12e, 20o).
    "valence_3d4s4p5s": ["Cr 3d", "Cr 4s", "Cr 4p", "Cr 5s"],

    # Compact double-shell check -> (12e, 22o), slightly above the default
    # MAX_ACTIVE_ORBITALS guard.  Raise that guard before running DMRG here.
    "double_shell_3d4s4d": ["Cr 3d", "Cr 4s", "Cr 4d"],
}

AVAS_PRESET = "valence_3d4s"
AVAS_LABELS = AVAS_LABEL_PRESETS[AVAS_PRESET]
# For the default def2-svp / Cr 3d+4s choice, 0.3 selects the usual
# (12e, 12o) Cr2 valence active space in the test environment used here.
AVAS_THRESHOLD = 0.3
# Use the calculation basis as the AVAS reference so that labels such as Cr 4p,
# Cr 5s, and Cr 4d actually select the corresponding def2-svp AO subspaces.
# Setting this to "minao" is useful for the compact valence space, but it makes
# the larger presets above collapse to the same selected space in this setup.
#
# Basis caveat: if BASIS is changed to "cc-pvdz", AVAS_MINAO = BASIS makes the
# valence_3d4s4p5s preset select (18e, 22o) at threshold 0.3.  To run a cc-pVDZ
# orbital calculation with the more compact (12e, 20o) AVAS space, set
# AVAS_MINAO = "def2-svp" or pass --avas-minao def2-svp.
AVAS_MINAO = BASIS
AVAS_WITH_IAO = False
AVAS_CANONICALIZE = True

# For the default Cr 3d/4s AVAS choice, the target is the usual (12e, 12o).
# For valence_3d4s4p5s use EXPECT_ACTIVE = (12, 20); for exploratory scans use
# None or the --no-expect-active command-line option.
EXPECT_ACTIVE = (12, 12)

# Safety check for exploratory AVAS choices.  This does not truncate orbitals;
# it just stops if AVAS selected a space larger than intended.
MAX_ACTIVE_ORBITALS = 20

# DMRG-CASCI controls.  Keep RUN_DMRG False for quick AVAS inspection.  Set it
# True once the active space looks right.
RUN_DMRG = False
BOND_DIM = 500
DMRG_TOL = 1.0e-8
DMRG_THREADS = None
DMRG_MEMORY_MB = None
BLOCK_EXE = None
MPI_PREFIX = None
SCRATCH_DIR = THIS_DIR / "scratch_block2"
RUNTIME_DIR = THIS_DIR / "runtime_block2"
AUTO_ORDER = True
BLOCK_EXTRA_KEYWORDS: List[str] = []

# Print spin-summed active-space natural orbital occupations after DMRG-CASCI.
# These are eigenvalues of the active-space 1-RDM and should sum to nelecas.
PRINT_NATURAL_OCCUPATIONS = True

# Block2 stores the final MPS in scratch files named F.MPS.KET.*.  This
# diagnostic clears stale MPS tensor files before a DMRG run, lets Block2 write
# the new MPS, then reads the resulting file sizes to estimate the actual number
# of stored real-double MPS parameters.  This captures the large reduction from
# particle-number, spin, and other symmetry blocking.
REPORT_MPS_PARAMETERS = True
CLEAN_MPS_FILES_BEFORE_DMRG = True
DELETE_MPS_FILES_AFTER_REPORT = True
CLEAN_BLOCK2_SCRATCH_AFTER_RUN = True

# Human-readable run summary.  The terminal output is useful while watching a
# run; this file is meant to be a compact record for later comparison.
WRITE_OUTPUT_FILE = True
OUTPUT_FILE = THIS_DIR / "cr2_avas_dmrg_report.txt"


def cr2_geometry(distance_angstrom: float) -> List[Atom]:
    """Return a Cr2 geometry centered on the origin in Angstrom."""
    half = 0.5 * distance_angstrom
    return [
        ("Cr", (0.0, 0.0, -half)),
        ("Cr", (0.0, 0.0, half)),
    ]


def geometry_to_pyscf_atom(geom: Sequence[Atom]) -> str:
    """Format a geometry as a PySCF atom string."""
    return "\n".join(
        f"{symbol:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}"
        for symbol, xyz in geom
    )


def active_electron_count(nelecas) -> int:
    """Return the total active-electron count from PySCF's nelecas value."""
    if isinstance(nelecas, (tuple, list)):
        return int(sum(nelecas))
    return int(nelecas)


def format_nelecas(nelecas) -> str:
    """Format PySCF's active-electron descriptor for output."""
    if isinstance(nelecas, (tuple, list)):
        return f"{sum(nelecas)} total ({nelecas[0]} alpha, {nelecas[1]} beta)"
    return str(nelecas)


def active_spin_counts(nelecas) -> Tuple[int, int]:
    """Return active alpha/beta electron counts from PySCF's nelecas value."""
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]), int(nelecas[1])
    nelec = int(nelecas)
    return (nelec + 1) // 2, nelec // 2


def determinant_coefficient_count(ncas: int, nelecas) -> int:
    """Return the alpha/beta determinant-tensor coefficient count."""
    nalpha, nbeta = active_spin_counts(nelecas)
    return comb(ncas, nalpha) * comb(ncas, nbeta)


def get_solver_scratch_dir(args, solver) -> Path | None:
    """Find the Block2 scratch directory used for MPS tensor files."""
    if args.scratch_dir is not None:
        return Path(args.scratch_dir)
    scratch = getattr(solver, "scratchDirectory", None)
    if scratch:
        return Path(scratch)
    return None


def sorted_mps_tensor_files(scratch_dir: Path) -> List[Path]:
    """Return Block2 final-MPS tensor files in deterministic site order."""
    def sort_key(path: Path) -> Tuple[int, str]:
        try:
            return int(path.name.rsplit(".", 1)[-1]), path.name
        except ValueError:
            return 10**9, path.name

    return sorted(scratch_dir.glob("F.MPS.KET.*"), key=sort_key)


def clean_mps_tensor_files(scratch_dir: Path | None) -> int:
    """Remove stale Block2 MPS tensor files before a fresh diagnostic run."""
    if scratch_dir is None:
        return 0

    removed = 0
    for path in sorted_mps_tensor_files(scratch_dir):
        path.unlink()
        removed += 1
    return removed


def clean_block2_scratch_dir(scratch_dir: Path | None) -> Dict[str, object]:
    """Clear Block2 scratch contents while leaving the scratch directory itself.

    This is intentionally a little conservative.  The script defaults point at
    dedicated scratch_block2 directories; if a dangerous path is accidentally
    supplied, refuse to clean it.
    """
    if scratch_dir is None:
        return {"skipped": True, "reason": "no scratch directory configured"}

    scratch_dir = Path(scratch_dir)
    if not scratch_dir.exists():
        scratch_dir.mkdir(parents=True, exist_ok=True)
        return {
            "skipped": False,
            "scratch_dir": str(scratch_dir),
            "removed_files": 0,
            "removed_dirs": 0,
        }

    resolved = scratch_dir.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), THIS_DIR.resolve(), THIS_DIR.parent.resolve()}
    if resolved in forbidden:
        return {"skipped": True, "reason": f"refusing to clean unsafe path {resolved}"}

    removed_files = 0
    removed_dirs = 0
    for child in scratch_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            removed_dirs += 1
        else:
            child.unlink()
            removed_files += 1

    return {
        "skipped": False,
        "scratch_dir": str(scratch_dir),
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
    }


def print_mps_parameter_report(args, solver, ncas: int, nelecas) -> Dict[str, object]:
    """Print an actual stored-MPS size estimate from Block2 tensor files.

    Block2 stores symmetry-blocked MPS tensors, so the file sizes are a useful
    proxy for the real number of stored MPS amplitudes after particle-number,
    spin, and any point-group blocking have reduced the dense tensor shapes.
    The reported double-word count includes small file metadata/header overhead,
    so it is a slight overestimate of the pure variational parameter count.
    """
    scratch_dir = get_solver_scratch_dir(args, solver)
    if scratch_dir is None:
        reason = "no Block2 scratch directory was found"
        print(f"\nMPS parameter report skipped: {reason}.")
        return {"skipped": True, "reason": reason}

    files = sorted_mps_tensor_files(scratch_dir)
    if not files:
        reason = f"no F.MPS.KET.* files in {scratch_dir}"
        print(f"\nMPS parameter report skipped: {reason}")
        return {"skipped": True, "reason": reason, "scratch_dir": str(scratch_dir)}

    byte_count = sum(path.stat().st_size for path in files)
    stored_parameters = (byte_count + 7) // 8
    full_coeffs = determinant_coefficient_count(ncas, nelecas)
    compression = full_coeffs / stored_parameters if stored_parameters else float("inf")
    removed = 0

    print("\n==> Block2 MPS storage diagnostic <==")
    print(f"  scratch directory:       {scratch_dir}")
    print(f"  MPS tensor files:        {len(files)}")
    print(f"  MPS tensor bytes:        {byte_count}")
    print(f"  stored parameter bound:  {stored_parameters}")
    print(f"  full CAS coefficients:   {full_coeffs}")
    print(f"  compression factor:      {compression:.3e}")
    print("  note: parameter bound is ceil(bytes/8) and includes small file metadata")

    if args.delete_mps_files_after_report:
        removed = clean_mps_tensor_files(scratch_dir)
        print(f"  removed MPS files:       {removed}")

    return {
        "skipped": False,
        "scratch_dir": str(scratch_dir),
        "tensor_file_count": len(files),
        "tensor_bytes": byte_count,
        "stored_parameter_bound": stored_parameters,
        "full_cas_coefficients": full_coeffs,
        "compression_factor": compression,
        "removed_mps_files": removed,
    }


def import_pyscf():
    """Import the PySCF modules used by this script."""
    from pyscf import gto, mcscf, scf
    from pyscf.mcscf import avas

    return gto, scf, mcscf, avas


def build_mean_field(args):
    """Build and run the mean-field reference used by AVAS."""
    gto, scf, _mcscf, _avas = import_pyscf()

    mol = gto.Mole()
    mol.build(
        atom=geometry_to_pyscf_atom(cr2_geometry(args.distance)),
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit="Angstrom",
        symmetry=args.symmetry,
        verbose=args.verbose,
        max_memory=args.memory_mb,
    )

    if args.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)

    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = args.scf_max_cycle
    mf.level_shift = args.scf_level_shift
    mf.init_guess = args.scf_init_guess

    if args.density_fit:
        mf = mf.density_fit()
    if args.use_newton:
        mf = mf.newton()

    print("\n==> Mean-field reference <==")
    print(f"  system:            Cr2")
    print(f"  Cr-Cr distance:    {args.distance:.6f} Angstrom")
    print(f"  basis:             {args.basis}")
    print(f"  charge/spin:       {args.charge} / {args.spin}")
    print(f"  atoms:             {mol.natm}")
    print(f"  electrons:         {mol.nelectron}")
    print(f"  SCF type:          {'RHF' if args.spin == 0 else 'ROHF'}")
    print(f"  density fitting:   {args.density_fit}")
    print(f"  Newton SCF:        {args.use_newton}")

    escf = mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge; tune SCF options before AVAS/DMRG.")

    print(f"  SCF energy:        {escf:.12f} Eh")
    return mol, mf


def run_avas(args, mf):
    """Run AVAS for the requested Cr atomic-orbital labels."""
    _gto, _scf, _mcscf, avas = import_pyscf()

    avas_obj = avas.AVAS(
        mf,
        args.avas_labels,
        threshold=args.avas_threshold,
        minao=args.avas_minao,
        with_iao=args.avas_with_iao,
        canonicalize=args.avas_canonicalize,
        verbose=args.verbose,
    )
    ncas, nelecas, mo_coeff = avas_obj.kernel()
    nelec_active = active_electron_count(nelecas)

    print("\n==> AVAS active space <==")
    print(f"  preset:            {args.avas_preset}")
    print(f"  AO labels:         {args.avas_labels}")
    print(f"  threshold:         {args.avas_threshold}")
    print(f"  minao:             {args.avas_minao}")
    print(f"  with IAO:          {args.avas_with_iao}")
    print(f"  canonicalize:      {args.avas_canonicalize}")
    print(f"  active electrons:  {format_nelecas(nelecas)}")
    print(f"  active orbitals:   {ncas}")
    print(f"  compact notation:  ({nelec_active}e, {ncas}o)")

    if args.expect_active is not None:
        expected_e, expected_o = args.expect_active
        if (nelec_active, int(ncas)) != (expected_e, expected_o):
            raise RuntimeError(
                "Unexpected AVAS active space: "
                f"got ({nelec_active}e, {ncas}o), "
                f"expected ({expected_e}e, {expected_o}o)."
            )

    if args.max_active_orbitals is not None and int(ncas) > args.max_active_orbitals:
        raise RuntimeError(
            "AVAS selected more orbitals than requested: "
            f"got {ncas}, max is {args.max_active_orbitals}. "
            "Raise MAX_ACTIVE_ORBITALS or choose a smaller preset/threshold."
        )

    return int(ncas), nelecas, mo_coeff


def import_dmrgscf():
    """Import the optional PySCF DMRG plugin."""
    try:
        return importlib.import_module("pyscf.dmrgscf")
    except ImportError as exc:
        raise RuntimeError(
            "PySCF's DMRG plugin is not importable as pyscf.dmrgscf. "
            "Use an environment with pyscf-dmrgscf and block2 installed, such "
            "as new_pyscf_env."
        ) from exc


def find_block_exe(args) -> str | None:
    """Find a Block/Block2 executable from explicit option, env, or PATH."""
    if args.block_exe is not None:
        return str(args.block_exe)
    if os.environ.get("BLOCKEXE"):
        return os.environ["BLOCKEXE"]
    return shutil.which("block2main") or shutil.which("block2")


def configure_dmrg_solver(args, mol):
    """Create a DMRGCI solver for an unrelaxed CASCI calculation."""
    dmrgscf = import_dmrgscf()
    block_exe = find_block_exe(args)

    if block_exe is not None and hasattr(dmrgscf, "settings"):
        dmrgscf.settings.BLOCKEXE = block_exe
    if args.mpi_prefix is not None and hasattr(dmrgscf, "settings"):
        dmrgscf.settings.MPIPREFIX = args.mpi_prefix

    solver = dmrgscf.DMRGCI(mol, maxM=args.bond_dim, tol=args.dmrg_tol)

    if hasattr(solver, "maxM"):
        solver.maxM = args.bond_dim
    if hasattr(solver, "tol"):
        solver.tol = args.dmrg_tol
    if hasattr(solver, "nroots"):
        solver.nroots = 1
    if hasattr(solver, "spin"):
        solver.spin = mol.spin
    if args.scratch_dir is not None and hasattr(solver, "scratchDirectory"):
        solver.scratchDirectory = str(args.scratch_dir)
    if args.runtime_dir is not None and hasattr(solver, "runtimeDir"):
        solver.runtimeDir = str(args.runtime_dir)
    if args.dmrg_threads is not None and hasattr(solver, "num_thrds"):
        solver.num_thrds = args.dmrg_threads
    if args.dmrg_memory_mb is not None and hasattr(solver, "memory"):
        solver.memory = args.dmrg_memory_mb

    extra_keywords = list(args.block_extra_keyword)
    if not args.auto_order and not any(k.lower() == "noreorder" for k in extra_keywords):
        # Block2 automatically applies its Fiedler orbital ordering when no
        # explicit reorder keyword is present.  To disable that machinery, emit
        # noreorder.  This never changes the fact that the calculation is CASCI:
        # AVAS orbitals are not optimized by the DMRG solver.
        extra_keywords.append("noreorder")
    if extra_keywords and hasattr(solver, "block_extra_keyword"):
        solver.block_extra_keyword = extra_keywords

    return solver


def run_dmrg_casci(args, mf, ncas: int, nelecas, mo_coeff):
    """Run unrelaxed DMRG-CASCI in the AVAS orbital basis."""
    _gto, _scf, mcscf, _avas = import_pyscf()

    casci = mcscf.CASCI(mf, ncas, nelecas)
    casci.fcisolver = configure_dmrg_solver(args, mf.mol)
    casci.verbose = args.verbose

    scratch_dir = get_solver_scratch_dir(args, casci.fcisolver)
    if args.report_mps_parameters and args.clean_mps_files_before_dmrg:
        removed = clean_mps_tensor_files(scratch_dir)
        if removed:
            print(f"\nRemoved {removed} stale Block2 MPS tensor files before DMRG.")

    if hasattr(casci, "canonicalization"):
        casci.canonicalization = False

    print("\n==> Unrelaxed AVAS-DMRG-CASCI <==")
    print(f"  DMRG bond dimension: {args.bond_dim}")
    print(f"  DMRG tolerance:      {args.dmrg_tol}")
    print(f"  auto orbital order:  {args.auto_order}")
    print("  orbital relaxation:  disabled (CASCI, not DMRG-SCF)")

    result = casci.kernel(mo_coeff)
    energy = float(result[0])

    print("\n==> Result <==")
    print(f"  AVAS-DMRG-CASCI energy: {energy:.12f} Eh")

    occupations = compute_active_natural_occupations(casci, ncas, nelecas)
    if args.print_natural_occupations:
        print_active_natural_occupations(occupations)

    mps_report = None
    if args.report_mps_parameters:
        mps_report = print_mps_parameter_report(args, casci.fcisolver, ncas, nelecas)

    scratch_cleanup = None
    if args.clean_block2_scratch_after_run:
        scratch_cleanup = clean_block2_scratch_dir(scratch_dir)
        if scratch_cleanup.get("skipped"):
            print(f"\nBlock2 scratch cleanup skipped: {scratch_cleanup.get('reason')}")
        else:
            print(
                "\nCleared Block2 scratch directory: "
                f"{scratch_cleanup['scratch_dir']} "
                f"({scratch_cleanup['removed_files']} files, "
                f"{scratch_cleanup['removed_dirs']} directories)"
            )

    return {
        "energy": energy,
        "bond_dim": args.bond_dim,
        "tol": args.dmrg_tol,
        "auto_order": args.auto_order,
        "orbital_relaxation": "disabled (CASCI, not DMRG-SCF)",
        "natural_occupations": occupations,
        "mps_report": mps_report,
        "scratch_cleanup": scratch_cleanup,
    }


def compute_active_natural_occupations(casci, ncas: int, nelecas) -> List[float]:
    """Return active-space natural occupations from the DMRG 1-RDM.

    The PySCF DMRG solver returns the spin-summed spatial active-space 1-RDM.
    Its eigenvalues are the active-space natural orbital occupations, with
    values between 0 and 2 for a closed-shell spatial-orbital representation.
    This is still an analysis of the unrelaxed CASCI wavefunction; no orbitals
    are optimized or fed back into the DMRG calculation.
    """
    import numpy as np

    dm1 = casci.fcisolver.make_rdm1(casci.ci, ncas, nelecas)
    dm1 = 0.5 * (dm1 + dm1.T.conj())
    occupations = np.linalg.eigvalsh(dm1)
    occupations = np.sort(occupations)[::-1]
    return [float(occ) for occ in occupations]


def print_active_natural_occupations(occupations: Sequence[float]) -> None:
    """Print active-space natural occupations."""
    print("\n==> Active-space natural occupations <==")
    print(f"  trace(1-RDM):       {sum(occupations):.12f}")
    for idx, occ in enumerate(occupations, start=1):
        print(f"  NO {idx:3d}:          {occ: .12f}")


def parse_expect_active(text: str) -> Tuple[int, int]:
    """Parse --expect-active values like '12,12' or '12e,12o'."""
    import re

    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) != 2:
        raise argparse.ArgumentTypeError("Expected active space as 'nelec,norb'.")
    return nums[0], nums[1]


def write_run_report(args, scf_energy: float, ncas: int, nelecas, dmrg_result) -> None:
    """Write a compact, readable record of the AVAS/DMRG calculation."""
    if not args.write_output_file:
        return

    nalpha, nbeta = active_spin_counts(nelecas)
    nelec_active = active_electron_count(nelecas)
    full_coeffs = determinant_coefficient_count(ncas, nelecas)
    mps_report = dmrg_result.get("mps_report") if dmrg_result else None
    scratch_cleanup = dmrg_result.get("scratch_cleanup") if dmrg_result else None
    occupations = dmrg_result.get("natural_occupations") if dmrg_result else None

    lines = [
        "Cr2 AVAS-DMRG-CASCI Run Report",
        "=" * 32,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Workflow",
        "--------",
        "PySCF mean-field -> AVAS active space -> unrelaxed DMRG-CASCI",
        "Orbital relaxation: disabled (CASCI, not DMRG-SCF)",
        "",
        "System",
        "------",
        f"Molecule: Cr2",
        f"Cr-Cr distance / Angstrom: {args.distance:.12f}",
        f"Charge: {args.charge}",
        f"Spin (N_alpha - N_beta): {args.spin}",
        f"Basis: {args.basis}",
        f"Symmetry enabled: {args.symmetry}",
        f"Memory / MB: {args.memory_mb}",
        "",
        "Mean Field",
        "----------",
        f"Reference: {'RHF' if args.spin == 0 else 'ROHF'}",
        f"SCF convergence tolerance: {args.scf_conv_tol:.6e}",
        f"SCF max cycles: {args.scf_max_cycle}",
        f"SCF level shift: {args.scf_level_shift}",
        f"SCF initial guess: {args.scf_init_guess}",
        f"Density fitting: {args.density_fit}",
        f"Newton SCF: {args.use_newton}",
        f"HF energy / Eh: {scf_energy:.15f}",
        "",
        "AVAS",
        "----",
        f"Preset: {args.avas_preset}",
        f"Selected AO labels: {args.avas_labels}",
        f"Threshold: {args.avas_threshold}",
        f"Reference/minao basis: {args.avas_minao}",
        f"With IAO: {args.avas_with_iao}",
        f"Canonicalize: {args.avas_canonicalize}",
        f"Active electrons: {nelec_active}",
        f"Active alpha electrons: {nalpha}",
        f"Active beta electrons: {nbeta}",
        f"Active orbitals: {ncas}",
        f"Compact notation: ({nelec_active}e, {ncas}o)",
        f"Full alpha/beta CAS coefficients: {full_coeffs}",
        "",
        "DMRG",
        "----",
    ]

    if dmrg_result:
        lines.extend(
            [
                "Run DMRG: yes",
                f"DMRG energy / Eh: {dmrg_result['energy']:.15f}",
                f"Bond dimension maxM: {dmrg_result['bond_dim']}",
                f"DMRG tolerance: {dmrg_result['tol']:.6e}",
                f"Auto orbital ordering: {dmrg_result['auto_order']}",
                f"DMRG threads: {args.dmrg_threads}",
                f"DMRG memory / MB: {args.dmrg_memory_mb}",
                f"Block executable: {args.block_exe}",
                f"MPI prefix: {args.mpi_prefix}",
                f"Scratch directory: {args.scratch_dir}",
                f"Runtime directory: {args.runtime_dir}",
                f"Block extra keywords: {args.block_extra_keyword}",
            ]
        )
    else:
        lines.append("Run DMRG: no")

    lines.extend(["", "MPS Storage", "-----------"])
    if isinstance(mps_report, dict) and not mps_report.get("skipped"):
        lines.extend(
            [
                f"MPS scratch directory: {mps_report['scratch_dir']}",
                f"MPS tensor files counted: {mps_report['tensor_file_count']}",
                f"MPS tensor bytes: {mps_report['tensor_bytes']}",
                f"Stored MPS parameter bound: {mps_report['stored_parameter_bound']}",
                f"Full CAS coefficients: {mps_report['full_cas_coefficients']}",
                f"Compression factor: {mps_report['compression_factor']:.6e}",
                f"Removed MPS tensor files after count: {mps_report['removed_mps_files']}",
                "Parameter bound is ceil(bytes/8) and includes small Block2 file metadata.",
            ]
        )
    elif isinstance(mps_report, dict):
        lines.append(f"MPS report skipped: {mps_report.get('reason')}")
    else:
        lines.append("MPS report unavailable.")

    lines.extend(["", "Block2 Scratch Cleanup", "----------------------"])
    if isinstance(scratch_cleanup, dict) and not scratch_cleanup.get("skipped"):
        lines.extend(
            [
                f"Scratch directory cleared: {scratch_cleanup['scratch_dir']}",
                f"Removed files: {scratch_cleanup['removed_files']}",
                f"Removed directories: {scratch_cleanup['removed_dirs']}",
                "The scratch directory itself is left in place for future runs.",
            ]
        )
    elif isinstance(scratch_cleanup, dict):
        lines.append(f"Scratch cleanup skipped: {scratch_cleanup.get('reason')}")
    else:
        lines.append("Scratch cleanup was not run.")

    lines.extend(["", "Active Natural Occupations", "--------------------------"])
    if occupations:
        lines.append(f"Trace: {sum(occupations):.12f}")
        for idx, occ in enumerate(occupations, start=1):
            lines.append(f"NO {idx:3d}: {occ: .12f}")
    else:
        lines.append("Not available because DMRG was not run.")

    lines.extend(
        [
            "",
            "Notes",
            "-----",
            "Natural occupations are eigenvalues of the spin-summed active-space 1-RDM.",
            "The MPS parameter count is inferred from Block2 F.MPS.KET.* tensor files.",
        ]
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text("\n".join(lines) + "\n")
    print(f"\nWrote run report: {args.output_file}")


def default_avas_labels(preset: str) -> List[str]:
    """Return a copy of the labels for an AVAS preset."""
    if preset not in AVAS_LABEL_PRESETS:
        valid = ", ".join(sorted(AVAS_LABEL_PRESETS))
        raise ValueError(f"Unknown AVAS preset {preset!r}. Valid presets: {valid}")
    return list(AVAS_LABEL_PRESETS[preset])


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run PySCF RHF/ROHF, select Cr AVAS orbitals, and optionally run "
            "an unrelaxed DMRG-CASCI calculation."
        )
    )

    parser.add_argument("--distance", type=float, default=CR_CR_DISTANCE_ANGSTROM)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--charge", type=int, default=CHARGE)
    parser.add_argument("--spin", type=int, default=SPIN)
    parser.add_argument("--symmetry", default=SYMMETRY)
    parser.add_argument("--memory-mb", type=int, default=MEMORY_MB)
    parser.add_argument("--verbose", type=int, default=VERBOSE)

    parser.add_argument("--scf-conv-tol", type=float, default=SCF_CONV_TOL)
    parser.add_argument("--scf-max-cycle", type=int, default=SCF_MAX_CYCLE)
    parser.add_argument("--scf-level-shift", type=float, default=SCF_LEVEL_SHIFT)
    parser.add_argument("--scf-init-guess", default=SCF_INIT_GUESS)
    parser.add_argument("--density-fit", action="store_true", default=DENSITY_FIT)
    parser.add_argument("--use-newton", action="store_true", default=USE_NEWTON)

    parser.add_argument("--avas-preset", default=AVAS_PRESET, choices=sorted(AVAS_LABEL_PRESETS))
    parser.add_argument("--avas-labels", nargs="+")
    parser.add_argument("--avas-threshold", type=float, default=AVAS_THRESHOLD)
    parser.add_argument("--avas-minao", default=AVAS_MINAO)
    parser.add_argument("--avas-with-iao", action="store_true", default=AVAS_WITH_IAO)
    parser.add_argument("--no-avas-canonicalize", dest="avas_canonicalize", action="store_false")
    parser.set_defaults(avas_canonicalize=AVAS_CANONICALIZE)

    parser.add_argument("--expect-active", type=parse_expect_active, default=EXPECT_ACTIVE)
    parser.add_argument(
        "--no-expect-active",
        dest="expect_active",
        action="store_const",
        const=None,
        help="Disable the active-space size check when exploring AVAS presets.",
    )
    parser.add_argument("--max-active-orbitals", type=int, default=MAX_ACTIVE_ORBITALS)
    parser.add_argument("--run-dmrg", dest="run_dmrg", action="store_true", default=RUN_DMRG)
    parser.add_argument("--avas-only", dest="run_dmrg", action="store_false")

    parser.add_argument("--bond-dim", type=int, default=BOND_DIM)
    parser.add_argument("--dmrg-tol", type=float, default=DMRG_TOL)
    parser.add_argument("--dmrg-threads", type=int, default=DMRG_THREADS)
    parser.add_argument("--dmrg-memory-mb", type=int, default=DMRG_MEMORY_MB)
    parser.add_argument("--block-exe", type=Path, default=BLOCK_EXE)
    parser.add_argument("--mpi-prefix", default=MPI_PREFIX)
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=RUNTIME_DIR)
    parser.add_argument("--no-auto-order", dest="auto_order", action="store_false")
    parser.set_defaults(auto_order=AUTO_ORDER)
    parser.add_argument(
        "--block-extra-keyword",
        action="append",
        default=list(BLOCK_EXTRA_KEYWORDS),
        help="Extra raw keyword passed to the Block/Block2 DMRG backend.",
    )
    parser.add_argument(
        "--no-natural-occupations",
        dest="print_natural_occupations",
        action="store_false",
        help="Do not print active-space natural occupations after DMRG-CASCI.",
    )
    parser.set_defaults(print_natural_occupations=PRINT_NATURAL_OCCUPATIONS)
    parser.add_argument(
        "--no-mps-parameter-report",
        dest="report_mps_parameters",
        action="store_false",
        help="Do not print the Block2 MPS file-size parameter diagnostic.",
    )
    parser.set_defaults(report_mps_parameters=REPORT_MPS_PARAMETERS)
    parser.add_argument(
        "--keep-old-mps-files",
        dest="clean_mps_files_before_dmrg",
        action="store_false",
        help="Do not remove stale F.MPS.KET.* files before the DMRG run.",
    )
    parser.set_defaults(clean_mps_files_before_dmrg=CLEAN_MPS_FILES_BEFORE_DMRG)
    parser.add_argument(
        "--delete-mps-files-after-report",
        dest="delete_mps_files_after_report",
        action="store_true",
        help="Delete F.MPS.KET.* files after counting their stored parameters.",
    )
    parser.add_argument(
        "--keep-mps-files-after-report",
        dest="delete_mps_files_after_report",
        action="store_false",
        help="Keep F.MPS.KET.* files after counting their stored parameters.",
    )
    parser.set_defaults(delete_mps_files_after_report=DELETE_MPS_FILES_AFTER_REPORT)
    parser.add_argument(
        "--keep-block2-scratch",
        dest="clean_block2_scratch_after_run",
        action="store_false",
        help="Keep Block2 scratch files after the run report is written.",
    )
    parser.set_defaults(clean_block2_scratch_after_run=CLEAN_BLOCK2_SCRATCH_AFTER_RUN)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument(
        "--no-output-file",
        dest="write_output_file",
        action="store_false",
        help="Do not write the human-readable run report.",
    )
    parser.set_defaults(write_output_file=WRITE_OUTPUT_FILE)

    args = parser.parse_args(argv)
    if args.avas_labels is None:
        args.avas_labels = default_avas_labels(args.avas_preset)
    else:
        args.avas_preset = "custom"

    if args.scratch_dir is not None:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    if args.runtime_dir is not None:
        args.runtime_dir.mkdir(parents=True, exist_ok=True)

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _mol, mf = build_mean_field(args)
    ncas, nelecas, mo_coeff = run_avas(args, mf)
    dmrg_result = None

    if not args.run_dmrg:
        print("\nAVAS-only mode requested; DMRG-CASCI was not launched.")
        write_run_report(args, float(mf.e_tot), ncas, nelecas, dmrg_result)
        return 0

    dmrg_result = run_dmrg_casci(args, mf, ncas, nelecas, mo_coeff)
    write_run_report(args, float(mf.e_tot), ncas, nelecas, dmrg_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
