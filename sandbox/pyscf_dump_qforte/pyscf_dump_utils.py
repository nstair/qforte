"""Utilities for writing qforte-readable PySCF active-space dumps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np


Atom = Tuple[str, Tuple[float, float, float]]


def json_safe(value: Any) -> Any:
    """Convert NumPy-heavy metadata to JSON-serializable Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def geometry_to_pyscf_atom(geom: Sequence[Atom]) -> str:
    """Format a qforte-style geometry as a PySCF atom string."""
    return "\n".join(
        f"{symbol:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}"
        for symbol, xyz in geom
    )


def active_electron_count(nelecas: int | Sequence[int]) -> int:
    """Return total active electrons from PySCF's nelecas object."""
    if isinstance(nelecas, (tuple, list)):
        return int(sum(nelecas))
    return int(nelecas)


def active_spin_counts(nelecas: int | Sequence[int]) -> tuple[int, int]:
    """Return active alpha/beta electron counts from PySCF's nelecas object."""
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]), int(nelecas[1])
    nelec = int(nelecas)
    return (nelec + 1) // 2, nelec // 2


def closed_shell_hf_reference(num_electrons: int, num_orbitals: int) -> list[int]:
    """Return qforte's interleaved-spin closed-shell occupation reference."""
    if num_electrons < 0 or num_electrons > 2 * num_orbitals:
        raise ValueError("Invalid active electron/orbital count for HF reference.")
    return [1] * num_electrons + [0] * (2 * num_orbitals - num_electrons)


def normalize_name(name: str) -> str:
    clean = name.strip().lower().replace("_", "-")
    aliases = {"napthalene": "naphthalene"}
    return aliases.get(clean, clean)


def parse_acene_geometry_file(path: Path, acene: str, state: str) -> list[Atom]:
    """Read one comment-headed acene geometry block from acene_geoms.txt."""
    header_re = re.compile(r"^#\s*([^,#]+?)\s*,\s*(singlet|triplet)\s*$", re.I)
    atom_re = re.compile(
        r"^\s*([A-Za-z]+)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*$"
    )

    wanted = (normalize_name(acene), normalize_name(state))
    current_key = None
    blocks: dict[tuple[str, str], list[Atom]] = {}

    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        header_match = header_re.match(line)
        if header_match:
            current_key = (
                normalize_name(header_match.group(1)),
                normalize_name(header_match.group(2)),
            )
            blocks[current_key] = []
            continue

        if not line.strip() or line.lstrip().startswith("#"):
            continue

        atom_match = atom_re.match(line)
        if atom_match and current_key is not None:
            symbol = atom_match.group(1)
            xyz = tuple(float(atom_match.group(i)) for i in range(2, 5))
            blocks[current_key].append((symbol, xyz))  # type: ignore[arg-type]
            continue

        raise ValueError(f"Could not parse geometry line {line_no} in {path}: {line!r}")

    if wanted not in blocks:
        raise ValueError(f"Geometry block {wanted!r} not found in {path}.")
    return blocks[wanted]


def write_pyscf_dump(
    dump_path: Path,
    mo_oeis: np.ndarray,
    mo_teis: np.ndarray,
    metadata: dict[str, Any],
    write_fcidump: bool = True,
) -> None:
    """Write qforte metadata/integrals and, optionally, a standard FCIDUMP."""
    dump_path = Path(dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    mo_oeis = np.asarray(mo_oeis, dtype=float)
    mo_teis = np.asarray(mo_teis, dtype=float)
    metadata = dict(metadata)
    metadata.setdefault("dump_format", "qforte_pyscf_dump")
    metadata.setdefault("dump_version", 1)
    metadata.setdefault("integral_layout", "pyscf_ao2mo_kernel_spatial")
    metadata.setdefault("num_active_orbitals", int(mo_oeis.shape[0]))

    np.savez_compressed(
        dump_path,
        metadata_json=json.dumps(json_safe(metadata), sort_keys=True, indent=2),
        mo_oeis=mo_oeis,
        mo_teis=mo_teis,
    )

    if not write_fcidump:
        return

    try:
        from pyscf.tools import fcidump
    except ImportError:
        return

    nmo = int(mo_oeis.shape[0])
    nelec = int(metadata["num_active_electrons"])
    nalpha = int(metadata.get("num_alpha", (nelec + 1) // 2))
    nbeta = int(metadata.get("num_beta", nelec // 2))
    scalar_energy = float(metadata.get("scalar_energy", 0.0))
    fcidump_path = dump_path.with_suffix(".FCIDUMP")
    fcidump.from_integrals(
        str(fcidump_path),
        mo_oeis,
        mo_teis,
        nmo,
        nelec,
        nuc=scalar_energy,
        ms=abs(nalpha - nbeta),
    )
