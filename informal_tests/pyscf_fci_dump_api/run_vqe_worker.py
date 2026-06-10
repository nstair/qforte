"""Internal worker for one algorithm molecule-build parity run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import pyscf_dump_informal_common as common


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "__complex_ndarray__": True,
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return value.tolist()
    if isinstance(value, complex):
        return {"__complex__": True, "real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main():
    if len(sys.argv) == 4:
        algorithm = "vqe"
        system = sys.argv[1]
        build_type = sys.argv[2]
        output_json = Path(sys.argv[3])
    elif len(sys.argv) == 5:
        algorithm = sys.argv[1]
        system = sys.argv[2]
        build_type = sys.argv[3]
        output_json = Path(sys.argv[4])
    else:
        raise SystemExit("usage: run_vqe_worker.py [ALGORITHM] SYSTEM BUILD_TYPE OUTPUT_JSON")

    mol = common.build_molecule(system, build_type)
    if algorithm == "vqe":
        record = common.run_uccsd_vqe_algorithm(mol)
    elif algorithm == "srqk":
        record = common.run_srqk_algorithm(mol)
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(json_safe(record), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
