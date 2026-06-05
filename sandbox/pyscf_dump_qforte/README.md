PySCF Dump to QForte Sandbox
============================

This sandbox separates the workflow into two stages:

1. Use PySCF to generate active-space integral dumps.
2. Use qforte with `build_type="pyscf_dump"` to run UCCSD-VQE from the dump.

Suggested commands:

```bash
conda run -n forte_pyscf_env python sandbox/pyscf_dump_qforte/dump_n2_sto3g.py
conda run -n forte_pyscf_env python sandbox/pyscf_dump_qforte/dump_naphthalene_avas_ccpvdz.py

conda run -n qfe_env_v1 python sandbox/pyscf_dump_qforte/run_uccsd_vqe_from_dump.py --system n2
conda run -n qfe_env_v1 python sandbox/pyscf_dump_qforte/run_uccsd_vqe_from_dump.py --system naphthalene --maxiter 1
```

The `.npz` dump is the qforte-facing file.  A conventional `FCIDUMP` is also
written when PySCF's fcidump helper is available, but qforte needs metadata
that the plain FCIDUMP format does not normally carry.
