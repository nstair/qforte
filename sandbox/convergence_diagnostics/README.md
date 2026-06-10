# VQE/tUCC Convergence Diagnostics

This sandbox suite compares practical convergence behavior for SD, GSD/GSDx,
and k-UpCCGSD/k-UpCCGSDx ansatzes. It is not a pytest suite; it is meant for
longer exploratory runs that produce decision-oriented summaries.

Run from the repository root using the QForte environment:

```bash
conda run -n qfe_env_v1 python sandbox/convergence_diagnostics/run_convergence_diagnostics.py --mode fast
```

Available modes:

- `fast`: small smoke-style comparison with BeH2/H4, SD and 1-UpCCGSD, and core optimizers.
- `medium`: includes BeH2 C1/D2h, GSD/GSDx, and k=1,2,3 UpCCGSD variants with the key lbfgs_qf options.
- `slow`: broader matrix including history-reset and pool-ordering comparisons.

Useful debug option:

```bash
conda run -n qfe_env_v1 python sandbox/convergence_diagnostics/run_convergence_diagnostics.py --mode fast --limit-runs 2
```

Each run writes a timestamped directory:

```text
sandbox/convergence_diagnostics/results/<timestamp>/
```

Outputs:

- `summary.md`: human-readable report with decision-oriented comparisons.
- `summary.csv`: one scalar row per calculation.
- `raw_results.json`: full configs, result rows, pool checks, and trajectories.
- `trajectories.csv`: one energy row per reported iteration.
- `logs/*.log`: raw QForte output for each calculation.

The summary explicitly addresses:

- whether exact Hessian diagonal preconditioning helped;
- scipy BFGS vs scipy L-BFGS-B;
- late energy drops and the iteration where they occurred;
- zero vs MP2 initial amplitudes;
- target BLOCK and Newton-CG accelerations;
- L-BFGS history reset after BLOCK/NCG;
- pool ordering choices;
- GSD/GSDx and k-UpCCGSD/k-UpCCGSDx signature-set equality;
- C1 vs D2h pool-size and final-amplitude differences.

The main driver has editable sections near the top:

```python
RUN_MODE = "fast"
MOLECULES_TO_RUN = None
POOLS_TO_RUN = None
CONFIGS_TO_RUN = None
```

Set any of those to explicit lists to restrict the matrix without deleting the
mode defaults.

