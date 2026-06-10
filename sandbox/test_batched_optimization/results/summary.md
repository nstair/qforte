# Batched Optimization Summary

- pool: `1-UpCCGSD`
- init_amps: `zero`
- general_ex_pool_order: `particle_hole_first`
- cycles: `1`

| optimizer | batch type | status | E_final | err FCI | ||g|| | nit | nfev | njev | params | PH | GEN | nnz | sequence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bfgs_qf | none | ok | -15.4792216267 | +2.519e-03 | 1.945e-04 | 100 | 199 | 101 | 63 | 36 | 27 | 63 | ALL |
| bfgs_qf | half_sweep | ok | -15.4722538306 | +9.487e-03 | 7.306e-03 | 9 | 11 | 11 | 63 | 36 | 27 | 29 | PH -> GEN |
| bfgs_qf | full_sweep | ok | -15.4722570566 | +9.484e-03 | 1.896e-04 | 12 | 16 | 16 | 63 | 36 | 27 | 29 | PH -> GEN -> GEN -> PH |
| bfgs_qf | half_sweep_then_all | ok | -15.4792043535 | +2.537e-03 | 1.991e-03 | 109 | 132 | 112 | 63 | 36 | 27 | 41 | PH -> GEN -> ALL |
| bfgs_qf | full_sweep_then_all | ok | -15.4792083215 | +2.533e-03 | 1.372e-03 | 112 | 138 | 117 | 63 | 36 | 27 | 41 | PH -> GEN -> GEN -> PH -> ALL |
| lbfgs_qf | none | ok | -15.4791970278 | +2.544e-03 | 4.411e-04 | 100 | 125 | 101 | 63 | 36 | 27 | 41 | ALL |
| lbfgs_qf | half_sweep | ok | -15.4722538335 | +9.487e-03 | 7.305e-03 | 9 | 11 | 11 | 63 | 36 | 27 | 29 | PH -> GEN |
| lbfgs_qf | full_sweep | ok | -15.4722570587 | +9.484e-03 | 1.895e-04 | 12 | 16 | 16 | 63 | 36 | 27 | 29 | PH -> GEN -> GEN -> PH |
| lbfgs_qf | half_sweep_then_all | ok | -15.4792019303 | +2.539e-03 | 1.284e-03 | 109 | 141 | 112 | 63 | 36 | 27 | 41 | PH -> GEN -> ALL |
| lbfgs_qf | full_sweep_then_all | ok | -15.4722954168 | +9.446e-03 | 9.674e-05 | 49 | 55 | 54 | 63 | 36 | 27 | 37 | PH -> GEN -> GEN -> PH -> ALL |
| BFGS | none | ok | -15.4722872917 | +9.454e-03 | 2.349e-04 | 45 | 54 | 54 | 63 | 36 | 27 | 63 | ALL |
| BFGS | half_sweep | ok | -15.4722528768 | +9.488e-03 | 7.036e-03 | 16 | 29 | 29 | 63 | 36 | 27 | 29 | PH -> GEN |
| BFGS | full_sweep | ok | -15.4722557741 | +9.485e-03 | 2.264e-04 | 23 | 42 | 42 | 63 | 36 | 27 | 29 | PH -> GEN -> GEN -> PH |
| BFGS | half_sweep_then_all | ok | -15.4722872822 | +9.454e-03 | 1.770e-04 | 49 | 71 | 71 | 63 | 36 | 27 | 57 | PH -> GEN -> ALL |
| BFGS | full_sweep_then_all | ok | -15.4722872795 | +9.454e-03 | 9.053e-05 | 52 | 76 | 76 | 63 | 36 | 27 | 53 | PH -> GEN -> GEN -> PH -> ALL |
| L-BFGS-B | none | ok | -15.4722558001 | +9.485e-03 | 6.101e-04 | 22 | 26 | 26 | 63 | 36 | 27 | 29 | ALL |
| L-BFGS-B | half_sweep | ok | -15.4722529933 | +9.488e-03 | 7.095e-03 | 15 | 20 | 20 | 63 | 36 | 27 | 29 | PH -> GEN |
| L-BFGS-B | full_sweep | ok | -15.4722558207 | +9.485e-03 | 9.286e-04 | 18 | 26 | 26 | 63 | 36 | 27 | 29 | PH -> GEN -> GEN -> PH |
| L-BFGS-B | half_sweep_then_all | ok | -15.4722558262 | +9.485e-03 | 9.346e-04 | 18 | 25 | 25 | 63 | 36 | 27 | 29 | PH -> GEN -> ALL |
| L-BFGS-B | full_sweep_then_all | ok | -15.4722558938 | +9.485e-03 | 1.080e-03 | 19 | 29 | 29 | 63 | 36 | 27 | 29 | PH -> GEN -> GEN -> PH -> ALL |

## Per-Batch Details

### bfgs_qf / half_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.244e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538306, reduced ||g||=9.430e-05, full ||g||=7.306e-03, nit=4

### bfgs_qf / full_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.244e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538306, reduced ||g||=9.430e-05, full ||g||=7.306e-03, nit=4
- batch 3 `GEN`: active=27, E=-15.4722538306 -> -15.4722538306, reduced ||g||=9.430e-05, full ||g||=7.306e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722538306 -> -15.4722570566, reduced ||g||=1.909e-05, full ||g||=1.896e-04, nit=3

### bfgs_qf / half_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.244e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538306, reduced ||g||=9.430e-05, full ||g||=7.306e-03, nit=4
- batch 3 `ALL`: active=63, E=-15.4722538306 -> -15.4792043535, reduced ||g||=1.991e-03, full ||g||=1.991e-03, nit=100

### bfgs_qf / full_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.244e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538306, reduced ||g||=9.430e-05, full ||g||=7.306e-03, nit=4
- batch 3 `GEN`: active=27, E=-15.4722538306 -> -15.4722538306, reduced ||g||=9.430e-05, full ||g||=7.306e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722538306 -> -15.4722570566, reduced ||g||=1.909e-05, full ||g||=1.896e-04, nit=3
- batch 5 `ALL`: active=63, E=-15.4722570566 -> -15.4792083215, reduced ||g||=1.372e-03, full ||g||=1.372e-03, nit=100

### lbfgs_qf / half_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.259e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538335, reduced ||g||=9.287e-05, full ||g||=7.305e-03, nit=4

### lbfgs_qf / full_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.259e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538335, reduced ||g||=9.287e-05, full ||g||=7.305e-03, nit=4
- batch 3 `GEN`: active=27, E=-15.4722538335 -> -15.4722538335, reduced ||g||=9.287e-05, full ||g||=7.305e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722538335 -> -15.4722570587, reduced ||g||=1.910e-05, full ||g||=1.895e-04, nit=3

### lbfgs_qf / half_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.259e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538335, reduced ||g||=9.287e-05, full ||g||=7.305e-03, nit=4
- batch 3 `ALL`: active=63, E=-15.4722538335 -> -15.4792019303, reduced ||g||=1.284e-03, full ||g||=1.284e-03, nit=100

### lbfgs_qf / full_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823028, reduced ||g||=1.259e-05, full ||g||=5.384e-03, nit=5
- batch 2 `GEN`: active=27, E=-15.4720823028 -> -15.4722538335, reduced ||g||=9.287e-05, full ||g||=7.305e-03, nit=4
- batch 3 `GEN`: active=27, E=-15.4722538335 -> -15.4722538335, reduced ||g||=9.287e-05, full ||g||=7.305e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722538335 -> -15.4722570587, reduced ||g||=1.910e-05, full ||g||=1.895e-04, nit=3
- batch 5 `ALL`: active=63, E=-15.4722570587 -> -15.4722954168, reduced ||g||=9.674e-05, full ||g||=9.674e-05, nit=37

### BFGS / half_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823024, reduced ||g||=7.347e-05, full ||g||=5.384e-03, nit=14
- batch 2 `GEN`: active=27, E=-15.4720823024 -> -15.4722528768, reduced ||g||=1.459e-04, full ||g||=7.036e-03, nit=2

### BFGS / full_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823024, reduced ||g||=7.347e-05, full ||g||=5.384e-03, nit=14
- batch 2 `GEN`: active=27, E=-15.4720823024 -> -15.4722528768, reduced ||g||=1.459e-04, full ||g||=7.036e-03, nit=2
- batch 3 `GEN`: active=27, E=-15.4722528768 -> -15.4722528768, reduced ||g||=1.459e-04, full ||g||=7.036e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722528768 -> -15.4722557741, reduced ||g||=9.352e-05, full ||g||=2.264e-04, nit=7

### BFGS / half_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823024, reduced ||g||=7.347e-05, full ||g||=5.384e-03, nit=14
- batch 2 `GEN`: active=27, E=-15.4720823024 -> -15.4722528768, reduced ||g||=1.459e-04, full ||g||=7.036e-03, nit=2
- batch 3 `ALL`: active=63, E=-15.4722528768 -> -15.4722872822, reduced ||g||=1.770e-04, full ||g||=1.770e-04, nit=33

### BFGS / full_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720823024, reduced ||g||=7.347e-05, full ||g||=5.384e-03, nit=14
- batch 2 `GEN`: active=27, E=-15.4720823024 -> -15.4722528768, reduced ||g||=1.459e-04, full ||g||=7.036e-03, nit=2
- batch 3 `GEN`: active=27, E=-15.4722528768 -> -15.4722528768, reduced ||g||=1.459e-04, full ||g||=7.036e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722528768 -> -15.4722557741, reduced ||g||=9.352e-05, full ||g||=2.264e-04, nit=7
- batch 5 `ALL`: active=63, E=-15.4722557741 -> -15.4722872795, reduced ||g||=9.053e-05, full ||g||=9.053e-05, nit=29

### L-BFGS-B / half_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720822200, reduced ||g||=6.624e-04, full ||g||=5.422e-03, nit=13
- batch 2 `GEN`: active=27, E=-15.4720822200 -> -15.4722529933, reduced ||g||=1.314e-04, full ||g||=7.095e-03, nit=2

### L-BFGS-B / full_sweep

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720822200, reduced ||g||=6.624e-04, full ||g||=5.422e-03, nit=13
- batch 2 `GEN`: active=27, E=-15.4720822200 -> -15.4722529933, reduced ||g||=1.314e-04, full ||g||=7.095e-03, nit=2
- batch 3 `GEN`: active=27, E=-15.4722529933 -> -15.4722529933, reduced ||g||=1.314e-04, full ||g||=7.095e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722529933 -> -15.4722558207, reduced ||g||=9.127e-04, full ||g||=9.286e-04, nit=3

### L-BFGS-B / half_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720822200, reduced ||g||=6.624e-04, full ||g||=5.422e-03, nit=13
- batch 2 `GEN`: active=27, E=-15.4720822200 -> -15.4722529933, reduced ||g||=1.314e-04, full ||g||=7.095e-03, nit=2
- batch 3 `ALL`: active=63, E=-15.4722529933 -> -15.4722558262, reduced ||g||=9.346e-04, full ||g||=9.346e-04, nit=3

### L-BFGS-B / full_sweep_then_all

- batch 1 `PH`: active=36, E=-15.4556677735 -> -15.4720822200, reduced ||g||=6.624e-04, full ||g||=5.422e-03, nit=13
- batch 2 `GEN`: active=27, E=-15.4720822200 -> -15.4722529933, reduced ||g||=1.314e-04, full ||g||=7.095e-03, nit=2
- batch 3 `GEN`: active=27, E=-15.4722529933 -> -15.4722529933, reduced ||g||=1.314e-04, full ||g||=7.095e-03, nit=0
- batch 4 `PH`: active=36, E=-15.4722529933 -> -15.4722558207, reduced ||g||=9.127e-04, full ||g||=9.286e-04, nit=3
- batch 5 `ALL`: active=63, E=-15.4722558207 -> -15.4722558938, reduced ||g||=1.080e-03, full ||g||=1.080e-03, nit=1

## Expected Error Demos

- `non-generalized SD pool`: Batched optimization requested, but the GEN batch is empty. Use a generalized pool with particle-hole-first ordering.
- `missing particle_hole_first`: batched_opt_type requires general_ex_pool_order="particle_hole_first" so PH/GEN batches are unambiguous.
- `jacobi unsupported`: batched_opt_type is only supported for energy/gradient optimizers, not Jacobi/residual optimizers.