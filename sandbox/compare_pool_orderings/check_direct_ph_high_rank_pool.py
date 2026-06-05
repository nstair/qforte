"""Diagnostic for direct particle-hole triples/quadruples.

This is a readable sandbox check, not a formal pytest test.  It validates the
closed-shell C1 rank-count formula for the direct S/D/T/Q particle-hole pool
builder and confirms the generated operators obey the stored irrep filter.
"""

import math
import time

import qforte as qf


NOCC = 3
NVIR = 4


def reference(nocc=NOCC, nvir=NVIR):
    return [1] * (2 * nocc) + [0] * (2 * nvir)


def expected_rank_count(nocc, nvir, rank):
    """Closed-shell p-h count: sum_a C(o,a)C(v,a)C(o,r-a)C(v,r-a)."""
    total = 0
    for nalpha in range(rank + 1):
        nbeta = rank - nalpha
        if nalpha > nocc or nalpha > nvir or nbeta > nocc or nbeta > nvir:
            continue
        total += (
            math.comb(nocc, nalpha)
            * math.comb(nvir, nalpha)
            * math.comb(nocc, nbeta)
            * math.comb(nvir, nbeta)
        )
    return total


def expected_pool_count(pool_name, nocc=NOCC, nvir=NVIR):
    max_rank_by_pool = {
        "S": 1,
        "SD": 2,
        "SDT": 3,
        "SDTQ": 4,
    }
    return sum(
        expected_rank_count(nocc, nvir, rank)
        for rank in range(1, max_rank_by_pool[pool_name] + 1)
    )


def build_pool(pool_name, nocc=NOCC, nvir=NVIR, irreps=None, target_irrep=0):
    pool = qf.SQOpPool()
    pool.set_orb_spaces(reference(nocc, nvir))
    if irreps is not None:
        pool.set_orb_irreps(irreps, target_irrep)
    start = time.perf_counter()
    pool.fill_pool(pool_name)
    elapsed = time.perf_counter() - start
    return pool, elapsed


def particle_hole_rank(op, nocc=NOCC):
    """Return the rank from the excitation-oriented term of T - T^dagger."""
    for _, creators, annihilators in op.terms():
        creators_are_virtual = all((idx // 2) >= nocc for idx in creators)
        annihilators_are_occ = all((idx // 2) < nocc for idx in annihilators)
        if creators_are_virtual and annihilators_are_occ:
            return len(creators)
    return None


def rank_counts(pool, nocc=NOCC):
    counts = {}
    for _, op in pool.terms():
        rank = particle_hole_rank(op, nocc)
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def term_irrep(irreps, creators, annihilators):
    sym = 0
    for idx in list(creators) + list(annihilators):
        sym ^= irreps[idx // 2]
    return sym


def count_bad_irrep_terms(pool, irreps, target_irrep):
    bad = 0
    for _, op in pool.terms():
        for _, creators, annihilators in op.terms():
            if term_irrep(irreps, creators, annihilators) != target_irrep:
                bad += 1
    return bad


def count_empty_ops(pool):
    return sum(1 for _, op in pool.terms() if len(op.terms()) == 0)


def check_c1_counts():
    print("\n== Direct p-h S/D/T/Q C1 counts ==")
    print(f"nocc={NOCC}, nvir={NVIR}")
    print(
        f"{'pool':8s} {'observed':>9s} {'expected':>9s} {'time(s)':>10s} "
        f"{'rank counts':>24s}"
    )
    print("-" * 72)

    expected_fixed = {
        "S": 24,
        "SD": 204,
        "SDT": 644,
        "SDTQ": 1064,
    }

    for pool_name in ["S", "SD", "SDT", "SDTQ"]:
        pool, elapsed = build_pool(pool_name)
        observed = len(pool)
        expected = expected_pool_count(pool_name)
        counts = rank_counts(pool)
        print(
            f"{pool_name:8s} {observed:9d} {expected:9d} {elapsed:10.6f} "
            f"{str(counts):>24s}"
        )
        if observed != expected:
            raise AssertionError(f"{pool_name}: observed {observed}, expected {expected}")
        if observed != expected_fixed[pool_name]:
            raise AssertionError(f"{pool_name}: expected fixed count {expected_fixed[pool_name]}")
        if count_empty_ops(pool) != 0:
            raise AssertionError(f"{pool_name}: found empty simplified operators")


def check_irrep_filtering_for_high_ranks():
    print("\n== Direct p-h high-rank irrep filtering ==")
    nocc = 3
    nvir = 4
    irreps = [0, 1, 0, 1, 1, 0, 1]
    target_irrep = 0

    for pool_name in ["SDT", "SDTQ"]:
        c1_pool, _ = build_pool(pool_name, nocc, nvir, irreps=[0] * (nocc + nvir))
        restricted_pool, _ = build_pool(pool_name, nocc, nvir, irreps=irreps, target_irrep=target_irrep)
        bad_terms = count_bad_irrep_terms(restricted_pool, irreps, target_irrep)
        print(
            f"{pool_name:8s} C1={len(c1_pool):5d} restricted={len(restricted_pool):5d} "
            f"bad_terms={bad_terms}"
        )
        if bad_terms != 0:
            raise AssertionError(f"{pool_name}: found {bad_terms} symmetry-disallowed terms")
        if len(restricted_pool) >= len(c1_pool):
            raise AssertionError(f"{pool_name}: artificial irrep filter did not reduce the pool")


def main():
    print("Expected rank counts:")
    for rank in range(1, 5):
        print(f"  rank {rank}: {expected_rank_count(NOCC, NVIR, rank)}")
    check_c1_counts()
    check_irrep_filtering_for_high_ranks()
    print("\nDirect particle-hole high-rank checks completed successfully.")


if __name__ == "__main__":
    main()
