"""Check second-quantized excitation-pool content.

This is a sandbox diagnostic, not a pytest test.  It focuses on operator
membership and symmetry filtering; ordering variants such as GSDx and
1-UpCCGSDx should reorder the same generators, not enlarge the set.
"""

import math

import qforte as qf


NOCC = 3
NVIR = 4


def reference(nocc, nvir):
    """Closed-shell spin-orbital reference with occupied orbitals first."""
    return [1] * (2 * nocc) + [0] * (2 * nvir)


def expected_particle_hole_rank_count(nocc, nvir, rank):
    """Closed-shell C1 count for spin-conserving rank-r p-h generators."""
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


def expected_particle_hole_pool_count(nocc, nvir, max_rank):
    return sum(expected_particle_hole_rank_count(nocc, nvir, r) for r in range(1, max_rank + 1))


def build_pool(pool_name, nocc=NOCC, nvir=NVIR, irreps=None, target_irrep=0):
    pool = qf.SQOpPool()
    pool.set_orb_spaces(reference(nocc, nvir))
    if irreps is not None:
        pool.set_orb_irreps(irreps, target_irrep)

    if pool_name == "1-UpCCGSD":
        pool.fill_pool_kUpCCGSD(1)
    elif pool_name == "1-UpCCGSDx":
        pool.fill_pool_kUpCCGSDx(1)
    else:
        pool.fill_pool(pool_name)
    return pool


def clean_float(value):
    return 0.0 if abs(value) < 1.0e-12 else round(float(value), 12)


def op_signature(op, sign=1.0):
    pieces = []
    for coeff, creators, annihilators in op.terms():
        pieces.append(
            (
                clean_float(sign * coeff.real),
                clean_float(sign * coeff.imag),
                tuple(creators),
                tuple(annihilators),
            )
        )
    return tuple(pieces)


def canonical_op_signature(op):
    """Canonicalize modulo an overall sign, which is theta -> -theta."""
    return min(op_signature(op, +1.0), op_signature(op, -1.0))


def pool_signature_set(pool):
    return {canonical_op_signature(op) for _, op in pool.terms()}


def compact_op_description(op):
    if len(op.terms()) == 0:
        return "<empty>"
    coeff, creators, annihilators = op.terms()[0]
    return f"{coeff.real:+.1f} C{list(creators)} A{list(annihilators)}"


def particle_hole_rank(op, nocc):
    """Return excitation rank if any term is a clean p-h excitation."""
    for _, creators, annihilators in op.terms():
        if len(creators) != len(annihilators):
            continue
        creators_are_virtual = all((idx // 2) >= nocc for idx in creators)
        annihilators_are_occ = all((idx // 2) < nocc for idx in annihilators)
        if creators_are_virtual and annihilators_are_occ:
            return len(creators)
    return None


def print_first_ops(pool, n=10):
    for idx, (_, op) in enumerate(pool.terms()[:n]):
        print(f"    {idx:4d}: {compact_op_description(op)}")


def term_irrep(irreps, creators, annihilators):
    sym = 0
    for idx in list(creators) + list(annihilators):
        sym ^= irreps[idx // 2]
    return sym


def pool_terms_obey_irrep(pool, irreps, target_irrep):
    bad = []
    for op_idx, (_, op) in enumerate(pool.terms()):
        for term_idx, (_, creators, annihilators) in enumerate(op.terms()):
            if term_irrep(irreps, creators, annihilators) != target_irrep:
                bad.append((op_idx, term_idx, list(creators), list(annihilators)))
    return bad


def check_particle_hole_counts():
    print("\n== C1 particle-hole pool counts ==")
    print(f"nocc={NOCC}, nvir={NVIR}")
    print(f"{'pool':10s} {'actual':>8s} {'expected':>8s} {'status':>8s}")
    print("-" * 40)
    for pool_name, max_rank in [("S", 1), ("SD", 2), ("SDT", 3), ("SDTQ", 4)]:
        pool = build_pool(pool_name)
        expected = expected_particle_hole_pool_count(NOCC, NVIR, max_rank)
        status = "PASS" if len(pool) == expected else "FAIL"
        print(f"{pool_name:10s} {len(pool):8d} {expected:8d} {status:>8s}")
        if status != "PASS":
            raise AssertionError(f"{pool_name} count {len(pool)} != expected {expected}")


def check_x_pool_set_equality():
    print("\n== x-pool set equality checks ==")
    comparisons = [("GSD", "GSDx"), ("1-UpCCGSD", "1-UpCCGSDx")]
    for base_name, x_name in comparisons:
        base = build_pool(base_name)
        xpool = build_pool(x_name)
        base_set = pool_signature_set(base)
        x_set = pool_signature_set(xpool)
        status = "PASS" if base_set == x_set and len(base) == len(xpool) else "FAIL"
        print(
            f"{base_name:12s} {len(base):5d} | {x_name:12s} {len(xpool):5d} | "
            f"set match: {status}"
        )
        if status != "PASS":
            print(f"  only in {base_name}: {len(base_set - x_set)}")
            print(f"  only in {x_name}: {len(x_set - base_set)}")
            raise AssertionError(f"{base_name}/{x_name} operator sets differ")

        print(f"  first 10 {x_name} operators:")
        print_first_ops(xpool)


def check_x_particle_hole_prefixes():
    print("\n== x-pool particle-hole prefixes ==")
    checks = [
        ("GSDx", expected_particle_hole_pool_count(NOCC, NVIR, 2), {1, 2}),
        ("1-UpCCGSDx", 3 * NOCC * NVIR, {1, 2}),
    ]

    for pool_name, prefix_size, allowed_ranks in checks:
        pool = build_pool(pool_name)
        ranks = [particle_hole_rank(op, NOCC) for _, op in pool.terms()]
        n_ph_total = sum(rank in allowed_ranks for rank in ranks)
        prefix_ok = all(rank in allowed_ranks for rank in ranks[:prefix_size])
        no_late_ph = all(rank not in allowed_ranks for rank in ranks[prefix_size:])
        status = "PASS" if prefix_ok and no_late_ph else "FAIL"
        print(
            f"{pool_name:12s} prefix={prefix_size:4d} p-h total={n_ph_total:4d} "
            f"status={status}"
        )
        if status != "PASS":
            raise AssertionError(f"{pool_name} particle-hole prefix check failed")


def check_irrep_filtering():
    print("\n== Irrep filtering check ==")
    nocc = 2
    nvir = 3
    # A small artificial non-C1 label pattern.  The allowed excitation irrep
    # is the XOR of all spatial irreps in the creator/annihilator lists.
    irreps = [0, 1, 0, 1, 1]
    target_irrep = 0

    c1_pool = build_pool("SD", nocc=nocc, nvir=nvir, irreps=[0] * (nocc + nvir))
    restricted_pool = build_pool("SD", nocc=nocc, nvir=nvir, irreps=irreps, target_irrep=target_irrep)
    bad = pool_terms_obey_irrep(restricted_pool, irreps, target_irrep)

    print(f"C1 SD count:          {len(c1_pool)}")
    print(f"restricted SD count:  {len(restricted_pool)}")
    print(f"bad restricted terms: {len(bad)}")

    if bad:
        raise AssertionError(f"Found disallowed symmetry terms: {bad[:3]}")
    if len(restricted_pool) >= len(c1_pool):
        raise AssertionError("Expected this artificial irrep pattern to reduce the SD pool.")


def main():
    check_particle_hole_counts()
    check_x_pool_set_equality()
    check_x_particle_hole_prefixes()
    check_irrep_filtering()
    print("\nPool validity checks completed successfully.")


if __name__ == "__main__":
    main()
