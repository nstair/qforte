import qforte as qf
from qforte.utils.trotterization import trotterize


def double_excitation_generator():
    op = qf.SQOperator()
    op.add_term(1.0, [2, 3], [1, 0])
    op.add_term(-1.0, [1, 0], [2, 3])
    return op


pool = qf.SQOpPool()
pool.add(1.0, double_excitation_generator())

jw_op = double_excitation_generator().jw_transform()
umu, phase = trotterize(jw_op)

assert phase == 1.0 + 0.0j
assert pool.count_cnot_for_jw_exponential() == umu.get_num_cnots()

print("sandbox tUCC CNOT count check passed")
