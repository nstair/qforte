import qforte as qf
import numpy as np

nel = 4
sz = 0
norb = 4

fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fc2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)



fc1.hartree_fock()
fc2.hartree_fock()

rand = True
if rand:
    np.random.seed(12)
    random_array = np.random.rand(fc1.get_state().shape()[0], fc1.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))
    Crand = qf.Tensor(fc1.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rnrm = Crand.norm()
    Crand.scale(1.0/rnrm)
    fc1.set_state(Crand)
    fc2.set_state(Crand)
    print(f"||Crand||: {fc1.get_state().norm()}")
    print(fc1.str(print_data=True))

print("\n SQOP Stuff")
print("===========================")
n32 = qf.SQOperator()
n32.add_term( -1.1, [5, 1], [5, 1])
n32.add_term( -1.1, [5, 1], [5, 1])

n54 = qf.SQOperator()
n54.add_term( -0.1, [6, 4], [6, 4])
n54.add_term( -0.1, [6, 4], [6, 4])

# TODO:(Nick) Verify that the taylor expansion of the above operators
# actyally produces the expected states.

Co = fc1.get_state_deep()

time = 1.0

print("\n Initial FCIcomp Stuff")
print("===========================")
print(fc1)

fc1.evolve_op_taylor(
    n32, 
    time,
    1e-15,
    30,
    True)

C1 = fc1.get_state_deep()
C1nrm = C1.norm()
C1.scale(1.0/C1nrm)
fc1.set_state(C1)


print(fc1.str(print_data=True, print_complex=False))
print(f"||C1||: {fc1.get_state().norm()}")
print("\n\n")

angles = np.linspace(0.113210,+0.113704, 10)

for theta in angles:
    k32 = qf.SQOperator()
    k32.add_term( +theta, [5], [1])
    k32.add_term( -theta, [1], [5])

    fc2.apply_sqop_evolution(
        time, 
        k32,
        True)

    # print(fc2)

    C2 = fc2.get_state_deep()

    dC1 = fc1.get_state_deep()
    dC1.subtract(C2)

    V1 = np.real(Co.vector_dot(C1))
    V2 = np.real(Co.vector_dot(C2))

    print(f"theta: {theta:+6.6f} ||dC||: {dC1.norm():6.6f} |dV|: {np.abs(V1-V2):6.6f}  V1: {V1:6.6f}  V2: {V2:6.6f}" )



