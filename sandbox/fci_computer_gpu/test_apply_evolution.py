import qforte as qf
import numpy as np

RAND = False


nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()

#fci_comp_gpu = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)
#fci_comp_gpu.hartree_fock()

fci_comp_thrust = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)
fci_comp_thrust.hartree_fock_cpu()

if(RAND):
    # random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    # random = np.array(random_array, dtype = np.dtype(np.complex128))
    random = np.ones((fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1]))
    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)
    fci_comp.set_state(Crand)
    #fci_comp_gpu.set_state(Crand)
    fci_comp_thrust.set_state(Crand)

print("\n Initial FCIcomp Stuff")
print("===========================")
print(fci_comp)
print("===========================")
print(fci_comp_thrust)

sq_terms = [
    #(+0.123, [2, 3], [0, 1]), # 2body ab
    (+0.704645, [7, 6], [3, 2]), # 2body ab
    (+0.4, [6], [0]), # 1bdy-a
    (+0.4, [7], [3]), # 1bdy-a
    (+0.4, [2], [2])
]

time = 1.0
print_imag = True

pool = qf.SQOpPool()

for sq_term in sq_terms:

    sqop = qf.SQOperator()
    sqop.add_term(sq_term[0], sq_term[1], sq_term[2])
    sqop.add_term(np.conj(sq_term[0]), sq_term[2][::-1], sq_term[1][::-1])

    pool.add_term(1.0, sqop)

print("\n SQOP Pool Stuff")
print("===========================")
print(pool)

fci_comp.evolve_pool_trotter_basic(
    pool,
    antiherm=False)

fci_comp_thrust.to_gpu()
fci_comp_thrust.evolve_pool_trotter_basic_gpu(
    pool,
    antiherm=False)
fci_comp_thrust.to_cpu()

print("\n Final FCIcomp Stuff")
print("===========================")
Ctemp = fci_comp.get_state_deep()
cnrm = Ctemp.norm()
print(f"||C||: {cnrm}")
print(fci_comp.str(print_data=True, print_complex=print_imag))

Ctemp_thrust_shape = fci_comp_thrust.get_shape()
Ctemp_thrust = qf.Tensor(Ctemp_thrust_shape, "Ctemp_thrust")
fci_comp_thrust.copy_to_tensor_cpu(Ctemp_thrust)
cnrm_thrust = Ctemp_thrust.norm()
print(f"||C_thrust||: {cnrm_thrust}")
print(fci_comp_thrust.str(print_data=True, print_complex=print_imag))

difference = qf.Tensor(Ctemp_thrust_shape, "difference")
fci_comp_thrust.copy_to_tensor_cpu(difference)
difference.subtract(Ctemp)
cnrm_thrust = difference.norm()
print(f"||C_thrust - C||: {cnrm_thrust}")
