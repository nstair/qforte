import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    ('H', (0., 0., 9.0)), 
    ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)),
    # ('H', (0., 0.,12.0)),
    # ('H', (0., 0.,13.0)),
    # ('H', (0., 0.,14.0)),
    ]

# geom = [
#     ('H', (0., 0., 1.0)), 
#     ('C', (0., 0., 2.0)),
#     ('C', (0., 0., 3.0)),
#     ('H', (0., 0., 4.0))
#     ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g', run_fci=0)

timer = qf.local_timer()
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

fci_comp_thrust = qf.FCIComputerThrust(
    nel=nel, 
    sz=sz, 
    norb=norb,
    on_gpu=False,
    data_type="real")

# reference = 'random'
reference = 'hf'

if(reference == 'hf'):
    fci_comp1.hartree_fock()
    fci_comp2.hartree_fock()
    fci_comp_thrust.hartree_fock_cpu()


# elif(reference == 'random'):
#     np.random.seed(42)
#     random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
#     random = np.array(random_array, dtype = np.dtype(np.complex128))

#     Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
#     Crand.fill_from_nparray(random.ravel(), Crand.shape())
#     rand_nrm = Crand.norm()
#     Crand.scale(1/rand_nrm)

#     fci_comp.set_state(Crand)
    
sqham = mol.sq_hamiltonian
# sqham.simplify()


sd_pool = qf.SQOpPool()
sd_pool.set_orb_spaces(ref)
sd_pool.fill_pool("SD")
# print(sd_pool)

sd_gpu = qf.SQOpPoolThrust(data_type="real")
sd_gpu.set_orb_spaces(ref)
sd_gpu.fill_pool("SD")

fci_comp_thrust.populate_index_arrays_for_pool_evo(sd_gpu)


len = sd_gpu.check_mu_tuple_container_sizes()

print(f"len {len}")

# mu = 2

# print(f"\n\n ===> mu: {mu} <=== \n")

# print(hp_gpu.terms()[mu])

# hp_gpu.print_mu_tuple_dims(mu)
# print("")
# hp_gpu.print_mu_tuple_elements(mu)


# print(hp_gpu)

time = 0.1

N = 1
r = 1
order = 1

print(f"dt:    {time}")
print(f"r:     {r}")
print(f"order: {order}")

fci_comp_thrust.to_gpu()

for _ in range(N):
# Call Trotter for fci_comp1
    timer.reset()
    fci_comp1.evolve_pool_trotter(
        sd_pool,
        time,
        r,
        order,
        antiherm=True,
        adjoint=False)
    timer.record('trotter fci_comp1')


    # print(fci_comp1.str(print_complex=False))
    # print(fci_comp1.get_state().norm())

    # Call full taylor evolution for fci_comp2
    # fci_comp2.evolve_op_taylor(
    #     sqham,
    #     time,
    #     1.0e-15,
    #     30)

    timer.reset()

    fci_comp_thrust.evolve_pool_trotter_gpu(
        sd_gpu,
        time,
        r,
        order,
        antiherm=True,
        adjoint=False)
    
    timer.record('trotter fci_comp_thrust')

    # print(fci_comp2)
    # print(fci_comp2.get_state().norm())

    fci_comp_thrust.to_cpu()

    C1 = fci_comp1.get_state_deep()
    C1_dup = fci_comp1.get_state_deep()
    # C2 = fci_comp2.get_state_deep()

    print(" coppying to C3 ")

    C3 = qf.Tensor(C1.shape(), "C3")
    fci_comp_thrust.copy_to_tensor_cpu(C3)

    print(" done coppying to C3 ")

    # print(f"C1: {C1}")
    # print(f"C2: {C2}")
    # print(f"C3: {C3}")

    # C1.subtract(C2)
    # C2.subtract(C3)
    C1_dup.subtract(C3)

    

    # print(C1)
    # print(C3)
    # print(f"deltaC.norm() {C1.norm()}")
    # print(f"deltaC_thrust.norm() {C2.norm()}")
    print(f"||C1 - C3|| {C1_dup.norm()}")

    fci_comp_thrust.to_gpu()


fci_thrust_timer = fci_comp_thrust.get_acc_timer()
print(fci_thrust_timer.acc_str_table())

print(timer)

print("I get here")

