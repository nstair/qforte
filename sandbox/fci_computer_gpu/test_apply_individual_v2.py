import qforte as qf
import numpy as np
nel = 14
sz = 0
norb = 14


fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
# fci_comp.hartree_fock()

fci_comp2 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
# fci_comp2.hartree_fock()


# print(fci_comp.str(print_data=True))
# print(fci_comp2.str(print_data=True))

reference = 'random'

if(reference == 'hf'):
    fci_comp.hartree_fock()
    fci_comp2.hartree_fock()

elif(reference == 'random'):
    random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fci_comp.set_state(Crand)
    fci_comp2.set_state(Crand)

print("\n SQOP Stuff")
print("===========================")
sqop = qf.SQOperator()
# sqop.add_term(1.6, [], [])

# beta
# sqop.add_term(3.0, [5], [1])
# sqop.add_term(3.0, [1], [5])

# # alfa
# sqop.add_term(2.0, [4], [0])
# sqop.add_term(2.0, [0], [4])

# # alfa alfa
# sqop.add_term(4.5, [6, 4], [2, 0])
# sqop.add_term(4.5, [0, 2], [4, 6])

# alfa beta
sqop.add_term(5.5, [16, 15], [1, 0])
# sqop.add_term(5.5, [1, 0], [5, 4])
# print(sqop)

print("hello world")


# print("\n Initial FCIcomp Stuff")
# print("===========================")
# print(fci_comp)
# print(fci_comp2)

my_timer = qf.local_timer()
my_timer.reset()

fci_comp.do_on_cpu()
fci_comp.apply_sqop(sqop)
print("-----------------------------------------------------------------------")
my_timer.record("normal")

# fci_comp3.apply_sqop(sqop)
fci_comp2.do_on_gpu()
my_timer.reset()
fci_comp2.apply_sqop(sqop)
my_timer.record("gpu")

print("\n Final FCIcomp Stuff")
print("===========================")
# print(fci_comp)
# print(fci_comp2)

C1 = fci_comp.get_state_deep()
C2 = fci_comp2.get_state_deep()

C1.subtract(C2)
print(f"\n\n||dC||: {C1.norm():6.6f}\n\n")

print(my_timer)