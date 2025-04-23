import qforte as qf
import numpy as np

nel = 4
sz = 0
norb = 4

fci_comp = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)
fci_comp.hartree_fock()

fci_comp2 = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)
fci_comp2.hartree_fock()

print(fci_comp.str(print_data=True))
print(fci_comp2.str(print_data=True))

reference = "random"

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

    fci_comp.set_state_from_tensor(Crand)
    fci_comp2.set_state_from_tensor(Crand)



# fci_comp.to_cpu()
fci_comp2.to_gpu()

print("\n SQOP Stuff")
print("===========================")
sqop = qf.SQOperator()
sqop.add_term(3.0, [5], [1])
sqop.add_term(3.0, [1], [5])

sqop.add_term(2.0, [4], [0])
sqop.add_term(2.0, [0], [4])
print(sqop)




print("\n Initial FCIcomp Stuff")
print("===========================")
# print(fci_comp)
# print(fci_comp2)

# fci_comp.to_cpu()
print("IOFHJEWOFJ(WE")
fci_comp.apply_sqop(sqop)
fci_comp2.apply_sqop_gpu(sqop)

fci_comp.to_cpu()
fci_comp2.to_cpu()

print("\n Final FCIcomp Stuff")
print("===========================")
print(fci_comp)
print(fci_comp2)