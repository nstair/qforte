import qforte as qf
import numpy as np
 
import time

# Define the reference and geometry lists.
geom = [
    ('C', (0., 0., 1.0)),
    ('H', (0., 0., 2.0)), 
    ('H', (0., 0., 3.0)),
    ('H', (0., 0., 4.0)), 
    ('H', (0., 0., 5.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0)),
    # ('H', (0., 0.,11.0)), 
    # ('H', (0., 0.,12.0))
    ]


timer = qf.local_timer()

timer.reset()
# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    build_qb_ham = False,
    run_fci=1)

timer.record('Run Psi4 and Initialize')
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

if(norb < 6): 
    print(f" nqbit:     {norb*2}")
    print(f" nel:       {nel}")
 
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fci_comp_gpu = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)


# reference = 'random'
reference = 'hf'

if(reference == 'hf'):
    fci_comp.hartree_fock()
    fci_comp_gpu.hartree_fock()

elif(reference == 'random'):
    random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fci_comp.set_state(Crand)
    fci_comp_gpu.set_state_from_tensor(Crand)


 
timer.reset()
fci_comp.apply_sqop(mol.sq_hamiltonian)

timer.record('normal')
E1 = fci_comp.get_hf_dot() 

fci_comp_gpu.to_gpu()
timer.reset()

fci_comp_gpu.apply_sqop(mol.sq_hamiltonian)
timer.record('gpu')
fci_comp_gpu.to_cpu()

E2 = fci_comp_gpu.get_hf_dot() 
 
if(norb < 6): 
    print("\n Final FCIcomp Stuff")
    print("===========================")
    print(fci_comp)
    print(fci_comp_gpu)
 
print("\n Timing")
print("======================================================")
print(timer)

print("\n Energetics")
print("======================================================")
print(f" Efci:               {mol.fci_energy}")
print(f" Ehf:                {mol.hf_energy}")
print(f" Enr:                {mol.nuclear_repulsion_energy}")
print(f" Eelec:              {mol.hf_energy - mol.nuclear_repulsion_energy}")
print(f" E1 (from cpu):   {E1}")
print(f" E2 (from gpu):   {E2}")