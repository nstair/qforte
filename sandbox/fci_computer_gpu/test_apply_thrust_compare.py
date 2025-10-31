import qforte as qf
import numpy as np
 
import time

np.random.seed(42)  # For reproducibility

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
    ('H', (0., 0., 10.0)),
    ('H', (0., 0., 11.0)),
    ('H', (0., 0., 12.0)),
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
fci_comp_thrust = qf.FCIComputerGPU(nel=nel, sz=sz, norb=norb)


reference = 'random'
#reference = 'hf'

if(reference == 'hf'):
    fci_comp.hartree_fock()
    fci_comp_gpu.hartree_fock()
    fci_comp_thrust.hartree_fock_cpu()

elif(reference == 'random'):
    random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fci_comp.set_state(Crand)
    fci_comp_gpu.set_state_from_tensor(Crand)
    fci_comp_thrust.set_state_from_tensor_cpu(Crand)

    print("\n Crand State ==============")
    print(Crand)
 
timer.reset()
fci_comp.apply_sqop(mol.sq_hamiltonian)
timer.record('normal')

fci_comp_gpu.to_gpu()
timer.reset()
fci_comp_gpu.apply_sqop(mol.sq_hamiltonian)
timer.record('gpu')
fci_comp_gpu.to_cpu()

fci_comp_thrust.to_gpu()
timer.reset()
fci_comp_thrust.apply_sqop_gpu(mol.sq_hamiltonian)
timer.record('thrust')
fci_comp_thrust.to_cpu()

# now instead of comparing enegrgies compare the norm of the difference between the states

# init state tensors for each computer
comp_state = fci_comp.get_state()
comp_state_gpu = qf.Tensor(comp_state.shape(), "comp_state_gpu")
comp_state_gpu2 = qf.Tensor(comp_state.shape(), "comp_state_gpu2")
comp_state_thrust = qf.Tensor(comp_state.shape(), "comp_state_thrust")

# copy the state tensors to the gpu and thrust versions
fci_comp_gpu.copy_to_tensor(comp_state_gpu)
fci_comp_gpu.copy_to_tensor(comp_state_gpu2)
fci_comp_thrust.copy_to_tensor_cpu(comp_state_thrust)

# subtract the original state from the gpu and thrust states
comp_state_gpu.subtract(comp_state)
comp_state_gpu2.subtract(comp_state_thrust)
comp_state_thrust.subtract(comp_state)

#print("\n thrust - comp state")
#print("======================================================")
#print(f" {comp_state_thrust}")

# compute the norms of the differences
norm_gpu = comp_state_gpu.norm()
norm_gpu2 = comp_state_gpu2.norm()
norm_thrust = comp_state_thrust.norm()

# get the energies from each computer
Ecomp = fci_comp.get_hf_dot()
Ecomp_gpu = fci_comp_gpu.get_hf_dot()
Ecomp_thrust = fci_comp_thrust.get_hf_dot()

if(norb < 6): 
    print("\n Final FCIcomp Stuff")
    print("===========================")
    print(fci_comp)
    print(fci_comp_gpu)
    print(fci_comp_thrust)


print("\n Timing")
print("======================================================")
print(timer)

print("\n Energetics")
print("======================================================")
print(f" Efci:               {mol.fci_energy}")
print(f" Ehf:                {mol.hf_energy}")
print(f" Enr:                {mol.nuclear_repulsion_energy}")
print(f" Eelec:              {mol.hf_energy - mol.nuclear_repulsion_energy}")

print("\n Norms of State Differences")
print("======================================================")
print(f" ||Cgpu - Ccomp||:           {norm_gpu}")
print(f" ||Cthrust - Ccomp||:        {norm_thrust}")
print(f" ||Cgpu  - Cthrust||:        {norm_gpu2}")

print("\n Energies")
print("======================================================")
print(f" E (from cpu):            {Ecomp}")
print(f" E (from gpu):            {Ecomp_gpu}")
print(f" E (from thrust):         {Ecomp_thrust}")


#Tensor: FCI Computer
#  Ndim  = 2
#  Size  = 36
#  Shape = (6,6)
#
#  Data:
#
#                   0            1            2            3            4            5
#      0   -0.1997705   -0.4588921   -0.1810259   -0.2179127    0.0919495   -0.0138702
#      1    0.0852668   -0.1449795   -0.0245644   -0.1022848    0.1916274   -0.0828110
#      2   -0.2480905    0.0858236    0.1173045    0.1084361    0.1431412    0.0710083
#      3   -0.0759035    0.0424627   -0.0268423    0.1121411    0.1263729    0.0957732
#      4   -0.1364915    0.0464349    0.1172260    0.0563248    0.2084580    0.1396020
#      5   -0.1057914    0.1130906    0.1266330    0.0217731    0.1330584    0.2201510
#
#
#
#TensorGPU: FCI Computer
#  Ndim  = 2
#  Size  = 36
#  Shape = (6,6)
#
#  Data:
#
#                   0            1            2            3            4            5
#      0   -0.1997705   -0.4588921   -0.1810259   -0.2179127    0.0919495   -0.0138702
#      1    0.0852668   -0.1449795   -0.0245644   -0.1022848    0.1916274   -0.0828110
#      2   -0.2480905    0.0858236    0.1173045    0.1084361    0.1431412    0.0710083
#      3   -0.0759035    0.0424627   -0.0268423    0.1121411    0.1263729    0.0957732
#      4   -0.1364915    0.0464349    0.1172260    0.0563248    0.2084580    0.1396020
#      5   -0.1057914    0.1130906    0.1266330    0.0217731    0.1330584    0.2201510