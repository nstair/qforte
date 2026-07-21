# comparing UCC on FCI and Fock computers
import qforte as qf
import numpy as np
from qforte.utils.trotterization import trotterize

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g', run_fci=1)
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref) # num electrons
sz = 0 # spin Z component
norb = int(len(ref) / 2) # number orbitals

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fock_comp = qf.Computer(norb * 2)
graph = qf.FCIGraph(int(norb / 2), int(norb / 2), norb)

# reference = 'random'
reference = 'hf'

if(reference == 'hf'):
    fci_comp.hartree_fock()
    
    Uhf = qf.utils.state_prep.build_Uprep(ref, 'occupation_list')
    fock_comp.apply_circuit(Uhf)
    
elif(reference == 'random'):
    # FCI Computer case
    np.random.seed(42)
    random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fci_comp.set_state(Crand)
    
    # FOCK Computer case
    state = []

    C = fci_comp.get_state()

    for alpha_idx in range(graph.get_lena()):
        for beta_idx in range(graph.get_lenb()):

            coeff = C.get([alpha_idx, beta_idx])

            if abs(coeff) < 1e-12:
                continue

            alpha_bits = int(graph.get_astr_at_idx(alpha_idx))
            beta_bits = int(graph.get_bstr_at_idx(beta_idx))

            qb = qf.QubitBasis()

            for i in range(norb):
                qb.set_bit(2 * i,     bool((alpha_bits >> i) & 1))
                qb.set_bit(2 * i + 1, bool((beta_bits >> i) & 1))

            state.append((qb, coeff))

    fock_comp.set_state(state)
    
# print(fci_comp)
# print(fock_comp)

# ===> evovle pool basic <====

ah = True
adj = False

print("\n Time Evo Settings")
print("===========================")
print(f" time:      {1.0}")
print(f" r:         {1}")
print(f" order:     {1}")
print(f" antiherm:  {ah}")
print(f" adjoint:   {adj}")
print("\n")

sd_pool = qf.SQOpPool()
sd_pool.set_orb_spaces(ref)
sd_pool.fill_pool("SD")
# print(sd_pool)

A = sd_pool.get_qubit_operator('commuting_grp_lex', qubit_excitations = False)

U, phase1 = trotterize(A, trotter_number=1)

# FCI Computer case - UCC
fci_comp.evolve_pool_trotter_basic(
    sd_pool,
    antiherm=ah,
    adjoint=adj)
print(fci_comp)

# Fock Computer case
fock_comp.apply_circuit(U)
print(fock_comp)

do_phase_compare = True
tensor_diff = fock_comp.get_fci_tensor_diff(fci_comp, do_phase_compare)
print(tensor_diff)