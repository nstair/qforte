import qforte as qf
import numpy as np

# Define the reference and geometry lists.

BeH2_geom = [
    ('H', (0., 0., 1.0)), 
    ('Be', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ]

H20_geom = [
    ('H', (0., 0., 1.0)), 
    ('O', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ]

N2_geom = [
    ('N', (0., 0., 1.0)), 
    ('N', (0., 0., 2.0)), 
    ]

LiH_geom = [
    ('Li', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ]



H2_geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ]


H4_geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ]

H6_geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ]

H8_geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    ('H', (0., 0., 5.0)), 
    ('H', (0., 0., 6.0)),
    ('H', (0., 0., 7.0)), 
    ('H', (0., 0., 8.0)),
    ]

H10_geom = [
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
    ]

H12_geom = [
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
    ('H', (0., 0.,11.0)), 
    ('H', (0., 0.,12.0))
    ]

geom_list = [
    H2_geom,
    H4_geom,
    # H6_geom,
    H8_geom,
    # BeH2_geom,
    LiH_geom,
    N2_geom,
    H20_geom,
    H10_geom,
    # H12_geom,
    ]



name_list = [
    'H2',
    'H4',
    # 'H6',
    'H8',
    # 'BeH2',
    'LiH',
    'N2',
    'H2O',
    'H10',
    # 'H12'
    ]

norb_lst = []


timer = qf.local_timer()


for geom, name in zip(geom_list, name_list):


    # timer.reset()
    # Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
    mol = qf.system_factory(
        build_type='psi4', 
        mol_geometry=geom, 
        basis='sto-3g', 
        build_qb_ham = False,
        run_fci=0)

    # timer.record('Run Psi4 and Initialize')
    
    ref = mol.hf_reference

    nel = sum(ref)
    sz = 0
    norb = int(len(ref) / 2)

    norb_lst.append(norb)

    # if(norb < 6): 
    #     print(f" nqbit:     {norb*2}")
    #     print(f" nel:       {nel}")
    
    fc1 = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
    fc1.hartree_fock()
    
    timer.reset()
    # fc1.apply_tensor_spat_012bdy(
    #     mol.nuclear_repulsion_energy, 
    #     mol.mo_oeis, 
    #     mol.mo_teis, 
    #     mol.mo_teis_einsum, 
    #     norb
    # )
    fc1.apply_sqop(mol.sq_hamiltonian)
    timer.record(f'fci  computer {name}')

    E1 = fc1.get_hf_dot() 

    # fock computer stuff

    fk1 = qf.Computer(norb * 2)
    Uhf = qf.utils.state_prep.build_Uprep(ref, 'occupation_list')
    fk1.apply_circuit(Uhf)

    timer.reset()
    fk1.apply_sq_operator(mol.sq_hamiltonian)
    timer.record(f'fock computer {name}')

    times = timer.get_timings()

    N = int(len(times) / 2)

    # print(N)
    print("\n\n Timing")
    print("======================================================")  

    for n in range(N):
        ifc = 2*n
        ifk = 2*n + 1

        _, tval_fc = times[ifc]
        _, tval_fk = times[ifk]

        line = f"{name_list[n]:8}  {tval_fk:e}   {tval_fc:e}  {tval_fk/tval_fc:e}  {norb_lst[n]*2:6}"
        print(line)

    print("\n")

    
        
    
print("")
print(timer)

    # print("\n Energetics")
    # print("======================================================")
    # print(f" Efci:               {mol.fci_energy}")
    # print(f" Ehf:                {mol.hf_energy}")
    # print(f" Enr:                {mol.nuclear_repulsion_energy}")
    # print(f" Eelec:              {mol.hf_energy - mol.nuclear_repulsion_energy}")
    # print(f" E1 (from tensor):   {E1}")