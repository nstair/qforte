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

    
    mol = qf.system_factory(
        build_type='psi4', 
        mol_geometry=geom, 
        symmetry='c1',
        basis='sto-3g',
        run_fci=1)
    
    
    alg_fci = qf.UCCNVQE(
        mol,
        computer_type = 'fci',
        apply_ham_as_tensor=False,
        verbose=False)

    timer.reset()
    alg_fci.run(
        opt_thresh=1.0e-4, 
        pool_type='SD',
        opt_maxiter=2
        )
    timer.record(f'fci  computer {name}')
    
    norb_lst.append(alg_fci._norb)

    
    alg_fock = qf.UCCNVQE(
        mol,
        computer_type = 'fock'
        )

    timer.reset()
    alg_fock.run(
        opt_thresh=1.0e-2, 
        pool_type='SD',
        opt_maxiter=2)
    timer.record(f'fock computer {name}')



    times = timer.get_timings()

    N = int(len(times) / 2)

    # print(N)
    print("\n\n Timing")
    print("================================================================")  

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


