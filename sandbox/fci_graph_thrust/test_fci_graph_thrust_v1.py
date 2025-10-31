import qforte as qf
import numpy as np


# function to run make_mapping_each many times
def run_make_mapping_each(n_runs=1000, qg=None, qg_thrust=None, timer=None):

    daga = list(range(0, 10))      # 10 creation operators
    undaga = list(range(5, 15))     # 10 annihilation operators

    timer.reset()
    for _ in range(n_runs):
        qg.make_mapping_each(True, daga, undaga)
    timer.record('Make Mapping Each (Original) x 1000')

    timer.reset()
    for _ in range(n_runs):
        qg_thrust.make_mapping_each(True, daga, undaga)
    timer.record('Make Mapping Each (Thrust) x 1000')



if __name__ == "__main__":
    na = 10
    nb = 10
    norb = 20

    timer = qf.local_timer()
    timer.reset()

    qg = qf.FCIGraph(na, nb, norb)
    qg_thrust = qf.FCIGraphGPU(na, nb, norb)

    timer.record('Initialize')

    # timing stuff
    run_make_mapping_each(n_runs=1000, qg=qg, qg_thrust=qg_thrust, timer=timer)

    # check for correctness

    na = 3
    nb = 3
    norb = 6

    qg = qf.FCIGraph(na, nb, norb)
    qg_thrust = qf.FCIGraphGPU(na, nb, norb)

    daga = [2, 3]
    undaga = [4, 5]

    dagb = [1]
    undagb = [3]

    [vala, sourcea, targeta, paritya] = qg.make_mapping_each(
        True,
        daga,
        undaga,
    )

    [valb, sourceb, targetb, parityb] = qg.make_mapping_each(
        False,
        dagb,
        undagb,
    )

    [vala_thrust, sourcea_thrust, targeta_thrust, paritya_thrust] = qg_thrust.make_mapping_each(
        True,
        daga,
        undaga,
    )

    [valb_thrust, sourceb_thrust, targetb_thrust, parityb_thrust] = qg_thrust.make_mapping_each(
        False,
        dagb,
        undagb,
    )


    print("\n========== Original Results ==========")
    print(f"acount {vala}")
    print(f" sourcea: {sourcea}")
    print(f" targeta: {targeta}")
    print(f" paritya: {paritya}")

    print(f"acount {valb}")
    print(f" sourceb: {sourceb}")
    print(f" targetb: {targetb}")
    print(f" parityb: {parityb}")

    print("\n========== Thrust Results ==========")
    print(f"acount {vala_thrust}")
    print(f" sourcea: {sourcea_thrust}")
    print(f" targeta: {targeta_thrust}")
    print(f" paritya: {paritya_thrust}")

    print(f"acount {valb_thrust}")
    print(f" sourceb: {sourceb_thrust}")
    print(f" targetb: {targetb_thrust}")
    print(f" parityb: {parityb_thrust}")

    print("\n Timing")
    print("======================================================")
    print(timer)

    # Compare results
    print("\n========== Comparison Results ==========")
    if np.array_equal(vala, vala_thrust):
        print("vala    ✓")
    else:
        print("vala    x")
    if np.array_equal(sourcea, sourcea_thrust):
        print("sourcea ✓")
    else:
        print("sourcea x")
    if np.array_equal(targeta, targeta_thrust):
        print("targeta ✓")
    else:
        print("targeta x")
    if np.array_equal(paritya, paritya_thrust):
        print("paritya ✓")
    else:
        print("paritya x")
    if np.array_equal(valb, valb_thrust):
        print("valb    ✓")
    else:
        print("valb    x")
    if np.array_equal(sourceb, sourceb_thrust):
        print("sourceb ✓")
    else:
        print("sourceb x")
    if np.array_equal(targetb, targetb_thrust):
        print("targetb ✓")
    else:
        print("targetb x")
    if np.array_equal(parityb, parityb_thrust):
        print("parityb ✓")
    else:
        print("parityb x")

# Correct Output
"""
vala 2
valb 6

 ==> result alfa <== 
 [[ 9. 13.  1.]
 [15. 14.  1.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]

 ==> result beta <== 
 [[ 4.  7.  1.]
 [ 7. 19.  0.]
 [ 8. 35.  0.]
 [16. 22.  1.]
 [17. 38.  1.]
 [19. 50.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]
"""

