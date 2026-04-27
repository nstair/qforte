import qforte as qf
import numpy as np
 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

def plot_vis_matrix(vis, g_mu_info_str=None, figsize=(6,6), annotate=True, fontsize=9, filename=None, cmap_name='Reds'):
    """
    Plot integer NxN array `vis` where zeros are blue and non-zero entries are shades of red:
      - 0 -> fixed blue
      - 1 .. max -> light -> dark red (1 = lightest)
    Returns (fig, ax).
    """
    v = np.asarray(vis)
    if v.ndim != 2 or v.shape[0] != v.shape[1]:
        # still works for rectangular arrays, but message is optional
        pass

    maxv = int(v.max())

    # Prepare RGBA image buffer
    img = np.zeros(v.shape + (4,), dtype=float)

    # Blue for zeros
    zero_rgba = mpl.colors.to_rgba('grey')
    img[v == 0] = zero_rgba

    # Reds colormap for non-zero entries, map 1 -> low intensity, maxv -> high intensity
    if maxv >= 1:
        cmap = cm.get_cmap(cmap_name)
        vals = v.astype(float)
        # normalized in [0,1] where value 1 -> 0, value maxv -> 1
        if maxv == 1:
            norm_vals = np.zeros_like(vals)
        else:
            norm_vals = (vals - 1.0) / float(maxv - 1)
            norm_vals = np.clip(norm_vals, 0.0, 1.0)
        rgba = cmap(norm_vals)   # RGBA for all entries
        mask = v != 0
        img[mask] = rgba[mask]

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper', interpolation='nearest')

    # grid / ticks
    ax.set_xticks(np.arange(v.shape[1]) + 0.0)
    ax.set_yticks(np.arange(v.shape[0]) + 0.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, v.shape[1] - 0.5)
    ax.set_ylim(v.shape[0] - 0.5, -0.5)

    # annotate numbers (choose white/black text based on luminance)
    if annotate:
        for (i, j), val in np.ndenumerate(v):
            if val == 0:
                continue
            r, g, b, a = img[i, j]
            # perceived luminance
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = 'white' if lum < 0.5 else 'black'
            ax.text(j, i, int(val), ha='center', va='center', color=txt_color, fontsize=fontsize)

    if g_mu_info_str:
        ax.set_title(f'Pairs for {g_mu_info_str}')
    else:
        ax.set_title('Mapping visualization (vis)')
    ax.set_xlabel('Beta str index')
    ax.set_ylabel('Alpha str index')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')

    return fig, ax

# Example usage:
# fig, ax = plot_vis_matrix(vis, filename="vis_map.png")
# plt.show()

def pairs_overlap_any(s1, t1, s2, t2):
    """
    Return 1 if ANY element in (s1 U t1) appears in (s2 U t2), else 0.
    Uses numpy vectorized membership testing for efficiency.
    """
    import numpy as np

    a = np.asarray(s1).ravel()
    b = np.asarray(t1).ravel()
    c = np.asarray(s2).ravel()
    d = np.asarray(t2).ravel()

    A = np.concatenate((a, b)) if (a.size + b.size) > 0 else np.array([], dtype=a.dtype)
    B = np.concatenate((c, d)) if (c.size + d.size) > 0 else np.array([], dtype=c.dtype)

    return int(np.any(np.isin(A, B)))

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
    ]


mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    basis='sto-3g', 
    run_fci=0,
    build_qb_ham = False,
    store_mo_ints=True,
    store_mo_ints_np=True,
    build_df_ham=0,
    df_icut=1.0e-6
    )
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print("\n")
print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
print("\n")

timer = qf.local_timer()

fc = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

vis = np.zeros(fc.get_state().shape(), dtype=int)
# print(vis)

nastrs, nbstrs = fc.get_state().shape()

fc.hartree_fock()

sqham = mol.sq_hamiltonian

hermitian_pairs = qf.SQOpPool()
hermitian_pairs.add_hermitian_pairs(1.0, sqham)


sd_pool = qf.SQOpPool()
sd_pool.set_orb_spaces(ref)
sd_pool.fill_pool("GSD")
# print(sd_pool)




print_text = False
save_figs = True


sqop_trgts = [6, 11]

left_mon_trgts = [([], [], [1], [3]), ([], [], [2], [4])]

spairs = []
tpairs = []


sqop_idx = 0
for sqop in sd_pool.terms():
# for sqop in hermitian_pairs.terms():

    ab_inds = sqop[1].get_unique_ab_inds()

    left_monomial = ab_inds[0]

    daga = ab_inds[0][0]
    undaga = ab_inds[0][1]
    dagb = ab_inds[0][2]
    undagb = ab_inds[0][3]
    
    [counta, sourcea, targeta, paritya] = fc.get_graph().make_mapping_each(
        True,
        daga,
        undaga,
    )

    [countb, sourceb, targetb, parityb] = fc.get_graph().make_mapping_each(
        False,
        dagb,
        undagb,
    )

    ta = np.zeros((counta,), dtype=int)
    sa = np.zeros((counta,), dtype=int)
    pa = np.zeros((counta,), dtype=int)

    tb = np.zeros((countb,), dtype=int)
    sb = np.zeros((countb,), dtype=int)
    pb = np.zeros((countb,), dtype=int)

    for ii in range(counta):
        sa[ii] = sourcea[ii]
        ta[ii] = fc.get_graph().get_aind_for_str(targeta[ii])
        pa[ii] = 1 - 2 * paritya[ii]    

    for ii in range(countb):
        sb[ii] = sourceb[ii]
        tb[ii] = fc.get_graph().get_bind_for_str(targetb[ii])
        pb[ii] = 1 - 2 * parityb[ii]    


    vis.fill(0)

    spair = np.zeros((counta * countb,), dtype = int)
    tpair = np.zeros((counta * countb,), dtype = int)

    # print(f"counta: {counta}")
    # print(f"countb: {countb}")

    for ii in range(counta):
        for jj in range(countb):
            vis[sa[ii], sb[jj]] = countb * ii + jj + 1
            vis[ta[ii], tb[jj]] = countb * ii + jj + 1

            spair[countb * ii + jj] = nbstrs * sa[ii] + sb[jj]
            tpair[countb * ii + jj] = nbstrs * ta[ii] + tb[jj]

    if print_text:
        print("\n\n ======> sqop <====== ")
        print(sqop[0])
        print(sqop[1])

        print(" ==> ab inds <== ")
        print(ab_inds)

        print(f"acount {counta}")
        print(f" sourcea: {sa}")
        print(f" targeta: {ta}")
        print(f" paritya: {pa}")

        print(f"acount {countb}")
        print(f" sourceb: {sb}")
        print(f" targetb: {tb}")
        print(f" parityb: {pb}")

        print("\n ==> vis <== ")
        print(vis)
        print("\n")

    if sqop_idx in sqop_trgts:
    # if left_monomial in left_mon_trgts:
        print("\n Plotting vis matrix for sqop index ", sqop_idx)

        print("\n\n ======> sqop <====== ")
        print(sqop[0])
        print(sqop[1])

        print(" ==> ab inds <== ")
        print(ab_inds)

        print(f"acount {counta}")
        print(f" sourcea: {sa}")
        print(f" targeta: {ta}")
        print(f" paritya: {pa}")

        print(f"acount {countb}")
        print(f" sourceb: {sb}")
        print(f" targetb: {tb}")
        print(f" parityb: {pb}")

        print("\n ==> vis <== ")
        print(vis)
        print("\n")

        print('\n ==> spair <== ')
        print(spair)
        print('\n ==> tpair <== ')
        print(tpair)

        if save_figs:
            f_access =  float(np.count_nonzero(vis) / vis.size)
            g_mu_info_str = f"{sqop[1]} f: {f_access:.3f}"
            fname = f"sandbox/fci_graph/pairs_vis_v1/sqop_idx_{sqop_idx}.png"

            fig, ax = plot_vis_matrix(vis, g_mu_info_str=g_mu_info_str, filename=fname)
            # plt.show()

    
    spairs.append(spair)
    tpairs.append(tpair)
    
    sqop_idx += 1

if(len(spairs) != len(tpairs)):
    raise ValueError("Index mismatch in spairs and tpairs lengths")

M = len(spairs)

overlap_mat = np.zeros((M, M), dtype=int)

for mu in range(M):
    for nu in range(M):
        s1 = spairs[mu]
        t1 = tpairs[mu]
        s2 = spairs[nu]
        t2 = tpairs[nu]

        overlap_mat[mu, nu] = pairs_overlap_any(s1, t1, s2, t2)


print("\n Pairs overlap matrix:")
print(overlap_mat)

zero_pairs = np.argwhere(overlap_mat == 0)

print("\n Zero-overlap pairs (mu, nu):")
for zp in zero_pairs:
    print(f" {zp[0]} , {zp[1]}")
    









