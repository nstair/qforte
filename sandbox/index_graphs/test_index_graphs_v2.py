# index graphs for single alpha and beta excitations using a UCC circuit

import qforte as qf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

def spin_orb_to_spatial(spin_orb):
    """Return (is_alpha, spatial_orbital) for qforte's spin-orbital indexing."""
    spin_orb = int(spin_orb)
    return spin_orb % 2 == 0, spin_orb // 2


def read_ucc_mapping_for_sq_term(graph, sq_term):

    coeff, crea_ops, ann_ops = sq_term

    if len(crea_ops) != 1 or len(ann_ops) != 1:
        raise ValueError("Expected a single excitation.")

    crea_spin = int(crea_ops[0])
    ann_spin  = int(ann_ops[0])

    crea_alpha, crea_spatial = spin_orb_to_spatial(crea_spin)
    ann_alpha, ann_spatial   = spin_orb_to_spatial(ann_spin)

    if crea_alpha != ann_alpha:
        raise ValueError("Spin-flipping excitation encountered.")

    dag = [crea_spatial]
    undag = [ann_spatial]

    count, source, target, parity = graph.make_mapping_each(
    crea_alpha,
    dag,
    undag,
    )

    # print("\n===== Raw make_mapping_each() output =====")
    # print("count :", count)
    # print("source:", list(source))
    # print("target:", list(target))
    # print("parity:", list(parity))
    
    # print("alpha strings:", list(graph.get_astr()))
    # print("beta strings :", list(graph.get_bstr()))

    return {
        "coeff": coeff,
        "is_alpha": crea_alpha,
        "spin_sector": "alpha" if crea_alpha else "beta",
        "source_orb": ann_spatial,
        "target_orb": crea_spatial,
        "count": count,
        "source": list(source[:count]),
        "target": list(target[:count]),
        "parity": list(parity[:count]),
    }


def expand_mapping_to_fci_pairs(mapping, graph):

    len_alpha = graph.get_lena()
    len_beta = graph.get_lenb()
    strings = graph.get_astr() if mapping["is_alpha"] else graph.get_bstr()

    # bitstring -> determinant index
    bit_to_index = {
        int(bit): idx
        for idx, bit in enumerate(strings)
    }

    pairs = []

    for source_idx, target_bit, parity in zip(
            mapping["source"],
            mapping["target"],
            mapping["parity"]):

        source_idx = int(source_idx)

        target_idx = bit_to_index[int(target_bit)]

        if mapping["is_alpha"]:

            for beta in range(len_beta):

                pairs.append({
                    "qubit_1_coord": (source_idx, beta),
                    "qubit_2_coord": (target_idx, beta),
                    "parity": parity,
                })

        else:

            for alpha in range(len_alpha):

                pairs.append({
                    "qubit_1_coord": (alpha, source_idx),
                    "qubit_2_coord": (alpha, target_idx),
                    "parity": parity,
                })

    return pairs


def make_ucc_color_matrix(ucc_pairs, graph):
    """Create a matrix with color codes for untouched, source, and target cells."""
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    color_matrix = np.zeros((len_alpha, len_beta), dtype=int)

    for pair in ucc_pairs:
        source_alpha, source_beta = pair["qubit_1_coord"]
        target_alpha, target_beta = pair["qubit_2_coord"]

        color_matrix[source_alpha, source_beta] = 1

        if color_matrix[target_alpha, target_beta] == 1:
            color_matrix[target_alpha, target_beta] = 3
        else:
            color_matrix[target_alpha, target_beta] = 2

    return color_matrix


def bitstring_label(index, bitstring, norb):
    """Format a determinant-string index with its orbital occupation bitstring."""
    occ = format(int(bitstring), f"0{norb}b")[::-1]
    return f"{index}: {occ}"


def determinant_axis_labels(graph):
    """Build alpha and beta axis labels from FCIGraph determinant strings."""
    norb = int(graph.get_astr_at_idx(0)).bit_length()
    for bitstring in list(graph.get_astr()) + list(graph.get_bstr()):
        norb = max(norb, int(bitstring).bit_length())

    alpha_labels = [
        bitstring_label(index, bitstring, norb)
        for index, bitstring in enumerate(graph.get_astr())
    ]
    beta_labels = [
        bitstring_label(index, bitstring, norb)
        for index, bitstring in enumerate(graph.get_bstr())
    ]

    return alpha_labels, beta_labels


def plot_ucc_color_matrix(color_matrix, ucc_pairs, mapping, graph, output_path):
    """Plot alpha rows by beta columns and save the color-coded matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = [
        "#d9d9d9c6",  # untouched
        "#0aa137",  # qubit 1 / source
        "#b3190b",  # qubit 2 / target
        "#96407e",  # both source and target
    ]
    labels = [
        "untouched",
        "qubit 1 / source",
        "qubit 2 / target",
        "source and target",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(color_matrix, cmap=cmap, norm=norm, origin="upper")
    
    # Draw one arrow per unique orbital excitation
    unique_excitations = []

    for pair in ucc_pairs:
        source_row, source_col = pair["qubit_1_coord"]
        target_row, target_col = pair["qubit_2_coord"]

        if mapping["is_alpha"]:
            # alpha excitation: same column, changing rows
            excitation = ("alpha", source_col, source_row, target_row)

        else:
            # beta excitation: same row, changing columns
            excitation = ("beta", source_row, source_col, target_col)

        if excitation not in unique_excitations:
            unique_excitations.append(excitation)


    if mapping["is_alpha"]:
        # Vertical arrows
        arrow_cols = np.linspace(
            color_matrix.shape[1] * 0.35,
            color_matrix.shape[1] * 0.65,
            len(unique_excitations),
        )

        for excitation, arrow_col in zip(unique_excitations, arrow_cols):

            _, col, source_row, target_row = excitation

            ax.annotate(
                "",
                xy=(arrow_col, target_row),
                xytext=(arrow_col, source_row),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    linewidth=2,
                ),
                zorder=5,
            )


    else:
        # Horizontal arrows
        arrow_rows = np.linspace(
            color_matrix.shape[0] * 0.35,
            color_matrix.shape[0] * 0.65,
            len(unique_excitations),
        )

        for excitation, arrow_row in zip(unique_excitations, arrow_rows):

            _, row, source_col, target_col = excitation

            ax.annotate(
                "",
                xy=(target_col, arrow_row),
                xytext=(source_col, arrow_row),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    linewidth=2,
                ),
                zorder=5,
            )

    alpha_labels, beta_labels = determinant_axis_labels(graph)

    ax.set_xlabel("beta determinant string")
    ax.set_ylabel("alpha determinant string")
    ax.set_title(
        "FCI determinant mapping for first UCC pool operator\n"
        f"{mapping['spin_sector']} excitation: {mapping['source_orb']} → {mapping['target_orb']}"
    )

    ax.set_xticks(np.arange(color_matrix.shape[1]))
    ax.set_yticks(np.arange(color_matrix.shape[0]))
    ax.set_xticklabels(beta_labels, rotation=45, ha="right")
    ax.set_yticklabels(alpha_labels)
    ax.set_xticks(np.arange(-0.5, color_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, color_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(colors, labels)
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.subplots_adjust(left=0.18, right=0.72, top=0.86, bottom=0.22)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)


# build H4 molecule
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0))
]

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

# build FCI graph object
graph = qf.FCIGraph(int(norb/2), int(norb/2), norb)

# build UCCsd pool for molecule
pool = qf.SQOpPool()
pool.set_orb_spaces(ref)
pool.fill_pool("SD")
print(pool)

# make individual second quantized operators representative of the ones in pool
op1 = pool.terms()[0]
print(op1[0])
print(op1[1])

subop1 = op1[1].terms()
print(subop1)

crea_op_idx = subop1[0][1] # spin orbital indexes , 2 is 1 in spatial
anna_op_idx = subop1[0][2] # 4 is 2 in spatial

# convert spin orbitral indices to spatial orbital
crea_op_idx_spat = crea_op_idx[0]//2
anna_op_idx_spat = anna_op_idx[0]//2
print(crea_op_idx_spat)
print(anna_op_idx_spat)

# write subfunction that reads info from each operator --> source, targ, parity lists
mapping_info = read_ucc_mapping_for_sq_term(graph, subop1[0])
print("\n Givens mapping info for first pool operator")
print("============================================")
print(f" spin sector:        {mapping_info['spin_sector']}")
print(f" source spatial orb: {mapping_info['source_orb']}")
print(f" target spatial orb: {mapping_info['target_orb']}")
print(f" count:              {mapping_info['count']}")
print(f" source:             {mapping_info['source']}")
print(f" target:             {mapping_info['target']}")
print(f" parity:             {mapping_info['parity']}")

# take the indices from the subfunction and mapping function and group them into Givens pairs
ucc_pairs = expand_mapping_to_fci_pairs(mapping_info, graph)
print("\n Givens-pair data prepared for later plotting")
print("=============================================")
print(f" matrix shape:           ({graph.get_lena()}, {graph.get_lenb()})")
print(f" Givens pair count:       {len(ucc_pairs)}")
# print(f" altered matrix coords:   {altered_matrix_coords}")
print(f" first Givens pairs:      {ucc_pairs[:10]}")

# read in indices and color coordinate them

color_matrix = make_ucc_color_matrix(ucc_pairs, graph)
print("\n Color-code matrix")
print("==================")
print(color_matrix)

# create matplot script
output_path = Path(__file__).resolve().parent / "first_UCCop_index_map1.png"
plot_ucc_color_matrix(color_matrix, ucc_pairs, mapping_info, graph, output_path)
print(f"\nSaved color-coded matrix graph to: {output_path}")