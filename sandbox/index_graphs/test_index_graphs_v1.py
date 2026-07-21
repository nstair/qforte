# index graphs for single alpha and beta excitations using a Givens gate

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


def read_givens_mapping_for_sq_term(graph, sq_term):
    """Read source, target, and parity lists for a single-excitation SQ term."""
    
    coeff, crea_ops, ann_ops = sq_term

    if len(crea_ops) != 1 or len(ann_ops) != 1:
        raise ValueError(
            "make_givens_mapping_each() expects one creation and one annihilation "
            f"operator, but got creation={crea_ops}, annihilation={ann_ops}"
        )

    crea_spin_orb = int(crea_ops[0])
    ann_spin_orb = int(ann_ops[0])

    crea_is_alpha, target_orb = spin_orb_to_spatial(crea_spin_orb)
    ann_is_alpha, source_orb = spin_orb_to_spatial(ann_spin_orb)
    
    source_orbs = [source_orb]
    target_orbs = [target_orb]

    if crea_is_alpha != ann_is_alpha:
        raise ValueError(
            "Givens mapping is spin preserving, but this term mixes alpha/beta "
            f"operators: creation={crea_spin_orb}, annihilation={ann_spin_orb}"
        )

    count, source, target, parity = graph.make_givens_mapping_each(
        crea_is_alpha,
        source_orbs,
        target_orbs,
    )

    return {
        "coeff": coeff,
        "is_alpha": crea_is_alpha,
        "spin_sector": "alpha" if crea_is_alpha else "beta",
        "creation_spin_orb": crea_spin_orb,
        "annihilation_spin_orb": ann_spin_orb,
        "source_orb": source_orb,
        "target_orb": target_orb,
        "count": count,
        "source": list(source),
        "target": list(target),
        "parity": list(parity),
    }


def expand_mapping_to_fci_givens_pairs(mapping, graph):
    """Expand spin-string mappings into FCI determinant pairs for Givens rotations.

    Each returned pair is one two-level rotation:
    qubit_1_coord = (source alpha index, source beta index)
    qubit_2_coord = (target alpha index, target beta index)
    """
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    givens_pairs = []

    for source_idx, target_idx, parity in zip(
        mapping["source"],
        mapping["target"],
        mapping["parity"],
    ):
        if mapping["is_alpha"]:
            beta_range = range(len_beta)
            for beta_idx in beta_range:
                qubit_1_coord = (source_idx, beta_idx)
                qubit_2_coord = (target_idx, beta_idx)
                givens_pairs.append({
                    "qubit_1_coord": qubit_1_coord,
                    "qubit_2_coord": qubit_2_coord,
                    "source_coord": qubit_1_coord,
                    "target_coord": qubit_2_coord,
                    "source_alpha_idx": source_idx,
                    "source_beta_idx": beta_idx,
                    "target_alpha_idx": target_idx,
                    "target_beta_idx": beta_idx,
                    "parity": parity,
                    "color_key": "positive" if parity > 0 else "negative",
                })
        else:
            alpha_range = range(len_alpha)
            for alpha_idx in alpha_range:
                qubit_1_coord = (alpha_idx, source_idx)
                qubit_2_coord = (alpha_idx, target_idx)
                givens_pairs.append({
                    "qubit_1_coord": qubit_1_coord,
                    "qubit_2_coord": qubit_2_coord,
                    "source_coord": qubit_1_coord,
                    "target_coord": qubit_2_coord,
                    "source_alpha_idx": alpha_idx,
                    "source_beta_idx": source_idx,
                    "target_alpha_idx": alpha_idx,
                    "target_beta_idx": target_idx,
                    "parity": parity,
                    "color_key": "positive" if parity > 0 else "negative",
                })

    altered_matrix_coords = sorted({
        coord
        for pair in givens_pairs
        for coord in (pair["qubit_1_coord"], pair["qubit_2_coord"])
    })

    return givens_pairs, altered_matrix_coords


def make_givens_color_matrix(givens_pairs, graph):
    """Create a matrix with color codes for untouched, source, and target cells."""
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    color_matrix = np.zeros((len_alpha, len_beta), dtype=int)

    for pair in givens_pairs:
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


def plot_givens_color_matrix(color_matrix, givens_pairs, mapping, graph, output_path):
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

    for pair in givens_pairs:
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
        "FCI determinant cells touched by SQ pool operator\n"
        f"{mapping['spin_sector']} move: {mapping['source_orb']} -> {mapping['target_orb']}"
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
mapping_info = read_givens_mapping_for_sq_term(graph, subop1[0])
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
givens_pairs, altered_matrix_coords = expand_mapping_to_fci_givens_pairs(mapping_info, graph)
print("\n Givens-pair data prepared for later plotting")
print("=============================================")
print(f" matrix shape:           ({graph.get_lena()}, {graph.get_lenb()})")
print(f" Givens pair count:       {len(givens_pairs)}")
print(f" altered matrix coords:   {altered_matrix_coords}")
print(f" first Givens pairs:      {givens_pairs[:10]}")

# read in indices and color coordinate them
color_matrix = make_givens_color_matrix(givens_pairs, graph)
print("\n Color-code matrix")
print("==================")
print(color_matrix)

# create matplot script
output_path = Path(__file__).resolve().parent / "first_SQop_index_map1.png"
plot_givens_color_matrix(color_matrix, givens_pairs, mapping_info, graph, output_path)
print(f"\nSaved color-coded matrix graph to: {output_path}")