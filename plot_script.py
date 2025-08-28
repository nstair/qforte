import matplotlib.pyplot as plt
import numpy as np

def plot_speedup(save_path='speedup_plot.png'):
    # Data
    systems = ['H4', 'LiH', 'H2O', 'H8', 'N2', 'H10', 'H12', 'H14']
    qubits = [8, 12, 14, 16, 20, 20, 24, 28]
    speedups = [0.012658228, 0.018315018, 0.022530329, 0.074185968, 0.17195946, 0.716910338, 3.205705143, 1.627771907]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(qubits, speedups, color='blue')

    # Logarithmic scale for y-axis
    ax.set_yscale('log')

    # Set integer ticks for x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Label each point with the system name
    for x, y, label in zip(qubits, speedups, systems):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,8), ha='center')

    # Labels and title
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Wall Time Speedup')
    ax.set_title('Wall Time Speedup vs Number of Qubits')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# To use:
plot_speedup()
