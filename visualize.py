"""Save the graph counterexample and degree-kernel similarity heatmap."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from baby_kernel import baby_kernel, enhanced_kernel, kernel_matrix

OUT = Path("results")
OUT.mkdir(exist_ok=True)

def counterexample_graphs():
    cycle = nx.cycle_graph(6)
    two_triangles = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    return cycle, two_triangles

def draw_counterexample():
    g1, g2 = counterexample_graphs()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    nx.draw_networkx(g1, pos=nx.circular_layout(g1), ax=axes[0],
                     node_color="lightblue", edge_color="gray", node_size=650)
    axes[0].set_title("Cycle C6\\n1 component, 0 triangles")
    nx.draw_networkx(g2, pos=nx.spring_layout(g2, seed=7), ax=axes[1],
                     node_color="lightgreen", edge_color="gray", node_size=650)
    axes[1].set_title("Two triangles\\n2 components, 2 triangles")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Same degree histogram, different graph structure")
    fig.tight_layout()
    fig.savefig(OUT / "isomorphism_counterexample.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def draw_heatmap():
    path = nx.to_numpy_array(nx.path_graph(3), dtype=int)
    triangle = nx.to_numpy_array(nx.complete_graph(3), dtype=int)
    relabelled_path = path[np.ix_([2, 0, 1], [2, 0, 1])]
    c6 = nx.to_numpy_array(nx.cycle_graph(6), dtype=int)
    two_triangles = nx.to_numpy_array(
        nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)), dtype=int
    )
    graphs = [path, triangle, relabelled_path, c6, two_triangles]
    labels = ["Path-3", "Triangle", "Relabelled path", "Cycle-6", "Two triangles"]
    K = kernel_matrix(graphs, kernel=baby_kernel)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(K, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Degree-histogram kernel similarity matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{K[i, j]:.0f}", ha="center", va="center",
                    color="white" if K[i, j] > K.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Raw kernel score")
    fig.tight_layout()
    fig.savefig(OUT / "kernel_similarity_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    draw_counterexample()
    draw_heatmap()
    g1, g2 = counterexample_graphs()
    A1 = nx.to_numpy_array(g1, dtype=int)
    A2 = nx.to_numpy_array(g2, dtype=int)
    print("Saved results/isomorphism_counterexample.png")
    print("Saved results/kernel_similarity_heatmap.png")
    print(f"Baseline K(C6, two triangles): {baby_kernel(A1, A2):.1f}")
    print(f"Enhanced K(C6, two triangles): {enhanced_kernel(A1, A2):.1f}")
