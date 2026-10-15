"""Reproducible mathematical experiments for the baby graph kernel."""
import numpy as np
from baby_kernel import (
    baby_kernel, enhanced_kernel, degree_histogram, kernel_matrix,
    validate_graph
)


def permute_graph(A, permutation):
    """Relabel nodes: A' = P A P^T, implemented by row/column reindexing."""
    return A[np.ix_(permutation, permutation)]


def make_cycle(n):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


def make_two_triangles():
    A = np.zeros((6, 6), dtype=int)
    for offset in (0, 3):
        for i in range(3):
            u, v = offset + i, offset + (i + 1) % 3
            A[u, v] = A[v, u] = 1
    return A


def main():
    path = np.array([[0,1,0],[1,0,1],[0,1,0]])
    triangle = np.array([[0,1,1],[1,0,1],[1,1,0]])
    permuted_path = permute_graph(path, [2, 0, 1])
    c6 = make_cycle(6)
    two_triangles = make_two_triangles()

    print("EXPERIMENT 1 — Isomorphic graphs under node relabelling")
    print("K(path, path)         =", baby_kernel(path, path))
    print("K(path, relabel(path))=", baby_kernel(path, permuted_path))
    print("Expected: same values because degree histograms are invariant to node order.\n")

    print("EXPERIMENT 2 — Non-isomorphic graphs with same degree histogram")
    print("C6 histogram:          ", degree_histogram(c6))
    print("Two-triangles histogram:", degree_histogram(two_triangles))
    print("Baseline K(C6, two triangles) =", baby_kernel(c6, two_triangles))
    print("Enhanced K(C6, two triangles)=", enhanced_kernel(c6, two_triangles))
    print("Expected: baseline cannot distinguish them; component/triangle features help.\n")

    print("EXPERIMENT 3 — Symmetry")
    print("K(path, triangle) =", baby_kernel(path, triangle))
    print("K(triangle, path) =", baby_kernel(triangle, path))
    print("Expected: equal, since dot product is symmetric.\n")

    print("EXPERIMENT 4 — Positive semidefinite kernel matrix")
    graphs = [path, triangle, permuted_path, c6, two_triangles]
    K = kernel_matrix(graphs, kernel=baby_kernel)
    eigvals = np.linalg.eigvalsh(K)
    print("Kernel matrix:\n", K)
    print("Eigenvalues:", np.round(eigvals, 8))
    print("PSD within tolerance:", bool(eigvals.min() >= -1e-8))


if __name__ == "__main__":
    main()
