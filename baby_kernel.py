"""Baby graph kernel implemented from scratch using NumPy.

This is an educational degree-histogram kernel, not a complete graph-isomorphism test.
"""
from __future__ import annotations
import numpy as np


def validate_graph(A: np.ndarray) -> np.ndarray:
    """Validate a simple, undirected, unweighted graph adjacency matrix."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("This implementation supports only 0/1 adjacency matrices.")
    if not np.array_equal(A, A.T):
        raise ValueError("Undirected graph adjacency matrix must be symmetric.")
    if np.any(np.diag(A) != 0):
        raise ValueError("Self-loops are not supported in this baby-kernel implementation.")
    return A.astype(int)


def degree_vector(A: np.ndarray) -> np.ndarray:
    """Degree of each node: d_i = sum_j A_ij."""
    A = validate_graph(A)
    return A.sum(axis=1)


def degree_histogram(A: np.ndarray, max_degree: int | None = None) -> np.ndarray:
    """h_G(d) = number of vertices of degree d."""
    degrees = degree_vector(A)
    if max_degree is None:
        max_degree = max(A.shape[0] - 1, int(degrees.max(initial=0)))
    if max_degree < int(degrees.max(initial=0)):
        raise ValueError("max_degree is smaller than a degree in the graph.")
    return np.bincount(degrees, minlength=max_degree + 1)


def feature_vector(A: np.ndarray, max_degree: int | None = None) -> np.ndarray:
    """Explicit feature map for the baseline kernel: degree histogram."""
    return degree_histogram(A, max_degree=max_degree).astype(float)


def baby_kernel(A: np.ndarray, B: np.ndarray) -> float:
    """K(A,B) = h_A^T h_B, padding both histograms to the same length."""
    A = validate_graph(A)
    B = validate_graph(B)
    max_degree = max(A.shape[0] - 1, B.shape[0] - 1)
    hA = feature_vector(A, max_degree=max_degree)
    hB = feature_vector(B, max_degree=max_degree)
    return float(hA @ hB)


def enhanced_feature_vector(A: np.ndarray, max_degree: int | None = None) -> np.ndarray:
    """A simple extension: degree histogram + component count + triangle count.

    These hand-designed features remain invariant to node relabelling. They still
    do not fully characterize graph structure or decide graph isomorphism.
    """
    A = validate_graph(A)
    h = feature_vector(A, max_degree=max_degree)
    n = A.shape[0]
    # Connected components, computed by a small DFS without a graph library.
    seen = set()
    components = 0
    for start in range(n):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    # Each triangle contributes 6 to trace(A^3), one per ordering/direction.
    triangles = int(round(np.trace(A @ A @ A) / 6))
    return np.concatenate([h.astype(float), np.array([components, triangles], dtype=float)])


def enhanced_kernel(A: np.ndarray, B: np.ndarray) -> float:
    """Dot-product kernel on the extended, fixed-length graph feature vectors."""
    A = validate_graph(A)
    B = validate_graph(B)
    max_degree = max(A.shape[0] - 1, B.shape[0] - 1)
    fA = enhanced_feature_vector(A, max_degree)
    fB = enhanced_feature_vector(B, max_degree)
    return float(fA @ fB)


def kernel_matrix(graphs, kernel=baby_kernel) -> np.ndarray:
    """Return the symmetric Gram/kernel matrix for a list of graphs."""
    n = len(graphs)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = kernel(graphs[i], graphs[j])
    return K


def demo():
    path3 = np.array([[0,1,0],[1,0,1],[0,1,0]])
    triangle = np.array([[0,1,1],[1,0,1],[1,1,0]])
    cycle6 = np.zeros((6,6), dtype=int)
    for i in range(6):
        cycle6[i, (i+1)%6] = cycle6[(i+1)%6, i] = 1
    two_triangles = np.zeros((6,6), dtype=int)
    for offset in (0, 3):
        for i in range(3):
            u, v = offset+i, offset+(i+1)%3
            two_triangles[u,v] = two_triangles[v,u] = 1
    print("Path-3 degrees:", degree_vector(path3))
    print("Path-3 histogram:", degree_histogram(path3))
    print("Triangle histogram:", degree_histogram(triangle))
    print("K(path3, triangle):", baby_kernel(path3, triangle))
    print("K(path3, path3):", baby_kernel(path3, path3))
    print("\nCounterexample: C6 vs two disconnected triangles")
    print("Baseline kernel:", baby_kernel(cycle6, two_triangles))
    print("Enhanced kernel:", enhanced_kernel(cycle6, two_triangles))
    print("Connected components:", enhanced_feature_vector(cycle6)[-2],
          "vs", enhanced_feature_vector(two_triangles)[-2])
    print("Triangle counts:", enhanced_feature_vector(cycle6)[-1],
          "vs", enhanced_feature_vector(two_triangles)[-1])


if __name__ == "__main__":
    demo()
