import numpy as np
import pytest
from baby_kernel import (
    degree_vector, degree_histogram, baby_kernel, enhanced_kernel,
    kernel_matrix
)
from experiments import make_cycle, make_two_triangles, permute_graph


def test_path_degrees_and_histogram():
    A = np.array([[0,1,0],[1,0,1],[0,1,0]])
    assert degree_vector(A).tolist() == [1,2,1]
    assert degree_histogram(A).tolist() == [0,2,1]


def test_kernel_symmetric():
    A = np.array([[0,1,0],[1,0,1],[0,1,0]])
    B = np.ones((3,3), dtype=int) - np.eye(3, dtype=int)
    assert baby_kernel(A, B) == baby_kernel(B, A)


def test_permutation_invariance():
    A = np.array([[0,1,0],[1,0,1],[0,1,0]])
    Ap = permute_graph(A, [2,0,1])
    assert baby_kernel(A, A) == baby_kernel(Ap, Ap)
    assert degree_histogram(A).tolist() == degree_histogram(Ap).tolist()


def test_degree_kernel_misses_counterexample():
    assert np.array_equal(
        degree_histogram(make_cycle(6)),
        degree_histogram(make_two_triangles())
    )
    assert baby_kernel(make_cycle(6), make_two_triangles()) == 36
    assert enhanced_kernel(make_cycle(6), make_two_triangles()) != 36


def test_gram_matrix_psd():
    path = np.array([[0,1,0],[1,0,1],[0,1,0]])
    triangle = np.ones((3,3), dtype=int) - np.eye(3, dtype=int)
    K = kernel_matrix([path, triangle, path])
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() >= -1e-8


def test_reject_asymmetric_adjacency():
    A = np.array([[0,1],[0,0]])
    with pytest.raises(ValueError):
        degree_vector(A)
