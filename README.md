# Baby Graph Kernel + Vanilla GCN

Two independent graph-learning implementations in one educational repository:
1. A degree-histogram graph kernel implemented from scratch using NumPy.
2. A two-layer Vanilla Graph Convolutional Network (GCN) implemented directly in PyTorch and trained on Zachary's Karate Club network.

## Project structure

```text
baby-kernel-gcn-assignment/
├── baby_kernel.py
├── experiments.py
├── graph_features.py (features are currently in baby_kernel.py)
├── gcn_model.py
├── train_gcn.py
├── tests/
│   └── test_baby_kernel.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Part 1 — Baby graph kernel

### Definition
For graph `G`, define the degree histogram `h_G(d)` as the number of nodes of degree `d`.

The baseline kernel is the dot product:

`K(G1, G2) = h_G1^T h_G2`

This is a kernel because it is an inner product in an explicit feature space. For any graphs `G_i` and real coefficients `c_i`:

`sum_i sum_j c_i c_j K(G_i,G_j) = ||sum_i c_i h_Gi||^2 >= 0`

So the Gram matrix is positive semidefinite (PSD), up to floating-point tolerance.

### Mathematical properties tested
- **Permutation invariance:** node relabelling changes adjacency-matrix row/column order but not the degree histogram.
- **Symmetry:** `K(G1,G2) = K(G2,G1)` because the dot product is symmetric.
- **PSD:** the kernel Gram matrix has nonnegative eigenvalues, within numerical tolerance.
- **Limitation:** same degree histograms do not imply graph isomorphism.

### Counterexample
A 6-cycle (`C6`) and two disjoint triangles both have six nodes, six edges, and degree histogram `[0, 0, 6, 0, ...]`. They are not isomorphic: `C6` is connected; two triangles have two connected components. The baseline degree kernel gives them the same score.

An optional enhanced feature vector adds connected-component count and triangle count. This demonstrates how hand-designed features can distinguish this specific counterexample, but it still is not a complete graph-isomorphism test.

### Run
```bash
python baby_kernel.py
python experiments.py
python -m pytest -q
```

## Part 2 — Vanilla GCN

Dataset: Zachary's Karate Club network supplied by NetworkX (34 nodes, 78 edges). Each node's club membership is a binary label. Identity features are used as a simple, transparent feature baseline.

The normalized adjacency is:

`A_hat = D_tilde^(-1/2) (A + I) D_tilde^(-1/2)`

Two-layer GCN:

`H1 = ReLU(A_hat X W0)`

`Z = A_hat H1 W1`

The model is trained with cross-entropy loss on training nodes and evaluated on held-out nodes. The split and random seed are fixed for reproducibility, but this tiny classic graph is an educational demonstration, not a statistically reliable benchmark. Test accuracy may vary with software versions and split choices.

Run:
```bash
python train_gcn.py
```


.
