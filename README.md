# Baby Graph Kernel + Vanilla GCN

This educational repository contains two independent graph-learning implementations:

1. A degree-histogram graph kernel implemented from scratch with NumPy.
2. A two-layer Vanilla Graph Convolutional Network implemented directly in PyTorch and trained on Zachary's Karate Club graph.

## Repository structure

```text
baby-kernel-gcn-assignment/
├── baby_kernel.py
├── experiments.py
├── visualize.py
├── gcn_model.py
├── train_gcn.py
├── tests/test_baby_kernel.py
├── results/
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Part 1 — Baby graph kernel

For graph \(G\), \(h_G(d)\) is the number of nodes of degree \(d\). The baseline kernel is:

\[
K(G_1,G_2)=h_{G_1}^{T}h_{G_2}
\]

Since this is a dot product in an explicit feature space, for real coefficients \(c_i\):

\[
\sum_i\sum_j c_i c_j K(G_i,G_j)
=\left\|\sum_i c_i h_{G_i}\right\|^2\geq 0
\]

This establishes positive semidefiniteness (PSD).

### Recorded experiments

| Experiment | Observed result | Interpretation |
|---|---:|---|
| Path-3 degree vector | `[1, 2, 1]` | Two endpoints of degree 1 and one centre of degree 2. |
| Path-3 histogram | `[0, 2, 1]` | Two nodes have degree 1; one has degree 2. |
| Triangle histogram | `[0, 0, 3]` | All three nodes have degree 2. |
| \(K(\mathrm{path},\mathrm{triangle})\) | 3 | Dot product of the histograms. |
| \(K(\mathrm{path},\mathrm{path})\) | 5 | Squared norm of the path histogram. |
| Baseline \(K(C_6,\mathrm{two\ triangles})\) | 36 | The baseline cannot distinguish this non-isomorphic pair. |
| Enhanced score for that pair | 38 | Component and triangle features distinguish this example. |

A cycle \(C_6\) and two disconnected triangles each have six nodes, six edges, and histogram `[0, 0, 6]`. The baseline score is \(6\cdot6=36\), even though the graphs are not isomorphic: the cycle has one connected component and the other graph has two. The enhanced kernel adds connected-component count and triangle count. It distinguishes this pair but is not a complete graph-isomorphism test.

### Mathematical properties

- **Permutation invariance:** relabelling nodes changes adjacency row/column order but not the degree histogram.
- **Symmetry:** \(K(G_1,G_2)=K(G_2,G_1)\), because dot products are symmetric.
- **PSD:** the recorded kernel matrix eigenvalues, rounded to eight decimals, were `[-0.00000000, -0.00000000, -0.00000000, 7.78727005, 83.21272995]`. The tiny negative zeros are floating-point rounding; the PSD check passed with tolerance \(10^{-8}\).

Run the experiments and tests:

```powershell
python baby_kernel.py
python experiments.py
python -m pytest -q
```

### Visualisations

```powershell
python visualize.py
```

The script saves `results/isomorphism_counterexample.png` and `results/kernel_similarity_heatmap.png`.

## Part 2 — Vanilla GCN

### Dataset

Zachary's Karate Club network from NetworkX has 34 nodes and 78 edges. Each node is labelled according to its recorded club affiliation. The initial node-feature matrix is the identity matrix, giving each node a unique one-hot feature.

### Model mathematics

Add self-loops and normalize the adjacency matrix:

\[
\tilde A=A+I,\qquad
\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}
\]

The two layers are:

\[
H^{(1)}=\operatorname{ReLU}(\hat A X W^{(0)}),\qquad
Z=\hat A H^{(1)}W^{(1)}
\]

The model is trained using cross-entropy loss on the training nodes, with Adam updating the weights.

### Recorded baseline result

For the fixed seed-42 run, there were 20 training nodes and 14 held-out test nodes:

| Metric | Observed result |
|---|---:|
| Training loss at epoch 200 | 0.0233 |
| Training accuracy | 100.0% |
| Test accuracy | 92.9% (13/14 nodes) |

Test accuracy was 100% at epoch 50 but 92.9% at epoch 200. With only 14 test nodes, one wrong prediction changes accuracy by about 7.14 percentage points. Treat this as an educational result, not a reliable estimate for larger graphs.

### Additional evaluation metrics

```powershell
python train_gcn.py
```

The updated script reports macro precision, macro recall, macro F1-score, a confusion matrix and a per-class report. If your output differs due to environment or version differences, record the values from your own run.

## Limitations

- Degree histograms do not capture all graph structure; non-isomorphic graphs may share them.
- The enhanced kernel is based on hand-designed features and is not a complete isomorphism algorithm.
- The GCN uses identity features and a small graph, so it is an educational demonstration rather than a production benchmark.
- Future work: compare to a majority-class baseline, test multiple stratified splits and investigate richer node features.

## Commit the improvements to GitHub

```powershell
python visualize.py
python -m pytest -q
python train_gcn.py
git add README.md train_gcn.py requirements.txt visualize.py .gitignore results/
git commit -m "Add kernel visualizations and GCN evaluation metrics"
git push
```
