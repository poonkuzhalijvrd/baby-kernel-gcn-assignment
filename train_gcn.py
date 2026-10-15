"""Train/evaluate Vanilla GCN on Zachary's Karate Club network (NetworkX dataset)."""
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from gcn_model import VanillaGCN, normalize_adjacency


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # A small real social network included with NetworkX.
    G = nx.karate_club_graph()
    A_np = nx.to_numpy_array(G, dtype=np.float32)
    # Club membership is the binary target label.
    labels_np = np.array([
        0 if G.nodes[i]["club"] == "Mr. Hi" else 1
        for i in range(G.number_of_nodes())
    ], dtype=np.int64)

    # Identity features: each node starts with a unique one-hot feature.
    X = torch.eye(G.number_of_nodes(), dtype=torch.float32)
    A = torch.tensor(A_np, dtype=torch.float32)
    A_hat = normalize_adjacency(A)
    y = torch.tensor(labels_np, dtype=torch.long)

    # Reproducible stratified split: 60% training per class, remaining test.
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels_np):
        ids = np.flatnonzero(labels_np == cls)
        rng.shuffle(ids)
        n_train = max(1, int(round(0.6 * len(ids))))
        train_idx.extend(ids[:n_train].tolist())
        test_idx.extend(ids[n_train:].tolist())
    train_idx = torch.tensor(sorted(train_idx), dtype=torch.long)
    test_idx = torch.tensor(sorted(test_idx), dtype=torch.long)

    model = VanillaGCN(input_dim=X.shape[1], hidden_dim=16, num_classes=2, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits = model(X, A_hat)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if epoch in (1, 20, 50, 100, 150, 200):
            model.eval()
            with torch.no_grad():
                pred = model(X, A_hat).argmax(dim=1)
                train_acc = (pred[train_idx] == y[train_idx]).float().mean().item()
                test_acc = (pred[test_idx] == y[test_idx]).float().mean().item()
            print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | "
                  f"train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")

    model.eval()
    with torch.no_grad():
        pred = model(X, A_hat).argmax(dim=1)
        test_acc = (pred[test_idx] == y[test_idx]).float().mean().item()
    print("\nDataset: Zachary Karate Club")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    print(f"Train nodes: {len(train_idx)} | Test nodes: {len(test_idx)}")
    print(f"Final test accuracy: {test_acc:.3f}")
    print("Note: this is a small educational demonstration; results can vary with split/seed.")


if __name__ == "__main__":
    main()
