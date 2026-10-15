"""Train/evaluate a two-layer Vanilla GCN on Zachary's Karate Club graph."""
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)
from gcn_model import VanillaGCN, normalize_adjacency


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    G = nx.karate_club_graph()
    A_np = nx.to_numpy_array(G, dtype=np.float32)
    labels_np = np.array([
        0 if G.nodes[i]["club"] == "Mr. Hi" else 1
        for i in range(G.number_of_nodes())
    ], dtype=np.int64)

    X = torch.eye(G.number_of_nodes(), dtype=torch.float32)
    A = torch.tensor(A_np, dtype=torch.float32)
    A_hat = normalize_adjacency(A)
    y = torch.tensor(labels_np, dtype=torch.long)

    # Reproducible stratified split: about 60% from each class for training.
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

    model = VanillaGCN(X.shape[1], 16, 2, dropout=0.3)
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
            train_acc = accuracy_score(y[train_idx].numpy(), pred[train_idx].numpy())
            test_acc = accuracy_score(y[test_idx].numpy(), pred[test_idx].numpy())
            print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | "
                  f"train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")

    model.eval()
    with torch.no_grad():
        pred = model(X, A_hat).argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()
    y_test = y_np[test_idx.numpy()]
    pred_test = pred[test_idx.numpy()]

    accuracy = accuracy_score(y_test, pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])

    print("\\nDataset: Zachary Karate Club")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    print(f"Train nodes: {len(train_idx)} | Test nodes: {len(test_idx)}")
    print(f"Final test accuracy: {accuracy:.3f}")
    print(f"Macro precision: {precision:.3f}")
    print(f"Macro recall:    {recall:.3f}")
    print(f"Macro F1-score:  {f1:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; labels [0, 1]):")
    print(cm)
    print("\\nPer-class report:")
    print(classification_report(
        y_test, pred_test, labels=[0, 1],
        target_names=["Mr. Hi", "Officer"], zero_division=0
    ))
    print("Note: small educational transductive demonstration; results may vary.")

if __name__ == "__main__":
    main()
