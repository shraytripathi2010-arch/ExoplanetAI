"""
train_cnn.py -- AstroNet-style local+global-view CNN, trained and evaluated
SEPARATELY from the existing classical (HistGradientBoosting) model -- not a
replacement. Same stratified split methodology (random_seed=42) as the
classical model's own training for direct comparability.

Reports REAL-only and REAL+SYNTHETIC results separately throughout, per the
explicit requirement that synthetic-trained results must never be silently
blended with real-data results.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_dataset.npz")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_results.json")
RANDOM_SEED = 42
# MPS backend doesn't support AdaptiveAvgPool1d with non-divisible input/
# output sizes (confirmed live -- RuntimeError on this exact architecture).
# The model is tiny (a few hundred floats per example, ~9k examples), so
# CPU is plenty fast here; not worth reshaping the architecture just to
# force GPU use on a dataset this small.
DEVICE = torch.device("cpu")


class ViewDataset(Dataset):
    def __init__(self, global_view, local_view, labels):
        self.g = torch.tensor(global_view, dtype=torch.float32).unsqueeze(1)
        self.l = torch.tensor(local_view, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.g[idx], self.l[idx], self.y[idx]


class LocalGlobalCNN(nn.Module):
    """Deliberately smaller than AstroNet's original (fewer conv layers,
    fewer channels) -- this project's real training set is ~1/3 AstroNet's
    size, so the model is scaled down to reduce overfitting risk rather
    than copying AstroNet's full-size architecture uncritically."""
    def __init__(self):
        super().__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 8 + 32 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, g, l):
        g_feat = self.global_conv(g).flatten(1)
        l_feat = self.local_conv(l).flatten(1)
        combined = torch.cat([g_feat, l_feat], dim=1)
        return self.head(combined).squeeze(-1)


def train_one_model(g_train, l_train, y_train, g_val, l_val, y_val, epochs=40, track_curve=False):
    torch.manual_seed(RANDOM_SEED)
    model = LocalGlobalCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = ViewDataset(g_train, l_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    g_val_t = torch.tensor(g_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    l_val_t = torch.tensor(l_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    curve = []
    best_val_auc, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        for gb, lb, yb in train_loader:
            gb, lb, yb = gb.to(DEVICE), lb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(gb, lb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_pred = torch.sigmoid(model(
                torch.tensor(g_train, dtype=torch.float32).unsqueeze(1).to(DEVICE),
                torch.tensor(l_train, dtype=torch.float32).unsqueeze(1).to(DEVICE))).cpu().numpy()
            val_pred = torch.sigmoid(model(g_val_t, l_val_t)).cpu().numpy()
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        curve.append({"epoch": epoch, "train_auc": train_auc, "val_auc": val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, curve, best_val_auc


def main():
    data = np.load(DATASET_PATH, allow_pickle=True)
    g, l, y, host, is_synth = data["global_view"], data["local_view"], data["label"], data["host"], data["is_synthetic"]

    real_mask = ~is_synth
    g_real, l_real, y_real, host_real = g[real_mask], l[real_mask], y[real_mask], host[real_mask]
    print(f"Real examples: {len(y_real)} (pos={y_real.sum()}, neg={len(y_real)-y_real.sum()})")

    # Same split methodology as the classical model: stratified, seed=42,
    # 80/20 -- for direct comparability of the resulting ROC-AUC numbers.
    idx = np.arange(len(y_real))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y_real)

    results = {}

    # ---- Experiment 1: REAL DATA ONLY ----
    idx_tr, idx_val = train_test_split(idx_train, test_size=0.15, random_state=RANDOM_SEED,
                                        stratify=y_real[idx_train])
    model_real, curve_real, best_val_auc = train_one_model(
        g_real[idx_tr], l_real[idx_tr], y_real[idx_tr],
        g_real[idx_val], l_real[idx_val], y_real[idx_val])
    model_real.eval()
    with torch.no_grad():
        test_pred = torch.sigmoid(model_real(
            torch.tensor(g_real[idx_test], dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(l_real[idx_test], dtype=torch.float32).unsqueeze(1).to(DEVICE))).cpu().numpy()
    test_auc_real = roc_auc_score(y_real[idx_test], test_pred)
    print(f"\n=== REAL DATA ONLY: test ROC-AUC = {test_auc_real:.4f} (val best = {best_val_auc:.4f}) ===")
    results["real_only"] = {"test_roc_auc": float(test_auc_real), "learning_curve": curve_real,
                             "n_train": len(idx_tr), "n_val": len(idx_val), "n_test": len(idx_test)}

    # ---- Experiment 2: REAL + SYNTHETIC (synthetic added to TRAIN only,
    # never to val/test -- test set must stay 100% real so the reported
    # number means what it says) ----
    synth_mask = is_synth
    g_synth, l_synth, y_synth = g[synth_mask], l[synth_mask], y[synth_mask]
    g_tr2 = np.concatenate([g_real[idx_tr], g_synth])
    l_tr2 = np.concatenate([l_real[idx_tr], l_synth])
    y_tr2 = np.concatenate([y_real[idx_tr], y_synth])
    print(f"\nReal+synthetic training set: {len(y_tr2)} ({len(idx_tr)} real + {len(y_synth)} synthetic)")

    model_synth, curve_synth, best_val_auc2 = train_one_model(
        g_tr2, l_tr2, y_tr2, g_real[idx_val], l_real[idx_val], y_real[idx_val])
    model_synth.eval()
    with torch.no_grad():
        test_pred2 = torch.sigmoid(model_synth(
            torch.tensor(g_real[idx_test], dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(l_real[idx_test], dtype=torch.float32).unsqueeze(1).to(DEVICE))).cpu().numpy()
    test_auc_synth = roc_auc_score(y_real[idx_test], test_pred2)
    print(f"=== REAL + SYNTHETIC (test set still 100% real): test ROC-AUC = {test_auc_synth:.4f} ===")
    results["real_plus_synthetic"] = {"test_roc_auc": float(test_auc_synth), "learning_curve": curve_synth,
                                       "n_train": len(y_tr2), "n_val": len(idx_val), "n_test": len(idx_test)}

    results["classical_model_baseline_test_roc_auc"] = 0.9031559838011451

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")
    print(f"\nFor comparison, classical model test ROC-AUC: 0.9032")


if __name__ == "__main__":
    main()
