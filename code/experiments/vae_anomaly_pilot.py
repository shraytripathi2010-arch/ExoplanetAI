"""
vae_anomaly_pilot.py -- BOUNDED pilot for the VAE anomaly-detection proposal.

Runs against the criteria PRE-REGISTERED in RESULTS_SUMMARY.md before any code
was written, so the verdict cannot be moved after seeing the numbers:

  SC1 (does it see transits at all): reconstruction error must separate
       transit-bearing from no-transit light curves at AUC > 0.75. Below that
       it is not detecting transits and nothing else matters.
  SC2 (is it additive): its flags must disagree with the DEPLOYED
       multivariate_ood_flag on a material fraction of the candidate pool.
       Full agreement means it is a slower reimplementation of a shipped model.
  SC3 (is it meaningful): VAE-flagged anomalies must be enriched in
       independently-vetted-bad candidates relative to the pool baseline.
  KILL: if reconstruction error correlates with var_oot_rms at |r| > 0.6, it is
       a stellar-VARIABILITY detector, not an anomaly detector. Stop and report;
       do not tune around it.

PHASE 1 (this file) = SC1 + KILL. SC2/SC3 need phase-folded views for the
unknown candidate pool, which do not exist yet and cost real work to build --
so they are deliberately NOT built until Phase 1 passes. That is what a kill
criterion is for.

WHAT "NORMAL" MEANS HERE
------------------------
The VAE trains unsupervised on real, TRAIN-SPLIT, label-0 curves (n=924) --
TOI false positives, i.e. things that looked transit-like and were not. It
never sees a label-1 curve or a test-split host. SC1 then asks whether real
transits (test split) reconstruct worse than real non-transits (test split).

A LIMITATION OF THIS PILOT, STATED UP FRONT
--------------------------------------------
The views are phase-folded AT THE KNOWN PERIOD, produced by
`phase_fold_views.py` for the earlier CNN work. That presupposes TLS already
found the period -- so this VAE sits DOWNSTREAM of detection and is not the
independent "flag weird light curves" detector the original proposal imagined.
It is the cheap version the pre-registered scope called for (data prep already
existed). If SC1 passes, a real version would have to run on unfolded data,
which is a much larger build. Recorded so the result is not over-read.

Touches nothing in production: reads cnn_dataset.npz and training.csv, writes
only its own outputs.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
NPZ = os.path.join(HERE, "cnn_dataset.npz")
SPLIT = os.path.join(ROOT, "data", "training_dataset", "split_manifest.json")
TRAINING = os.path.join(ROOT, "data", "training_dataset", "training.csv")
OUT_JSON = os.path.join(HERE, "vae_anomaly_pilot_results.json")
OUT_CSV = os.path.join(HERE, "vae_anomaly_pilot_scores.csv")

SEED = 20260811
LATENT = 8
EPOCHS = 300
BATCH = 64
BETA = 1.0            # KL weight
LR = 1e-3
AUC_FLOOR = 0.75      # SC1
KILL_CORR = 0.60      # KILL


def load():
    d = np.load(NPZ, allow_pickle=True)
    gv, lv = d["global_view"].astype("float32"), d["local_view"].astype("float32")
    lab, syn = d["label"], d["is_synthetic"]
    host = np.array([str(x) for x in d["host"]])
    m = json.load(open(SPLIT))
    tr = np.isin(host, list(map(str, m["train_hosts"])))
    te = np.isin(host, list(map(str, m["test_hosts"])))
    return gv, lv, lab, syn, host, tr, te


class VAE:
    """Small 1D-conv VAE, two branches (global 201 / local 61). Deliberately
    tiny: 924 training curves cannot support anything larger."""

    def __init__(self, seed=SEED):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.torch, self.nn = torch, nn

        class Net(nn.Module):
            def __init__(self, latent):
                super().__init__()
                self.eg = nn.Sequential(
                    nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
                    nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(), nn.Flatten())
                self.el = nn.Sequential(
                    nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
                    nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(), nn.Flatten())
                ng, nl = 32 * 51, 32 * 16
                self.mu = nn.Linear(ng + nl, latent)
                self.lv = nn.Linear(ng + nl, latent)
                self.dg = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 201))
                self.dl = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 61))

            def encode(self, g, l):
                h = self.torch_cat(self.eg(g), self.el(l))
                return self.mu(h), self.lv(h)

            @staticmethod
            def torch_cat(a, b):
                import torch
                return torch.cat([a, b], dim=1)

            def forward(self, g, l):
                import torch
                mu, lv = self.encode(g, l)
                std = torch.exp(0.5 * lv)
                z = mu + std * torch.randn_like(std)
                return self.dg(z), self.dl(z), mu, lv

        self.net = Net(LATENT)

    def fit(self, g, l, gval, lval):
        torch, nn = self.torch, self.nn
        opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        G = torch.tensor(g).unsqueeze(1); L = torch.tensor(l).unsqueeze(1)
        GV = torch.tensor(gval).unsqueeze(1); LV = torch.tensor(lval).unsqueeze(1)
        n = len(G)
        best, bad, best_state = np.inf, 0, None
        for ep in range(EPOCHS):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                gb, lb = G[idx], L[idx]
                rg, rl, mu, lvar = self.net(gb, lb)
                rec = ((rg - gb.squeeze(1)) ** 2).sum(1).mean() + \
                      ((rl - lb.squeeze(1)) ** 2).sum(1).mean()
                kl = (-0.5 * (1 + lvar - mu ** 2 - lvar.exp()).sum(1)).mean()
                loss = rec + BETA * kl
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                rg, rl, mu, lvar = self.net(GV, LV)
                v = (((rg - GV.squeeze(1)) ** 2).sum(1) +
                     ((rl - LV.squeeze(1)) ** 2).sum(1)).mean().item()
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 30:
                    print(f"  early stop at epoch {ep} (val recon {best:.4f})", flush=True)
                    break
            if ep % 25 == 0:
                print(f"  epoch {ep:3d}  val recon {v:.4f}", flush=True)
        if best_state:
            self.net.load_state_dict(best_state)
        return best

    def recon_error(self, g, l, n_samples=8):
        """Mean per-example reconstruction MSE, averaged over latent draws."""
        torch = self.torch
        self.net.eval()
        G = torch.tensor(g).unsqueeze(1); L = torch.tensor(l).unsqueeze(1)
        acc = np.zeros(len(g), dtype="float64")
        with torch.no_grad():
            for _ in range(n_samples):
                rg, rl, _, _ = self.net(G, L)
                e = (((rg - G.squeeze(1)) ** 2).mean(1) +
                     ((rl - L.squeeze(1)) ** 2).mean(1))
                acc += e.numpy()
        return acc / n_samples


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main():
    from scipy.stats import spearmanr, pearsonr
    gv, lv, lab, syn, host, tr, te = load()
    real = ~syn

    trn = real & tr & (lab == 0)                 # the VAE's "normal" set
    print(f"training on real TRAIN-split label-0: n={trn.sum()}")
    idx = np.where(trn)[0]
    rng = np.random.default_rng(SEED); rng.shuffle(idx)
    nval = max(60, int(0.15 * len(idx)))
    vi, ti = idx[:nval], idx[nval:]
    print(f"  fit {len(ti)}  val {len(vi)}")

    m = VAE()
    m.fit(gv[ti], lv[ti], gv[vi], lv[vi])

    # ---- SC1, primary: real transits vs real non-transits, TEST split only
    t0 = real & te & (lab == 0)
    t1 = real & te & (lab == 1)
    sel = np.where(t0 | t1)[0]
    err = m.recon_error(gv[sel], lv[sel])
    y = (lab[sel] == 1).astype(int)
    a_real = auc(y, err)
    print(f"\nSC1 (real transit vs no-transit, TEST split): "
          f"n1={int(y.sum())} n0={int((1-y).sum())}  AUC = {a_real:.4f}")

    # ---- SC1, literal pre-registered form: INJECTED transits vs no-transit
    inj = np.where(syn)[0]
    err_inj = m.recon_error(gv[inj], lv[inj])
    err_t0 = m.recon_error(gv[np.where(t0)[0]], lv[np.where(t0)[0]])
    y2 = np.r_[np.ones(len(err_inj)), np.zeros(len(err_t0))]
    s2 = np.r_[err_inj, err_t0]
    a_inj = auc(y2, s2)
    print(f"SC1 (INJECTED transit vs no-transit)        : "
          f"n1={len(err_inj)} n0={len(err_t0)}  AUC = {a_inj:.4f}")

    # ---- KILL: is reconstruction error just a variability meter?
    tdf = pd.read_csv(TRAINING)[["host", "var_oot_rms"]]
    tdf["host"] = tdf.host.astype(str)
    allreal = np.where(real)[0]
    err_all = m.recon_error(gv[allreal], lv[allreal])
    j = pd.DataFrame({"host": host[allreal], "err": err_all}).merge(tdf, on="host", how="left")
    j = j.dropna(subset=["var_oot_rms"])
    pr = pearsonr(j.err, j.var_oot_rms); sp = spearmanr(j.err, j.var_oot_rms)
    print(f"\nKILL check (n={len(j)}): Pearson r = {pr[0]:+.4f} (p={pr[1]:.2e})   "
          f"Spearman rho = {sp[0]:+.4f} (p={sp[1]:.2e})")

    killed = max(abs(pr[0]), abs(sp[0])) > KILL_CORR
    sc1 = a_real > AUC_FLOOR

    print("\n" + "=" * 62)
    print(f"SC1  transit separation AUC > {AUC_FLOOR}      : "
          f"{'PASS' if sc1 else 'FAIL'}  ({a_real:.4f})")
    print(f"KILL |corr(recon, var_oot_rms)| > {KILL_CORR}   : "
          f"{'FIRED' if killed else 'not fired'} "
          f"({max(abs(pr[0]), abs(sp[0])):.4f})")
    print("=" * 62)
    if killed:
        print("VERDICT: KILL CRITERION FIRED -- reconstruction error is a "
              "stellar-variability meter.\n         Stop. Do not tune around it. "
              "SC2/SC3 not run.")
    elif not sc1:
        print("VERDICT: SC1 FAILED -- the VAE does not see transits. "
              "SC2/SC3 not run,\n         since an anomaly detector blind to the "
              "signal of interest cannot be additive.")
    else:
        print("VERDICT: Phase 1 PASSED -- proceed to build candidate-pool views "
              "for SC2/SC3.")

    pd.DataFrame({"host": j.host, "recon_err": j.err,
                  "var_oot_rms": j.var_oot_rms}).to_csv(OUT_CSV, index=False)
    json.dump({
        "n_train": int(trn.sum()), "latent": LATENT,
        "sc1_auc_real": a_real, "sc1_auc_injected": a_inj, "auc_floor": AUC_FLOOR,
        "kill_pearson": float(pr[0]), "kill_spearman": float(sp[0]),
        "kill_threshold": KILL_CORR, "kill_fired": bool(killed),
        "sc1_pass": bool(sc1),
        "verdict": ("KILL" if killed else ("SC1_FAIL" if not sc1 else "PHASE1_PASS")),
    }, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
