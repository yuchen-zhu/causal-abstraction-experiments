"""
Experiment 3: Neural Network Representation Surgery
====================================================

Tests whether the complementary-variable framework enables *surgical*
interventions on neural network activations: changing one chosen concept
direction without perturbing others or producing off-target output changes.

Data-generating process
-----------------------
  K̄  independent latent concepts  Z_k ~ N(0,1),  k = 1,...,K̄
  X   = W_mix @ Z + noise          (D-dim mixed features fed to the MLP)
  Y   = a^T Z + ε                  (scalar label depending on all concepts)

The MLP  X → h → Ŷ  is trained on (X, Y).  We pick a hidden layer of
width K and call its activations h ∈ R^K the "micro state".

Linear probe
------------
  A linear probe  p ∈ R^{K̄xK}  is learned by OLS: regress Z on h
  across a held-out set.  This gives the aggregation map p: h → Z̄ = ph.

Macro causal story (choice of Ā)
---------------------------------
  We commit to  Ȳ = Ā Z̄ + N̄  as the macro model.  Ā ∈ R^{1xK̄}  is
  estimated by OLS regression of Y on Z̄ = ph across the training set.
  This determines the complementary variable:
    q(h) = (J_head - Ā p) h        (locally linear; J_head = ∂Ŷ/∂h)
  and the q-constraint on a shift δ:
    (J_head - Ā p) δ = 0           (preserve unexplained output)

Three intervention strategies
------------------------------
  1. Min-norm steering: min-norm δ with  p δ = Δ·e₁  only.
     The standard "concept steering vector" — no output constraint.
  2. q-constrained:    δ satisfying  p δ = Δ·e₁  AND  (J_head-Āp)δ = 0.
     Our framework.  Should give ΔŶ = Ā[0]·Δ  and zero off-target concept change.
  3. Random unconstrained: many random δ with  p δ = Δ·e₁.
     Shows the variance of naive approaches.

Metrics (per test sample)
--------------------------
  On-target:  Δz̄₁  =  p[0] δ   (should equal Δ for all methods)
  Off-target: Δz̄_k = p[k] δ   (should equal 0 for q-constrained)
  ΔŶ actual:  head(h+δ) - head(h)
  Macro pred: Ā[0] · Δ
  ΔŶ error:   |ΔŶ_actual - Ā[0]·Δ|
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found — using numpy-only fallback MLP.")

rng = np.random.default_rng(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_dataset(N=8000, K_bar=4, D=40, sigma_mix=0.2):
    """
    Returns X (NxD), Y (N,), Z (NxK̄), W_mix (DxK̄), alpha (K̄,).
    The causal structure is  Z → X (mixing)  and  Z → Y (label).
    """
    alpha  = rng.standard_normal(K_bar)               # true concept→label weights
    W_mix  = rng.standard_normal((D, K_bar)) / np.sqrt(K_bar)
    Z      = rng.standard_normal((N, K_bar))
    X      = Z @ W_mix.T + sigma_mix * rng.standard_normal((N, D))
    Y      = Z @ alpha + 0.2 * rng.standard_normal(N)
    return X, Y, Z, W_mix, alpha


# ──────────────────────────────────────────────────────────────────────────────
# MLP (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class MLP(nn.Module):
        """
        Architecture:  D  →[ReLU]→  hidden  →[ReLU]→  K  →[ReLU]→  hidden//2  →  1
        We expose the K-dimensional 'probe layer' activations h.
        """
        def __init__(self, D, K=64, hidden=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(D, hidden), nn.ReLU(),
                nn.Linear(hidden, K), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(K, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            h = self.encoder(x)
            y = self.head(h).squeeze(-1)
            return y, h

        def get_h(self, x_np):
            x_t = torch.tensor(x_np, dtype=torch.float32)
            with torch.no_grad():
                _, h = self.forward(x_t)
            return h.numpy()

        def predict_from_h(self, h_t):
            """Forward from probe layer; h_t is a torch tensor (requires_grad)."""
            return self.head(h_t).squeeze(-1)

        def jacobian_head(self, h_np):
            """
            Returns J_head = ∂Ŷ/∂h  ∈ R^{1xK}  evaluated at h_np (numpy array).
            """
            h_t = torch.tensor(h_np, dtype=torch.float32, requires_grad=True)
            y   = self.predict_from_h(h_t)
            y.backward()
            return h_t.grad.detach().numpy()   # shape (K,)


    def train_mlp(X_tr, Y_tr, D, K=64, hidden=128, epochs=60, lr=1e-3, batch=256):
        model = MLP(D, K, hidden)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X_tr, dtype=torch.float32)
        Y_t = torch.tensor(Y_tr, dtype=torch.float32)
        N   = len(X_tr)

        for epoch in range(epochs):
            idx  = torch.randperm(N)
            X_t, Y_t = X_t[idx], Y_t[idx]
            for start in range(0, N, batch):
                xb = X_t[start:start+batch]
                yb = Y_t[start:start+batch]
                y_hat, _ = model(xb)
                loss = loss_fn(y_hat, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    y_all, _ = model(X_t)
                    mse = loss_fn(y_all, Y_t).item()
                print(f"    epoch {epoch+1:>3}/{epochs}  train MSE = {mse:.4f}")

        return model


# ──────────────────────────────────────────────────────────────────────────────
# Linear probe and macro story
# ──────────────────────────────────────────────────────────────────────────────

def learn_probe(h, Z):
    """
    OLS: Z = h @ w + b.  Returns p = w^T  ∈ R^{K̄xK}.
    The rows of p are the concept directions in activation space.
    """
    h_aug  = np.hstack([h, np.ones((len(h), 1))])       # (N x K+1)
    W_aug  = pinv(h_aug) @ Z                             # (K+1 x K̄)
    p      = W_aug[:-1].T                                # (K̄ x K)
    return p


def estimate_A_bar(h, Y, p):
    """
    Estimate macro coefficient Ā ∈ R^{1xK̄} by OLS:  Y = Ā @ (p @ h) + noise.
    This is the 'macro causal story': how much each extracted concept drives Y.
    """
    Z_bar = (p @ h.T).T                                  # (N x K̄)
    Z_aug = np.hstack([Z_bar, np.ones((len(Y), 1))])
    coef  = pinv(Z_aug) @ Y                              # (K̄+1,)
    A_bar = coef[:-1]                                    # (K̄,)
    return A_bar


# ──────────────────────────────────────────────────────────────────────────────
# Intervention strategies
# ──────────────────────────────────────────────────────────────────────────────

def delta_min_norm(p, delta_z_bar):
    """
    Strategy 1 — minimum-norm steering.
    Solve: min ||δ||  s.t.  p δ = Δz̄.
    This is the standard 'concept vector' intervention.
    Result is pinv(p) @ Δz̄.
    """
    return pinv(p) @ delta_z_bar


def delta_q_constrained(p, A_bar, J_head, delta_z_bar):
    """
    Strategy 2 — q-constrained intervention.
    Solve: min ||δ||  s.t.
      (1)  p δ = Δz̄                 [hit macro target]
      (2)  (J_head - A_bar @ p) δ = 0  [preserve complementary variable q]

    J_head: (K,) gradient of Ŷ w.r.t. h  [the row vector of the Jacobian]
    A_bar:  (K̄,) macro coefficients
    """
    q_row = (J_head - A_bar @ p).reshape(1, -1)   # (1 x K)
    C     = np.vstack([p, q_row])                  # (K̄+1 x K)
    b     = np.concatenate([delta_z_bar, [0.0]])   # (K̄+1,)
    return pinv(C) @ b


def delta_random_unconstrained(p, delta_z_bar, n=50):
    """
    Strategy 3 — many random δ satisfying only  p δ = Δz̄.
    Returns list of n vectors.
    """
    delta_0 = pinv(p) @ delta_z_bar
    NS      = _null_space(p)
    deltas  = [delta_0 + NS @ (rng.standard_normal(NS.shape[1]) * rng.uniform(0.5, 2.0))
               for _ in range(n)]
    return deltas


def _null_space(M, rcond=1e-10):
    """Null space of M via SVD."""
    _, s, Vt = np.linalg.svd(M, full_matrices=True)
    rank = (s > rcond * s[0]).sum()
    return Vt[rank:].T


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_intervention(delta, h, p, A_bar, model, delta_target):
    """
    Given a shift δ applied to a single activation vector h, compute:
      on_target:   p[0] @ δ  (should = Δ)
      off_targets: p[1:] @ δ (should = 0 for q-constrained)
      dY_actual:   head(h+δ) - head(h)  [via forward pass]
      dY_macro:    A_bar[0] * Δ         [macro model prediction]
    """
    on_target  = float(p[0] @ delta)
    off_target = p[1:] @ delta                            # (K̄-1,)

    h_new = h + delta
    if TORCH_AVAILABLE:
        with torch.no_grad():
            y_old = model.predict_from_h(
                torch.tensor(h,     dtype=torch.float32)).item()
            y_new = model.predict_from_h(
                torch.tensor(h_new, dtype=torch.float32)).item()
        dY_actual = y_new - y_old
    else:
        dY_actual = float("nan")

    dY_macro = float(A_bar[0] * delta_target)

    return {
        "on_target":  on_target,
        "off_target": off_target,
        "dY_actual":  dY_actual,
        "dY_macro":   dY_macro,
        "dY_error":   abs(dY_actual - dY_macro),
    }


def run_experiment(X, Y, Z, K=64, delta_target=1.0, n_test=200, n_random=50,
                   verbose=True):
    """
    Full experiment pipeline:
      1. Train MLP
      2. Learn probe p and macro story A_bar
      3. For each test sample: compute J_head, build all three δ vectors
      4. Evaluate metrics
    """
    N     = len(X)
    D     = X.shape[1]
    K_bar = Z.shape[1]
    split = int(0.8 * N)

    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]
    Z_tr, Z_te = Z[:split], Z[split:]

    # ── 1. Train MLP ──────────────────────────────────────────────────────────
    if verbose:
        print("  Training MLP...")
    model = train_mlp(X_tr, Y_tr, D=D, K=K, epochs=80, lr=3e-3)

    # ── 2. Learn probe and macro story ────────────────────────────────────────
    h_tr = model.get_h(X_tr)
    h_te = model.get_h(X_te)

    p     = learn_probe(h_tr, Z_tr)
    A_bar = estimate_A_bar(h_tr, Y_tr, p)

    if verbose:
        # Report probe quality
        Z_hat = (p @ h_te.T).T
        r2    = [np.corrcoef(Z_te[:, k], Z_hat[:, k])[0, 1]**2 for k in range(K_bar)]
        print(f"  Probe R² per concept: {[f'{v:.2f}' for v in r2]}")
        print(f"  Estimated Ā (macro coefficients): {np.round(A_bar, 3)}")
        print()

    # ── 3. Evaluate interventions on test samples ──────────────────────────────
    delta_z_bar = np.zeros(K_bar)
    delta_z_bar[0] = delta_target          # shift first concept by Δ

    results = {
        "min_norm": [],
        "q_constrained": [],
        "random": [],                       # pooled across all samplesxdraws
    }
    n_test = min(n_test, len(X_te))

    for i in range(n_test):
        h_i = h_te[i]

        # Jacobian of head at h_i
        J_i = model.jacobian_head(h_i)     # (K,)

        # Strategy 1: min-norm
        d1 = delta_min_norm(p, delta_z_bar)
        results["min_norm"].append(
            evaluate_intervention(d1, h_i, p, A_bar, model, delta_target))

        # Strategy 2: q-constrained
        d2 = delta_q_constrained(p, A_bar, J_i, delta_z_bar)
        results["q_constrained"].append(
            evaluate_intervention(d2, h_i, p, A_bar, model, delta_target))

        # Strategy 3: random (n_random draws per sample)
        for d3 in delta_random_unconstrained(p, delta_z_bar, n=n_random):
            results["random"].append(
                evaluate_intervention(d3, h_i, p, A_bar, model, delta_target))

    if verbose:
        _print_summary(results, K_bar, delta_target)

    return results, p, A_bar, model


def _print_summary(results, K_bar, delta_target):
    """Print per-strategy summary statistics."""
    print(f"  Target concept shift Δ = {delta_target}")
    print()
    print(f"  {'Strategy':<22}  {'On-target Δz̄₁':>16}  "
          f"{'Max off-target |Δz̄_k|':>23}  {'|ΔY - Ā₀Δ|':>14}")
    print("  " + "─" * 82)

    labels = [("min_norm",      "Min-norm steering"),
              ("q_constrained", "q-constrained"),
              ("random",        "Random unconstrained")]

    for key, label in labels:
        r = results[key]
        on  = np.array([x["on_target"]  for x in r])
        off = np.array([np.max(np.abs(x["off_target"])) for x in r])
        err = np.array([x["dY_error"]   for x in r])
        print(f"  {label:<22}  "
              f"{on.mean():>+8.4f} ± {on.std():>6.4f}  "
              f"{off.mean():>14.4f} ± {off.std():>6.4f}  "
              f"{err.mean():>10.4f} ± {err.std():>6.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(results, K_bar, delta_target, A_bar,
                 save_path="experiment3_nn_surgery.png"):

    labels = {
        "min_norm":      "Min-norm steering",
        "q_constrained": "q-constrained (ours)",
        "random":        "Random unconstrained",
    }
    colors = {
        "min_norm":      "steelblue",
        "q_constrained": "seagreen",
        "random":        "tomato",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Experiment 3 — Neural Network Representation Surgery", fontsize=14)

    # ── Panel 1: on-target concept change (should = Δ for all) ─────────────────
    ax = axes[0]
    for key in ["min_norm", "q_constrained", "random"]:
        vals = [x["on_target"] for x in results[key]]
        ax.hist(vals, bins=40, alpha=0.6, color=colors[key], label=labels[key])
    ax.axvline(delta_target, color="black", lw=2, linestyle="--", label=f"Target Δ={delta_target}")
    ax.set_xlabel("Δz̄₁  (on-target concept change)")
    ax.set_ylabel("Count")
    ax.set_title("On-target effect\n(all methods should hit Δ)")
    ax.legend(fontsize=8)

    # ── Panel 2: max off-target concept change (should = 0 for q-constrained) ──
    ax = axes[1]
    for key in ["min_norm", "q_constrained", "random"]:
        vals = [np.max(np.abs(x["off_target"])) for x in results[key]]
        ax.hist(vals, bins=40, alpha=0.6, color=colors[key], label=labels[key])
    ax.set_xlabel("max_k |Δz̄_k|  for k ≠ 1  (off-target concept change)")
    ax.set_ylabel("Count")
    ax.set_title("Off-target concept effects\n(should be 0 for q-constrained)")
    ax.legend(fontsize=8)

    # ── Panel 3: prediction error  |ΔY_actual - Ā₀·Δ| ─────────────────────────
    ax = axes[2]
    for key in ["min_norm", "q_constrained", "random"]:
        errs = [x["dY_error"] for x in results[key]]
        ax.hist(np.log10(np.array(errs) + 1e-8), bins=40,
                alpha=0.6, color=colors[key], label=labels[key])
    ax.axvline(-6, color="black", lw=1, linestyle=":", label="1e-6 threshold")
    ax.set_xlabel("log₁₀|ΔŶ_actual − Ā₀·Δ|  (output prediction error)")
    ax.set_ylabel("Count")
    ax.set_title("Output prediction error\n(q-constrained ≈ 0, others large)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved to: {save_path}")
    plt.show()


def plot_concept_heatmap(results, K_bar, delta_target,
                         save_path="experiment3_concept_shifts.png"):
    """
    Heatmap of mean |Δz̄_k| per concept dimension and strategy.
    Rows = strategies, columns = concept dimensions.
    """
    strategies = ["random", "min_norm", "q_constrained"]
    labels     = ["Random unconstrained", "Min-norm steering", "q-constrained (ours)"]

    mat = np.zeros((len(strategies), K_bar))
    for i, key in enumerate(strategies):
        r   = results[key]
        on  = np.mean([x["on_target"]  for x in r])
        off = np.array([x["off_target"] for x in r])   # (N, K̄-1)
        mat[i, 0]  = abs(on)
        mat[i, 1:] = np.mean(np.abs(off), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, K_bar * 1.5), 4))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(K_bar))
    ax.set_xticklabels(
        [f"z̄₁ (target)" if k == 0 else f"z̄{k+1} (off-target)" for k in range(K_bar)],
        rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(strategies)):
        for j in range(K_bar):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if mat[i,j] < mat.max()*0.6 else "white")
    plt.colorbar(im, ax=ax, label="Mean |Δz̄_k|")
    ax.set_title("Mean absolute concept-shift per strategy\n"
                 "(column 0 = target; columns 1+ = off-target spill)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for Experiment 3.  "
                           "Install with: pip install torch")

    print("=" * 65)
    print("EXPERIMENT 3 — Neural Network Representation Surgery")
    print("=" * 65)

    K_bar  = 4     # number of latent concepts
    D      = 40    # input feature dimension
    K      = 64    # hidden-layer width (micro state dimension)
    delta  = 1.5   # target concept shift magnitude

    print(f"\n  K̄={K_bar} concepts,  D={D} input features,  K={K} hidden units")
    print(f"  Target shift Δ = {delta}  (concept 1 only)\n")

    X, Y, Z, W_mix, alpha_true = generate_dataset(N=8000, K_bar=K_bar, D=D)
    print(f"  True a = {np.round(alpha_true, 3)}")
    print()

    results, p, A_bar, model = run_experiment(
        X, Y, Z, K=K, delta_target=delta, n_test=300, n_random=30)

    plot_results(results, K_bar, delta, A_bar)
    plot_concept_heatmap(results, K_bar, delta)

    print("\nDone.")
