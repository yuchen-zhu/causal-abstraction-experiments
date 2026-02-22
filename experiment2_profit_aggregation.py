"""
Experiment 2: Profit Aggregation Simulation
============================================

Micro SCM (retailer with K products):
  X_k  :=  N_k          (sales of product k, driven by exogenous noise)
  Y    :=  a^T X + N_Y  (total profit; ak = unit margin of product k)

Aggregation map p:
  X̄  =  Σk X_k          (total category sales — the KPI stakeholders track)
  Ȳ  =  Y               (profit is already a scalar, directly observed)

Macro causal story (choice of Ā / β):
  Commit to  Ȳ = β X̄ + N̄_Y  as the macro model.
  β is a modelling decision — e.g. β = mean(ak) says "the macro model
  treats the portfolio as if every product had the average margin."
  This determines the complementary variable:
    q(x, n_Y) = (a - β·1)^T x + N_Y  =  Y - β X̄   (unexplained profit)
  which is exactly the residual not captured by the macro story.

Implementation constraints for a shift intervention δ_x on the X variables:
  (1)  1^T δ_x = Δx̄            [hits the total-sales macro target]
  (2)  (a - β·1)^T δ_x = 0     [preserves q, i.e. the unexplained profit]

Consequence:  ΔY = a^T δ_x = β·Δx̄  (exact, for any q-constrained impl.)
              ΔY = a^T δ_x           (varies, for naive implementations)

Part A:  Naive vs q-constrained interventions
  Compare the distribution of actual ΔY across many implementations of each
  type, and measure the profit prediction error vs the macro prediction β·Δx̄.

Part B:  Constrained business levers
  In practice the company can only use a small number of levers:
    - Pricing discount:  one uniform shift per sub-category (G groups)
    - Promotion:         equal additive shift to top-M products by sales rank
    - Shelf placement:   shift proportional to margin rank

  For each lever, find the minimum-norm solution in the lever subspace that
  (a) matches the macro target and (b) satisfies the q constraint (if feasible).
  Report: feasibility, constraint violation, sparsity, and business alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm

rng = np.random.default_rng(42)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_retailer(K=500, lognormal_mean=0.5, lognormal_sigma=0.6):
    """
    Sample product margins a ~ LogNormal.  Returns a (K,) and sub-category
    assignments (K,) where sub-categories are contiguous blocks of size K//G.
    """
    alpha = rng.lognormal(mean=lognormal_mean, sigma=lognormal_sigma, size=K)
    return alpha


def choose_beta(alpha, method="mean"):
    """
    Choose the macro-level elasticity β — the 'macro causal story'.
    method='mean'  : β = E[ak]  (average-margin story)
    method='median': β = median(ak)
    method='ols'   : β = Cov(Y, X̄) / Var(X̄) = Σak / K  (same as mean here
                     since Xk are i.i.d.; included for conceptual clarity)
    """
    if method == "mean":
        return np.mean(alpha)
    elif method == "median":
        return np.median(alpha)
    elif method == "ols":
        # Under the micro SCM with i.i.d. unit-variance Xk:
        # Cov(Y, X̄) = Var(Xk) * Σak / K = mean(a)
        return np.mean(alpha)
    raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────────────────────────────────────
# Part A — naive vs q-constrained implementations
# ──────────────────────────────────────────────────────────────────────────────

def sample_naive_implementations(delta_x_bar, K, n=200):
    """
    Sample n shift vectors δ satisfying only  Σk δk = Δx̄.
    Strategy: uniform base shift + random mean-zero perturbation.
    """
    base = np.ones(K) * delta_x_bar / K
    deltas = [base]
    for _ in range(n - 1):
        perturb = rng.standard_normal(K)
        perturb -= perturb.mean()                    # project to null space of 1^T
        perturb *= rng.uniform(0.3, 3.0)             # random magnitude
        deltas.append(base + perturb)
    return deltas


def sample_q_constrained_implementations(delta_x_bar, alpha, beta, n=200):
    """
    Sample n shift vectors δ satisfying both:
      (1)  Σk δk = Δx̄
      (2)  Σk (ak - β) δk = 0
    Via pseudo-inverse particular solution + null-space perturbations.
    """
    K = len(alpha)
    c = alpha - beta
    C = np.vstack([np.ones(K), c])          # (2 × K) constraint matrix
    b = np.array([delta_x_bar, 0.0])

    delta_0 = pinv(C) @ b                   # minimum-norm particular solution

    # Null space: (K-2)-dimensional, orthogonal to both 1 and (a-β)
    # We construct it cheaply: random vectors projected onto the null space
    def proj_null(v):
        """Project v onto null space of C."""
        return v - pinv(C) @ (C @ v)

    deltas = [delta_0]
    for _ in range(n - 1):
        v = rng.standard_normal(K)
        deltas.append(delta_0 + proj_null(v) * rng.uniform(0.3, 3.0))
    return deltas


def actual_delta_Y(delta_x, alpha):
    """ΔY = a^T δ_x  (exact for root-cause SCM with no downstream feedback)."""
    return float(alpha @ delta_x)


def run_part_a(alpha, beta, delta_x_bar=10.0, n_impl=300, verbose=True):
    """
    Part A: compare ΔY distributions for naive vs q-constrained implementations.

    Returns:
      naive_dY  -- (n_impl,) array of ΔY values for naive implementations
      q_dY      -- (n_impl,) array of ΔY values for q-constrained implementations
      pred_dY   -- scalar macro model prediction β * Δx̄
    """
    pred_dY = beta * delta_x_bar

    naive_deltas = sample_naive_implementations(delta_x_bar, len(alpha), n_impl)
    q_deltas     = sample_q_constrained_implementations(delta_x_bar, alpha, beta, n_impl)

    naive_dY = np.array([actual_delta_Y(d, alpha) for d in naive_deltas])
    q_dY     = np.array([actual_delta_Y(d, alpha) for d in q_deltas])

    if verbose:
        print(f"  Macro prediction  β·Δx̄ = {beta:.4f} × {delta_x_bar} = {pred_dY:.4f}")
        print()
        print(f"  Naive ({n_impl} draws):")
        print(f"    mean ΔY = {naive_dY.mean():.4f}  ±  {naive_dY.std():.4f}")
        rel_err = np.abs(naive_dY - pred_dY) / (np.abs(pred_dY) + 1e-8)
        print(f"    off-target rate (>10% error): {(rel_err > 0.10).mean():.1%}")
        print()
        print(f"  q-constrained ({n_impl} draws):")
        print(f"    mean ΔY = {q_dY.mean():.4f}  ±  {q_dY.std():.6f}  (std should ≈ 0)")
        rel_err_q = np.abs(q_dY - pred_dY) / (np.abs(pred_dY) + 1e-8)
        print(f"    max relative error vs macro pred: {rel_err_q.max():.2e}  (should ≈ machine ε)")
        print()
        print(f"  Null-space dimension of q-constraint: {len(alpha) - 2}")
        print(f"    (K - 2 = {len(alpha)} - 2 = {len(alpha) - 2} free directions)")

    return naive_dY, q_dY, pred_dY


def plot_part_a(naive_dY, q_dY, pred_dY, save_path="experiment2a_profit.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 2A — Profit Change ΔY Across Naive vs q-Constrained Implementations",
                 fontsize=13)

    # Left: full distributions
    ax = axes[0]
    ax.hist(naive_dY, bins=40, alpha=0.7, color="tomato",   label="Naive")
    ax.hist(q_dY,     bins=40, alpha=0.7, color="steelblue", label="q-constrained")
    ax.axvline(pred_dY, color="black", lw=2, linestyle="--", label=f"Macro pred β·Δx̄ = {pred_dY:.2f}")
    ax.set_xlabel("ΔY (actual profit change)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of actual profit change\nacross implementations")
    ax.legend()

    # Right: relative error vs macro prediction (log scale for q-constrained)
    ax = axes[1]
    rel_naive = np.abs(naive_dY - pred_dY) / (np.abs(pred_dY) + 1e-8)
    rel_q     = np.abs(q_dY     - pred_dY) / (np.abs(pred_dY) + 1e-8)
    ax.hist(np.log10(rel_naive + 1e-16), bins=40, alpha=0.7, color="tomato",   label="Naive")
    ax.hist(np.log10(rel_q     + 1e-16), bins=40, alpha=0.7, color="steelblue", label="q-constrained")
    ax.axvline(np.log10(0.10), color="orange", linestyle="--", label="10% error threshold")
    ax.axvline(-12,            color="black",  linestyle=":",  label="machine ε")
    ax.set_xlabel("log₁₀(|ΔY - β·Δx̄| / |β·Δx̄|)  [relative error]")
    ax.set_ylabel("Count")
    ax.set_title("Relative profit prediction error\n(log scale)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Plot saved to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Part B — business levers
# ──────────────────────────────────────────────────────────────────────────────

def build_levers(alpha, G=8, M=40):
    """
    Build three lever matrices L ∈ ℝ^{K × d}.  Each column of L is one
    unit "direction" the company can push.  An intervention in this lever is
    δ = L @ t  for some parameters t ∈ ℝ^d.

    Returns dict: lever_name -> L matrix
    """
    K = len(alpha)
    levers = {}

    # ── 1. Pricing discount: one uniform shift per sub-category (G groups) ────
    # Products are sorted by margin and divided into G quantile buckets.
    # Each bucket has one shared discount parameter.
    buckets = np.array_split(np.argsort(alpha), G)  # sort by margin, split
    L_price = np.zeros((K, G))
    for g, bucket in enumerate(buckets):
        L_price[bucket, g] = 1.0 / len(bucket)      # unit: avg shift within bucket
    levers["Pricing\n(G=8 margin buckets)"] = L_price

    # ── 2. Promotion: equal shift to top-M products by margin ─────────────────
    top_M = np.argsort(alpha)[-M:]                   # top-M by margin
    L_promo = np.zeros((K, 1))
    L_promo[top_M, 0] = 1.0 / M                     # unit: avg shift across top-M
    levers[f"Promotion\n(top-{M} by margin)"] = L_promo

    # ── 3. Shelf placement: shift proportional to margin rank ─────────────────
    ranks = np.argsort(np.argsort(alpha)).astype(float) + 1.0  # rank 1..K
    L_shelf = (ranks / ranks.sum()).reshape(K, 1)
    levers["Shelf placement\n(margin-ranked)"] = L_shelf

    return levers


def solve_lever(L, alpha, beta, delta_x_bar):
    """
    Find minimum-norm t* such that δ = L @ t satisfies:
      (1)  1^T δ = Δx̄            [macro target]
      (2)  (a-β)^T δ = 0         [q constraint]

    Returns:
      delta        -- the recovered shift vector (K,)
      macro_viol   -- |1^T δ - Δx̄|  (should be ≈ 0)
      q_viol       -- |(a-β)^T δ|    (0 = feasible, >0 = infeasible lever)
      actual_dY    -- a^T δ  (actual profit change)
    """
    K = len(alpha)
    c = alpha - beta

    # Constraint matrix in t-space:  [1^T L; c^T L] t = [Δx̄, 0]
    C_t = np.vstack([np.ones(K) @ L, c @ L])   # (2 × d)
    b   = np.array([delta_x_bar, 0.0])

    t_star = pinv(C_t) @ b                      # min-norm least-squares in t
    delta  = L @ t_star

    macro_viol = abs(np.ones(K) @ delta - delta_x_bar)
    q_viol     = abs(c @ delta)
    actual_dY  = alpha @ delta

    return delta, macro_viol, q_viol, actual_dY


def run_part_b(alpha, beta, delta_x_bar=10.0, G=8, M=40, verbose=True):
    """
    Part B: test each business lever for feasibility and interpret the solution.

    Returns list of result dicts, one per lever.
    """
    pred_dY = beta * delta_x_bar
    levers  = build_levers(alpha, G=G, M=M)
    results = []

    if verbose:
        print(f"  Macro prediction β·Δx̄ = {pred_dY:.4f}")
        print(f"  {'Lever':<35}  {'Macro viol':>12}  {'q viol':>12}  {'ΔY actual':>12}  {'Sparsity':>10}")
        print("  " + "─" * 90)

    for name, L in levers.items():
        delta, mv, qv, dY = solve_lever(L, alpha, beta, delta_x_bar)
        sparsity = (np.abs(delta) > 1e-6 * norm(delta)).sum()
        rel_err  = abs(dY - pred_dY) / (abs(pred_dY) + 1e-8)
        feasible = qv < 1e-6 * abs(delta_x_bar)

        results.append({
            "name":     name,
            "delta":    delta,
            "macro_viol": mv,
            "q_viol":   qv,
            "actual_dY": dY,
            "pred_dY":  pred_dY,
            "rel_err":  rel_err,
            "sparsity": sparsity,
            "feasible": feasible,
        })

        if verbose:
            label = name.replace("\n", " ")
            print(f"  {label:<35}  {mv:>12.2e}  {qv:>12.4f}  {dY:>12.4f}  {sparsity:>10d}")

    if verbose:
        print()
        print("  Interpretation of q-constrained solutions (where feasible):")
        for r in results:
            if r["feasible"]:
                print(f"\n  [{r['name'].replace(chr(10),' ')}]  (q-feasible ✓)")
                _interpret_delta(r["delta"], alpha, beta)
            else:
                label = r["name"].replace("\n", " ")
                print(f"\n  [{label}]  q-infeasible — q violation = {r['q_viol']:.4f}")
                print(f"    Best achievable ΔY = {r['actual_dY']:.4f}  "
                      f"(macro pred = {pred_dY:.4f},  rel err = {r['rel_err']:.1%})")

    return results


def _interpret_delta(delta, alpha, beta, top_n=8):
    """
    Provide business-interpretable summary of a shift vector δ:
    - Which products receive the largest up/down shifts?
    - Are the largest-shifted products close to or far from margin β?
    """
    K = len(alpha)
    top_up   = np.argsort(delta)[-top_n:][::-1]
    top_down = np.argsort(delta)[:top_n]

    print(f"    Top {top_n} products receiving positive shift:")
    print(f"    {'Product':>10}  {'δk':>10}  {'ak (margin)':>14}  {'ak - β':>10}")
    for k in top_up:
        print(f"    {k:>10d}  {delta[k]:>+10.4f}  {alpha[k]:>14.4f}  {alpha[k]-beta:>+10.4f}")

    print(f"    Top {top_n} products receiving negative shift:")
    for k in top_down:
        print(f"    {k:>10d}  {delta[k]:>+10.4f}  {alpha[k]:>14.4f}  {alpha[k]-beta:>+10.4f}")

    # Correlation between δ and (a - β): should be ≈ 0 for q-constrained
    corr = np.corrcoef(delta, alpha - beta)[0, 1]
    # Correlation between δ and margin rank
    rank_corr = np.corrcoef(delta, np.argsort(np.argsort(alpha)))[0, 1]
    print(f"    Corr(δk, ak-β) = {corr:+.4f}  (0 = q-neutral, ±1 = margin-biased)")
    print(f"    Corr(δk, rank(ak)) = {rank_corr:+.4f}  (business direction: promote high-margin?)")


def plot_part_b(results, alpha, beta, save_path="experiment2b_levers.png"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=False)
    fig.suptitle("Experiment 2B — Business Lever Solutions: Shift δk vs Product Margin ak",
                 fontsize=13)

    for ax, r in zip(axes, results):
        delta = r["delta"]
        sc = ax.scatter(alpha, delta, c=alpha - beta, cmap="RdBu_r",
                        alpha=0.5, s=8, vmin=-(alpha.max()-beta), vmax=(alpha.max()-beta))
        ax.axhline(0,    color="k",      lw=0.8, linestyle="--")
        ax.axvline(beta, color="orange", lw=1.0, linestyle="--", label=f"β={beta:.2f}")
        ax.set_xlabel("Product margin ak")
        ax.set_ylabel("Shift δk")
        label = r["name"].replace("\n", " ")
        status = "✓ feasible" if r["feasible"] else f"✗ q-viol={r['q_viol']:.3f}"
        ax.set_title(f"{label}\n{status}\nΔY={r['actual_dY']:.2f}  pred={r['pred_dY']:.2f}")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="ak − β")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("EXPERIMENT 2 — Profit Aggregation Simulation")
    print("=" * 65)

    K     = 500
    alpha = setup_retailer(K=K)
    beta  = choose_beta(alpha, method="mean")
    delta_x_bar = 20.0          # target: increase total sales by 20 units

    print(f"\n  K = {K} products")
    print(f"  a ~ LogNormal(0.5, 0.6):  mean={alpha.mean():.3f}, "
          f"std={alpha.std():.3f}, range=[{alpha.min():.2f}, {alpha.max():.2f}]")
    print(f"  Macro story: β = mean(a) = {beta:.4f}")
    print(f"  Target total-sales shift Δx̄ = {delta_x_bar}")

    # ── Part A ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PART A — Naive vs q-Constrained Implementations")
    print("─" * 65 + "\n")
    naive_dY, q_dY, pred_dY = run_part_a(alpha, beta, delta_x_bar, n_impl=400)
    plot_part_a(naive_dY, q_dY, pred_dY)

    # ── Part B ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("PART B — Business Lever Feasibility and Interpretability")
    print("─" * 65 + "\n")
    results = run_part_b(alpha, beta, delta_x_bar, G=8, M=40)
    plot_part_b(results, alpha, beta)

    print("\nDone.")
