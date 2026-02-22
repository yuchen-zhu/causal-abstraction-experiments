"""
Experiment 1: Toy Linear Model — Validating the Complementary Variable Construction
=====================================================================================

Validates Theorem 1 from Chapter 4.2:

  Given a linear micro SCM  x = Ax + n,  a full-rank aggregation map  p,  and the
  complementary variable  q = [pA - Āp, p],  ALL micro implementations of any macro
  shift intervention produce the SAME macro causal effect.

Theory recap
------------
Micro SCM:   x = Ax + n   =>   solution  s(n) = (I - A)^{-1} n
Macro var:   x̄ = p x
Complement:  q(x, n) = (pA - Āp) x + p n

A micro shift intervention  i(z) = z + δ  applied post-structural-function gives:
  s^(i)(n) = (I - A)^{-1} (n + δ)

A valid *implementation* of macro shift  Δx̄  w.r.t. (p, q) must satisfy:
  (1)  p δ = Δx̄                             [hits the macro target]
  (2)  (pA - Āp)(I - A)^{-1} δ = 0         [preserves complementary variable q]

Theorem 1 says: all δ satisfying (1)+(2) yield the same  p s^(i)(n),  for all n.

Sub-experiments
---------------
A–C) Linear validation sweep:
     Single illustrative draw with verbose output, followed by a sweep over
     random (A, p, Ā) parameters and K values.  Shows the implementation gap
     drops to machine-epsilon with q, while remaining large without q.
E)   DAG subcase:
     Both A and Ā are restricted to be DAGs (strictly lower-triangular).
     This makes the micro and macro causal graphs directly readable.
     We print the edge structures and show how q constrains implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from numpy.linalg import norm, pinv, matrix_rank

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────────────────────────
# Constructors
# ──────────────────────────────────────────────────────────────────────────────

def make_stable_A(K, spectral_radius=0.7, rng=rng):
    """Random stable matrix: scale so that its spectral radius < target."""
    A = rng.standard_normal((K, K))
    eigvals = np.linalg.eigvals(A)
    A = A / (np.max(np.abs(eigvals)) + 1e-8) * spectral_radius
    return A


def make_dag_A(K, edge_prob=0.4, weight_scale=0.4, rng=rng):
    """
    Random linear-DAG weight matrix.  Nodes are in topological order 0,…,K-1.
    A[i,j] != 0 only when j < i  (j is a parent of i), sampled with
    probability edge_prob.  Strictly lower-triangular => always a valid DAG.
    """
    A = np.zeros((K, K))
    for i in range(1, K):
        for j in range(i):
            if rng.random() < edge_prob:
                A[i, j] = rng.standard_normal() * weight_scale
    return A


def make_partition_p(K, K_bar, rng=rng):
    """
    Constructive aggregation map: partition the K micro-nodes into K_bar
    non-overlapping groups and set p_{k̄} to the uniform average of the nodes
    in group k̄.  This makes p literally a 'group summary' operator and ensures
    the macro variables are disjoint projections of the micro state.

    Returns:
      p      -- (K_bar × K) aggregation matrix
      groups -- list of K_bar lists of node indices (the partition)
    """
    perm   = rng.permutation(K)
    groups = [sorted(perm[i::K_bar].tolist()) for i in range(K_bar)]
    P      = np.zeros((K_bar, K))
    for k_bar, grp in enumerate(groups):
        P[k_bar, grp] = 1.0 / len(grp)   # uniform group mean
    return P, groups


# ──────────────────────────────────────────────────────────────────────────────
# Core mathematical helpers
# ──────────────────────────────────────────────────────────────────────────────

def solution(n, A):
    """s(n) = (I - A)^{-1} n   [micro SCM solution, batched over columns of n]"""
    return np.linalg.solve(np.eye(A.shape[0]) - A, n)


def intervened_solution(delta, n, A):
    """s^(i)(n) = (I - A)^{-1} (n + δ)  for constant shift intervention δ."""
    return solution(n + delta[:, None], A)


def make_q_matrices(p, A, A_bar):
    """
    Returns (Q_x, Q_n) such that q(x, n) = Q_x @ x + Q_n @ n,
    where  Q_x = pA - Āp,  Q_n = p.
    """
    return p @ A - A_bar @ p, p   # (Q_x, Q_n)


def find_implementations(p, A, A_bar, delta_x_bar, n_implementations=5, rng=rng):
    """
    Find micro shift vectors δ satisfying BOTH constraints:
      (1)  p δ = Δx̄
      (2)  (pA - Āp)(I-A)^{-1} δ = 0

    Particular solution via pseudo-inverse; additional solutions by sampling
    the null space of the combined constraint matrix.
    """
    K    = A.shape[0]
    Q_x, _ = make_q_matrices(p, A, A_bar)
    IminA_inv = np.linalg.inv(np.eye(K) - A)

    M = np.vstack([p, Q_x @ IminA_inv])                          # (2K̄ × K)
    b = np.concatenate([delta_x_bar, np.zeros(p.shape[0])])       # (2K̄,)

    delta_0 = pinv(M) @ b
    NS      = null_space(M)                                        # (K × d)

    deltas = [delta_0]
    for _ in range(n_implementations - 1):
        deltas.append(delta_0 + NS @ rng.standard_normal(NS.shape[1]))
    return deltas


def find_unconstrained_implementations(p, delta_x_bar, n_implementations=5, rng=rng):
    """
    Find micro shift vectors δ satisfying ONLY:
      (1)  p δ = Δx̄
    (naive baseline — no q constraint)
    """
    delta_0 = pinv(p) @ delta_x_bar
    NS      = null_space(p)

    deltas = [delta_0]
    for _ in range(n_implementations - 1):
        deltas.append(delta_0 + NS @ rng.standard_normal(NS.shape[1]))
    return deltas


def implementation_gap(deltas, n_batch, A, p, rng=rng):
    """
    Max pairwise macro-effect difference across implementations, over n_batch
    random noise draws:
      ε_impl = max_{i,j} ‖ p s^(δ_i)(n) - p s^(δ_j)(n) ‖_F / √n_batch
    """
    N = rng.standard_normal((A.shape[0], n_batch))
    effects = [p @ intervened_solution(d, N, A) for d in deltas]

    max_gap = 0.0
    for i in range(len(effects)):
        for j in range(i + 1, len(effects)):
            max_gap = max(max_gap, norm(effects[i] - effects[j], "fro") / np.sqrt(n_batch))
    return max_gap


# ──────────────────────────────────────────────────────────────────────────────
# Sub-experiments A–C (combined): linear validation + sweep
# ──────────────────────────────────────────────────────────────────────────────

def _single_trial(K, K_bar, n_impl, n_batch):
    """One random draw; returns (gap_with_q, gap_without_q)."""
    A     = make_stable_A(K, rng=rng)
    p, _  = make_partition_p(K, K_bar, rng=rng)
    A_bar = rng.standard_normal((K_bar, K_bar)) * 0.5

    delta_x_bar = rng.standard_normal(K_bar)

    gap_with    = implementation_gap(
        find_implementations(p, A, A_bar, delta_x_bar, n_impl), n_batch, A, p)
    gap_without = implementation_gap(
        find_unconstrained_implementations(p, delta_x_bar, n_impl), n_batch, A, p)

    return gap_with, gap_without


def run_linear_validation(
        K_values=(20, 50, 100), K_bar=3,
        n_trials=200, n_impl=6, n_batch=200,
        save_path="experiment1_linear_validation.png"):
    """
    Sub-experiments A–C (combined):

    1. Illustrative single draw (K=50):  print gap with and without q.
    2. Parameter sweep over K values:    collect gap distributions and plot.

    The aggregation map p is always constructive (partition of nodes into
    K_bar non-overlapping groups with uniform weights).
    The macro story Ā is drawn randomly — any Ā is valid by Theorem 1.
    """
    # ── 1. Single illustrative draw ───────────────────────────────────────────
    K_demo = 50
    gw, gn = _single_trial(K_demo, K_bar, n_impl=8, n_batch=1000)
    print(f"  Single draw (K={K_demo}, K̄={K_bar}, 8 implementations, 1000 noise draws):")
    print(f"    Implementation gap  WITH  q :  {gw:.2e}  (should be ≈ machine ε)")
    print(f"    Implementation gap WITHOUT q :  {gn:.4f}  (should be non-zero)")
    print()

    # ── 2. Parameter sweep ────────────────────────────────────────────────────
    results = {}
    for K in K_values:
        print(f"  Sweep K={K} ({n_trials} trials)...", end=" ", flush=True)
        rows = [_single_trial(K, K_bar, n_impl, n_batch) for _ in range(n_trials)]
        gws, gns = zip(*rows)
        results[K] = (np.array(gws), np.array(gns))
        print(f"median gap WITH q: {np.median(gws):.2e}  WITHOUT: {np.median(gns):.4f}")

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 1 — Implementation Gap: With vs Without q", fontsize=14)

    ax = axes[0]
    for K in K_values:
        gws, _ = results[K]
        ax.hist(np.log10(gws + 1e-16), bins=30, alpha=0.6, label=f"K={K}")
    ax.axvline(-12, color="red", linestyle="--", alpha=0.7, label="machine ε")
    ax.set_xlabel("log₁₀(gap WITH q)")
    ax.set_ylabel("Count")
    ax.set_title("With q constraint  (should be at machine-ε)")
    ax.legend()

    ax = axes[1]
    for K in K_values:
        _, gns = results[K]
        ax.hist(gns, bins=30, alpha=0.6, label=f"K={K}")
    ax.set_xlabel("Gap WITHOUT q")
    ax.set_ylabel("Count")
    ax.set_title("Without q constraint  (non-zero, scales with heterogeneity of A)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Plot saved to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-experiment E: DAG subcase — interpretable causal structure
# ──────────────────────────────────────────────────────────────────────────────

def print_dag_structure(A, label="DAG", threshold=1e-8):
    """Print the edge list of a DAG weight matrix."""
    print(f"  {label} edges (parent -> child, weight):")
    any_edge = False
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if abs(A[i, j]) > threshold:
                print(f"    X{j} -> X{i}  (w={A[i, j]:+.3f})")
                any_edge = True
    if not any_edge:
        print("    (no edges)")


def _print_implementation_table(deltas, groups):
    """
    Print group-mean shifts for each implementation.
    Observable: group means are identical across all implementations (Theorem 1).
    Hidden:     within-group shifts differ (that is the null-space freedom).
    """
    print("  q-constrained implementations — group-mean shifts:")
    header = "  Group".ljust(34) + "".join(f"  impl{i+1:>2}" for i in range(len(deltas)))
    print(header)
    print("  " + "─" * (32 + 8 * len(deltas)))
    for k_bar, grp in enumerate(groups):
        label = f"  X̄{k_bar}  ← avg({['X'+str(g) for g in grp]})".ljust(34)
        row   = label + "".join(f"  {np.mean(d[grp]):+6.3f}" for d in deltas)
        print(row)
    print()
    print("  ↑ Group means are identical across implementations (Theorem 1).")
    print("    Within-group distributions differ freely (null-space freedom).")


def run_dag_example(K=12, K_bar=3, edge_prob=0.4, n_impl=6, n_batch=500):
    """
    Sub-experiment E: both A (micro) and Ā (macro) are DAGs.
    p is the constructive partition map (group averages).
    Prints edge structure, group membership, and the implementation table.
    """
    A         = make_dag_A(K, edge_prob=edge_prob, rng=rng)
    A_bar     = make_dag_A(K_bar, edge_prob=0.5, rng=rng)
    p, groups = make_partition_p(K, K_bar, rng=rng)
    delta_x_bar = rng.standard_normal(K_bar)

    print(f"  K={K} micro nodes,  K̄={K_bar} macro nodes\n")
    print_dag_structure(A,     label="Micro DAG (A)")
    print()
    print_dag_structure(A_bar, label="Macro DAG (Ā)")
    print()
    print("  Aggregation map p (partition):")
    for k_bar, grp in enumerate(groups):
        print(f"    X̄{k_bar}  =  avg({['X'+str(g) for g in grp]})")
    print()

    deltas_q   = find_implementations(p, A, A_bar, delta_x_bar, n_impl)
    deltas_nav = find_unconstrained_implementations(p, delta_x_bar, n_impl)

    gap_q   = implementation_gap(deltas_q,   n_batch, A, p)
    gap_nav = implementation_gap(deltas_nav, n_batch, A, p)

    print(f"  Target Δx̄ = {np.round(delta_x_bar, 3)}")
    print(f"  Implementation gap  WITH  q : {gap_q:.2e}")
    print(f"  Implementation gap WITHOUT q : {gap_nav:.4f}")
    print()
    _print_implementation_table(deltas_q, groups)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("EXPERIMENT 1 — Linear Causal Model: Complementary Variable")
    print("=" * 65)

    # ── A–C: linear validation sweep ──────────────────────────────────────────
    print("\n[A–C] Linear validation sweep\n")
    run_linear_validation(K_values=(20, 50, 100), K_bar=3, n_trials=200)

    # ── E: DAG subcase ────────────────────────────────────────────────────────
    print("\n[E] DAG subcase — interpretable structure\n")
    run_dag_example(K=12, K_bar=3, edge_prob=0.4, n_impl=6)

    print("\nDone.")
