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
A) Implementation gap with q:
     Draw multiple δ satisfying (1)+(2) and verify  ε_impl = ‖Δ p s^(i)‖ ≈ 0.
B) Implementation gap without q:
     Draw multiple δ satisfying only (1) and show  ε_impl  is large.
C) Parameter sweep:
     Repeat A & B across many random (A, p, Ā) draws and plot distributions.
D) Ablation on Ā:
     Fix A and p; vary Ā and show validity holds but macro coefficient changes.
E) DAG subcase:
     Both A and Ā are restricted to be DAGs (strictly lower-triangular, i.e.
     variables are in topological order).  This makes the micro and macro causal
     graphs directly readable.  We print the edge structures, run the
     implementation gap check, and visualise how the micro DAG is abstracted into
     the macro DAG via p, alongside q's constraint on valid interventions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from numpy.linalg import norm, pinv, matrix_rank

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_stable_A(K, spectral_radius=0.7, rng=rng):
    """Random stable matrix: scale so that its spectral radius < target."""
    A = rng.standard_normal((K, K))
    eigvals = np.linalg.eigvals(A)
    A = A / (np.max(np.abs(eigvals)) + 1e-8) * spectral_radius
    return A


def make_full_rank_p(K_bar, K, rng=rng):
    """Random full-rank aggregation map p ∈ ℝ^{K̄ × K}."""
    while True:
        P = rng.standard_normal((K_bar, K))
        if matrix_rank(P) == K_bar:
            return P


def solution(n, A):
    """s(n) = (I - A)^{-1} n   [micro SCM solution, batch over columns of n]"""
    IminA = np.eye(A.shape[0]) - A
    return np.linalg.solve(IminA, n)


def intervened_solution(delta, n, A):
    """s^(i)(n) = (I - A)^{-1} (n + δ)  for constant shift intervention δ."""
    return solution(n + delta[:, None], A)


def make_q_matrices(p, A, A_bar):
    """
    Returns (Q_x, Q_n) such that q(x, n) = Q_x @ x + Q_n @ n,
    where  Q_x = pA - Āp,  Q_n = p.
    """
    Q_x = p @ A - A_bar @ p   # (K̄ × K)
    Q_n = p                    # (K̄ × K)
    return Q_x, Q_n


def q_value(x, n, Q_x, Q_n):
    """Evaluate complementary variable q(x, n) = Q_x x + Q_n n."""
    return Q_x @ x + Q_n @ n


def find_implementations(p, A, A_bar, delta_x_bar, n_implementations=5, rng=rng):
    """
    Find n_implementations micro shift vectors δ satisfying BOTH:
      (1)  p δ = Δx̄
      (2)  (pA - Āp)(I-A)^{-1} δ = 0

    Returns a list of δ vectors.
    """
    K = A.shape[0]
    Q_x, _ = make_q_matrices(p, A, A_bar)
    IminA_inv = np.linalg.inv(np.eye(K) - A)

    # Combined constraint matrix M δ = b
    #   Row block 1 (K̄ rows):  p δ = Δx̄
    #   Row block 2 (K̄ rows):  Q_x (I-A)^{-1} δ = 0
    M = np.vstack([p, Q_x @ IminA_inv])           # (2K̄ × K)
    b = np.concatenate([delta_x_bar, np.zeros(p.shape[0])])  # (2K̄,)

    # Particular solution via pseudo-inverse
    delta_particular = pinv(M) @ b

    # Null space of M — free directions
    NS = null_space(M)  # (K × d) where d = nullity of M

    deltas = [delta_particular]
    for _ in range(n_implementations - 1):
        t = rng.standard_normal(NS.shape[1])
        deltas.append(delta_particular + NS @ t)

    return deltas


def find_unconstrained_implementations(p, delta_x_bar, n_implementations=5, rng=rng):
    """
    Find n_implementations micro shift vectors δ satisfying ONLY:
      (1)  p δ = Δx̄
    (no q constraint — naive baseline)
    """
    delta_particular = pinv(p) @ delta_x_bar
    NS = null_space(p)

    deltas = [delta_particular]
    for _ in range(n_implementations - 1):
        t = rng.standard_normal(NS.shape[1])
        deltas.append(delta_particular + NS @ t)

    return deltas


def implementation_gap(deltas, n_batch, A, p):
    """
    Compute the max pairwise macro-effect difference across implementations:
      ε_impl = max_{i,j} ‖ p s^(δ_i)(n) - p s^(δ_j)(n) ‖_F / √(n_batch)
    averaged over n_batch noise draws.
    """
    n_batch_vecs = rng.standard_normal((A.shape[0], n_batch))  # (K × n_batch)
    macro_effects = []
    for delta in deltas:
        x_int = intervened_solution(delta, n_batch_vecs, A)   # (K × n_batch)
        macro_effects.append(p @ x_int)                        # (K̄ × n_batch)

    max_gap = 0.0
    for i in range(len(macro_effects)):
        for j in range(i + 1, len(macro_effects)):
            gap = norm(macro_effects[i] - macro_effects[j], "fro") / np.sqrt(n_batch)
            max_gap = max(max_gap, gap)
    return max_gap


# ──────────────────────────────────────────────────────────────────────────────
# Sub-experiment A & B: single draw, implementation gap with and without q
# ──────────────────────────────────────────────────────────────────────────────

def run_single_example(K=50, K_bar=3, n_impl=8, n_batch=500, verbose=True):
    """
    One random draw: compare implementation gap WITH vs WITHOUT the q constraint.
    Returns (gap_with_q, gap_without_q).
    """
    A     = make_stable_A(K)
    p     = make_full_rank_p(K_bar, K)
    A_bar = rng.standard_normal((K_bar, K_bar)) * 0.5   # arbitrary macro story
    delta_x_bar = rng.standard_normal(K_bar)

    # WITH q constraint
    deltas_q   = find_implementations(p, A, A_bar, delta_x_bar, n_impl)
    gap_with_q = implementation_gap(deltas_q, n_batch, A, p)

    # WITHOUT q constraint (naive)
    deltas_naive    = find_unconstrained_implementations(p, delta_x_bar, n_impl)
    gap_without_q   = implementation_gap(deltas_naive, n_batch, A, p)

    if verbose:
        print(f"  K={K}, K̄={K_bar}, {n_impl} implementations, {n_batch} noise draws")
        print(f"  Implementation gap  WITH  q : {gap_with_q:.2e}")
        print(f"  Implementation gap WITHOUT q : {gap_without_q:.4f}")
        print()

    return gap_with_q, gap_without_q


# ──────────────────────────────────────────────────────────────────────────────
# Sub-experiment C: parameter sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_parameter_sweep(n_trials=200, K_values=(20, 50, 100), K_bar=3, n_impl=6, n_batch=200):
    """
    Sweep over random draws and K values, collecting implementation gaps.
    Returns dict: K -> (gaps_with_q, gaps_without_q).
    """
    results = {}
    for K in K_values:
        print(f"  Sweeping K={K} ({n_trials} trials)...")
        gaps_with, gaps_without = [], []
        for _ in range(n_trials):
            gw, gn = run_single_example(K=K, K_bar=K_bar, n_impl=n_impl,
                                         n_batch=n_batch, verbose=False)
            gaps_with.append(gw)
            gaps_without.append(gn)
        results[K] = (np.array(gaps_with), np.array(gaps_without))
    return results


def plot_sweep_results(results, save_path="experiment1_sweep.png"):
    K_values = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 1 — Implementation Gap: With vs Without q", fontsize=14)

    # Left: with q (should be ≈ 0 — log scale)
    ax = axes[0]
    for K in K_values:
        gaps_with, _ = results[K]
        ax.hist(np.log10(gaps_with + 1e-16), bins=30, alpha=0.6, label=f"K={K}")
    ax.set_xlabel("log₁₀(implementation gap WITH q)")
    ax.set_ylabel("Count")
    ax.set_title("With q constraint\n(should be at machine-epsilon)")
    ax.legend()
    ax.axvline(-14, color="red", linestyle="--", label="machine ε (~1e-14)")

    # Right: without q
    ax = axes[1]
    for K in K_values:
        _, gaps_without = results[K]
        ax.hist(gaps_without, bins=30, alpha=0.6, label=f"K={K}")
    ax.set_xlabel("Implementation gap WITHOUT q")
    ax.set_ylabel("Count")
    ax.set_title("Without q constraint\n(heterogeneous, non-zero)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved sweep plot to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-experiment D: ablation on Ā
# ──────────────────────────────────────────────────────────────────────────────

def run_abar_ablation(K=50, K_bar=3, n_impl=6, n_batch=500, n_abar=20):
    """
    Fix A and p; vary Ā randomly.  For each Ā:
      - implementation gap with q  (should always be ≈ 0)
      - macro coefficient  Ā  (should change with Ā choice)
    """
    A = make_stable_A(K)
    p = make_full_rank_p(K_bar, K)
    delta_x_bar = rng.standard_normal(K_bar)

    gaps = []
    A_bars_norm = []

    for _ in range(n_abar):
        A_bar = rng.standard_normal((K_bar, K_bar)) * rng.uniform(0.1, 2.0)
        deltas = find_implementations(p, A, A_bar, delta_x_bar, n_impl)
        gap    = implementation_gap(deltas, n_batch, A, p)
        gaps.append(gap)
        A_bars_norm.append(norm(A_bar, "fro"))

    return np.array(A_bars_norm), np.array(gaps)


def plot_abar_ablation(A_bars_norm, gaps, save_path="experiment1_abar.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(A_bars_norm, np.log10(gaps + 1e-16), alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(-12, color="red", linestyle="--", label="machine ε threshold (~1e-12)")
    ax.set_xlabel("‖Ā‖_F  (macro story coefficient magnitude)")
    ax.set_ylabel("log₁₀(implementation gap WITH q)")
    ax.set_title("Ablation on Ā: validity holds for all Ā choices")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved Ā ablation plot to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-experiment E: DAG subcase — interpretable causal structure
# ──────────────────────────────────────────────────────────────────────────────

def make_dag_A(K, edge_prob=0.4, weight_scale=0.4, rng=rng):
    """
    Random linear-DAG weight matrix.  Nodes are in topological order 0,…,K-1.
    A[i,j] != 0 only when j < i  (j is a parent of i), sampled with
    probability edge_prob, with Gaussian weights scaled so the matrix is stable.
    """
    A = np.zeros((K, K))
    for i in range(1, K):
        for j in range(i):            # j is a potential parent of i
            if rng.random() < edge_prob:
                A[i, j] = rng.standard_normal() * weight_scale
    return A


def make_dag_A_bar(K_bar, edge_prob=0.5, weight_scale=0.4, rng=rng):
    """
    Random DAG weight matrix for the macro model (same convention as make_dag_A).
    """
    return make_dag_A(K_bar, edge_prob=edge_prob, weight_scale=weight_scale, rng=rng)


def make_group_p(K, K_bar, rng=rng):
    """
    Build an interpretable aggregation map p by partitioning the K micro-nodes
    into K_bar roughly equal groups, then assigning uniform weights within each
    group.  This gives p a clear "group average" meaning.

    Returned alongside groups, a list of lists of node indices, for display.
    """
    # Randomly permute nodes then split into K_bar groups
    perm = rng.permutation(K)
    groups = [sorted(perm[i::K_bar].tolist()) for i in range(K_bar)]

    P = np.zeros((K_bar, K))
    for bar_k, grp in enumerate(groups):
        P[bar_k, grp] = 1.0 / len(grp)   # uniform average of group
    return P, groups


def print_dag_structure(A, label="Micro DAG", threshold=1e-8):
    """Print the adjacency list of a DAG weight matrix."""
    K = A.shape[0]
    print(f"  {label} edges (j -> i, weight):")
    any_edge = False
    for i in range(K):
        for j in range(K):
            if abs(A[i, j]) > threshold:
                print(f"    X{j} -> X{i}  (w={A[i,j]:+.3f})")
                any_edge = True
    if not any_edge:
        print("    (no edges — empty graph)")


def run_dag_example(K=12, K_bar=3, edge_prob=0.4, n_impl=8, n_batch=500, verbose=True):
    """
    Sub-experiment E: restrict both A (micro) and Ā (macro) to be DAGs.

    The aggregation map p gives a "group-average" interpretation:
      macro node k̄  =  average of micro nodes in group k̄.

    We then verify:
      - Implementation gap WITH q ≈ 0   (Theorem 1 must still hold)
      - Implementation gap WITHOUT q is non-zero
    and print the graph structure for interpretability.
    """
    A       = make_dag_A(K, edge_prob=edge_prob, rng=rng)
    A_bar   = make_dag_A_bar(K_bar, rng=rng)
    p, groups = make_group_p(K, K_bar, rng=rng)
    delta_x_bar = rng.standard_normal(K_bar)

    if verbose:
        print(f"  K={K} micro nodes,  K̄={K_bar} macro nodes")
        print()
        print_dag_structure(A, label="Micro DAG (A)")
        print()
        print_dag_structure(A_bar, label="Macro DAG (Ā)")
        print()
        print("  Aggregation map  p  (group membership):")
        for bar_k, grp in enumerate(groups):
            print(f"    Macro node X̄{bar_k}  =  avg({['X'+str(g) for g in grp]})")
        print()

    # Implementations WITH q
    deltas_q    = find_implementations(p, A, A_bar, delta_x_bar, n_impl)
    gap_with_q  = implementation_gap(deltas_q, n_batch, A, p)

    # Implementations WITHOUT q
    deltas_naive   = find_unconstrained_implementations(p, delta_x_bar, n_impl)
    gap_without_q  = implementation_gap(deltas_naive, n_batch, A, p)

    if verbose:
        print(f"  Target macro shift Δx̄ = {np.round(delta_x_bar, 3)}")
        print(f"  Implementation gap  WITH  q : {gap_with_q:.2e}")
        print(f"  Implementation gap WITHOUT q : {gap_without_q:.4f}")
        print()
        _print_dag_implementation_insight(deltas_q, groups, K_bar)

    return gap_with_q, gap_without_q


def _print_dag_implementation_insight(deltas, groups, K_bar):
    """
    Interpret the implementations found under the q constraint:
    for each macro group, show mean and std of the shift applied to its members.
    """
    print("  Interpretation of q-constrained implementations:")
    print("  (each column is one implementation; rows are macro groups)")
    header = "  Group".ljust(30) + "".join(f"  impl{i+1:>3}" for i in range(len(deltas)))
    print(header)
    print("  " + "-" * (28 + 9 * len(deltas)))
    for bar_k, grp in enumerate(groups):
        row = f"  X̄{bar_k} ({['X'+str(g) for g in grp]})".ljust(30)
        for delta in deltas:
            group_shift = delta[grp]
            row += f"  {np.mean(group_shift):+6.3f}"
        print(row)
    print()
    print("  Note: within-group shifts vary across implementations (freedom)")
    print("        but p-weighted sums remain identical (Theorem 1).")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("EXPERIMENT 1 — Linear Causal Model: Complementary Variable")
    print("=" * 65)

    # ── A & B: Single example ─────────────────────────────────────────────────
    print("\n[A & B] Single random draw — implementation gap comparison")
    run_single_example(K=50, K_bar=3)

    # ── C: Parameter sweep ────────────────────────────────────────────────────
    print("[C] Parameter sweep over K values (200 trials each) ...")
    sweep_results = run_parameter_sweep(
        n_trials=200, K_values=(20, 50, 100), K_bar=3, n_impl=6, n_batch=200
    )

    print("\n  Summary (median implementation gaps):")
    print(f"  {'K':>6}  {'WITH q (median)':>20}  {'WITHOUT q (median)':>20}")
    print("  " + "-" * 50)
    for K in sorted(sweep_results):
        gw, gn = sweep_results[K]
        print(f"  {K:>6}  {np.median(gw):>20.2e}  {np.median(gn):>20.4f}")
    print()

    plot_sweep_results(sweep_results)

    # ── D: Ā ablation ─────────────────────────────────────────────────────────
    print("[D] Ablation on Ā — varying macro causal story (20 draws) ...")
    A_bars_norm, gaps = run_abar_ablation(K=50, K_bar=3, n_abar=20)
    plot_abar_ablation(A_bars_norm, gaps)

    # ── E: DAG subcase ────────────────────────────────────────────────────────
    print("[E] DAG subcase — interpretable causal structure")
    print("  Both micro (A) and macro (Ā) are restricted to DAGs.")
    print("  p gives group-average macro variables.\n")
    run_dag_example(K=12, K_bar=3, edge_prob=0.4, n_impl=6, n_batch=500)

    print("Done.")
