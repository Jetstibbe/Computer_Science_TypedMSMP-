# msmp_typed_bootstrap_grids.py
from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import List, Set, Tuple, Dict as DictType

import numpy as np
from numpy.typing import NDArray

from minhash_typed import minhash
from lsh_typed import lsh

from msmp_typed_variants import (
    load_products_from_raw,
    build_msmp_token_sets,
    filter_msmp_by_df,
    build_typed_msmp_variants,
    encode_token_sets,
    build_duplicates_matrix,
    generate_r_b_pairs,
    lsh_metrics,
)

from typed_msmp_main import msm_clustering, cluster_f1_on_test


def bootstrap_indices(
    N: int,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Bootstrap split on indices [0..N):

      - sample N indices with replacement
      - train = unique indices in the sample
      - test  = out-of-bag indices
    """
    sample = rng.integers(low=0, high=N, size=N, endpoint=False)
    train = np.unique(sample)
    mask = np.ones(N, dtype=bool)
    mask[train] = False
    test = np.where(mask)[0]
    return train, test


def build_all_token_variants(
    products: List,
    min_df: int = 1,
) -> Tuple[
    List[Set[int]],
    List[Set[int]],
    List[Set[int]],
]:
    """
    Build MSMP+ and two typed variants, and encode all to integer sets.
    """
    # 1) MSMP+ string token sets
    msmp_sets_str = build_msmp_token_sets(products)
    msmp_sets_str = filter_msmp_by_df(msmp_sets_str, min_df=min_df)

    # 2) typed variants on top of MSMP+
    typed_only_str, typed_plus_raw_str = build_typed_msmp_variants(
        products,
        msmp_sets_str,
    )

    # 3) integer encoding (separate vocab per variant)
    msmp_sets_int, _ = encode_token_sets(msmp_sets_str)
    typed_only_int, _ = encode_token_sets(typed_only_str)
    typed_plus_raw_int, _ = encode_token_sets(typed_plus_raw_str)

    return msmp_sets_int, typed_only_int, typed_plus_raw_int


def run_variant_with_bootstrap_and_eps_grid(
    variant: str,
    json_path: Path,
    n_hashes: int,
    seed: int,
    bootstraps: int,
    eps_grid: List[float],
    out_path: Path,
) -> None:
    """
    Full pipeline for a single variant:

      - build feature sets over all products
      - for each bootstrap:
          * bootstrap split (train/test) on [0..N)
          * MinHash+LSH on train and on test
          * epsilon-grid tuning on TRAIN via MSM clustering
          * MSM clustering + LSH metrics on TEST at best_eps
      - aggregate over bootstraps
      - write results to txt/csv
    """

    # 1) products + token sets + global duplicates matrix
    products_all = load_products_from_raw(json_path)
    N = len(products_all)

    msmp_sets_int, typed_only_int, typed_plus_raw_int = build_all_token_variants(
        products_all,
        min_df=1,
    )

    dup_all = build_duplicates_matrix(products_all)

    # 2) prepare MinHash signatures (full set; reused per bootstrap)
    if variant == "msmp":
        feature_sets = msmp_sets_int
        v_label = "MSMP_PLUS"
    elif variant == "typed_only":
        feature_sets = typed_only_int
        v_label = "MSMP_PLUS_TYPED_ONLY"
    elif variant == "typed_plus_raw":
        feature_sets = typed_plus_raw_int
        v_label = "MSMP_PLUS_TYPED_PLUS_RAW"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    print(f"Variant = {v_label}, n_hashes = {n_hashes}")

    # 3) r,b-pairs
    rb_pairs: List[Tuple[int, int]] = generate_r_b_pairs(n_hashes)
    print(f"#(r,b) pairs: {len(rb_pairs)}")

    # Accumulators over bootstraps
    agg_lsh_t: DictType[Tuple[int, int], List[float]] = {}
    agg_lsh_cand: DictType[Tuple[int, int], List[float]] = {}
    agg_lsh_dup: DictType[Tuple[int, int], List[float]] = {}
    agg_lsh_pq: DictType[Tuple[int, int], List[float]] = {}
    agg_lsh_pc: DictType[Tuple[int, int], List[float]] = {}
    agg_lsh_f1s: DictType[Tuple[int, int], List[float]] = {}

    agg_clu_prec: DictType[Tuple[int, int], List[float]] = {}
    agg_clu_rec: DictType[Tuple[int, int], List[float]] = {}
    agg_clu_f1: DictType[Tuple[int, int], List[float]] = {}

    rng = np.random.default_rng(seed)

    # 3) bootstrap loop
    for b_id in range(bootstraps):
        print(f"\n=== Bootstrap {b_id+1}/{bootstraps} for {v_label} ===", flush=True)

        train_idx, test_idx = bootstrap_indices(N, rng)

        if train_idx.size == 0 or test_idx.size == 0:
            print("Empty train or test set; skipping bootstrap.", flush=True)
            continue

        train_sets = [feature_sets[i] for i in train_idx]
        test_sets = [feature_sets[i] for i in test_idx]

        if feature_sets:
            universe_size = max((max(s) if s else 0) for s in feature_sets) + 1
        else:
            universe_size = 1

        sig_train: NDArray[np.int64] = minhash(
            sets=train_sets,
            num_hashes=n_hashes,
            universe_size=universe_size,
            seed=seed + b_id,
        )
        sig_test: NDArray[np.int64] = minhash(
            sets=test_sets,
            num_hashes=n_hashes,
            universe_size=universe_size,
            seed=seed + b_id,
        )

        train_products = [products_all[i] for i in train_idx]
        test_products = [products_all[i] for i in test_idx]

        dup_train = dup_all[np.ix_(train_idx, train_idx)]
        dup_test = dup_all[np.ix_(test_idx, test_idx)]

        T_train = len(train_products)
        T_test = len(test_products)

        total_pairs_train = T_train * (T_train - 1) // 2
        total_pairs_test = T_test * (T_test - 1) // 2

        # 4) loop over LSH (r,b) configurations
        for r, b in rb_pairs:
            print(f"  (r={r}, b={b})", flush=True)

            # LSH on train
            cand_train_local: Set[Tuple[int, int]] = lsh(
                signatures=sig_train,
                bands=b,
                rows_per_band=r,
            )
            # LSH on test
            cand_test_local: Set[Tuple[int, int]] = lsh(
                signatures=sig_test,
                bands=b,
                rows_per_band=r,
            )

            # TRAIN: epsilon grid tuning
            best_eps = None
            best_f1_train = -1.0

            for eps in eps_grid:
                # MSM clustering on train
                labels_train, _ = msm_clustering(
                    train_products,
                    candidate_pairs=cand_train_local,
                    k_qgram=3,
                    mu=0.650,
                    gamma=0.756,
                    distance_threshold=eps,
                )

                # F1 op train-subset
                all_train_indices = np.arange(T_train, dtype=np.int64)
                m_train = cluster_f1_on_test(
                    labels_train,
                    dup_train,
                    all_train_indices,
                )
                f1_tr = float(m_train["f1"])

                print(
                    f"    [TRAIN] eps={eps:.2f} → F1={f1_tr:.4f}",
                    flush=True,
                )

                if f1_tr > best_f1_train:
                    best_f1_train = f1_tr
                    best_eps = eps

            if best_eps is None:
                print("    No valid eps on train; skip (r,b).", flush=True)
                continue

            # TEST: LSH metrics
            lsh_info_test = lsh_metrics(
                candidate_pairs=cand_test_local,
                duplicates_matrix=dup_test,
                total_pairs=total_pairs_test,
            )

            # TEST: MSM with best_eps
            labels_test, _ = msm_clustering(
                test_products,
                candidate_pairs=cand_test_local,
                k_qgram=3,
                mu=0.650,
                gamma=0.756,
                distance_threshold=best_eps,
            )

            all_test_indices = np.arange(T_test, dtype=np.int64)
            m_test = cluster_f1_on_test(
                labels_test,
                dup_test,
                all_test_indices,
            )

            print(
                f"    [TEST] best_eps={best_eps:.2f} → "
                f"LSH_F1*={lsh_info_test['f1_star']:.4f}, "
                f"CLU_F1={float(m_test['f1']):.4f}",
                flush=True,
            )

            key = (r, b)

            # aggregate LSH metrics
            agg_lsh_t.setdefault(key, []).append(lsh_info_test["t"])
            agg_lsh_cand.setdefault(key, []).append(lsh_info_test["candidate_pairs"])
            agg_lsh_dup.setdefault(key, []).append(lsh_info_test["duplicates_found_lsh"])
            agg_lsh_pq.setdefault(key, []).append(lsh_info_test["pq_lsh"])
            agg_lsh_pc.setdefault(key, []).append(lsh_info_test["pc_lsh"])
            agg_lsh_f1s.setdefault(key, []).append(lsh_info_test["f1_star"])

            # aggregate clustering metrics
            agg_clu_prec.setdefault(key, []).append(float(m_test["precision"]))
            agg_clu_rec.setdefault(key, []).append(float(m_test["recall"]))
            agg_clu_f1.setdefault(key, []).append(float(m_test["f1"]))

    # 5) write averages per (r, b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        f_out.write(
            "(variant,r,b),"
            "t_lsh,cand_pairs_lsh,dup_lsh,PQ_lsh,PC_lsh,F1star_lsh,"
            "F1_clu,Prec_clu,Rec_clu\n"
        )

        for (r, b) in sorted(agg_lsh_t.keys()):
            t_avg = float(np.mean(agg_lsh_t[(r, b)]))
            cand_avg = float(np.mean(agg_lsh_cand[(r, b)]))
            dup_avg = float(np.mean(agg_lsh_dup[(r, b)]))
            pq_avg = float(np.mean(agg_lsh_pq[(r, b)]))
            pc_avg = float(np.mean(agg_lsh_pc[(r, b)]))
            f1s_avg = float(np.mean(agg_lsh_f1s[(r, b)]))

            f1c_avg = float(np.mean(agg_clu_f1.get((r, b), [0.0])))
            prec_avg = float(np.mean(agg_clu_prec.get((r, b), [0.0])))
            rec_avg = float(np.mean(agg_clu_rec.get((r, b), [0.0])))

            f_out.write(
                f"({v_label},{r},{b}),"
                f"{t_avg:.6f},{cand_avg:.1f},{dup_avg:.1f},"
                f"{pq_avg:.6f},{pc_avg:.6f},{f1s_avg:.6f},"
                f"{f1c_avg:.6f},{prec_avg:.6f},{rec_avg:.6f}\n"
            )

    print(f"Done: {v_label} → {out_path}")

def main() -> None:

    if __name__ == "__main__" and len(sys.argv) == 1:
        # Default values for local debugging
        sys.argv.extend([
            "--variant", "typed_plus_raw",
            "--json", "/Users/jetstibbe/Documents/caise_typed/data/TVs-all-merged.json",
            "--n-hashes", "840",
            "--bootstraps", "5",
            "--seed", "42",
            "--out", "msm_debug_output.txt",
        ])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        choices=["msmp", "typed_only", "typed_plus_raw"],
        required=True,
        help="Which feature variant to use",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        required=True,
        help="Path to TVs-all-merged.json",
    )
    parser.add_argument(
        "--n-hashes",
        dest="n_hashes",
        type=int,
        default=840,
        help="Number of MinHash-functions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        required=True,
        help="Output txt/csv file.",
    )
    args = parser.parse_args()

    # same epsilon grid as in the main experiments
    eps_grid = [0.25, 0.38, 0.52]

    run_variant_with_bootstrap_and_eps_grid(
        variant=args.variant,
        json_path=args.json_path,
        n_hashes=args.n_hashes,
        seed=args.seed,
        bootstraps=args.bootstraps,
        eps_grid=eps_grid,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
