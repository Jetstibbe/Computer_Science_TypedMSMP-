# typed_msmp_main.py
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple, Optional, Dict as DictType

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from encoder_typed import Vocabulary, TitleModelWordEncoder, TypedModelWordEncoder
from minhash_typed import minhash
from lsh_typed import lsh
from msmp_typed_variants import build_msmp_token_sets, encode_token_sets, filter_msmp_by_df


# ---------------------------------------------------------------------------
# Data model + load
# ---------------------------------------------------------------------------

@dataclass
class Product:
    title: str
    model_id: str
    shop: str
    url: str
    features: Dict[str, str]


def _norm_id(v) -> str:
    return str(v).strip()


def load_products_from_typed_json(path: Path) -> List[Product]:
    """
    Load JSON:
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected top-level dict in typed JSON")

    products: List[Product] = []
    for model_key, variants in data.items():
        if not isinstance(variants, list):
            continue
        for obj in variants:
            model_id = _norm_id(obj.get("modelID", model_key))
            p = Product(
                title=str(obj.get("title", "")).strip(),
                model_id=model_id,
                shop=str(obj.get("shop", "")).strip(),
                url=str(obj.get("url", "")).strip(),
                features=dict(obj.get("featuresMap", {})),
            )
            products.append(p)
    return products


# ---------------------------------------------------------------------------
# LSH-features: title modelwords ∪ typed model words
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LSH-features: MSMP+ tokens (titel-modelwoorden + numerieke waarden uit features)
# ---------------------------------------------------------------------------

def build_feature_sets(
    products: Sequence[Product],
) -> Tuple[List[Set[int]], Vocabulary]:
    """
    Bouw de binaire representatie precies zoals in msmp_typed_variants:
      - model words uit titels
      - numerieke measurement tokens uit alle feature-values
    en encodeer die naar int-ids.
    """
    # 1) MSMP+ string-tokens (gebruikt titles + features)
    msmp_sets = build_msmp_token_sets(products)          # set[str] per product

    # 2) optioneel: DF-filter (kun je op 1 laten staan)
    msmp_sets = filter_msmp_by_df(msmp_sets, min_df=1)

    # 3) encode naar int-features met gedeelde vocabulary
    feature_sets, vocab = encode_token_sets(msmp_sets)   # List[Set[int]], Vocabulary
    return feature_sets, vocab


def stats_unique_tokens(products: Sequence[Product]) -> Tuple[int, int]:
    """
    Alle unieke string-tokens die in MSMP+ gebruikt worden tellen.
    Tweede getal is hier gewoon hetzelfde (we hebben maar één vocab),
    maar ik laat de signatuur intact.
    """
    msmp_sets = build_msmp_token_sets(products)
    msmp_sets = filter_msmp_by_df(msmp_sets, min_df=1)

    feature_sets, vocab = encode_token_sets(msmp_sets)
    vocab_size = vocab.size()
    return vocab_size, vocab_size


# ---------------------------------------------------------------------------
# Gold standard duplicate matrix
# ---------------------------------------------------------------------------

def build_duplicates_matrix(products: Sequence[Product]) -> NDArray[np.int64]:
    N = len(products)
    dup = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i + 1, N):
            if products[i].model_id == products[j].model_id:
                dup[i, j] = 1
                dup[j, i] = 1
    return dup


# ---------------------------------------------------------------------------
# LSH metrics (PQ, PC, F1*)
# ---------------------------------------------------------------------------

def lsh_metrics(
    candidate_pairs: Set[Tuple[int, int]],
    duplicates_matrix: NDArray[np.int64],
    total_pairs: int,
) -> DictType[str, float]:
    if not candidate_pairs:
        return {
            "t": 0.0,
            "pq_lsh": 0.0,
            "pc_lsh": 0.0,
            "f1_star": 0.0,
            "candidate_pairs": 0.0,
            "duplicates_found_lsh": 0.0,
        }

    num_comp = len(candidate_pairs)
    dup_found = 0
    for i, j in candidate_pairs:
        a, b = (i, j) if i < j else (j, i)
        if duplicates_matrix[a, b] == 1:
            dup_found += 1

    gold = np.triu(duplicates_matrix, k=1)
    total_dup = float(gold.sum())

    pq = dup_found / num_comp if num_comp > 0 else 0.0
    pc = dup_found / total_dup if total_dup > 0 else 0.0
    f1_star = 0.0 if pq + pc == 0.0 else 2 * pq * pc / (pq + pc)
    t = num_comp / float(total_pairs)

    return {
        "t": t,
        "pq_lsh": pq,
        "pc_lsh": pc,
        "f1_star": f1_star,
        "candidate_pairs": float(num_comp),
        "duplicates_found_lsh": float(dup_found),
    }


# ---------------------------------------------------------------------------
# MSM-similarities
# ---------------------------------------------------------------------------

_BRANDS = [
    "akai", "alba", "apple", "arcam", "arise", "bang", "bpl", "bush", "cge",
    "changhong", "compal", "curtis", "durabrand", "element", "finlux",
    "fujitsu", "funai", "google", "haier", "hisense", "hitachi", "itel",
    "jensen", "jvc", "kogan", "konka", "lg", "loewe", "magnavox", "marantz",
    "memorex", "micromax", "metz", "onida", "panasonic", "pensonic", "philips",
    "planar", "proscan", "rediffusion", "saba", "salora", "samsung", "sansui",
    "sanyo", "seiki", "sharp", "skyworth", "sony", "tatung", "tcl",
    "telefunken", "thomson", "toshiba", "tpv", "tp vision", "vestel",
    "videocon", "vizio", "vu", "walton", "westinghouse", "xiaomi", "zenith",
]


def brands_differ(a: Product, b: Product) -> bool:
    """
    Return True if brand evidence differs between the two products.
    """
    a_str = (a.title + " " + " ".join(a.features.values())).lower()
    b_str = (b.title + " " + " ".join(b.features.values())).lower()
    for brand in _BRANDS:
        a_has = brand in a_str
        b_has = brand in b_str
        if a_has != b_has:
            return True
    return False


def qgrams(text: str, k: int) -> Set[str]:
    text = text or ""
    if len(text) < k:
        return set()
    if len(text) == k:
        return {text}
    return {text[i: i + k] for i in range(len(text) - k + 1)}


def overlap_score(g1: Set[str], g2: Set[str]) -> float:
    if not g1 and not g2:
        return 0.0
    n1 = len(g1)
    n2 = len(g2)
    dist = len(g1.symmetric_difference(g2))
    return (n1 + n2 - dist) / (n1 + n2)


def jaccard(v1: Set[str], v2: Set[str]) -> float:
    if not v1 or not v2:
        return 0.0
    inter = len(v1 & v2)
    uni = len(v1 | v2)
    return inter / uni


def cosine_like(v1: Set[str], v2: Set[str]) -> float:
    if not v1 and not v2:
        return 0.0
    inter = len(v1 & v2)
    return inter / (math.sqrt(len(v1)) + math.sqrt(len(v2)))


_NOISE_WORDS = {"and", "or"}
_NOISE_CHARS = re.compile(r"[&/\-]")


def clean_title_tokens(title: str) -> Set[str]:
    cleaned = _NOISE_CHARS.sub(" ", (title or "").lower())
    toks = cleaned.split()
    return {t for t in toks if t not in _NOISE_WORDS}


_MODELWORD_PATTERN = re.compile(r"[A-Za-z0-9]*\d+[A-Za-z0-9]*")


def extract_model_words_from_text(text: str) -> Set[str]:
    return {m.group(0) for m in _MODELWORD_PATTERN.finditer(text or "")}


def extract_model_words_from_attributes(attrs: Dict[str, str]) -> Set[str]:
    result: Set[str] = set()
    for v in attrs.values():
        result |= extract_model_words_from_text(v)
    return result


def title_similarity(t1: str, t2: str) -> float:
    """
    Title similarity as in MSM:
    bag-of-words cosine_like, backed by model word cosine_like.
    Returns -1.0 if titles are too dissimilar to be used.
    """
    alpha = 0.602
    if not t1 or not t2:
        return -1.0
    bow1 = clean_title_tokens(t1)
    bow2 = clean_title_tokens(t2)
    if cosine_like(bow1, bow2) > alpha:
        return 1.0
    mw1 = extract_model_words_from_text(t1)
    mw2 = extract_model_words_from_text(t2)
    sim2 = cosine_like(mw1, mw2)
    if sim2 > 0:
        return sim2
    return -1.0


def msm_clustering(
    products: Sequence[Product],
    candidate_pairs: Optional[Set[Tuple[int, int]]] = None,
    *,
    k_qgram: int = 3,
    mu: float = 0.650,
    gamma: float = 0.756,
    distance_threshold: float = 0.522,
) -> Tuple[NDArray[np.int64], int]:
    """
    MSM similarity followed by agglomerative clustering.

    Parameters
    ----------
    products
        List of Product objects.
    candidate_pairs
        Optional LSH block of (i, j) pairs. If None, all pairs are compared.
    k_qgram, mu, gamma, distance_threshold
        MSM and clustering hyperparameters.

    Returns
    -------
    labels
        Cluster labels per product.
    num_comparisons_made
        Number of MSM comparisons carried out.
    """
    N = len(products)
    big = 1e9
    distances = np.full((N, N), big, dtype=float)
    np.fill_diagonal(distances, 0.0)
    comparisons = np.zeros((N, N), dtype=int)

    def set_distance(i: int, j: int, value: float, counted: bool) -> None:
        distances[i, j] = value
        distances[j, i] = value
        if counted:
            a, b = (i, j) if i <= j else (j, i)
            comparisons[a, b] = 1

    if candidate_pairs is None:
        index_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    else:
        index_pairs = []
        for i, j in candidate_pairs:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            index_pairs.append((a, b))

    for i, j in index_pairs:
        pi = products[i]
        pj = products[j]

        if pi.shop == pj.shop or brands_differ(pi, pj):
            continue

        sim_sum = 0.0
        m = 0
        w_sum = 0.0

        non_i: Dict[str, str] = dict(pi.features)
        non_j: Dict[str, str] = dict(pj.features)

        for key_i, val_i in pi.features.items():
            for key_j, val_j in pj.features.items():
                key_sim = overlap_score(qgrams(key_i, k_qgram), qgrams(key_j, k_qgram))
                if key_sim <= gamma:
                    continue
                val_sim = overlap_score(qgrams(val_i, k_qgram), qgrams(val_j, k_qgram))
                weight = key_sim
                sim_sum += weight * val_sim
                m += 1
                w_sum += weight
                non_i.pop(key_i, None)
                non_j.pop(key_j, None)

        avg_sim = sim_sum / w_sum if w_sum > 0 else 0.0
        mw_perc = jaccard(
            extract_model_words_from_attributes(non_i),
            extract_model_words_from_attributes(non_j),
        )
        t_sim = title_similarity(pi.title.lower(), pj.title.lower())

        min_feats = min(len(pi.features), len(pj.features))
        if min_feats == 0:
            continue

        if t_sim < 0:
            theta1 = m / min_feats
            theta2 = 1.0 - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_perc
        else:
            theta1 = (1.0 - mu) * m / min_feats
            theta2 = 1.0 - mu - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_perc + mu * t_sim

        set_distance(i, j, 1.0 - h_sim, counted=True)

    model = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        linkage="complete",
        metric="precomputed",
    )
    model.fit(distances)
    labels = model.labels_
    num_comparisons_made = int(comparisons.sum())
    return labels, num_comparisons_made


# ---------------------------------------------------------------------------
# Cluster-F1 on test-set (bootstrapping)
# ---------------------------------------------------------------------------

def cluster_f1_on_test(
    labels: NDArray[np.int64],
    duplicates_matrix: NDArray[np.int64],
    test_indices: NDArray[np.int64],
) -> DictType[str, float]:
    """
    Compute TP/FP/TN/FN, precision, recall, F1 on pairs restricted to test_indices.
    """
    N = labels.shape[0]
    test_mask = np.zeros(N, dtype=bool)
    test_mask[test_indices] = True

    TP = FP = TN = FN = 0

    for i in range(N):
        if not test_mask[i]:
            continue
        for j in range(i + 1, N):
            if not test_mask[j]:
                continue
            same_cluster = labels[i] == labels[j]
            is_dup = duplicates_matrix[i, j] == 1

            if same_cluster and is_dup:
                TP += 1
            elif same_cluster and not is_dup:
                FP += 1
            elif not same_cluster and not is_dup:
                TN += 1
            else:
                FN += 1

    TP = float(TP)
    FP = float(FP)
    TN = float(TN)
    FN = float(FN)

    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }


# ---------------------------------------------------------------------------
# Bootstrapping helper
# ---------------------------------------------------------------------------

def bootstrap_indices(
    N: int,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Bootstrap split:
      - draw N indices with replacement
      - train = unique indices in the sample
      - test  = out-of-bag indices
    """
    sample = rng.integers(low=0, high=N, size=N, endpoint=False)
    train = np.unique(sample)
    mask = np.ones(N, dtype=bool)
    mask[train] = False
    test = np.where(mask)[0]
    return train, test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Typed-model-words LSH + MSM clustering met 5 bootstraps"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to json",
    )
    parser.add_argument(
        "--num-hashes",
        type=int,
        default=800,
        help="Aantal MinHash-functies n.",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=10,
        help="LSH rows per band (r).",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=80,
        help="LSH number of bands (b).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed voor hashfuncties en bootstraps.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=5,
        help="Aantal bootstraps voor MSM-evaluatie.",
    )
    parser.add_argument(
        "--no-lsh",
        action="store_true",
        help="MSM op alle paren (geen LSH; candidate_pairs=None).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.650,
        help="Gewicht op titel-similariteit.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.756,
        help="Threshold voor key-similarity (q-grams).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.522,
        help="distance_threshold voor AgglomerativeClustering.",
    )
    parser.add_argument(
        "--k-qgram",
        type=int,
        default=3,
        help="q-gramlengte voor keys/values.",
    )

    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    default_data = here.parent / "data" / "TVs-all-merged.json"
    data_path = Path(args.data) if args.data is not None else default_data

    products = load_products_from_typed_json(data_path)
    N = len(products)
    print(f"Loaded {N} products from {data_path}")

    title_vocab_size, typed_vocab_size = stats_unique_tokens(products)
    print(f"Unique title model words: {title_vocab_size}")

    duplicates_matrix = build_duplicates_matrix(products)
    total_pairs = N * (N - 1) // 2
    print(f"Total possible pairs: {total_pairs}")

    rng = np.random.default_rng(args.seed)

    # LSH once (no bootstrap for LSH itself)
    if args.no_lsh:
        candidate_pairs = None
        print("No LSH: MSM will consider all pairs.")
        lsh_info = None
    else:
        feature_sets, vocab = build_feature_sets(products)
        sig = minhash(
            sets=feature_sets,
            num_hashes=args.num_hashes,
            universe_size=vocab.size(),
            seed=args.seed,
        )
        print(f"Signature length: {sig.shape[1] if sig.size else 0}")

        if args.r * args.b > args.num_hashes:
            raise ValueError("r * b must be <= num_hashes")

        candidate_pairs = lsh(sig, bands=args.b, rows_per_band=args.r)

        lsh_info = lsh_metrics(candidate_pairs, duplicates_matrix, total_pairs)
        print("\n=== LSH metrics (global, no bootstrap) ===")
        print(f"(r={args.r}, b={args.b})")
        print(f"t = {lsh_info['t']:.6f}")
        print(f"candidate_pairs = {lsh_info['candidate_pairs']:.1f}")
        print(f"duplicates_found_lsh = {lsh_info['duplicates_found_lsh']:.1f}")
        print(f"PQ_lsh = {lsh_info['pq_lsh']:.6f}")
        print(f"PC_lsh = {lsh_info['pc_lsh']:.6f}")
        print(f"F1*_lsh = {lsh_info['f1_star']:.6f}")

    # MSM + clustering with bootstrap evaluation on held-out pairs
    print("\n=== MSM + Agglomerative clustering with bootstraps (test F1) ===")

    f1_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []

    for k in range(args.bootstraps):
        train_idx, test_idx = bootstrap_indices(N, rng)

        if test_idx.size == 0:
            print(f"Bootstrap {k+1}: empty test set, skipping.")
            continue

        labels, num_comp = msm_clustering(
            products,
            candidate_pairs=candidate_pairs,
            k_qgram=args.k_qgram,
            mu=args.mu,
            gamma=args.gamma,
            distance_threshold=args.eps,
        )

        m = cluster_f1_on_test(labels, duplicates_matrix, test_idx)

        f1_list.append(m["f1"])
        prec_list.append(m["precision"])
        rec_list.append(m["recall"])

        print(
            f"Bootstrap {k+1}: "
            f"test_size={len(test_idx)}, "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, "
            f"F1={m['f1']:.4f}"
        )

    if f1_list:
        avg_f1 = float(np.mean(f1_list))
        avg_prec = float(np.mean(prec_list))
        avg_rec = float(np.mean(rec_list))
    else:
        avg_f1 = avg_prec = avg_rec = 0.0

    print("\n=== Average over bootstraps (test-set cluster F1) ===")
    print(f"precision_avg = {avg_prec:.4f}")
    print(f"recall_avg    = {avg_rec:.4f}")
    print(f"F1_avg        = {avg_f1:.4f}")


if __name__ == "__main__":
    main()
