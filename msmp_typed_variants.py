# msmp_typed_variants.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

import math
from encoder_typed import Vocabulary, TitleModelWordEncoder
from minhash_typed import minhash
from lsh_typed import lsh


# ============================================================
#  Basic helpers
# ============================================================

NUM_ONLY_RE = re.compile(r"^\d+(?:\.\d+)?$")

_NUM_RE = re.compile(r"(?<!\d)[-+]?\d*\.?\d+")

def round_half_up(x: float) -> int:
    """
    Round half up: 1.5 -> 2, -1.5 -> -2.
    """
    if x >= 0:
        return int(math.floor(x + 0.5))
    else:
        return int(math.ceil(x - 0.5))

def extract_number(s: str) -> float | None:
    """
    Extract first numeric substring as float, if present.
    """
    if not s:
        return None
    m = _NUM_RE.findall(s)
    if not m:
        return None
    try:
        return float(m[0])
    except ValueError:
        return None


# ------------------------------------------------------------
#  Unit normalization (inch / hz / nit / degrees)
# ------------------------------------------------------------

UNIT_INCH_RE = re.compile(
    r"\binches\b|\binch\b|\s*-inch\b|\s+inch\b",
    re.IGNORECASE,
)

UNIT_HZ_RE = re.compile(
    r'\bhertz\b|\bhz\b|\s+-hz\b|\s+hz\b',
    re.IGNORECASE,
)

UNIT_NIT_RE = re.compile(
    r'\bcd/?m2\b|\bcd/m\u00b2\b|\bcdm2\b|\bnits?\b',
    re.IGNORECASE,
)

UNIT_DEG_RE = re.compile(
    r'\u00b0|\bdegrees?\b|\bdeg\b|\bdeg\.\b',
    re.IGNORECASE,
)


def normalize_units_basic(s: str) -> str:
    """
    Basic normalization of units and numeric formatting in free text.
    - normalize decimal/thousand separators
    - map inch/hz/nit/degree variants to a canonical token
    """
    if not s:
        return ""
    text = str(s)
    text = text.replace("\u00c2", "")
    text = text.lower()

    # comma as decimal separator -> dot: 2,5 -> 2.5
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)

    # comma as thousands separator -> remove: 1,000 -> 1000
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    text = UNIT_INCH_RE.sub("inch", text)
    text = UNIT_HZ_RE.sub("hz", text)
    text = UNIT_NIT_RE.sub("nit", text)
    text = UNIT_DEG_RE.sub("deg", text)
    return text


# ------------------------------------------------------------
#  Fractions such as 80-7/8", "32 1/2"
# ------------------------------------------------------------

FRACTION_RANGE_RE = re.compile(r"(\d+)-(\d+)/(\d+)")
FRACTION_SPACE_RE = re.compile(r"(\d+)\s+(\d+)/(\d+)")

def parse_fraction_ranges(text: str) -> list[float]:
    """
    Parse mixed-number fractions (e.g. '80-7/8', '32 1/2') into floats.
    """
    vals: list[float] = []
    for m in FRACTION_RANGE_RE.finditer(text):
        a = float(m.group(1))
        b = float(m.group(2))
        c = float(m.group(3)) or 1.0
        vals.append(a + b / c)
    for m in FRACTION_SPACE_RE.finditer(text):
        a = float(m.group(1))
        b = float(m.group(2))
        c = float(m.group(3)) or 1.0
        vals.append(a + b / c)
    return vals


# ============================================================
#  Data model
# ============================================================

@dataclass
class Product:
    title: str
    model_id: str
    shop: str
    url: str
    features: Dict[str, str]


def load_products_from_raw(path: Path) -> List[Product]:
    """
    Load products from the TVs-all-merged JSON structure.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    products: List[Product] = []
    for model_key, variants in data.items():
        if not isinstance(variants, list):
            continue
        for obj in variants:
            model_id = str(obj.get("modelID", model_key)).strip()
            p = Product(
                title=str(obj.get("title", "")).strip(),
                model_id=model_id,
                shop=str(obj.get("shop", "")).strip(),
                url=str(obj.get("url", "")).strip(),
                features=dict(obj.get("featuresMap", {})),
            )
            products.append(p)
    return products


def build_duplicates_matrix(products: Sequence[Product]) -> NDArray[np.int64]:
    """
    Build a symmetric duplicates matrix based on model_id equality.
    """
    N = len(products)
    dup = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i + 1, N):
            if products[i].model_id == products[j].model_id:
                dup[i, j] = 1
                dup[j, i] = 1
    return dup


# ============================================================
#  MSMP+: title model words + numeric values
# ============================================================

PLUS_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)(?P<unit>[A-Za-z]+)?$")


def is_measurement_decimal(num: str, unit: str | None) -> bool:
    """
    Decide whether a numeric token should be treated as a 'measurement decimal'
    (i.e., kept as a value token in MSMP+).
    """
    if unit:
        return True
    if "." in num:
        return True
    try:
        v = float(num)
    except ValueError:
        return False
    return v >= 5.0



def extract_measurement_decimals(raw_text: str) -> list[str]:
    """
    Extract numeric tokens from free text that qualify as measurement decimals.
    """
    tokens: list[str] = []

    # 1) fracties uit RAW tekst (met quotes)
    for v in parse_fraction_ranges(raw_text):
        tokens.append(str(v))

    # 2) losse numerieken / numeriek+letters uit genormaliseerde tekst
    text = normalize_units_basic(raw_text)  # hier mag " → inch worden

    for m in PLUS_RE.finditer(text):
        num = m.group("num")
        if not num:
            continue
        tokens.append(num)

    return tokens


def build_msmp_token_sets(products: Sequence[Product]) -> List[Set[str]]:
    """
    Build MSMP+ style token sets:
      - model words from titles
      - numeric measurement tokens from all attribute values.
    """   
    enc = TitleModelWordEncoder(vocab=Vocabulary())

    MW_title_global: Set[str] = set()
    title_mw_per_product: List[Set[str]] = []

    # title model words: must contain at least one digit and one non-digit
    for p in products:
        norm_title = enc.normalize_units(p.title or "")
        mw_title = set(enc.extract_model_words(norm_title))
        filtered = {t for t in mw_title if re.search(r"\d", t) and re.search(r"\D", t)}
        title_mw_per_product.append(filtered)
        MW_title_global.update(filtered)

    msmp_sets: List[Set[str]] = []

    for p, mw_title in zip(products, title_mw_per_product):
        toks: Set[str] = set()
        # title model words
        toks |= mw_title

        # numeric measurement values from all attributes
        for raw_v in p.features.values():
            if not raw_v:
                continue
            raw_text = str(raw_v)
            for num in extract_measurement_decimals(raw_text):
                toks.add(num)

        msmp_sets.append(toks)

    return msmp_sets



def filter_msmp_by_df(
    msmp_raw: List[Set[str]],
    min_df: int = 1,
) -> List[Set[str]]:
    """
    Document-frequency filtering over MSMP+ token sets.
    """
    from collections import Counter

    freq = Counter()
    for s in msmp_raw:
        freq.update(s)
    msmp_filtered: List[Set[str]] = []
    for s in msmp_raw:
        msmp_filtered.append({mw for mw in s if freq[mw] >= min_df})
    return msmp_filtered


# ============================================================
#  Typing of numeric MSMP+ tokens
# ============================================================

NUM_ONLY_RE = re.compile(r"^\d+(?:\.\d+)?$")

RATIO_RE = re.compile(r"(\d+)\s*:\s*(\d+)")
RES_RE = re.compile(r"(\d+)\s*[xX×]\s*(\d+)(?:\s*[xX×]\s*(\d+))?")
MIXED_RES_RE  = re.compile(r"^(\d+)p$", re.IGNORECASE)
MIXED_HZ_RE   = re.compile(r"^(\d+)hz$", re.IGNORECASE)
MIXED_INCH_RE = re.compile(r"^(\d+)inch$", re.IGNORECASE)


def infer_num_label(num: str, contexts: list[str]) -> str:
    """
    Infer a coarse attribute label for a numeric value based on its context.
    """
    if not contexts:
        return "real" if "." in num else "int"

    for ctx in contexts:
        raw = ctx
        c = ctx.lower()

        # resolution via tokens such as 720p
        if re.search(rf"\b{re.escape(num)}p\b", c):
            return "res"

        # Hz
        if re.search(rf"\b{re.escape(num)}\s*hz\b", c):
            return "hz"

        # response time
        if re.search(rf"\b{re.escape(num)}\s*ms\b", c) or \
           "response time" in c or "millisecond" in c:
            return "time"

        # brightness
        if re.search(rf"\b{re.escape(num)}\b", c) and (
            "nit" in c or "cd/m2" in c or "cd/m²" in c or "brightness" in c
        ):
            return "bright"

        # weight
        if re.search(rf"\b{re.escape(num)}\b", c) and (
            "weight" in c or "lb" in c or "lbs" in c or "pound" in c or "kg" in c
        ):
            return "weight"

        # power
        if re.search(rf"\b{re.escape(num)}\b", c) and (
            "watt" in c or "wattage" in c or "power consumption" in c
        ):
            return "power"

        # year
        if re.search(rf"\b{re.escape(num)}\b", c) and ("year" in c or "years" in c):
            return "year"

        # inch: directly connected to inch or a quote
        if re.search(rf'\b{re.escape(num)}\s*inch\b', c) or \
        re.search(rf'{re.escape(num)}\s*["\u201d]', ctx):
            return "inch"

        # mm / cm
        if re.search(rf"\b{re.escape(num)}\b", c) and (
            "mm" in c or "millimeter" in c or "millimetre" in c
        ):
            return "mm"
        if re.search(rf"\b{re.escape(num)}\b", c) and (
            "cm" in c or "centimeter" in c or "centimetre" in c
        ):
            return "cm"

    # fallback
    return "real" if "." in num else "int"


typed_nums: Set[str] = set()
skip_nums: Set[str] = set()

def build_typed_msmp_variants(
    products: Sequence[Product],
    msmp_sets: Sequence[Set[str]],
) -> tuple[list[Set[str]], list[Set[str]]]:
    """
    Build two variants on top of MSMP+ token sets:
      - typed_only: only typed numeric tokens and mixed model words
      - typed_plus_raw: typed tokens + original MSMP+ tokens.
    """
    typed_only_sets: list[Set[str]] = []
    typed_plus_raw_sets: list[Set[str]] = []

    for prod, msmp_s in zip(products, msmp_sets):
        title_norm = normalize_units_basic(prod.title or "")

        # 1) attribute values as context strings
        value_contexts: list[str] = []
        for raw_v in prod.features.values():
            if not raw_v:
                continue
            v_norm = normalize_units_basic(str(raw_v))
            value_contexts.append(v_norm)

        all_contexts = ["title " + title_norm] + value_contexts

        out_only: Set[str] = set()
        out_plus: Set[str] = set()

        skip_nums: Set[str] = set()
        typed_nums: Set[str] = set()

        # 2) ratio patterns A:B encoded as single tokens
        for ctx in all_contexts:
            for a, b in RATIO_RE.findall(ctx):
                token = f"{a}:{b}_ratio"
                out_only.add(token)
                out_plus.add(token)
                skip_nums.add(a)
                skip_nums.add(b)

        # 3) A x B (x C) groups -> whole group gets a single unit
        for ctx in all_contexts:
            for m in RES_RE.finditer(ctx):
                a, b, c = m.group(1), m.group(2), m.group(3)
                nums_in_group = [n for n in (a, b, c) if n is not None]

                ctx_lower = ctx.lower()

                if "mm" in ctx_lower or "millimeter" in ctx_lower or "millimetre" in ctx_lower:
                    unit = "mm"
                elif "cm" in ctx_lower or "centimeter" in ctx_lower or "centimetre" in ctx_lower:
                    unit = "cm"
                elif "inch" in ctx_lower or "inches" in ctx_lower or '"' in ctx:
                    unit = "inch"
                elif "kg" in ctx_lower or "kilogram" in ctx_lower or "lb" in ctx_lower or "lbs" in ctx_lower:
                    unit = "weight"
                else:
                    # fallback: resolution only if at least one value >= 100
                    try:
                        max_val = max(float(n) for n in nums_in_group)
                    except ValueError:
                        continue
                    if max_val >= 100:
                        unit = "res"
                    else:
                        # no clear unit and all < 100 -> leave for generic loop
                        continue

                for n in nums_in_group:
                    if n is None:
                        continue
                    try:
                        base_int = int(round(float(n)))
                    except ValueError:
                        continue
                    num_str = str(base_int)

                    tok = f"{num_str}_{unit}"
                    out_only.add(tok)
                    out_plus.add(tok)
                    skip_nums.add(n)



        # 4) mixed title tokens (1080p, 60hz, 32inch)
        for tok in msmp_s:
            if NUM_ONLY_RE.fullmatch(tok):
                continue

            out_only.add(tok)
            out_plus.add(tok)

            m = MIXED_RES_RE.match(tok)   
            if m:
                n = m.group(1)
                base_int = int(round(float(n)))
                t = f"{base_int}_res"
                out_only.add(t)
                out_plus.add(t)
                typed_nums.add(n)
                skip_nums.add(n)
                continue

            m = MIXED_HZ_RE.match(tok)    
            if m:
                n = m.group(1)
                base_int = int(round(float(n)))
                t = f"{base_int}_hz"
                out_only.add(t)
                out_plus.add(t)
                typed_nums.add(n)
                skip_nums.add(n)
                continue

            m = MIXED_INCH_RE.match(tok)  
            if m:
                n = m.group(1)
                base_int = int(round(float(n)))
                t = f"{base_int}_inch"
                out_only.add(t)
                out_plus.add(t)
                typed_nums.add(n)
                skip_nums.add(n)
                continue

        # 5) numeric MSMP+ tokens: round + label, unless already handled
        for tok in msmp_s:
            if not NUM_ONLY_RE.fullmatch(tok):
                continue
            if tok in skip_nums:
                continue

            pattern = rf"(?<!\d){re.escape(tok)}(?!\d)"
            ctx_for_num: list[str] = []
            for ctx in all_contexts:
                if re.search(pattern, ctx):
                    ctx_for_num.append(ctx)

            # numeric value; include fractional forms when present
            val: float | None = None
            for ctx in ctx_for_num:
                frac_vals = parse_fraction_ranges(ctx)  
                if frac_vals:
                    val = frac_vals[0]  
                    break

            if val is None:
                try:
                    val = float(tok)
                except ValueError:
                    val = None

            if val is not None:
                base_int = round_half_up(val)
                base_str = str(base_int)
            else:
                base_str = tok


            label = infer_num_label(base_str, ctx_for_num)
            typed_tok = f"{base_str}_{label}"

            out_only.add(typed_tok)
            out_plus.add(tok)
            out_plus.add(typed_tok)

        typed_only_sets.append(out_only)
        typed_plus_raw_sets.append(out_plus)

    return typed_only_sets, typed_plus_raw_sets




# ============================================================
#  Encode + LSH
# ============================================================

def encode_token_sets(token_sets: Sequence[Set[str]]) -> Tuple[List[Set[int]], Vocabulary]:
    """
    Encode string-token sets to integer feature-id sets with a shared vocabulary.
    """
    vocab = Vocabulary()
    encoded: List[Set[int]] = []
    for s in token_sets:
        ids: Set[int] = set()
        for tok in sorted(s):
            ids.add(vocab.get_id(tok))
        encoded.append(ids)
    return encoded, vocab


def lsh_metrics(
    candidate_pairs: Set[Tuple[int, int]],
    duplicates_matrix: NDArray[np.int64],
    total_pairs: int,
) -> Dict[str, float]:
    """
    Compute LSH-based blocking metrics:
      - t (fraction of comparisons)
      - PQ (pair quality)
      - PC (pair completeness)
      - F1* (harmonic mean of PQ and PC).
    """
    if not candidate_pairs:
        return {
            "candidate_pairs": 0.0,
            "duplicates_found_lsh": 0.0,
            "pq_lsh": 0.0,
            "pc_lsh": 0.0,
            "f1_star": 0.0,
            "t": 0.0,
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
    f1_star = 0.0 if pq + pc == 0.0 else (2 * pq * pc) / (pq + pc)
    t = num_comp / float(total_pairs) if total_pairs > 0 else 0.0

    return {
        "candidate_pairs": float(num_comp),
        "duplicates_found_lsh": float(dup_found),
        "pq_lsh": float(pq),
        "pc_lsh": float(pc),
        "f1_star": float(f1_star),
        "t": float(t),
    }


def generate_r_b_pairs(n_hashes: int) -> List[Tuple[int, int]]:
    """
    Generate all (rows_per_band, bands) pairs compatible with the
    exact and plus-one layouts used by the LSH implementation.
    """
    pairs: Set[Tuple[int, int]] = set()

    for r in range(1, n_hashes + 1):
        # first (exact) layout: bands * r == n_hashes
        if n_hashes % r == 0:
            b = n_hashes // r
            if b >= 1:
                pairs.add((r, b))

        # second (plus-one) layout: bands * r == n_hashes + 1, only if r > 1
        if r > 1 and (n_hashes + 1) % r == 0:
            b = (n_hashes + 1) // r
            if b >= 1:
                pairs.add((r, b))

    return sorted(pairs)



def run_lsh_grid_for_variant(
    label: str,
    feature_sets: Sequence[Set[int]],
    vocab: Vocabulary,
    duplicates_matrix: NDArray[np.int64],
    n_hashes: int,
    seed: int,
) -> None:
    """
    Run MinHash+LSH for a variant over a grid of (rows, bands) settings,
    printing LSH metrics per configuration.
    """

    N = len(feature_sets)
    if N == 0:
        print(f"=== {label} ===")
        print("Geen producten.")
        return

    total_pairs = N * (N - 1) // 2

    print(f"=== Variant: {label} ===")
    print(f"#objects={N}, |V|={vocab.size()}, n_hashes={n_hashes}")
    print("r,b,t,candidate_pairs,duplicates_found,PQ,PC,F1*")

    sig = minhash(
        sets=feature_sets,
        num_hashes=n_hashes,
        universe_size=vocab.size(),
        seed=seed,
    )

    pairs = generate_r_b_pairs(n_hashes)

    for r, b in pairs:
        cand = lsh(signatures=sig, bands=b, rows_per_band=r)
        m = lsh_metrics(cand, duplicates_matrix, total_pairs)
        print(
            f"{r},{b},"
            f"{m['t']:.6f},"
            f"{m['candidate_pairs']:.1f},"
            f"{m['duplicates_found_lsh']:.1f},"
            f"{m['pq_lsh']:.6f},"
            f"{m['pc_lsh']:.6f},"
            f"{m['f1_star']:.6f}"
        )
    print()


def dump_preprocessed_tokens(
    path: Path,
    products: Sequence[Product],
    token_sets: Sequence[Set[str]],
) -> None:
    out: List[Dict[str, object]] = []
    for i, (p, toks) in enumerate(zip(products, token_sets)):
        out.append(
            {
                "index": i,
                "modelID": p.model_id,
                "shop": p.shop,
                "title": p.title,
                "tokens": sorted(toks),
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=Path,
        default=Path("/Users/jetstibbe/Documents/caise_typed/data/TVs-all-merged.json"),
    )
    parser.add_argument(
        "--n_hashes",
        type=int,
        default=840,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="lsh",
    )
    args = parser.parse_args()

    products = load_products_from_raw(args.json_path)
    dup = build_duplicates_matrix(products)

    # MSMP+ base
    msmp_raw = build_msmp_token_sets(products)
    msmp_sets = filter_msmp_by_df(msmp_raw, min_df=args.min_df)

    # typed variants
    typed_only_sets, typed_plus_raw_sets = build_typed_msmp_variants(products, msmp_sets)

    # JSON-dumps
    dump_preprocessed_tokens(
        Path(f"{args.out_prefix}_msmp_plus.json"),
        products,
        msmp_sets,
    )
    dump_preprocessed_tokens(
        Path(f"{args.out_prefix}_msmp_plus_typed_only.json"),
        products,
        typed_only_sets,
    )
    dump_preprocessed_tokens(
        Path(f"{args.out_prefix}_msmp_plus_typed_plus_raw.json"),
        products,
        typed_plus_raw_sets,
    )

    # integer-encoding
    msmp_sets_int, vocab_msmp = encode_token_sets(msmp_sets)
    typed_only_int, vocab_typed_only = encode_token_sets(typed_only_sets)
    typed_plus_raw_int, vocab_typed_plus_raw = encode_token_sets(typed_plus_raw_sets)

    # run grids
    run_lsh_grid_for_variant(
        label="MSMP_PLUS",
        feature_sets=msmp_sets_int,
        vocab=vocab_msmp,
        duplicates_matrix=dup,
        n_hashes=args.n_hashes,
        seed=args.seed,
    )

    run_lsh_grid_for_variant(
        label="MSMP_PLUS_TYPED_ONLY",
        feature_sets=typed_only_int,
        vocab=vocab_typed_only,
        duplicates_matrix=dup,
        n_hashes=args.n_hashes,
        seed=args.seed,
    )

    run_lsh_grid_for_variant(
        label="MSMP_PLUS_TYPED_PLUS_RAW",
        feature_sets=typed_plus_raw_int,
        vocab=vocab_typed_plus_raw,
        duplicates_matrix=dup,
        n_hashes=args.n_hashes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
