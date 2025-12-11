# lsh_typed.py
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray


def lsh(
    signatures: NDArray[np.int64],
    bands: int,
    rows_per_band: int,
) -> Set[Tuple[int, int]]:
    """
    Standard LSH banding on MinHash signatures.

    Supports two layouts:

    1) Exact:
       bands * rows_per_band == signature length

    2) Plus-one:
       bands * rows_per_band == signature length + 1 and rows_per_band > 1

       In this case:
       - (bands - 1) bands use `rows_per_band` rows
       - the last band uses `rows_per_band - 1` rows
       so that all rows of the signature are used exactly once.

    Parameters
    ----------
    signatures : ndarray of shape (n_objects, sig_len)
        MinHash signatures per object.
    bands : int
        Number of bands.
    rows_per_band : int
        Number of rows per band (for the exact layout).

    Returns
    -------
    candidates : set of (int, int)
        Candidate pairs (i, j) with i < j produced by LSH.
    """
    n_obj, sig_len = signatures.shape
    if n_obj == 0:
        return set()

    exact = (bands * rows_per_band == sig_len)
    plus_one = (bands * rows_per_band == sig_len + 1 and rows_per_band > 1)

    if not exact and not plus_one:
        raise ValueError(
            f"Invalid (bands, rows_per_band) for sig_len={sig_len}: "
            f"bands * rows_per_band = {bands * rows_per_band}, "
            f"expected {sig_len} or {sig_len + 1}"
        )

    candidates: Set[Tuple[int, int]] = set()

    if exact:
        # First layout: each band has #rows_per_band rows.
        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_block = signatures[:, start:end]

            buckets: Dict[Tuple[int, ...], List[int]] = {}

            for i in range(n_obj):
                key = tuple(int(x) for x in band_block[i, :])
                bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [i]
                else:
                    bucket.append(i)

            for idxs in buckets.values():
                if len(idxs) < 2:
                    continue
                idxs.sort()
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        candidates.add((idxs[i], idxs[j]))

        return candidates

    # Plus-one layout:
    # #(bands - 1) bands of #rows_per_band rows,
    # 1 last band of #(rows_per_band - 1) rows.
    full_band_rows = rows_per_band
    last_band_rows = rows_per_band - 1

    total_rows = (bands - 1) * full_band_rows + last_band_rows
    if total_rows != sig_len:
        raise ValueError(
            f"Internal error in plus-one layout: total_rows={total_rows}, sig_len={sig_len}"
        )

    start = 0
    for b in range(bands):
        if b < bands - 1:
            r = full_band_rows
        else:
            r = last_band_rows

        end = start + r
        band_block = signatures[:, start:end]

        buckets: Dict[Tuple[int, ...], List[int]] = {}

        for i in range(n_obj):
            key = tuple(int(x) for x in band_block[i, :])
            bucket = buckets.get(key)
            if bucket is None:
                buckets[key] = [i]
            else:
                bucket.append(i)

        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            idxs.sort()
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    candidates.add((idxs[i], idxs[j]))

        start = end

    return candidates
