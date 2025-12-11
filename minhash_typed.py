# minhash_typed.py
from __future__ import annotations

from typing import List, Sequence, Set, Tuple
import random

import numpy as np
from numpy.typing import NDArray


def _pick_prime_above(n: int) -> int:
    """
    Return a fixed large prime strictly above n.
    """
    primes = [
        2147483647,
        4294967311,
        1610612741,
        805306457,
        3221225473,
    ]
    n = max(2, n)
    for p in primes:
        if p > n:
            return p
    return primes[0]


def _generate_hash_params(
    num_hashes: int,
    universe_size: int,
    rng: random.Random,
) -> List[Tuple[int, int, int]]:
    """
    Generate (a, b, p) parameters for num_hashes hash functions
    of the form h(x) = (a * x + b) mod p.
    """
    p = _pick_prime_above(universe_size)
    params: List[Tuple[int, int, int]] = []
    for _ in range(num_hashes):
        a = rng.randint(1, p - 1)
        b = rng.randint(0, p - 1)
        params.append((a, b, p))
    return params


def minhash(
    sets: Sequence[Set[int]],
    num_hashes: int,
    universe_size: int,
    seed: int = 0,
) -> NDArray[np.int64]:
    """
    Compute MinHash signatures for a sequence of feature-id sets.

    Parameters
    ----------
    sets : sequence of sets of int
        The feature sets per object.
    num_hashes : int
        Number of hash functions (signature length).
    universe_size : int
        Upper bound on feature ids; used to choose a prime.
    seed : int, default 0
        Random seed for hash parameter generation.

    Returns
    -------
    signatures : ndarray of shape (n_objects, num_hashes)
        MinHash signatures.
    """
    n_obj = len(sets)
    if n_obj == 0 or num_hashes <= 0:
        return np.zeros((0, num_hashes), dtype=np.int64)

    rng = random.Random(seed)
    params = _generate_hash_params(num_hashes, max(universe_size, 1), rng)

    inf = np.iinfo("int64").max
    sig = np.full((n_obj, num_hashes), fill_value=inf, dtype=np.int64)

    for i, features in enumerate(sets):
        if not features:
            continue
        for h_idx, (a, b, p) in enumerate(params):
            m = None
            for x in features:
                hv = (a * x + b) % p
                if m is None or hv < m:
                    m = hv
            if m is not None:
                sig[i, h_idx] = m

    return sig
