# encoder_typed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set
import re


class Vocabulary:
    """
    String-to-int vocabulary.
    Each distinct token is mapped to a unique integer id.
    """

    def __init__(self) -> None:
        self._token_to_id: Dict[str, int] = {}

    def get_id(self, token: str) -> int:
        t = str(token)
        if t in self._token_to_id:
            return self._token_to_id[t]
        idx = len(self._token_to_id)
        self._token_to_id[t] = idx
        return idx

    def size(self) -> int:
        return len(self._token_to_id)


@dataclass
class BagOfTokensEncoder:
    """
    Base encoder: collection of string tokens encoded to set of feature ids.
    """
    vocab: Vocabulary

    def _add(self, ids: Set[int], token: str) -> None:
        token = token.strip()
        if not token:
            return
        ids.add(self.vocab.get_id(token))

    def encode_tokens(self, tokens: Iterable[str]) -> Set[int]:
        ids: Set[int] = set()
        for tok in tokens:
            self._add(ids, str(tok))
        return ids


@dataclass
class TitleModelWordEncoder(BagOfTokensEncoder):
    """
    Title model-word encoder in the spirit of MSMP+:
    - normalises common unit variants (inch / hz)
    - then applies a model word regex on the normalized text.
    """

    # ModelWord_title-style pattern:
    # token must contain at least one digit and at least one letter/symbol.
    _pattern = re.compile(
        r"(?iu)\b(?=[A-Z0-9_/'-]*\d)(?=[A-Z0-9_/'-]*[A-Z_'-])[A-Z0-9_/'-]+\b"
    )

    @staticmethod
    def normalize_units(text: str) -> str:
        """
        Unit normalisation for inch / hz:

        - map variants of inch to 'inch'
        - map variants of hertz to 'hz'
        - merge number + unit into one token: '23 inch' -> '23inch'
        - return lowercase text
        """
        t = text
        if not t:
            return ""

        t = t.lower()

        # quotes as inch: 23" -> 23inch
        t = re.sub(r'(\d+)\s*["”]', r"\1inch", t)
        # t = re.sub(r"(\d+)[\s\W]*inch\b", r"\1inch", t)

        # variants of 'inch' -> 'inch'
        t = re.sub(r"\b(inches|inch)\b", "inch", t)
        t = re.sub(r"-inch", "inch", t)

        # variants of 'hz' / 'hertz' -> 'hz'
        t = re.sub(r"\b(hertz|hz)\b", "hz", t)
        t = re.sub(r"-hz", "hz", t)

        # merge: number + unit -> '23inch', '100hz'
        t = re.sub(r"(\d+)\s*inch", r"\1inch", t)
        t = re.sub(r"(\d+)\s*hz", r"\1hz", t)

        return t


    def extract_model_words(self, text: str) -> List[str]:
        """
        Apply unit normalisation, then extract model word tokens.
        """
        norm = self.normalize_units(text or "")
        candidates: List[str] = []
        seen = set()
        for m in self._pattern.finditer(norm):
            w = m.group(0)
            if w and w not in seen:
                seen.add(w)
                candidates.append(w)
        return candidates

    def encode(self, text: str) -> Set[int]:
        tokens = self.extract_model_words(text)
        return self.encode_tokens(tokens)

@dataclass
class TypedModelWordEncoder(BagOfTokensEncoder):
    """
    Encoder for typed model-word tokens.
    Encodes each typed token directly as a feature id.
    """

    def encode(self, typed_tokens: Iterable[str]) -> Set[int]:
        return self.encode_tokens(typed_tokens)
