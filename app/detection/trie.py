"""Character trie for name-dictionary lookups (Phase 5).

* ``roles(word)``       exact lookup, O(len(word)); returns e.g. {"first", "last"}.
* ``fuzzy(word, 1)``    names within Levenshtein distance k, for OCR slips such as
                        "Sharna" -> "Sharma" or "Arnit" -> "Amit".

Fuzzy search walks the trie with one row of the edit-distance matrix per node
and prunes any branch whose row minimum already exceeds k - the classic reason to
use a trie over a hash set for dictionary matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


class _Node:
    __slots__ = ("children", "roles")

    def __init__(self) -> None:
        self.children: Dict[str, "_Node"] = {}
        self.roles: Set[str] = set()      # non-empty => a word ends here


class NameTrie:
    def __init__(self) -> None:
        self._root = _Node()
        self.size = 0

    @staticmethod
    def normalize(word: str) -> str:
        return word.strip().lower()

    def insert(self, word: str, role: str) -> None:
        word = self.normalize(word)
        if not word:
            return
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
        if not node.roles:
            self.size += 1
        node.roles.add(role)

    def roles(self, word: str) -> Set[str]:
        node: Optional[_Node] = self._root
        for ch in self.normalize(word):
            node = node.children.get(ch)
            if node is None:
                return set()
        return set(node.roles)

    def __contains__(self, word: str) -> bool:
        return bool(self.roles(word))

    def fuzzy(self, word: str, max_edits: int = 1) -> List[Tuple[str, int, Set[str]]]:
        """All dictionary words within ``max_edits``: [(word, distance, roles)]."""
        word = self.normalize(word)
        first_row = list(range(len(word) + 1))
        results: List[Tuple[str, int, Set[str]]] = []

        def walk(node: _Node, prefix: str, prev_row: List[int]) -> None:
            for ch, child in node.children.items():
                row = [prev_row[0] + 1]
                for i in range(1, len(word) + 1):
                    cost = 0 if word[i - 1] == ch else 1
                    row.append(min(row[i - 1] + 1, prev_row[i] + 1, prev_row[i - 1] + cost))
                if child.roles and row[-1] <= max_edits:
                    results.append((prefix + ch, row[-1], set(child.roles)))
                if min(row) <= max_edits:
                    walk(child, prefix + ch, row)

        walk(self._root, "", first_row)
        return sorted(results, key=lambda r: (r[1], r[0]))

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_lists(cls, lists: Dict[str, Iterable[str]]) -> "NameTrie":
        trie = cls()
        for role, words in lists.items():
            for w in words:
                trie.insert(w, role)
        return trie

    @classmethod
    def from_dir(cls, names_dir: str | Path) -> "NameTrie":
        """Load firstName.txt / middleName.txt / lastName.txt (one name per line,
        '#' comments allowed). Missing files are skipped."""
        names_dir = Path(names_dir)
        files = {"first": "firstName.txt", "middle": "middleName.txt", "last": "lastName.txt"}
        lists = {}
        for role, fname in files.items():
            path = names_dir / fname
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                lists[role] = [ln.split("#")[0] for ln in lines if ln.split("#")[0].strip()]
        return cls.from_lists(lists)
