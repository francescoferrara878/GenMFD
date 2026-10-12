"""Shared code for the evaluation of GenMFD (Chapter 4 of the thesis).

Loads the six lexicons and the three corpora, applies the common design
choices (dominant foundation, Equality + Proportionality merged into Fairness,
base class excluded) and derives the gold foundations of each document.
"""
import json
import os
import re
from collections import defaultdict

import pandas as pd

FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "purity"]
LEXICONS = ["GenMFD (annotated)", "GenMFD (generated)", "MFD", "MFD 2.0", "eMFD", "MoralStrength"]
CORPORA = ["MFTC", "MFRC", "MFTCXplain"]

# Paths (override with environment variables if needed)
DATA = os.environ.get("GENMFD_DATA", "data")
GENMFD_DIR = os.environ.get("GENMFD_LEXICONS", os.path.join(os.path.dirname(__file__), "..", "lexicons"))
PATHS = {
    "genmfd_annotated": os.path.join(GENMFD_DIR, "genmfd_annotated.csv"),
    "genmfd_generated": os.path.join(GENMFD_DIR, "genmfd_generated.csv"),
    "mfd": os.path.join(DATA, "emfd/dictionaries/mft_original.dic"),
    "mfd2": os.path.join(DATA, "emfd/dictionaries/mfd2.0.dic"),
    "emfd": os.path.join(DATA, "emfd/dictionaries/emfd_scoring.csv"),
    "moralstrength": os.path.join(DATA, "moral-foundations/moralstrength/annotations/v1.1"),
    "mfrc": os.path.join(DATA, "final_mfrc_data.csv"),
    "mftc": os.path.join(DATA, "MFTC_V4_text.json"),
    "mftcxplain": os.path.join(DATA, "MFTCXplain/dataset/mftcxplain_en.json"),
}

# ---------------------------------------------------------------- tokenization
_URL = re.compile(r"http\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_TOKEN = re.compile(r"[a-zA-Z]+")


def tokenize(text):
    """Lowercase, remove URLs and @mentions, keep alphabetic tokens only."""
    text = _MENTION.sub(" ", _URL.sub(" ", text)).lower()
    return _TOKEN.findall(text)


# ------------------------------------------------------------------- lexicons
class Lexicon:
    """A lexicon as a set of (word, foundation) pairs, plus stems (MFD only)."""

    def __init__(self, name, pairs, stems=()):
        self.name = name
        self.pairs = set(pairs)          # exact words, e.g. ("suffering", "care")
        self.stems = set(stems)          # (stem, foundation) pairs matching by prefix
        self._cache = {}

    @property
    def all_pairs(self):
        """Entries as listed, with the asterisk of the stems (compared as strings in Task 3)."""
        return self.pairs | {(s + "*", f) for s, f in self.stems}

    def words_of(self, foundation):
        return {w for w, f in self.all_pairs if f == foundation}

    def foundations_of_token(self, token):
        hit = self._cache.get(token)
        if hit is None:
            hit = {f for f in FOUNDATIONS if (token, f) in self.pairs}
            hit |= {f for s, f in self.stems if token.startswith(s)}
            self._cache[token] = hit
        return hit

    def foundations_in(self, tokens):
        found = set()
        for t in set(tokens):
            found |= self.foundations_of_token(t)
        return found


_GEN_MAP = {"care": "care", "equality": "fairness", "proportionality": "fairness",
            "loyalty": "loyalty", "authority": "authority", "purity": "purity"}


def load_genmfd(path, name):
    df = pd.read_csv(path, index_col=0)
    dominant = df.idxmax(axis=1)                       # over all classes, base included
    pairs = {(str(w).lower(), _GEN_MAP[c]) for w, c in dominant.items() if c in _GEN_MAP}
    return Lexicon(name, pairs)


_DIC_MAP = {"care": "care", "fairness": "fairness", "loyalty": "loyalty",
            "authority": "authority", "sanctity": "purity"}


def load_dic(path, name):
    """LIWC-style .dic: poles merged; entries with '*' are stems; multiword discarded."""
    cats, pairs, stems = {}, set(), set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip("\n") for l in f]
    marks = [i for i, l in enumerate(lines) if l.strip() == "%"]
    for l in lines[marks[0] + 1:marks[1]]:
        parts = l.split()
        if len(parts) >= 2:
            cats[parts[0]] = parts[1].split(".")[0]
    for l in lines[marks[1] + 1:]:
        parts = l.split()
        if len(parts) < 2:
            continue
        # a multiword entry has several leading non-numeric tokens
        k = next((i for i, p in enumerate(parts) if p.isdigit()), len(parts))
        if k != 1:
            continue
        word = parts[0].lower()
        for c in parts[1:]:
            fnd = _DIC_MAP.get(cats.get(c))
            if fnd is None:
                continue                                # general 'moral' category
            (stems if word.endswith("*") else pairs).add(
                (word.rstrip("*"), fnd) if word.endswith("*") else (word, fnd))
    return Lexicon(name, pairs, stems)


def load_emfd(path, name="eMFD"):
    df = pd.read_csv(path).set_index("word")
    p = df[[c for c in df.columns if c.endswith("_p")]]
    names = {"care_p": "care", "fairness_p": "fairness", "loyalty_p": "loyalty",
             "authority_p": "authority", "sanctity_p": "purity"}
    dom = p.idxmax(axis=1).map(names)
    return Lexicon(name, {(str(w).lower(), f) for w, f in dom.items()})


def load_moralstrength(path, name="MoralStrength"):
    pairs = set()
    for f in FOUNDATIONS:
        df = pd.read_csv(os.path.join(path, f + ".tsv"), sep="\t")
        pairs |= {(str(w).lower(), f) for w in df.iloc[:, 0]}
    return Lexicon(name, pairs)


def load_lexicons():
    return {
        "GenMFD (annotated)": load_genmfd(PATHS["genmfd_annotated"], "GenMFD (annotated)"),
        "GenMFD (generated)": load_genmfd(PATHS["genmfd_generated"], "GenMFD (generated)"),
        "MFD": load_dic(PATHS["mfd"], "MFD"),
        "MFD 2.0": load_dic(PATHS["mfd2"], "MFD 2.0"),
        "eMFD": load_emfd(PATHS["emfd"]),
        "MoralStrength": load_moralstrength(PATHS["moralstrength"]),
    }


# -------------------------------------------------------------------- corpora
class Corpus:
    def __init__(self, name, texts, gold, eligible, all_docs=None):
        self.name = name
        self.texts = texts            # distinct texts (used in Tasks 2 and 4)
        self.tokens = [tokenize(t) for t in texts]
        self.gold = gold              # list of sets of foundations (None if not eligible)
        self.eligible = eligible      # list of bool
        # all documents of the corpus as released (used in Task 1); for MFTC the same
        # text may appear in several tweets, so this list is longer than `texts`
        self.doc_tokens = self.tokens if all_docs is None else [tokenize(t) for t in all_docs]


_MFTC_MAP = {"care": "care", "harm": "care", "fairness": "fairness", "cheating": "fairness",
             "loyalty": "loyalty", "betrayal": "loyalty", "authority": "authority",
             "subversion": "authority", "purity": "purity", "degradation": "purity"}


def _majority(label_sets, n):
    """Foundations selected by at least half of the n annotators."""
    cnt = defaultdict(int)
    for s in label_sets:
        for f in s:
            cnt[f] += 1
    return {f for f, c in cnt.items() if c / n >= 0.5}


def load_mftc():
    """All tweets are kept for Task 1; for Tasks 2 and 4 each distinct text is counted
    once and takes the foundations reached by at least one of its tweets."""
    with open(PATHS["mftc"], encoding="utf-8") as f:
        data = json.load(f)
    all_docs, order, gold, elig = [], [], {}, {}
    for block in data:
        for tw in block["Tweets"]:
            t = tw["tweet_text"]
            all_docs.append(t)
            anns = tw["annotations"]
            if t not in gold:
                order.append(t)
                gold[t], elig[t] = set(), False
            if len(anns) >= 2:
                elig[t] = True
                sets = [{_MFTC_MAP[l] for l in a["annotation"].split(",") if l in _MFTC_MAP} for a in anns]
                gold[t] |= _majority(sets, len(anns))
    return Corpus("MFTC", order, [gold[t] if elig[t] else None for t in order],
                  [elig[t] for t in order], all_docs=all_docs)


_MFRC_MAP = {"care": "care", "equality": "fairness", "proportionality": "fairness",
             "loyalty": "loyalty", "authority": "authority", "purity": "purity"}


def load_mfrc():
    df = pd.read_csv(PATHS["mfrc"])
    texts, gold, elig = [], [], []
    for t, g in df.groupby("text", sort=False):
        conf = g[g["confidence"] == "Confident"]
        texts.append(t)
        if len(conf) >= 2:
            sets = [{_MFRC_MAP[l.strip().lower()] for l in str(a).split(",")
                     if l.strip().lower() in _MFRC_MAP} for a in conf["annotation"]]
            gold.append(_majority(sets, len(conf)))
            elig.append(True)
        else:
            gold.append(None)
            elig.append(False)
    return Corpus("MFRC", texts, gold, elig)


_XPL_MAP = {"H": "care", "F": "fairness", "L": "loyalty", "A": "authority", "P": "purity"}


def load_mftcxplain():
    with open(PATHS["mftcxplain"], encoding="utf-8") as f:
        data = json.load(f)
    texts, gold = [], []
    for e in data:
        g = set()
        for k in ("mft_sentiment_first", "mft_sentiment_second", "mft_sentiment_third"):
            lab = e.get(k)
            if lab and lab[0] in _XPL_MAP and lab != "NM":
                g.add(_XPL_MAP[lab[0]])
        texts.append(e["tweets"])
        gold.append(g)
    return Corpus("MFTCXplain", texts, gold, [True] * len(texts))


def load_corpora():
    return {"MFTC": load_mftc(), "MFRC": load_mfrc(), "MFTCXplain": load_mftcxplain()}


# ---------------------------------------------------------------- predictions
import numpy as np


def predict(lexicon, token_lists):
    """Boolean matrix [documents, 5]: the lexicon has at least one word of the foundation."""
    out = np.zeros((len(token_lists), len(FOUNDATIONS)), dtype=bool)
    for i, toks in enumerate(token_lists):
        for f in lexicon.foundations_in(toks):
            out[i, FOUNDATIONS.index(f)] = True
    return out


def gold_matrix(corpus):
    """Boolean matrix [documents, 5] of the gold foundations (rows of non-eligible documents are False)."""
    g = np.zeros((len(corpus.gold), len(FOUNDATIONS)), dtype=bool)
    for i, s in enumerate(corpus.gold):
        for f in (s or ()):
            g[i, FOUNDATIONS.index(f)] = True
    return g


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def fleiss_kappa(matrix):
    """matrix: [documents, raters] of 0/1 judgments."""
    m = np.asarray(matrix, int)
    n_docs, n = m.shape
    n1 = m.sum(axis=1)
    n0 = n - n1
    p_i = (n1 * (n1 - 1) + n0 * (n0 - 1)) / (n * (n - 1))
    p1 = m.mean()
    pe = p1 ** 2 + (1 - p1) ** 2
    return (p_i.mean() - pe) / (1 - pe) if pe < 1 else float("nan")
