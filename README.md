# GenMFD: A Moral Lexicon Extracted from LLM-Generated Texts

Resources accompanying the master's thesis *GenMFD: A Moral Lexicon Extracted from LLM-Generated Texts* (Francesco Maria Ferrara, MSc in Linguistic Sciences, University of Turin, 2026).

GenMFD is a pair of word-level moral lexicons built through compositional semantics (TF-IDF word-by-foundation weights, normalized per word) from texts of the MoralTrajectories dataset.

## Contents

```
LICENSE                  MIT license (evaluation code)
lexicons/
  genmfd_annotated.csv   2,830 words; weights for 6 foundations
  genmfd_generated.csv   2,364 words; weights for 6 foundations + base
code/
  cs.py                  compositional-semantics lexicon extraction
  GenerateLexicons.ipynb notebook that builds the two lexicons (outputs cleared)
  NOTICE                 origin and license of the code
evaluation/
  common.py              loading of lexicons and corpora, dominant-foundation rule, gold labels
  task1_coverage.py      Task 1 (coverage)
  task2_agreement.py     Task 2 (Fleiss' and Cohen's kappa, raw agreement)
  task3_overlap.py       Task 3 (vocabulary overlap)
  task4_detection.py     Task 4 (precision, recall, F-score, baseline)
  download_data.sh       downloads the public lexicons and MFTCXplain
  run_all.sh             runs everything
```

## Lexicon format

Each CSV has one row per word (first column, unnamed) and one column per class. The weights of each row sum to 1.

- `genmfd_annotated.csv`: authority, care, equality, loyalty, proportionality, purity. Built from the spans highlighted by human annotators.
- `genmfd_generated.csv`: the same six foundations plus `base`, the weight of the condition in which the model was not prompted with any foundation. Built from LLM responses labeled by prompting condition.

Words whose dominant class is `base` are not associated with any foundation.

```python
import pandas as pd
lex = pd.read_csv("lexicons/genmfd_annotated.csv", index_col=0)
lex.loc["dignity"].idxmax()   # dominant foundation of a word
```

## Rebuilding the lexicons

`code/GenerateLexicons.ipynb` expects the MoralTrajectories generations and annotations, which are not redistributed here (see the project repository: https://github.com/oaraque/moral-trajectories). Dependencies: numpy, pandas, scikit-learn, nltk, gsitk, tqdm. Words occurring in fewer than 10 documents are discarded (`cutoff=10`).

## Reproducing the evaluation

From the repository root:

```
sh evaluation/download_data.sh      # eMFD repository (MFD, MFD 2.0, eMFD), MoralStrength, MFTCXplain
# place final_mfrc_data.csv (MFRC) and MFTC_V4_text.json (MFTC) in data/
sh evaluation/run_all.sh
```

The four task scripts write their tables to `outputs/`. Different locations can be set with the environment variables `GENMFD_DATA` and `GENMFD_LEXICONS`.

## Not included

The evaluation corpora (MFRC, MFTC, MFTCXplain) and the lexicons used for comparison (MFD, MFD 2.0, MoralStrength, eMFD) are distributed by their authors and must be obtained from the original sources.

## License

Unless otherwise stated, the code in this repository is released under the MIT License (see `LICENSE`). The lexicons in `lexicons/` are an exception: they are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The code in `code/` comes from the MoralTrajectories project (https://github.com/oaraque/moral-trajectories), is released under the MIT License, Copyright (c) 2026 Oscar Araque, and keeps its own notice in `code/NOTICE`.

## Citation

If you use GenMFD, please cite:

Ferrara, F. M. (2026). GenMFD: A Moral Lexicon Extracted from LLM-Generated Texts. Master's thesis, University of Turin.

```bibtex
@mastersthesis{ferrara2026genmfd,
  author = {Ferrara, Francesco Maria},
  title  = {{GenMFD}: A Moral Lexicon Extracted from {LLM}-Generated Texts},
  school = {University of Turin},
  year   = {2026},
  type   = {Master's thesis}
}
```
