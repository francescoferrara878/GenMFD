"""Task 4 - Foundation detection: precision, recall, F1 on the eligible documents of each corpus."""
import numpy as np
import pandas as pd
import os
import common as c

os.makedirs("outputs", exist_ok=True)

lex, cor = c.load_lexicons(), c.load_corpora()
rows = []
for cn in c.CORPORA:
    k = cor[cn]
    el = np.array(k.eligible)
    gold = c.gold_matrix(k)[el]
    n = int(el.sum())
    preds = {ln: c.predict(lex[ln], [t for t, e in zip(k.tokens, el) if e]) for ln in c.LEXICONS}
    for j, f in enumerate(c.FOUNDATIONS):
        g = gold[:, j]
        for ln in c.LEXICONS:
            p = preds[ln][:, j]
            tp, fp, fn, tn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()), int((~p & ~g).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            rows.append({"Corpus": cn, "Lexicon": ln, "Foundation": f, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                         "Precision": pr, "Recall": rc, "F1": f1, "N_eligible": n})
        pg = g.mean()
        rows.append({"Corpus": cn, "Lexicon": "Baseline (always positive)", "Foundation": f,
                     "TP": int(g.sum()), "FP": int((~g).sum()), "FN": 0, "TN": 0,
                     "Precision": pg, "Recall": 1.0, "F1": 2 * pg / (1 + pg), "N_eligible": n})
df = pd.DataFrame(rows)
macro = df.groupby(["Corpus", "Lexicon"])[["Precision", "Recall", "F1"]].mean().rename(columns={"F1": "macro_F1"})
pd.set_option("display.width", 250)
print(macro["macro_F1"].unstack(0).round(3).to_string())
df.to_csv("outputs/task4_detection_by_foundation.csv", index=False)
macro.to_csv("outputs/task4_macro.csv")
