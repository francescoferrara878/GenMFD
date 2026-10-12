"""Task 2 - Agreement: Fleiss' kappa among the six lexicons and Cohen's kappa between pairs,
on the documents annotated with each foundation."""
import itertools
import numpy as np
import pandas as pd
import os
import common as c

os.makedirs("outputs", exist_ok=True)

lex, cor = c.load_lexicons(), c.load_corpora()
fleiss, pair_rows = [], []
for cn in c.CORPORA:
    k = cor[cn]
    gold = c.gold_matrix(k)
    preds = {ln: c.predict(lex[ln], k.tokens) for ln in c.LEXICONS}
    for j, f in enumerate(c.FOUNDATIONS):
        idx = gold[:, j]
        m = np.stack([preds[ln][idx, j] for ln in c.LEXICONS], axis=1)
        fleiss.append({"Corpus": cn, "Foundation": f, "N_documents": int(idx.sum()),
                       "Fleiss_kappa": c.fleiss_kappa(m),
                       **{f"prev_{ln}": 100 * preds[ln][idx, j].mean() for ln in c.LEXICONS}})
        for a, b in itertools.combinations(c.LEXICONS, 2):
            kap, po = c.cohen_kappa(preds[a][idx, j], preds[b][idx, j])
            pair_rows.append({"Corpus": cn, "Foundation": f, "A": a, "B": b, "cohen_kappa": kap, "raw_agreement": po})
fl = pd.DataFrame(fleiss)
pw = pd.DataFrame(pair_rows)
pd.set_option("display.width", 250)
print(fl.round(3).to_string())
avg = pw.groupby(["Corpus", "A", "B"])[["cohen_kappa", "raw_agreement"]].mean().round(3)
print(avg.to_string())
fl.to_csv("outputs/task2_fleiss_by_foundation.csv", index=False)
pw.to_csv("outputs/task2_pairwise_by_foundation.csv", index=False)
avg.to_csv("outputs/task2_pairwise_mean_over_foundations.csv")
