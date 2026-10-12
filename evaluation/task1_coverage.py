"""Task 1 - Coverage: percentage of documents containing at least one foundation word."""
import pandas as pd
import os
import common as c

os.makedirs("outputs", exist_ok=True)

lex, cor = c.load_lexicons(), c.load_corpora()
rows = {}
for cn in c.CORPORA:
    rows[cn] = {}
    for ln in c.LEXICONS:
        pred = c.predict(lex[ln], cor[cn].doc_tokens)
        rows[cn][ln] = 100 * pred.any(axis=1).mean()
df = pd.DataFrame(rows).T[c.LEXICONS].round(1)
print(df.to_string())
df.to_csv("outputs/task1_coverage.csv")
