"""Task 3 - Vocabulary overlap: overlap coefficient on (word, foundation) pairs."""
import itertools
import pandas as pd
import os
import common as c

os.makedirs("outputs", exist_ok=True)

lex = c.load_lexicons()
rows = []
for a, b in itertools.combinations(c.LEXICONS, 2):
    for scope in ["all"] + c.FOUNDATIONS:
        A = lex[a].all_pairs if scope == "all" else {p for p in lex[a].all_pairs if p[1] == scope}
        B = lex[b].all_pairs if scope == "all" else {p for p in lex[b].all_pairs if p[1] == scope}
        inter = len(A & B)
        rows.append({"A": a, "B": b, "Foundation": scope, "size_A": len(A), "size_B": len(B),
                     "intersection": inter, "overlap": inter / min(len(A), len(B)) if min(len(A), len(B)) else float("nan")})
df = pd.DataFrame(rows)
print(df[df.Foundation == "all"].round(3).to_string())
df.to_csv("outputs/task3_overlap.csv", index=False)
