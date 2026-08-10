"""
Post-run analysis for the core research question:

    Does semantic inter-agent disagreement correlate with wrong answers?

Reads results.csv (produced by main.py) and splits questions into
semantic_disagreement = True vs. False, then reports the wrong-answer /
hallucination rate in each group under several complementary definitions.

Run AFTER the full pipeline finishes:  python analyze_disagreement.py
"""

import sys
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

df = pd.read_csv(CSV)
total = len(df)

# --- Booleans come back from CSV as strings ("True"/"False") or real bools ---
def as_bool(series):
    return series.astype(str).str.strip().str.lower().eq("true")

for col in [
    "semantic_disagreement",
    "judge_answer_a_correct",
    "judge_answer_b_correct",
    "judge_factual_disagreement",
    "proposed_multi_agent_correct",
    "qwen_correct",
    "qwen_correct_judged",
]:
    if col in df.columns:
        df[col] = as_bool(df[col])

# --- Validity filter: same definition main.py uses for its metrics ---
def is_valid(row):
    a, b = str(row.get("answer_a", "")), str(row.get("answer_b", ""))
    return bool(a) and not a.startswith("ERROR:") and bool(b) and not b.startswith("ERROR:")

df["_valid"] = df.apply(is_valid, axis=1)
valid = df[df["_valid"]].copy()
n = len(valid)

# --- Data-quality / caveat counts ---
agent_err   = total - n
judge_err   = int((valid["judge_selection_status"] == "JUDGE_ERROR").sum())
qwen_err    = int(valid["qwen_answer"].astype(str).str.startswith("ERROR:").sum())

print("=" * 64)
print("DATA QUALITY")
print("=" * 64)
print(f"Total rows in CSV:                 {total}")
print(f"Valid rows (both agents answered): {n}")
print(f"Rows dropped (agent ERROR):        {agent_err}")
print(f"JUDGE_ERROR rows (within valid):   {judge_err}")
print(f"Qwen ERROR rows (within valid):    {qwen_err}")

def pct(num, den):
    return f"{num/den*100:5.2f}%" if den else "  N/A"

# ---------------------------------------------------------------------------
# CORE FINDING: disagreement vs. wrong-answer rate
# ---------------------------------------------------------------------------
# Three complementary "wrong" definitions per group:
#   1. BOTH_WRONG rate  -> consensus/both-agent hallucination
#      (judge_selection_status == BOTH_WRONG)
#   2. Pipeline-incorrect rate -> no correct answer produced/selected
#      (proposed_multi_agent_correct == False; covers BOTH_WRONG + JUDGE_ERROR)
#   3. Agent-answer wrong rate -> fraction of individual agent answers judged
#      incorrect (out of 2 per question)
def group_stats(g):
    gn = len(g)
    both_wrong = int((g["judge_selection_status"] == "BOTH_WRONG").sum())
    pipe_wrong = int((~g["proposed_multi_agent_correct"]).sum())
    agent_wrong = int((~g["judge_answer_a_correct"]).sum() + (~g["judge_answer_b_correct"]).sum())
    return gn, both_wrong, pipe_wrong, agent_wrong

dis  = valid[valid["semantic_disagreement"]]
agr  = valid[~valid["semantic_disagreement"]]

dn, d_bw, d_pw, d_aw = group_stats(dis)
an, a_bw, a_pw, a_aw = group_stats(agr)

print("\n" + "=" * 64)
print("CORE FINDING: SEMANTIC DISAGREEMENT vs. WRONG-ANSWER RATE")
print("=" * 64)
print(f"{'Group':<28}{'N':>6}{'BOTH_WRONG':>13}{'Pipeline wrong':>16}{'Agent wrong':>14}")
print("-" * 77)
print(f"{'Disagreement = TRUE':<28}{dn:>6}{pct(d_bw,dn):>13}{pct(d_pw,dn):>16}{pct(d_aw,2*dn):>14}")
print(f"{'Disagreement = FALSE':<28}{an:>6}{pct(a_bw,an):>13}{pct(a_pw,an):>16}{pct(a_aw,2*an):>14}")
print("-" * 77)
print(f"{'ALL valid':<28}{n:>6}{pct(d_bw+a_bw,n):>13}{pct(d_pw+a_pw,n):>16}{pct(d_aw+a_aw,2*n):>14}")

# Relative risk on the BOTH_WRONG (consensus-hallucination) measure
if dn and an and a_bw:
    rr_bw = (d_bw/dn) / (a_bw/an)
    print(f"\nBOTH_WRONG relative risk (disagree / agree): {rr_bw:.2f}x")
if dn and an and a_pw:
    rr_pw = (d_pw/dn) / (a_pw/an)
    print(f"Pipeline-wrong relative risk (disagree / agree): {rr_pw:.2f}x")

# ---------------------------------------------------------------------------
# FAIR HEAD-TO-HEAD: pipeline vs. Qwen (Judge-graded)
# ---------------------------------------------------------------------------
# Pending-aware: the Qwen baseline may be filled in a separate later pass
# (qwen_pass.py). Rows with an empty qwen_answer have NOT had the baseline run yet, so
# the head-to-head is only valid once every valid row has a Qwen answer. Until then we
# print the pipeline number (which never depends on Qwen) and suppress the comparison,
# rather than dividing by empty cells and reporting a misleadingly large gap.
# NaN-safe: pandas 3.0's .astype(str) preserves NaN (so NaN.ne("nan") is True), so use
# .notna() as the primary test and treat blank / literal-"nan" text as not-filled too.
raw_ans        = valid["qwen_answer"]
ans_str        = raw_ans.fillna("").astype(str).str.strip()
attempted_mask = raw_ans.notna() & ans_str.ne("") & ans_str.str.lower().ne("nan")
qa             = int(attempted_mask.sum())     # rows where the baseline actually ran
pending        = n - qa

pipe_correct = int(valid["proposed_multi_agent_correct"].sum())

print("\n" + "=" * 64)
print("FAIR HEAD-TO-HEAD (Judge-graded)")
print("=" * 64)
print(f"Proposed multi-agent correctness:        {pct(pipe_correct,n)}  ({pipe_correct}/{n})")

if pending > 0:
    print(f"Qwen baseline:                           PENDING -- {pending} of {n} rows have no")
    print(f"                                         Qwen answer yet. Run qwen_pass.py, then")
    print(f"                                         re-run this script for the comparison.")
    if qa:
        print(f"                                         ({qa} row(s) filled so far -- partial, biased subset; not reported)")
else:
    not_err  = ~ans_str.str.startswith("ERROR:")
    qwen_judged = int((valid["qwen_correct_judged"] & attempted_mask & not_err).sum())
    qwen_sim    = int((valid["qwen_correct"]        & attempted_mask & not_err).sum())
    print(f"Qwen baseline (Judge-graded, fair):      {pct(qwen_judged,n)}  ({qwen_judged}/{n})")
    print(f"Qwen baseline (similarity, legacy):      {pct(qwen_sim,n)}  ({qwen_sim}/{n})")
    diff = (pipe_correct - qwen_judged) / n * 100
    print(f"Performance difference (fair):           {diff:+.2f} pp")
