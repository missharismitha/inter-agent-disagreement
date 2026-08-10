"""
Judge reliability assessment -- LOCAL ONLY, NO API CALLS.

Establishes the reliability of the Llama-4 Scout judge across the full 817-question
dataset without re-running any model on Groq. Four analyses:

  1. Convergent validity: does the LLM judge agree with an INDEPENDENT automatic grader
     built from TruthfulQA's own correct_answers / incorrect_answers sets via local
     sentence embeddings? (agreement %, Cohen's kappa, confusion matrix)
  2. Internal consistency: does the judge's structured output obey its own rules?
  3. Operational reliability: parse-success / fallback / JUDGE_ERROR rates.
  4. Stratified human-audit sample (oversampling judge-vs-reference disagreements).

Inputs : results.csv, offline TruthfulQA cache, all-MiniLM-L6-v2 (local).
Outputs: prints tables; writes judge_audit_sample.csv and _reliability_stats.json.

Run: python judge_reliability.py
"""
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import warnings
import logging

import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import cohen_kappa_score, confusion_matrix

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

CSV = "results.csv"


def as_bool(series):
    return series.astype(str).str.strip().str.lower().eq("true")


def pct(num, den):
    return f"{num/den*100:.1f}%" if den else "N/A"


def main():
    print("Loading results.csv ...")
    df = pd.read_csv(CSV)
    N = len(df)

    print("Loading TruthfulQA reference answers (offline) ...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    refs = {r["question"]: (list(r["correct_answers"]), list(r["incorrect_answers"])) for r in ds}

    print("Loading embedding model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded.\n")

    # ---------------------------------------------------------------------
    # Build an INDEPENDENT reference label for each candidate answer:
    #   correct iff max_sim(answer, correct_answers) > max_sim(answer, incorrect_answers)
    # The reference grader is embedding-based; the judge is LLM-based -> independent.
    # ---------------------------------------------------------------------
    def ref_label(answer, q):
        a = str(answer)
        if not a or a.startswith("ERROR:") or a.strip().lower() == "nan":
            return None
        cors, incs = refs.get(q, ([], []))
        if not cors or not incs:
            return None
        emb_a = model.encode(a, convert_to_tensor=True)
        emb_c = model.encode(cors, convert_to_tensor=True)
        emb_i = model.encode(incs, convert_to_tensor=True)
        max_c = float(util.cos_sim(emb_a, emb_c).max())
        max_i = float(util.cos_sim(emb_a, emb_i).max())
        return max_c > max_i  # True = reference says correct

    print("Computing independent reference labels for A / B / Qwen answers ...")
    judge_lab, ref_lab, who, rowidx = [], [], [], []
    for i, r in df.iterrows():
        q = r["question"]
        for col_ans, col_judge, tag in [
            ("answer_a", "judge_answer_a_correct", "agentA"),
            ("answer_b", "judge_answer_b_correct", "agentB"),
            ("qwen_answer", "qwen_correct_judged", "qwen"),
        ]:
            rl = ref_label(r[col_ans], q)
            if rl is None:
                continue
            jl = str(r[col_judge]).strip().lower() == "true"
            judge_lab.append(jl); ref_lab.append(rl); who.append(tag); rowidx.append(i)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{N} questions embedded")

    rel = pd.DataFrame({"row": rowidx, "who": who, "judge": judge_lab, "ref": ref_lab})

    def block(mask, name):
        sub = rel[mask]
        n = len(sub)
        if n == 0:
            return None
        agree = int((sub.judge == sub.ref).sum())
        kappa = cohen_kappa_score(sub.judge, sub.ref) if sub.judge.nunique() > 1 and sub.ref.nunique() > 1 else float("nan")
        cm = confusion_matrix(sub.ref, sub.judge, labels=[True, False])  # rows=ref, cols=judge
        return dict(n=n, agree=agree, agree_pct=round(agree / n * 100, 1),
                    kappa=round(float(kappa), 3) if kappa == kappa else None,
                    cm=cm.tolist())

    overall = block(rel.index.notnull(), "overall")
    agents  = block(rel.who.isin(["agentA", "agentB"]), "agents")
    qwen    = block(rel.who == "qwen", "qwen")

    print("\n" + "=" * 70)
    print("1. CONVERGENT VALIDITY  (LLM judge vs independent embedding reference)")
    print("=" * 70)
    print("Reference grader: TruthfulQA correct_answers vs incorrect_answers (local embeddings).")
    print("Independent of the judge (different mechanism) -> agreement is real evidence.\n")
    hdr = f"{'Group':<10}{'N':>7}{'Agree%':>9}{'Kappa':>8}"
    print(hdr); print("-" * len(hdr))
    for label, blk in [("Overall", overall), ("Agents", agents), ("Qwen", qwen)]:
        if blk:
            print(f"{label:<10}{blk['n']:>7}{blk['agree_pct']:>8}%{str(blk['kappa']):>8}")
    print("\nConfusion matrix (overall)  rows = reference, cols = judge:")
    cm = overall["cm"]
    print(f"                 judge=correct  judge=wrong")
    print(f"  ref=correct    {cm[0][0]:>10}    {cm[0][1]:>10}")
    print(f"  ref=wrong      {cm[1][0]:>10}    {cm[1][1]:>10}")
    print("\nNote: the embedding reference is itself imperfect, so this is CONVERGENT VALIDITY,")
    print("not gold-standard accuracy. Disagreements localize cases for the human audit (#4).")

    # ---------------------------------------------------------------------
    # 2. Internal consistency / rule adherence (results.csv only)
    # ---------------------------------------------------------------------
    a = as_bool(df.judge_answer_a_correct)
    b = as_bool(df.judge_answer_b_correct)
    st = df.judge_selection_status.astype(str)
    sel = df.judge_selected_answer.astype(str).str.upper()
    fac = as_bool(df.judge_factual_disagreement)
    pc = as_bool(df.proposed_multi_agent_correct)
    fa = df.final_answer.astype(str)

    judged = st != "JUDGE_ERROR"
    exp_status = pd.Series("BOTH_WRONG", index=df.index)
    exp_status[a & ~b] = "A_CORRECT"
    exp_status[~a & b] = "B_CORRECT"
    exp_status[a & b] = "BOTH_CORRECT"
    v_status = judged & (st != exp_status)
    exp_sel = exp_status.map({"A_CORRECT": "A", "B_CORRECT": "B", "BOTH_CORRECT": "BOTH", "BOTH_WRONG": "NONE"})
    v_sel = judged & (sel != exp_sel)
    v_fac = judged & (fac != (a != b))
    exp_pc = judged & exp_status.isin(["A_CORRECT", "B_CORRECT", "BOTH_CORRECT"])
    v_pc = judged & (pc != exp_pc)
    njudged = int(judged.sum())
    any_v = (v_status | v_sel | v_fac | v_pc) & judged
    consistent = njudged - int(any_v.sum())

    print("\n" + "=" * 70)
    print("2. INTERNAL CONSISTENCY  (judge JSON obeys its own rules)")
    print("=" * 70)
    print(f"Judged rows: {njudged}")
    print(f"  fully consistent          : {consistent} ({pct(consistent, njudged)})")
    print(f"  status<->verdict mismatch : {int(v_status.sum())}")
    print(f"  selected_answer mismatch  : {int(v_sel.sum())}")
    print(f"  factual_disagreement calc : {int(v_fac.sum())}")
    print(f"  pipeline_correct mapping  : {int(v_pc.sum())}")

    # ---------------------------------------------------------------------
    # 3. Operational reliability
    # ---------------------------------------------------------------------
    reason = df.judge_reason.astype(str)
    fallback = int(reason.str.contains("Fallback semantic similarity", case=False, na=False).sum())
    judge_err = int((st == "JUDGE_ERROR").sum())
    parsed_ok = njudged - fallback
    print("\n" + "=" * 70)
    print("3. OPERATIONAL RELIABILITY")
    print("=" * 70)
    print(f"  rows with a judge verdict : {njudged}/{N}")
    print(f"  clean JSON parse          : {parsed_ok} ({pct(parsed_ok, njudged)})")
    print(f"  similarity fallback used  : {fallback} ({pct(fallback, njudged)})")
    print(f"  JUDGE_ERROR               : {judge_err} ({pct(judge_err, N)})")

    # ---------------------------------------------------------------------
    # 4. Stratified human-audit sample (oversample disagreements)
    # ---------------------------------------------------------------------
    cat_map = json.load(open("_cat_map.json")) if os.path.exists("_cat_map.json") else {}
    df["_cat"] = df.question.map(cat_map)
    # per-question: did the reference disagree with the judge on ANY of A/B/Qwen?
    rel["disagree"] = rel.judge != rel.ref
    dis_rows = set(rel[rel.disagree].row)
    df["_has_disagree"] = df.index.isin(dis_rows)

    def trim(x, n=160):
        s = str(x).replace("\n", " ")
        return s[:n] + ("..." if len(s) > n else "")

    sample = []
    seen = set()
    # 1) oversample disagreement rows, spread across statuses
    for status in ["A_CORRECT", "B_CORRECT", "BOTH_CORRECT", "BOTH_WRONG"]:
        pool = df[(df.judge_selection_status == status) & df._has_disagree]
        for i in pool.index[:5]:
            if i not in seen:
                seen.add(i); sample.append(i)
    # 2) fill to ~30 with agreement rows spread across statuses
    for status in ["A_CORRECT", "B_CORRECT", "BOTH_CORRECT", "BOTH_WRONG"]:
        pool = df[(df.judge_selection_status == status) & ~df._has_disagree]
        for i in pool.index[:3]:
            if i not in seen and len(sample) < 30:
                seen.add(i); sample.append(i)

    audit = []
    for i in sample:
        r = df.loc[i]
        cors = refs.get(r.question, ([], []))[0]
        audit.append({
            "row": i, "category": r._cat, "question": trim(r.question, 120),
            "answer_a": trim(r.answer_a), "answer_b": trim(r.answer_b),
            "qwen_answer": trim(r.qwen_answer),
            "ground_truth": trim(r.ground_truth, 120),
            "correct_answers": trim(" | ".join(cors), 200),
            "judge_a_correct": r.judge_answer_a_correct,
            "judge_b_correct": r.judge_answer_b_correct,
            "qwen_correct_judged": r.qwen_correct_judged,
            "judge_status": r.judge_selection_status,
            "judge_reason": trim(r.judge_reason, 200),
            "ref_disagreed_somewhere": bool(r._has_disagree),
        })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv("judge_audit_sample.csv", index=False, encoding="utf-8")
    print("\n" + "=" * 70)
    print("4. HUMAN-AUDIT SAMPLE")
    print("=" * 70)
    print(f"  wrote judge_audit_sample.csv  ({len(audit_df)} rows; "
          f"{int(audit_df.ref_disagreed_somewhere.sum())} are judge-vs-reference disagreements)")

    # ---------------------------------------------------------------------
    # Persist stats for the report generator
    # ---------------------------------------------------------------------
    stats = {
        "overall": overall, "agents": agents, "qwen": qwen,
        "consistency_njudged": njudged, "consistency_consistent": consistent,
        "consistency_pct": round(consistent / njudged * 100, 1) if njudged else None,
        "v_status": int(v_status.sum()), "v_sel": int(v_sel.sum()),
        "v_fac": int(v_fac.sum()), "v_pc": int(v_pc.sum()),
        "op_parsed_ok": parsed_ok, "op_fallback": fallback, "op_judge_err": judge_err,
        "op_njudged": njudged, "N": N,
        "audit_rows": len(audit_df),
        "audit_disagreements": int(audit_df.ref_disagreed_somewhere.sum()),
    }
    json.dump(stats, open("_reliability_stats.json", "w"), indent=0)
    print("\nWrote _reliability_stats.json")


if __name__ == "__main__":
    main()
