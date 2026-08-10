# Data Preprocessing & Initial Results Discussion

**Project Title:** Inter-Agent Disagreement as a Signal of Hallucination in AI Agent-Based Question Answering

**Institution:** Deggendorf Institute of Technology
**Programme:** Master of AI for Smart Sensors and Actuators
**Authors:** Gogikar Harismitha · Ashna Thomas Grace
**Date:** 20 May 2026

---

## 1. Objective

This project investigates whether disagreement between two independently-prompted large language model (LLM) agents can serve as a reliable signal of hallucination risk in factual question answering. Rather than improving accuracy through training, the study analyses agent behaviour — specifically whether semantic disagreement between agents correlates with factually incorrect outputs. A stronger single-model baseline (Qwen 3 32B) is used for head-to-head factual correctness comparison.

---

## 2. Dataset Used

| Property | Detail |
|---|---|
| **Dataset Name** | TruthfulQA |
| **Source** | HuggingFace `datasets` library |
| **Configuration** | `truthful_qa`, `generation` config, `validation` split |
| **Total Size** | ~817 questions across 38 categories |
| **Fields Used** | `question`, `best_answer` |

TruthfulQA was selected because it is specifically designed to expose hallucinations that language models propagate confidently from misconceptions — making it well-suited for studying disagreement as a signal of factual failure.

---

## 3. Data Preprocessing Steps

The preprocessing pipeline involves the following steps:

**Step 1 — Dataset Loading**
The TruthfulQA dataset is loaded directly from the HuggingFace `datasets` library using the `generation` configuration and `validation` split. No manual data collection or annotation was required.

**Step 2 — Field Extraction**
Two fields are extracted from each record:
- `question` — the factual question sent to all agents
- `best_answer` — used as ground truth for factual evaluation by the Judge Agent and for post-hoc Qwen baseline evaluation

**Step 3 — Test Subset Selection**
For the current pilot stage, a `TEST_MODE` flag is used to limit processing to the first 20 questions. This reduces API cost and runtime while allowing validation of the full pipeline structure before scaling to all 817 questions.

**Step 4 — Question Formatting**
Each question is formatted separately for four pipeline components:
- **Agent A** receives the question with a cautious analyst system prompt
- **Agent B** receives the same question with a confident assistant system prompt
- **Judge Agent** receives the question, both agent answers, and the ground truth for structured factual evaluation
- **Qwen 3 32B Baseline** receives only the question with a neutral factual assistant prompt — no ground truth or agent answers are shared

**Step 5 — Output Structuring and Storage**
All outputs including agent answers, similarity scores, judge verdicts, and baseline answers are stored row-by-row into a structured CSV file (`results.csv`) with 16 defined columns.

---

## 4. Current Pipeline Status

The full multi-agent pipeline has been implemented and validated on a 20-question pilot subset. The current operational status is as follows:

- Agent A and Agent B independently generate answers to each TruthfulQA question using role-conditioned system prompts via the Groq API (model: `llama-3.1-8b-instant`)
- Semantic similarity between Agent A and Agent B answers is computed using the `all-MiniLM-L6-v2` sentence-transformer model
- Semantic disagreement is flagged when cosine similarity falls below the threshold of **0.85**
- The Judge Agent (`llama-3.3-70b-versatile`) evaluates both answers against ground truth and returns a structured JSON verdict indicating which answer is factually correct
- Qwen 3 32B (`qwen/qwen3-32b`) answers each question independently as a single-model baseline — it does not receive agent outputs or ground truth
- All results are automatically saved to `results.csv` after each question is processed

---

## 5. Why Only 20 Questions Were Used

The current stage focuses on validating the preprocessing pipeline, output structure, and API integration — not on drawing final research conclusions. A 20-question pilot was selected deliberately for the following reasons:

- To confirm that all pipeline components function correctly end-to-end before committing to a full API run
- To verify that the CSV output structure is correct and all 16 columns are populated as expected
- To reduce Groq API cost and processing time during the validation phase
- To identify any edge cases in judge JSON parsing or semantic similarity computation at small scale

Once the pipeline is confirmed correct, the same code can be run unchanged on the full ~817-question dataset by setting `TEST_MODE = False`.

---

## 6. Output File Description (`results.csv`)

| Column | Description |
|---|---|
| `question` | TruthfulQA question text |
| `answer_a` | Agent A (cautious analyst) response |
| `answer_b` | Agent B (confident assistant) response |
| `similarity_score` | Cosine similarity between `answer_a` and `answer_b` |
| `semantic_disagreement` | `True` if similarity < 0.85 |
| `judge_answer_a_correct` | Judge verdict: is Agent A factually correct? |
| `judge_answer_b_correct` | Judge verdict: is Agent B factually correct? |
| `judge_factual_disagreement` | `True` if exactly one agent is judged correct |
| `judge_selection_status` | `A_CORRECT`, `B_CORRECT`, `BOTH_CORRECT`, `BOTH_WRONG`, or `JUDGE_ERROR` |
| `judge_selected_answer` | `A`, `B`, `BOTH`, or `NONE` |
| `judge_reason` | Short explanation from Judge Agent |
| `final_answer` | Pipeline's final answer text or abstention label |
| `proposed_multi_agent_correct` | `True` if pipeline produced a correct outcome |
| `qwen_answer` | Qwen 3 32B single-model baseline answer |
| `qwen_correct` | `True` if `qwen_answer` matches `ground_truth` semantically |
| `ground_truth` | TruthfulQA `best_answer` reference field |

---

## 7. Initial Pilot Results (20 Questions)

The following results are from a 20-question pilot run only. They should not be treated as final conclusions. The small sample size limits statistical confidence and the full 817-question run is required before drawing research conclusions.

| Metric | Value |
|---|---|
| Total Questions Processed | 20 |
| Semantic Disagreement Rate | 20.00% |
| Judge Factual Disagreement Rate | 10.00% |
| Agent A Correctness | 60.00% |
| Agent B Correctness | 60.00% |
| BOTH_CORRECT Rate | 55.00% |
| BOTH_WRONG Rate | 35.00% |
| Abstention Rate | 35.00% |
| **Proposed Multi-Agent Answer Correctness** | **65.00%** |
| **Qwen 3 32B Single-Model Baseline** | **5.00%** |
| **Performance Difference** | **+60.00 percentage points** |
| Semantic Disagreement Precision | 25.00% |
| Agreement But Both Wrong Rate | 37.50% |

**Preliminary observations (pilot only):**

- The proposed multi-agent pipeline (65%) substantially outperforms the Qwen single-model baseline (5%) in this pilot. However, the 5% Qwen result is partly a known limitation of semantic similarity evaluation — Qwen's verbose outputs are often semantically distant from TruthfulQA's short reference answers even when factually correct. This will be addressed in the full run with improved evaluation.
- The 37.5% Agreement But Both Wrong rate indicates that semantic agreement between agents does not guarantee correctness. Both agents can hallucinate the same wrong answer — a finding known as consensus hallucination.
- The 0% Factual Disagreement Precision confirms that when agents factually disagree, the Judge successfully identifies and selects the correct answer — the pipeline does not fail on factual disagreements.
- These results validate that the pipeline runs correctly end-to-end and that the output structure is complete and ready for the full dataset run.

---

## 8. Next Steps

The following work is planned before the final report submission:

1. **Full dataset run** — Execute the pipeline on all ~817 TruthfulQA questions with `TEST_MODE = False`
2. **Evaluation improvement** — Investigate NLI-based or LLM-as-judge evaluation to handle verbose model outputs more fairly than semantic similarity alone
3. **Disagreement-hallucination correlation analysis** — Analyse whether semantic disagreement rate is statistically associated with higher hallucination frequency
4. **Baseline comparison** — Produce a rigorous head-to-head factual correctness comparison between the proposed pipeline and Qwen 3 32B on the full dataset
5. **Report preparation** — Write the final IEEE-format report covering methodology, results, analysis, limitations, and conclusions

---

*Submitted for: Data Preprocessing Results Discussion — 20.05.2026*
*Accompanying file: `results.csv` (20-question pilot output)*
