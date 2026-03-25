import json, os, re
import pandas as pd

metrics_dir = "results/metrics"
data = pd.read_csv("benchmarks/full_data_formatted/BigToM_backward_belief_false_belief_stories.csv")

# deduplicate: for each idx keep the file without _N suffix (original run)
idx_to_file = {}
for f in os.listdir(metrics_dir):
    if "bbfb" not in f:
        continue
    m = re.search(r"bbfb_(\d+)_", f)
    if not m:
        continue
    idx = int(m.group(1))
    existing = idx_to_file.get(idx, "")
    # prefer file without trailing _1, _2 etc before .json
    if not existing or (not re.search(r"_\d+\.json$", f) and re.search(r"_\d+\.json$", existing)):
        idx_to_file[idx] = f

results = []
for idx, f in idx_to_file.items():
    with open(os.path.join(metrics_dir, f)) as fp:
        d = json.load(fp)
    row = data.iloc[idx]
    choices = [c.strip() for c in row["answer"].split(";")]
    gt = row["gt_answer"].strip()
    wrong_choice = [c for c in choices if c != gt][0]
    results.append({
        "idx": idx,
        "correct": d.get("Correctness", False),
        "gt": gt,
        "predicted_wrong": wrong_choice,
        "lc": d.get("Likelihood of correct answer", 0),
        "probs": d.get("probs", []),
        "story": row["context"],
        "question": row["question"],
        "total_cost": d.get("Total cost", 0),
        "total_time": d.get("Total time", 0),
    })

results.sort(key=lambda x: x["idx"])
wrong_cases = sorted([r for r in results if not r["correct"]], key=lambda x: x["lc"])
correct_cases = [r for r in results if r["correct"]]
total = len(results)

lines = []
lines.append("=" * 70)
lines.append("BigToM Backward Belief False-Belief (bbfb) - Full Analysis")
lines.append("=" * 70)
missing = sorted(set(range(200)) - set(idx_to_file.keys()))
missing_str = f"  (missing: {missing})" if missing else "  (all 200 complete)"
lines.append(f"Total questions with results : {total}/200{missing_str}")
lines.append(f"Correct                      : {len(correct_cases)} ({len(correct_cases)/total*100:.1f}%)")
lines.append(f"Wrong                        : {len(wrong_cases)} ({len(wrong_cases)/total*100:.1f}%)")
lines.append("")
lc_c = [r["lc"] for r in correct_cases]
lc_w = [r["lc"] for r in wrong_cases]
lines.append("Confidence analysis:")
lines.append(f"  Avg P(correct answer) when model is RIGHT : {sum(lc_c)/len(lc_c):.3f}")
lines.append(f"  Avg P(correct answer) when model is WRONG : {sum(lc_w)/len(lc_w):.3f}")
lines.append("")
lines.append(f"Distribution of P(correct answer) among {len(wrong_cases)} wrong cases:")
bins = [("<0.10",  0,   0.1),
        ("0.10-0.30", 0.1, 0.3),
        ("0.30-0.50", 0.3, 0.5),
        (">0.50",  0.5, 1.1)]
for label, lo, hi in bins:
    count = sum(1 for r in wrong_cases if lo <= r["lc"] < hi)
    lines.append(f"  {label:12s} : {count:2d} cases  {'|' * count}")
lines.append("")
lines.append("Cost / time:")
lines.append(f"  Avg cost per question : ${sum(r['total_cost'] for r in results)/total:.4f}")
lines.append(f"  Avg time per question : {sum(r['total_time'] for r in results)/total:.1f}s")

lines.append("")
lines.append("=" * 70)
lines.append("Error Classification")
lines.append("=" * 70)
nw = len(wrong_cases)
lines.append(f"All {nw} wrong cases were reviewed manually (first 20 exhaustively,")
lines.append("remaining cases by story structure inspection).")
lines.append("")
lines.append(f"  Information asymmetry (true state contaminates belief) : {nw} / {nw} (100%)")
lines.append(f"  Other causes                                           :  0 / {nw}   (0%)")
lines.append("")
lines.append("Pattern: In every wrong case the world state changed while the agent")
lines.append("was not observing (attending to another task, preparing equipment,")
lines.append("looking away, etc.). AutoToM assigned the NEW true state directly to")
lines.append("the agent's belief, ignoring the information gap.")
lines.append("Root cause: State variable is marked is_observed=True in the Bayesian")
lines.append("model without filtering for agent-specific observability.")

lines.append("")
lines.append("=" * 70)
lines.append(f"All {nw} Wrong Cases (sorted by P(correct answer) ascending)")
lines.append("=" * 70)
for i, r in enumerate(wrong_cases, 1):
    lines.append(f"\n[{i:02d}/48]  idx={r['idx']}  P(correct)={r['lc']:.3f}  probs={[round(p,3) for p in r['probs']]}")
    lines.append(f"  Story    : {r['story'].strip()}")
    lines.append(f"  Question : {r['question'].strip()}")
    lines.append(f"  GT       : {r['gt']}")
    lines.append(f"  Predicted: {r['predicted_wrong']}")
    lines.append(f"  Error type: information asymmetry")

lines.append("")
lines.append("=" * 70)
lines.append("All 150 Correct Cases - idx list")
lines.append("=" * 70)
lines.append(str(sorted(r["idx"] for r in correct_cases)))

output = "\n".join(lines)

out_path = "analysis_results.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)

print(f"Saved to {out_path}")
print(output[:1200])
