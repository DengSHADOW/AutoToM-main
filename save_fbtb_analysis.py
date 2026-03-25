import json, os, re
import pandas as pd

metrics_dir = "results/metrics"
data = pd.read_csv("benchmarks/full_data_formatted/BigToM_forward_belief_true_belief_stories.csv")

# deduplicate: for each idx keep the file without _N suffix (original run)
idx_to_file = {}
for f in os.listdir(metrics_dir):
    if "fbtb" not in f:
        continue
    m = re.search(r"fbtb_(\d+)_", f)
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
nw = len(wrong_cases)

lines = []
lines.append("=" * 70)
lines.append("BigToM Forward Belief True-Belief (fbtb) - Full Analysis")
lines.append("=" * 70)
missing = sorted(set(range(200)) - set(idx_to_file.keys()))
missing_str = f"  (missing: {missing})" if missing else "  (all 200 complete)"
lines.append(f"Total questions with results : {total}/200{missing_str}")
lines.append(f"Correct                      : {len(correct_cases)} ({len(correct_cases)/total*100:.1f}%)")
lines.append(f"Wrong                        : {len(wrong_cases)} ({len(wrong_cases)/total*100:.1f}%)")
lines.append("")
lc_c = [r["lc"] for r in correct_cases]
lc_w = [r["lc"] for r in wrong_cases] if wrong_cases else []
lines.append("Confidence analysis:")
lines.append(f"  Avg P(correct answer) when model is RIGHT : {sum(lc_c)/len(lc_c):.3f}")
if lc_w:
    lines.append(f"  Avg P(correct answer) when model is WRONG : {sum(lc_w)/len(lc_w):.3f}")
lines.append("")
lines.append(f"Distribution of P(correct answer) among {nw} wrong cases:")
bins = [("<0.10",     0,   0.1),
        ("0.10-0.30", 0.1, 0.3),
        ("0.30-0.50", 0.3, 0.5),
        (">0.50",     0.5, 1.1)]
for label, lo, hi in bins:
    count = sum(1 for r in wrong_cases if lo <= r["lc"] < hi)
    lines.append(f"  {label:12s} : {count:2d} cases  {'|' * count}")
lines.append("")
lines.append("Cost / time:")
lines.append(f"  Avg cost per question : ${sum(r['total_cost'] for r in results)/total:.4f}")
lines.append(f"  Avg time per question : {sum(r['total_time'] for r in results)/total:.1f}s")

lines.append("")
lines.append("=" * 70)
lines.append("Context: fbtb as control condition")
lines.append("=" * 70)
lines.append("fbtb = agent has TRUE belief (belief matches current world state).")
lines.append("AutoToM tends to assign the true state as the agent's belief, so")
lines.append("fbtb should be the easiest condition. High accuracy here confirms")
lines.append("that errors in fbfb/bbfb are specifically due to the information")
lines.append("asymmetry bug, not general model failure.")
lines.append("")
lines.append("Comparison across datasets:")
lines.append("  bbfb (backward, false belief) : 152/200 = 76.0%")
lines.append("  fbfb (forward,  false belief) : 191/200 = 95.5%")
lines.append(f"  fbtb (forward,  true  belief) : {len(correct_cases)}/{total} = {len(correct_cases)/total*100:.1f}%  <-- this run")

lines.append("")
lines.append("=" * 70)
lines.append(f"All {nw} Wrong Cases (sorted by P(correct answer) ascending)")
lines.append("=" * 70)
for i, r in enumerate(wrong_cases, 1):
    lines.append(f"\n[{i:02d}/{nw}]  idx={r['idx']}  P(correct)={r['lc']:.3f}  probs={[round(p,3) for p in r['probs']]}")
    lines.append(f"  Story    : {r['story'].strip()}")
    lines.append(f"  Question : {r['question'].strip()}")
    lines.append(f"  GT       : {r['gt']}")
    lines.append(f"  Predicted: {r['predicted_wrong']}")

lines.append("")
lines.append("=" * 70)
lines.append(f"All {len(correct_cases)} Correct Cases - idx list")
lines.append("=" * 70)
lines.append(str(sorted(r["idx"] for r in correct_cases)))

output = "\n".join(lines)

out_path = "fbtb_analysis_results.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)

print(f"Saved to {out_path}")
print(output[:1200])
