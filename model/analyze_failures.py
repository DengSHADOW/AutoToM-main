"""
Analyze PAWM failures: cross-reference probs files (on vs off) to classify each case.
"""
import os, csv, re, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

ERROR_INDICES = [
    28, 68, 150, 95, 118, 8, 55, 22, 109, 121, 130, 132, 138, 142, 161,
    194, 199, 20, 10, 145, 129, 76, 43, 134, 67, 12, 14, 44, 47, 48, 49,
    60, 70, 90, 92, 96, 112, 125, 128, 136, 148, 149, 151, 158, 167, 168,
    188, 191,
]

rows = list(csv.reader(open('../benchmarks/full_data_formatted/BigToM_backward_belief_false_belief_stories.csv')))
gt_text   = {i: row[3].strip().rstrip('.') for i, row in enumerate(rows[1:])}
story_col = {i: row[0].strip()             for i, row in enumerate(rows[1:])}
question_col = {i: row[1].strip()          for i, row in enumerate(rows[1:])}

def parse_probs(s):
    nums = re.findall(r'[\d.eE+-]+', str(s))
    return [float(x) for x in nums if x not in ('e', 'E')]

def read_result(prefix, idx):
    fpath = f'../results/probs/automated_{prefix}_{idx}.csv'
    if not os.path.exists(fpath):
        return None, None
    df = pd.read_csv(fpath)
    col = df.columns[-1]
    choices = re.findall(r"'([^']+)'", col)
    probs = parse_probs(df.iloc[-1][col])
    if len(probs) < 2 or len(choices) < 2:
        return None, None
    pred_idx = probs.index(max(probs))
    return choices[pred_idx].rstrip('.'), probs

on_correct = on_wrong = on_missing = 0
off_correct = off_wrong = off_missing = 0

categories = {
    "pawm_fixed": [],       # wrong without, correct with
    "pawm_hurt":  [],       # correct without, wrong with
    "both_wrong": [],       # wrong in both
    "both_correct": [],     # correct in both (regression check)
    "no_on_data": [],       # PAWM on result missing (cached elsewhere)
}

print(f"{'idx':>4}  {'OFF':^8} {'ON':^8}  {'GT text (truncated)':40}")
print("-" * 70)

for idx in ERROR_INDICES:
    gt = gt_text[idx]
    pred_off, probs_off = read_result('pawm48_off', idx)
    pred_on,  probs_on  = read_result('pawm48_on',  idx)

    ok_off = (pred_off is not None and pred_off == gt)
    ok_on  = (pred_on  is not None and pred_on  == gt)

    if pred_on is None:
        cat = "no_on_data"
    elif ok_off and ok_on:
        cat = "both_correct"
    elif not ok_off and ok_on:
        cat = "pawm_fixed"
    elif ok_off and not ok_on:
        cat = "pawm_hurt"
    else:
        cat = "both_wrong"

    categories[cat].append(idx)

    off_str = "✓" if ok_off else ("✗" if pred_off else "?")
    on_str  = "✓" if ok_on  else ("✗" if pred_on  else "?")
    print(f"{idx:>4}  {off_str:^8} {on_str:^8}  {gt[:40]}")

print()
print("=" * 70)
for cat, idxs in categories.items():
    print(f"  {cat:20s}: {len(idxs):2d}  indices={idxs}")

print()
print("Cases where PAWM ON result is missing (cached under different name):")
for idx in categories["no_on_data"]:
    print(f"  idx={idx}: story='{story_col[idx][:80]}...'")
