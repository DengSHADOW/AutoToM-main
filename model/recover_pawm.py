"""
Recover PAWM run results from saved probs files.
Run: python recover_pawm.py
"""
import pandas as pd, os, csv, re, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ERROR_INDICES = [
    28,68,150,95,118,8,55,22,109,121,130,132,138,142,161,
    194,199,20,10,145,129,76,43,134,67,12,14,44,47,48,49,
    60,70,90,92,96,112,125,128,136,148,149,151,158,167,168,188,191,
]

# Load GT answer text from CSV
rows = list(csv.reader(open('../benchmarks/full_data_formatted/BigToM_backward_belief_false_belief_stories.csv')))
gt_text = {i: row[3].strip().rstrip('.') for i, row in enumerate(rows[1:])}

def parse_probs(s):
    nums = re.findall(r'[\d.eE+-]+', str(s))
    return [float(x) for x in nums if x not in ('e', 'E')]

correct = 0
done = 0
missing = []

for rank, idx in enumerate(ERROR_INDICES):
    fpath = f'../results/probs/automated_pawm48_on_{idx}.csv'
    if not os.path.exists(fpath):
        missing.append((rank + 1, idx))
        continue

    df = pd.read_csv(fpath)
    col = df.columns[-1]
    choices_in_col = re.findall(r"'([^']+)'", col)
    probs = parse_probs(df.iloc[-1][col])

    if len(probs) < 2 or len(choices_in_col) < 2:
        print(f'  rank={rank+1:02d} idx={idx:3d} PARSE ERROR col={col}')
        missing.append((rank + 1, idx))
        continue

    pred_idx = probs.index(max(probs))
    predicted_text = choices_in_col[pred_idx].rstrip('.')
    gt = gt_text[idx]
    ok = (predicted_text == gt)

    if ok:
        correct += 1
    done += 1
    status = 'OK' if ok else 'WRONG'
    print(f'  rank={rank+1:02d} idx={idx:3d} [{status}] probs={[round(p,3) for p in probs]}')

print()
print(f'Recovered: {correct}/{done} correct ({100*correct/done:.1f}%)')
print(f'Missing (not yet run): ranks {[r for r,i in missing]}, indices {[i for r,i in missing]}')
if missing:
    print(f'Resume with: python eval_pawm_48.py --start_rank {missing[0][0]} --prev_correct {correct}')
