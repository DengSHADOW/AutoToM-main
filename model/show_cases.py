"""Show story+question for specific indices."""
import csv, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

TARGET = [8, 150, 118, 48, 151]  # pawm_hurt + both_wrong

rows = list(csv.reader(open('../benchmarks/full_data_formatted/BigToM_backward_belief_false_belief_stories.csv')))
headers = rows[0]

for idx in TARGET:
    row = rows[idx + 1]
    print(f"\n{'='*70}")
    print(f"idx={idx}")
    print(f"Story:    {row[0].strip()}")
    print(f"Question: {row[1].strip()}")
    print(f"GT:       {row[3].strip()}")
