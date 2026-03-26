"""
Diagnose PAWM on a single case.
Usage: python diagnose_pawm.py --idx 118
"""
import argparse, csv, random, numpy as np, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import utils, probs as probs_module

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=118)
parser.add_argument("--no_pawm", action="store_true")
args = parser.parse_args()

random.seed(42); np.random.seed(42)
utils.set_global_seed(42); probs_module.set_global_seed(42)

# Load data
rows = list(csv.reader(open('../benchmarks/full_data_formatted/BigToM_backward_belief_false_belief_stories.csv')))
gt_text = rows[args.idx + 1][3].strip()
choices_raw = [c.strip() for c in rows[args.idx + 1][2].replace('[','').replace(']','').split(';')]

from DataLoader import load_full_dataset
data = load_full_dataset("BigToM_bbfb")
story, question, choices, correct_answer = data[args.idx]

print("=" * 70)
print(f"DIAGNOSE: idx={args.idx}, PAWM={'OFF' if args.no_pawm else 'ON'}")
print("=" * 70)
print(f"Story: {story}")
print(f"Question: {question}")
print(f"Choices: {choices}")
print(f"GT text: {gt_text}")
print(f"GT letter: {correct_answer}")
print()

# Monkey-patch apply_pawm to log everything
import pawm as pawm_module
original_apply = pawm_module.apply_pawm

def verbose_apply_pawm(time_variables, story, inf_agent_name, llm):
    print("\n" + "="*50)
    print("[DIAGNOSE] apply_pawm called")
    print(f"[DIAGNOSE] inf_agent_name: {inf_agent_name}")
    print(f"[DIAGNOSE] len(time_variables): {len(time_variables)}")
    for i, tv in enumerate(time_variables):
        if "State" in tv:
            print(f"[DIAGNOSE] timestep {i} State BEFORE: {tv['State'].possible_values}")
            print(f"[DIAGNOSE] timestep {i} State is_observed: {tv['State'].is_observed}")
    print("="*50 + "\n")
    result = original_apply(time_variables, story, inf_agent_name, llm)
    print("\n" + "="*50)
    print(f"[DIAGNOSE] apply_pawm returned: {result}")
    for i, tv in enumerate(time_variables):
        if "State" in tv:
            print(f"[DIAGNOSE] timestep {i} State AFTER: {tv['State'].possible_values}")
    print("="*50 + "\n")
    return result

pawm_module.apply_pawm = verbose_apply_pawm

from ProbSolver import ProblemSolver

def argmax(lst): return lst.index(max(lst))
def letter_to_num(l): return {'A':0,'B':1,'C':2,'D':3}.get(l.strip(), 0)

solver = ProblemSolver(
    story=story, question=question, choices=choices, K=1,
    assigned_model=["State", "Observation", "Belief", "Action", "Goal"],
    model_name="automated",
    episode_name=f"diag_{'off' if args.no_pawm else 'on'}_{args.idx}",
    llm="gpt-4o", verbose=False,
    dataset_name="BigToM_bbfb",
    hypo_method="guided", nested=None,
    answerfunc=argmax, back_inference=True,
    reduce_hypotheses=True, seed=42,
    use_pawm=not args.no_pawm,
)

final_probs, _ = solver.solve()

print("\n" + "="*70)
print("RESULT")
print("="*70)
pred_idx = argmax(final_probs)
gt_idx = letter_to_num(correct_answer)
print(f"Probs: {[round(p,3) for p in final_probs]}")
print(f"Predicted: {choices[pred_idx]}")
print(f"GT:        {choices[gt_idx]}")
print(f"{'✓ CORRECT' if pred_idx == gt_idx else '✗ WRONG'}")
