"""
Evaluate PAWM on the 48 known error cases from BigToM bbfb.

These are the cases where AutoToM (without PAWM) answered wrong,
all due to information asymmetry (agent absent during state change).

Usage:
    python eval_pawm_48.py              # PAWM on, should fix errors
    python eval_pawm_48.py --no_pawm    # baseline reproduction
"""

import sys
import os
import argparse
import numpy as np
from copy import deepcopy
from datetime import datetime


class Tee:
    """Write to both terminal and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# The 48 wrong case indices from analysis_results.txt
ERROR_INDICES = [
    28, 68, 150, 95, 118, 8, 55, 22, 109, 121, 130, 132, 138, 142, 161,
    194, 199, 20, 10, 145, 129, 76, 43, 134, 67, 12, 14, 44, 47, 48, 49,
    60, 70, 90, 92, 96, 112, 125, 128, 136, 148, 149, 151, 158, 167, 168,
    188, 191,
]

def letter_to_number_mapping(letter):
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(letter, 0)

def argmax(lst):
    return lst.index(max(lst))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_pawm", action="store_true", help="Disable PAWM (baseline reproduction).")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_rank", type=int, default=0, help="Resume from this rank (1-based).")
    parser.add_argument("--prev_correct", type=int, default=0, help="Carry over correct count from previous partial run.")
    args = parser.parse_args()

    use_pawm = not args.no_pawm
    mode_str = "pawm_on" if use_pawm else "baseline"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../results/log_pawm48_{mode_str}_{timestamp}.txt"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    tee = Tee(log_path)
    sys.stdout = tee

    print(f"{'='*60}")
    print(f"PAWM Evaluation on 48 Error Cases")
    print(f"Mode: {'PAWM ON' if use_pawm else 'Baseline (no PAWM)'}")
    print(f"Log: {os.path.abspath(log_path)}")
    print(f"{'='*60}\n")

    # Import here so we're in the right working directory context
    from DataLoader import load_full_dataset
    from ProbSolver import ProblemSolver

    import random
    import utils
    import probs
    random.seed(args.seed)
    np.random.seed(args.seed)
    utils.set_global_seed(args.seed)
    probs.set_global_seed(args.seed)

    data = load_full_dataset("BigToM_bbfb")

    correct = args.prev_correct
    results_log = []

    for rank, idx in enumerate(ERROR_INDICES):
        if rank < args.start_rank - 1:
            continue
        story, question, choices, correct_answer = data[idx]

        print(f"\n[{rank+1:02d}/48] idx={idx}")
        print(f"  Story: {story[:120]}...")
        print(f"  Question: {question}")
        print(f"  GT: {correct_answer}")

        solver = ProblemSolver(
            story=story,
            question=question,
            choices=choices,
            K=args.K,
            assigned_model=["State", "Observation", "Belief", "Action", "Goal"],
            model_name="automated",
            episode_name=f"pawm48_{'on' if use_pawm else 'off'}_{idx}",
            llm=args.llm,
            verbose=args.verbose,
            dataset_name="BigToM_bbfb",
            hypo_method="guided",
            nested=None,
            answerfunc=argmax,
            back_inference=True,
            reduce_hypotheses=True,
            seed=args.seed,
            use_pawm=use_pawm,
        )

        try:
            final_probs, _ = solver.solve()
        except Exception as e:
            print(f"  ERROR: {e}")
            results_log.append({"idx": idx, "correct": False, "error": str(e)})
            continue

        if final_probs is None:
            print(f"  SKIPPED (model returned None)")
            results_log.append({"idx": idx, "correct": False, "error": "None result"})
            continue

        answer_idx = argmax(final_probs)
        gt_idx = letter_to_number_mapping(correct_answer)
        is_correct = (answer_idx == gt_idx)

        if is_correct:
            correct += 1

        print(f"  Probs: {[round(p, 3) for p in final_probs]}")
        print(f"  Predicted: {choices[answer_idx]}")
        print(f"  GT answer: {choices[gt_idx]}")
        print(f"  {'✓ CORRECT' if is_correct else '✗ WRONG'}")

        results_log.append({
            "idx": idx,
            "correct": is_correct,
            "probs": final_probs,
            "predicted": choices[answer_idx],
            "gt": choices[gt_idx],
        })

        # Running tally
        done = rank + 1
        print(f"  Running: {correct}/{done} = {100*correct/done:.1f}%")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({'PAWM ON' if use_pawm else 'Baseline'})")
    print(f"{'='*60}")
    print(f"Correct: {correct} / {len(results_log)}")
    if results_log:
        print(f"Accuracy: {100*correct/len(results_log):.1f}%")
    print(f"Baseline (AutoToM alone): 0 / 48 = 0.0%")

    sys.stdout = tee.terminal
    tee.close()
    print(f"\nLog saved to: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
