"""
Evaluate PAWM on all 200 BigToM bbfb cases.

Usage:
    python eval_full_200.py               # PAWM on (full evaluation)
    python eval_full_200.py --no_pawm     # baseline (no PAWM)
    python eval_full_200.py --start_idx 50 --prev_correct 43   # resume
"""

import sys
import os
import argparse
import numpy as np

def letter_to_number_mapping(letter):
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(letter, 0)

def argmax(lst):
    return lst.index(max(lst))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_pawm", action="store_true")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0, help="Resume from this dataset index.")
    parser.add_argument("--prev_correct", type=int, default=0, help="Correct count from prior partial run.")
    args = parser.parse_args()

    use_pawm = not args.no_pawm
    print(f"{'='*60}")
    print(f"Full 200-case Evaluation — BigToM bbfb")
    print(f"Mode: {'PAWM ON' if use_pawm else 'Baseline (no PAWM)'}")
    print(f"{'='*60}\n")

    from DataLoader import load_full_dataset
    from ProbSolver import ProblemSolver

    import random, utils, probs as probs_module
    random.seed(args.seed)
    np.random.seed(args.seed)
    utils.set_global_seed(args.seed)
    probs_module.set_global_seed(args.seed)

    data = load_full_dataset("BigToM_bbfb")

    correct = args.prev_correct
    total = 0
    results_log = []

    for idx in range(len(data)):
        if idx < args.start_idx:
            continue

        story, question, choices, correct_answer = data[idx]
        tag = "on" if use_pawm else "off"

        print(f"\n[{idx+1:03d}/200] idx={idx}")
        print(f"  Story: {story[:100]}...")
        print(f"  GT: {correct_answer}")

        solver = ProblemSolver(
            story=story,
            question=question,
            choices=choices,
            K=args.K,
            assigned_model=["State", "Observation", "Belief", "Action", "Goal"],
            model_name="automated",
            episode_name=f"full200_{tag}_{idx}",
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
            total += 1
            continue

        if final_probs is None:
            print(f"  SKIPPED (None result)")
            results_log.append({"idx": idx, "correct": False, "error": "None"})
            total += 1
            continue

        answer_idx = argmax(final_probs)
        gt_idx = letter_to_number_mapping(correct_answer)
        is_correct = (answer_idx == gt_idx)

        if is_correct:
            correct += 1
        total += 1

        print(f"  Probs: {[round(p, 3) for p in final_probs]}")
        print(f"  Predicted: {choices[answer_idx]}")
        print(f"  GT:        {choices[gt_idx]}")
        print(f"  {'✓ CORRECT' if is_correct else '✗ WRONG'}  |  Running: {correct}/{total} = {100*correct/total:.1f}%")

        results_log.append({"idx": idx, "correct": is_correct})

    print(f"\n{'='*60}")
    print(f"FINAL — {'PAWM ON' if use_pawm else 'Baseline'}")
    print(f"{'='*60}")
    print(f"Correct: {correct} / {total}")
    if total:
        print(f"Accuracy: {100*correct/total:.1f}%")

if __name__ == "__main__":
    main()
