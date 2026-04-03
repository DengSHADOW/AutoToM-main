"""
Evaluate AutoToM baseline on Hi-ToM len1 (200 questions across 10 files).

Dataset: 10 files × 20 questions = 200 total
  - HiToM_len1_tell{0,1}_order{0..4}
  (tell1 = story tells agent B about move; tell0 = no_tell)

Usage:
    python eval_hitom_baseline.py                 # all 200 questions
    python eval_hitom_baseline.py --tell_only      # tell subset (100)
    python eval_hitom_baseline.py --no_tell_only   # no_tell subset (100)
    python eval_hitom_baseline.py --start_file 3   # resume from file index
"""

import sys
import os
import argparse
from copy import deepcopy
import numpy as np


def argmax(lst):
    return lst.index(max(lst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_file", type=int, default=0, help="Resume from this file index (0-9).")
    parser.add_argument("--prev_correct", type=int, default=0)
    parser.add_argument("--prev_total", type=int, default=0)
    parser.add_argument("--tell_only", action="store_true", help="Only run tell subset.")
    parser.add_argument("--no_tell_only", action="store_true", help="Only run no_tell subset.")
    args = parser.parse_args()

    from DataLoader import load_full_dataset
    from ProbSolver import ProblemSolver

    import random, utils, probs as probs_module
    random.seed(args.seed)
    np.random.seed(args.seed)
    utils.set_global_seed(args.seed)
    probs_module.set_global_seed(args.seed)

    # Build list of (dataset_name, tell_flag, file_order) to evaluate
    datasets = []
    for tell in [0, 1]:
        for order in range(5):
            ds_name = f"HiToM_len1_tell{tell}_order{order}"
            if args.tell_only and tell == 0:
                continue
            if args.no_tell_only and tell == 1:
                continue
            datasets.append(ds_name)

    print("=" * 60)
    print("Hi-ToM len1 Baseline Evaluation")
    print(f"Datasets to run: {len(datasets)} files × 20 questions")
    print("=" * 60)

    correct = args.prev_correct
    total = args.prev_total
    results_log = []

    for file_idx, dataset_name in enumerate(datasets):
        if file_idx < args.start_file:
            continue

        print(f"\n{'─'*50}")
        print(f"[File {file_idx+1}/{len(datasets)}] {dataset_name}")
        print(f"{'─'*50}")

        data = load_full_dataset(dataset_name)

        for q_idx, (story, question, shuffled_choices, correct_answer_int) in enumerate(data):
            global_idx = file_idx * 20 + q_idx
            orig_choices = deepcopy(shuffled_choices)

            # Filter choices: keep only those appearing in the story (HiToM has 15+ choices)
            filtered_choices = [c for c in orig_choices if c in story]
            if not filtered_choices:
                filtered_choices = orig_choices  # fallback: use all

            # True answer text and its index in filtered choices
            true_answer_word = orig_choices[correct_answer_int]
            gt_id = -1
            for j, c in enumerate(filtered_choices):
                if c == true_answer_word:
                    gt_id = j

            if gt_id == -1:
                # True answer was filtered out — include it back
                filtered_choices.append(true_answer_word)
                gt_id = len(filtered_choices) - 1

            print(f"\n  [{global_idx+1:03d}] Q{q_idx} | {dataset_name}")
            print(f"  Story: {story[:100].strip()}...")
            print(f"  Question: {question}")
            print(f"  GT: {true_answer_word}  (id={gt_id} in {len(filtered_choices)} filtered choices)")

            solver = ProblemSolver(
                story=story,
                question=question,
                choices=filtered_choices,
                K=args.K,
                assigned_model=["State", "Observation", "Belief", "Action", "Goal"],
                model_name="automated",
                episode_name=f"hitom_baseline_{dataset_name}_{q_idx}",
                llm=args.llm,
                verbose=args.verbose,
                dataset_name=dataset_name,
                hypo_method="guided",
                nested=None,
                answerfunc=argmax,
                back_inference=True,
                reduce_hypotheses=True,
                recursion_depth=0,
                seed=args.seed,
                use_pawm=False,
            )

            try:
                final_probs, _ = solver.solve()
            except Exception as e:
                print(f"  ERROR: {e}")
                results_log.append({
                    "dataset": dataset_name, "q_idx": q_idx,
                    "correct": False, "error": str(e)
                })
                total += 1
                continue

            if final_probs is None:
                print(f"  SKIPPED (None result)")
                results_log.append({
                    "dataset": dataset_name, "q_idx": q_idx,
                    "correct": False, "error": "None"
                })
                total += 1
                continue

            answer_idx = argmax(final_probs)
            is_correct = (answer_idx == gt_id)

            if is_correct:
                correct += 1
            total += 1

            probs_rounded = [round(p, 3) for p in final_probs]
            print(f"  Probs: {probs_rounded}")
            print(f"  Predicted: {filtered_choices[answer_idx]}")
            print(f"  GT:        {true_answer_word}")
            print(f"  {'✓ CORRECT' if is_correct else '✗ WRONG'}  |  Running: {correct}/{total} = {100*correct/total:.1f}%")

            results_log.append({
                "dataset": dataset_name, "q_idx": q_idx, "correct": is_correct
            })

    print(f"\n{'='*60}")
    print(f"FINAL — Hi-ToM len1 Baseline")
    print(f"{'='*60}")
    print(f"Correct: {correct} / {total}")
    if total:
        print(f"Accuracy: {100*correct/total:.1f}%")

    # Breakdown by tell/no_tell
    tell_correct = sum(1 for r in results_log if r.get("correct") and "tell1" in r["dataset"])
    tell_total = sum(1 for r in results_log if "tell1" in r["dataset"])
    no_tell_correct = sum(1 for r in results_log if r.get("correct") and "tell0" in r["dataset"])
    no_tell_total = sum(1 for r in results_log if "tell0" in r["dataset"])
    if tell_total:
        print(f"  Tell subset:    {tell_correct}/{tell_total} = {100*tell_correct/tell_total:.1f}%")
    if no_tell_total:
        print(f"  No-tell subset: {no_tell_correct}/{no_tell_total} = {100*no_tell_correct/no_tell_total:.1f}%")


if __name__ == "__main__":
    main()
