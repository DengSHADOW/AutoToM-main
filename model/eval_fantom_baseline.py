"""
Evaluate AutoToM baseline on FANToM inaccessible (false-belief) first-order.

Usage:
    python eval_fantom_baseline.py                    # first 20 questions
    python eval_fantom_baseline.py --n 50             # first 50 questions
    python eval_fantom_baseline.py --start_idx 20     # resume from idx 20
    python eval_fantom_baseline.py --use_pawm         # with PAWM enabled
"""

import sys, os, csv, argparse, random
import numpy as np
from copy import deepcopy
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))


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


def argmax(lst):
    return lst.index(max(lst))


def load_fantom(subset="inaccessible_full_context_first-order"):
    path = f"../benchmarks/full_data_formatted/FANToM_tom_belief_{subset}.csv"
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    data = []
    for row in rows:
        story, question, choices_str, gt = row[0], row[1], row[2], row[3]
        choices = eval(choices_str)
        data.append((story, question, choices, gt))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of questions to run")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_pawm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--subset", default="inaccessible_full_context_first-order")
    parser.add_argument("--prev_correct", type=int, default=0)
    parser.add_argument("--prev_total", type=int, default=0)
    args = parser.parse_args()

    mode_str = "pawm_on" if args.use_pawm else "baseline"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subset_short = args.subset.replace("_full_context_", "_").replace("_short_context_", "_short_")
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../results/log_fantom_{subset_short}_{mode_str}_{timestamp}.txt"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    tee = Tee(log_path)
    sys.stdout = tee

    from ProbSolver import ProblemSolver
    import utils, probs as probs_module

    random.seed(args.seed)
    np.random.seed(args.seed)
    utils.set_global_seed(args.seed)
    probs_module.set_global_seed(args.seed)

    data = load_fantom(args.subset)
    end_idx = min(args.start_idx + args.n, len(data))

    print("=" * 60)
    print(f"FANToM Baseline Evaluation: {args.subset}")
    print(f"Questions {args.start_idx} to {end_idx-1} (total {end_idx - args.start_idx})")
    print(f"PAWM: {'ON' if args.use_pawm else 'OFF'}")
    print(f"Log: {os.path.abspath(log_path)}")
    print("=" * 60)

    correct = args.prev_correct
    total = args.prev_total
    errors = []

    for idx in range(args.start_idx, end_idx):
        story, question, choices, gt = data[idx]

        # Shuffle choices, track GT index
        shuffled = deepcopy(choices)
        random.seed(args.seed + idx)
        random.shuffle(shuffled)
        gt_id = -1
        for j, c in enumerate(shuffled):
            if c.strip() == gt.strip():
                gt_id = j
        if gt_id == -1:
            print(f"  [{idx}] WARNING: GT not found in choices, skipping")
            continue

        print(f"\n{'─'*50}")
        print(f"[{idx}/{end_idx-1}] Question: {question[:100]}...")
        print(f"  GT: {gt[:80]}...")
        print(f"  Choices: {len(shuffled)}")

        solver = ProblemSolver(
            story=story,
            question=question,
            choices=shuffled,
            K=1,
            assigned_model=["State", "Observation", "Belief", "Action", "Goal"],
            model_name="automated",
            episode_name=f"fantom_{args.subset}_{idx}",
            llm=args.llm,
            verbose=args.verbose,
            dataset_name="FANToM-1st_FB_full",
            hypo_method="guided",
            nested=None,
            answerfunc=argmax,
            back_inference=True,
            reduce_hypotheses=True,
            seed=args.seed,
            use_pawm=args.use_pawm,
        )

        try:
            final_probs, _ = solver.solve()
        except Exception as e:
            print(f"  ERROR: {e}")
            total += 1
            errors.append({"idx": idx, "error": str(e)})
            continue

        if not final_probs:
            print(f"  SKIPPED (None or empty result)")
            total += 1
            continue

        pred_idx = argmax(final_probs)
        is_correct = (pred_idx == gt_id)
        if is_correct:
            correct += 1
        total += 1

        probs_r = [round(p, 3) for p in final_probs]
        mark = "✓" if is_correct else "✗"
        print(f"  Probs: {probs_r}")
        print(f"  Predicted: {shuffled[pred_idx][:80]}...")
        print(f"  {mark}  Running: {correct}/{total} = {100*correct/total:.1f}%")

        if not is_correct:
            errors.append({
                "idx": idx,
                "question": question,
                "gt": gt,
                "predicted": shuffled[pred_idx],
                "probs": probs_r,
            })

    print(f"\n{'='*60}")
    print(f"FINAL — FANToM {args.subset}")
    print(f"PAWM: {'ON' if args.use_pawm else 'OFF'}")
    print(f"Correct: {correct} / {total} = {100*correct/total:.1f}%")
    print(f"{'='*60}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            if "error" in e:
                print(f"  idx={e['idx']}: {e['error']}")
            else:
                print(f"  idx={e['idx']}: predicted='{e['predicted'][:60]}...' gt='{e['gt'][:60]}...'")

    sys.stdout = tee.terminal
    tee.close()
    print(f"\nLog saved to: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
