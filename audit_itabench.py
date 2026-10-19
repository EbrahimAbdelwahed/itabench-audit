"""
Structural audit of five ITA-Bench tasks. No model is needed: everything is computed from the data and
from the official task definitions (SapienzaNLP/ita-bench, pinned commit, run with lm-eval 0.4.8).

For each task:
- chance: mean of 1/k over questions (k = number of options)
- chance_official_scoring (TruthfulQA mc1 only): chance under the official target, which lm-eval reads as
  the set of correct indices {0, 1} (see README, finding 1)
- always_same_position: accuracy of always picking the most frequent gold position
- gold_position: distribution of the gold index as produced by the official code
- GSM8K only: distractor pattern and accuracy of "pick the median number" (the question is never read)

Usage: python audit_itabench.py --tasks-dir ita-bench/tasks > results/audit.json
"""
import argparse
import collections
import json
import statistics
import sys

from lm_eval.tasks import TaskManager

TASKS = [
    "itabench_arc_challenge_it-it",
    "itabench_hellaswag_it-it",
    "itabench_boolq_with_passages_it-it",
    "itabench_truthful_qa_mc1_it-it",
    "itabench_gsm8k_multichoice_it-it",
]


def as_float(s: str) -> float | None:
    try:
        return float(str(s).strip().replace(",", "."))
    except ValueError:
        return None


def distractor_offsets(nums: list[float], gold: int) -> tuple:
    return tuple(sorted(round(x - nums[gold], 3) for i, x in enumerate(nums) if i != gold))


def audit(task) -> dict:
    docs = list(task.test_docs() if task.has_test_docs() else task.validation_docs())
    golds, ks, official_chance = [], [], []
    median_hits, offsets = 0, collections.Counter()
    for doc in docs:
        choices = task.doc_to_choice(doc)
        target = task.doc_to_target(doc)
        k = len(choices)
        if isinstance(target, list):
            # Official TruthfulQA mc1 target: label vector [1, 0, 0, ...]. lm-eval treats a list target as
            # the set of correct indices, so a prediction of 0 or 1 is scored as correct.
            official_chance.append(len({t for t in target if t < k}) / k)
            gold = target.index(1)
        else:
            gold = choices.index(target) if isinstance(target, str) else int(target)
        golds.append(gold)
        ks.append(k)
        nums = [as_float(c) for c in choices]
        if None not in nums and k >= 3:
            median_hits += nums.index(statistics.median_low(nums)) == gold
            offsets[distractor_offsets(nums, gold)] += 1
    n = len(docs)
    pos = collections.Counter(golds)
    out = {
        "n": n,
        "n_options": {str(k): v for k, v in sorted(collections.Counter(ks).items())},
        "chance": round(100 * statistics.mean(1 / k for k in ks), 2),
        "always_same_position": round(100 * pos.most_common(1)[0][1] / n, 2),
        "gold_position": {str(k): v for k, v in sorted(pos.items())},
    }
    if official_chance:
        out["chance_official_scoring"] = round(100 * statistics.mean(official_chance), 2)
    if "gsm8k" in task.config.task:
        out["median_number_heuristic"] = round(100 * median_hits / n, 2)  # non-numeric questions count as wrong
        out["distractor_offsets_top"] = [
            {"offsets_from_gold": list(o), "questions": c} for o, c in offsets.most_common(5)
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="ita-bench/tasks", help="tasks/ folder of a SapienzaNLP/ita-bench checkout")
    args = ap.parse_args()
    tasks = TaskManager(include_path=args.tasks_dir).load_task_or_group(TASKS)
    result = {name: audit(tasks[name]) for name in TASKS}
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
