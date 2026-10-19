"""
ITA-Bench evaluation with three protocols, on the official SapienzaNLP task definitions (lm-eval 0.4.8).

--official     the official tasks, unchanged
(default)      corrected: options shuffled per question with fixed seeds (0, 1, 2), so a tie can no longer
               favour the gold position; TruthfulQA mc1 target replaced by the index of the single correct answer
--no-context   corrected, and the question/passage is removed: the model scores the options alone.
               This measures how much of the score comes from the prior over the answers.

Usage:
    python harness.py --model sapienzanlp/Minerva-350M-base-v1.0 --out results/models/minerva-350m_shuffled.json
"""
import argparse
import hashlib
import importlib.metadata
import json
import random
import statistics
import sys
import time
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

DEFAULT_TASKS = [
    "itabench_arc_challenge_it-it",
    "itabench_hellaswag_it-it",
    "itabench_boolq_with_passages_it-it",
    "itabench_truthful_qa_mc1_it-it",
    "itabench_gsm8k_multichoice_it-it",
]


def gold_index(task, doc, choices: list[str]) -> int:
    gold = task._orig_doc_to_target(doc)
    if isinstance(gold, list):  # TruthfulQA mc1: label vector [1, 0, 0, ...]
        return gold.index(1)
    return choices.index(gold) if isinstance(gold, str) else int(gold)


def permutation(doc_text: str, n: int, seed: int) -> list[int]:
    key = hashlib.sha256(f"{seed}|{doc_text}".encode()).hexdigest()
    order = list(range(n))
    random.Random(key).shuffle(order)
    return order


def patch_task(task, seed: int, no_context: bool) -> None:
    """Replace doc_to_choice / doc_to_target / doc_to_text on the task instance."""
    task._orig_doc_to_choice = task.doc_to_choice
    task._orig_doc_to_target = task.doc_to_target
    task._orig_doc_to_text = task.doc_to_text

    def order_for(doc):
        choices = task._orig_doc_to_choice(doc)
        return choices, permutation(str(task._orig_doc_to_text(doc)), len(choices), seed)

    def doc_to_choice(doc, *a, **k):
        choices, order = order_for(doc)
        return [choices[i] for i in order]

    def doc_to_target(doc, *a, **k):
        choices, order = order_for(doc)
        return order.index(gold_index(task, doc, choices))

    def doc_to_text(doc, *a, **k):
        return "" if no_context else task._orig_doc_to_text(doc)

    task.doc_to_choice = doc_to_choice
    task.doc_to_target = doc_to_target
    task.doc_to_text = doc_to_text
    task.multiple_target = False


def run(lm, tasks_dir: str, task_names: list[str], seed: int | None, no_context: bool, limit: int | None) -> dict:
    tasks = TaskManager(include_path=tasks_dir).load_task_or_group(task_names)
    if seed is not None:
        for task in tasks.values():
            patch_task(task, seed, no_context)
    res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks.values()), num_fewshot=0,
                                  limit=limit, log_samples=False, bootstrap_iters=1000)
    out = {}
    for name, r in res["results"].items():
        out[name] = {k.split(",")[0]: round(100 * v, 2) for k, v in r.items()
                     if k.split(",")[0] in ("acc", "acc_norm", "acc_stderr", "acc_norm_stderr")
                     and isinstance(v, (int, float))}
    return out


def summarize(per_seed: dict[int, dict]) -> dict:
    summary = {}
    for task in next(iter(per_seed.values())):
        row = {}
        for metric in ("acc", "acc_norm"):
            vals = [per_seed[s][task][metric] for s in per_seed if metric in per_seed[s][task]]
            if vals:
                row[metric] = round(statistics.mean(vals), 2)
                row[f"{metric}_std_seed"] = round(statistics.pstdev(vals), 2)
                row[f"{metric}_stderr"] = per_seed[next(iter(per_seed))][task].get(f"{metric}_stderr")
        summary[task] = row
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local folder (HF format)")
    ap.add_argument("--tasks-dir", default="ita-bench/tasks", help="tasks/ folder of a SapienzaNLP/ita-bench checkout")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--official", action="store_true", help="official protocol, no corrections")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    lm = HFLM(pretrained=args.model, batch_size=args.batch_size, device=args.device, dtype=args.dtype)
    if args.official:
        per_seed = {-1: run(lm, args.tasks_dir, args.tasks, None, False, args.limit)}
    else:
        per_seed = {s: run(lm, args.tasks_dir, args.tasks, s, args.no_context, args.limit) for s in args.seeds}
    report = {
        "model": args.model,
        "protocol": "official" if args.official else ("no_context" if args.no_context else "shuffled"),
        "seeds": list(per_seed),
        "limit": args.limit,
        "summary": summarize(per_seed),
        "per_seed": per_seed,
        "seconds": round(time.time() - t0),
        "lm_eval_version": importlib.metadata.version("lm_eval"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
