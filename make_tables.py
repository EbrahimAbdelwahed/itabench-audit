"""
Build the README tables and the CSV files from results/.

Usage: python make_tables.py   (writes results/summary.csv and results/chart_data.csv, prints Markdown tables)
"""
import csv
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TASKS = [  # (task, short name, metric): acc_norm for ARC/HellaSwag as in ITA-Bench, acc otherwise
    ("itabench_arc_challenge_it-it", "ARC-C", "acc_norm"),
    ("itabench_hellaswag_it-it", "HellaSwag", "acc_norm"),
    ("itabench_boolq_with_passages_it-it", "BoolQ", "acc"),
    ("itabench_truthful_qa_mc1_it-it", "TruthfulQA mc1", "acc"),
    ("itabench_gsm8k_multichoice_it-it", "GSM8K-MC", "acc"),
]
MODELS = [  # (file prefix, display name)
    ("minerva-350m", "Minerva-350M-base (public)"),
    ("e1", "it-SML E1: Wikipedia only"),
    ("e2", "it-SML E2: Wikipedia + 25% Pleias SYNTH-IT"),
    ("e4", "it-SML E4: synthetic only"),
    ("e5", "it-SML E5: Wikipedia + all synthetic"),
    ("e5_sft", "it-SML E5 + SFT v1"),
    ("e5_sft_v2", "it-SML E5 + SFT v2 (saw GSM8K train)"),
]
PROTOCOLS = ["official", "shuffled", "no_context"]


def score(prefix: str, protocol: str, task: str, metric: str) -> float:
    summary = json.loads((RESULTS / "models" / f"{prefix}_{protocol}.json").read_text())["summary"]
    return summary[task][metric]


def fmt(x: float) -> str:
    return f"{x:.1f}"


def main() -> int:
    audit = json.loads((RESULTS / "audit.json").read_text())
    rows = [(p, name, proto, short, score(p, proto, t, m)) for p, name in MODELS
            for proto in PROTOCOLS for t, short, m in TASKS]
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "model", "protocol", "task", "score"])
        w.writerows(rows)

    # Chart data: per task, the model-free baselines next to what models actually score
    tq, gsm, bq = (audit[t] for t in ("itabench_truthful_qa_mc1_it-it", "itabench_gsm8k_multichoice_it-it",
                                      "itabench_boolq_with_passages_it-it"))
    chart = [
        ("BoolQ", "chance", bq["chance"]),
        ("BoolQ", "always answer 'Sì' (no model)", bq["always_same_position"]),
        ("TruthfulQA mc1", "chance, correct target", tq["chance"]),
        ("TruthfulQA mc1", "chance, official target", tq["chance_official_scoring"]),
        ("GSM8K-MC", "chance", gsm["chance"]),
        ("GSM8K-MC", "pick the median number (no model)", gsm["median_number_heuristic"]),
    ]
    for proto in ("official", "shuffled"):
        chart.append(("TruthfulQA mc1", f"Minerva-350M, {proto} protocol",
                      score("minerva-350m", proto, TASKS[3][0], "acc")))
    chart.append(("GSM8K-MC", "Minerva-350M", score("minerva-350m", "shuffled", TASKS[4][0], "acc")))
    chart.append(("GSM8K-MC", "Minerva-350M, question removed", score("minerva-350m", "no_context", TASKS[4][0], "acc")))
    with open(RESULTS / "chart_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "series", "score"])
        w.writerows(chart)

    head = "| Model | " + " | ".join(s for _, s, _ in TASKS) + " | Mean |\n|---|" + "---|" * (len(TASKS) + 1)
    for proto in PROTOCOLS:
        print(f"\n### {proto}\n\n{head}")
        for p, name in MODELS:
            vals = [score(p, proto, t, m) for t, _, m in TASKS]
            mean = fmt(statistics.mean(vals)) if proto != "no_context" else "–"
            print(f"| {name} | " + " | ".join(fmt(v) for v in vals) + f" | {mean} |")

    print("\n### TruthfulQA mc1: official minus corrected\n\n| Model | Official | Corrected | Gap |\n|---|---|---|---|")
    for p, name in MODELS:
        o, c = score(p, "official", TASKS[3][0], "acc"), score(p, "shuffled", TASKS[3][0], "acc")
        print(f"| {name} | {fmt(o)} | {fmt(c)} | +{fmt(o - c)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
