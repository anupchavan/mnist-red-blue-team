#!/usr/bin/env python3
"""Consolidate the repository's saved MNIST red/blue-team JSON results.

This script does not retrain models or alter source result files. It creates a
flat CSV, a Markdown summary, an SVG robustness plot, and consistency checks.
Only the Python standard library is required.

Run from the repository root:
    python analyze_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CSV_FIELDS = [
    "experiment", "scenario", "attack", "epsilon", "n", "accuracy", "loss",
    "macro_f1", "attack_success_rate", "detection_precision",
    "detection_recall", "flagged_samples", "true_positives", "poisoned_samples",
]


def read_json(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def metric_row(experiment: str, scenario: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = {key: "" for key in CSV_FIELDS}
    row.update({
        "experiment": experiment,
        "scenario": scenario,
        "n": metrics.get("n", ""),
        "accuracy": metrics.get("accuracy", ""),
        "loss": metrics.get("loss", ""),
        "macro_f1": metrics.get("macro_f1", ""),
    })
    row.update({key: value for key, value in extra.items() if key in row})
    return row


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    baseline = read_json("baseline_clean.json")
    if baseline:
        rows.append(metric_row("baseline", "clean", baseline))

    evasion = read_json("baseline_evasion.json")
    if evasion:
        for scenario, metrics in evasion.items():
            attack, _, epsilon = scenario.rpartition("_")
            rows.append(metric_row("baseline_evasion", scenario, metrics,
                                   attack=attack, epsilon=epsilon))

    advtrained = read_json("advtrained.json")
    if advtrained:
        for scenario, metrics in advtrained.items():
            attack, _, epsilon = scenario.rpartition("_")
            rows.append(metric_row("adversarial_training", scenario, metrics,
                                   attack=attack if scenario != "clean" else "",
                                   epsilon=epsilon if scenario != "clean" else ""))

    poisoning = read_json("poisoning.json")
    if poisoning:
        for rate, experiment in poisoning.items():
            for split in ("clean", "triggered"):
                metrics = experiment.get(split, {})
                rows.append(metric_row(
                    "backdoor_poisoning", f"poison_rate_{rate}_{split}", metrics,
                    attack="BadNets_backdoor", epsilon="", 
                    attack_success_rate=experiment.get("attack_success_rate", "") if split == "triggered" else "",
                ))

    for method, filename in (("ICA", "sanitised_ica.json"), ("PCA", "sanitised_pca.json")):
        data = read_json(filename)
        if not data:
            continue
        detection = data.get("detection", {})
        for split in ("clean", "triggered"):
            rows.append(metric_row(
                f"backdoor_sanitization_{method}", split, data.get(split, {}),
                attack="BadNets_backdoor" if split == "triggered" else "",
                attack_success_rate=data.get("attack_success_rate", "") if split == "triggered" else "",
                detection_precision=detection.get("precision", ""),
                detection_recall=detection.get("recall", ""),
                flagged_samples=detection.get("flagged", ""),
                true_positives=detection.get("true_positives", ""),
                poisoned_samples=detection.get("poisoned", ""),
            ))
    return rows


def fmt_pct(value: Any) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def write_csv(rows: list[dict[str, Any]]) -> Path:
    output = RESULTS / "summary_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def robustness_svg() -> Path | None:
    baseline = read_json("baseline_evasion.json")
    defended = read_json("advtrained.json")
    if not baseline or not defended:
        return None

    width, height = 820, 500
    left, right, top, bottom = 90, 30, 55, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    epsilons = [0.1, 0.2, 0.3]
    series = [
        ("FGSM baseline", [baseline[f"fgsm_{e:.1f}"]["accuracy"] for e in epsilons]),
        ("FGSM adversarially trained", [defended[f"whitebox_fgsm_{e:.1f}"]["accuracy"] for e in epsilons]),
        ("PGD baseline", [baseline[f"pgd_{e:.1f}"]["accuracy"] for e in epsilons]),
        ("PGD adversarially trained", [defended[f"whitebox_pgd_{e:.1f}"]["accuracy"] for e in epsilons]),
    ]
    # SVG uses a 0-100% y-axis; no plotting package is needed.
    def x(i: int) -> float:
        return left + i * plot_w / (len(epsilons) - 1)
    def y(acc: float) -> float:
        return top + (1.0 - acc) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="410" y="28" font-family="sans-serif" font-size="19" text-anchor="middle" fill="#111">Accuracy under evasion attacks</text>']
    for tick in range(0, 101, 20):
        yy = top + (100 - tick) / 100 * plot_h
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-14}" y="{yy+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#333">{tick}%</text>')
    for i, eps in enumerate(epsilons):
        parts.append(f'<text x="{x(i):.1f}" y="{height-bottom+25}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333">{eps:.1f}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">Perturbation strength (epsilon)</text>')
    parts.append(f'<text x="23" y="{top+plot_h/2}" transform="rotate(-90 23 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">Accuracy</text>')

    # Explicit series styles make the generated figure legible in the report.
    styles = [("#2563eb", "solid"), ("#2563eb", "dash"), ("#dc2626", "solid"), ("#dc2626", "dash")]
    for idx, ((label, values), (color, dash)) in enumerate(zip(series, styles)):
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
        dash_attr = ' stroke-dasharray="7 5"' if dash == "dash" else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>')
        for i, value in enumerate(values):
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="4" fill="{color}"/>')
        lx = left + (idx % 2) * 330
        ly = height - 58 + (idx // 2) * 20
        parts.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+28}" y2="{ly-4}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        parts.append(f'<text x="{lx+36}" y="{ly}" font-family="sans-serif" font-size="12" fill="#222">{label}</text>')
    parts.append('<text x="410" y="486" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">Generated from saved JSON results; see report for test-set details.</text>')
    parts.append('</svg>')
    output = RESULTS / "robustness_comparison.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def consistency_checks() -> list[str]:
    messages: list[str] = []
    for filename in ("baseline_clean.json", "baseline_evasion.json", "advtrained.json", "poisoning.json", "sanitised_ica.json", "sanitised_pca.json"):
        data = read_json(filename)
        if not data:
            continue
        def check_metrics(label: str, metrics: dict[str, Any]) -> None:
            confusion = metrics.get("confusion")
            n = metrics.get("n")
            accuracy = metrics.get("accuracy")
            if isinstance(confusion, list) and isinstance(n, int) and accuracy is not None:
                total = sum(sum(row) for row in confusion if isinstance(row, list))
                diagonal = sum(row[i] for i, row in enumerate(confusion) if isinstance(row, list) and i < len(row))
                if total != n:
                    messages.append(f"WARNING {filename}:{label}: confusion total {total} != n={n}")
                calculated = diagonal / total if total else math.nan
                if total and abs(calculated - float(accuracy)) > 1e-6:
                    messages.append(f"WARNING {filename}:{label}: confusion accuracy {calculated:.6f} != saved accuracy {float(accuracy):.6f}")
        if filename == "baseline_clean.json":
            check_metrics("clean", data)
        elif filename in ("baseline_evasion.json", "advtrained.json"):
            for label, metrics in data.items():
                if isinstance(metrics, dict):
                    check_metrics(label, metrics)
        elif filename == "poisoning.json":
            for label, group in data.items():
                for split in ("clean", "triggered"):
                    check_metrics(f"{label}/{split}", group.get(split, {}))
        else:
            for split in ("clean", "triggered"):
                check_metrics(split, data.get(split, {}))
    return messages or ["All available confusion-matrix totals and derived accuracies are consistent with the saved metrics."]


def write_markdown(rows: list[dict[str, Any]], checks: list[str], svg_path: Path | None) -> Path:
    by_key = {(r["experiment"], r["scenario"]): r for r in rows}
    lines = [
        "# Consolidated MNIST Red/Blue-Team Results", "",
        "> Generated automatically from the JSON files in `results/` by `python analyze_results.py`. "
        "This is a summary of recorded results, not an independent rerun of the experiments.", "",
        "## Baseline and evasion robustness", "",
        "| Scenario | Accuracy | Macro F1 | Samples |", "|---|---:|---:|---:|",
    ]
    for scenario in ("clean", "fgsm_0.1", "fgsm_0.2", "fgsm_0.3", "pgd_0.1", "pgd_0.2", "pgd_0.3"):
        key = ("baseline", "clean") if scenario == "clean" else ("baseline_evasion", scenario)
        row = by_key.get(key)
        if row:
            lines.append(f"| {scenario} | {fmt_pct(row['accuracy'])} | {fmt_pct(row['macro_f1'])} | {row['n'] or '—'} |")
    lines += ["", "## Adversarial-training evaluation", "", "| Scenario | Accuracy | Macro F1 | Samples |", "|---|---:|---:|---:|"]
    for scenario in ("clean", "whitebox_fgsm_0.1", "whitebox_fgsm_0.2", "whitebox_fgsm_0.3", "whitebox_pgd_0.1", "whitebox_pgd_0.2", "whitebox_pgd_0.3", "whitebox_pgd100x5_0.3_subset"):
        row = by_key.get(("adversarial_training", scenario))
        if row:
            lines.append(f"| {scenario} | {fmt_pct(row['accuracy'])} | {fmt_pct(row['macro_f1'])} | {row['n'] or '—'} |")
    lines += ["", "## Backdoor poisoning and sanitization", "", "| Experiment | Scenario | Accuracy | Attack success rate | Detection precision | Detection recall |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["experiment"].startswith("backdoor_"):
            lines.append(f"| {r['experiment']} | {r['scenario']} | {fmt_pct(r['accuracy'])} | {fmt_pct(r['attack_success_rate'])} | {fmt_pct(r['detection_precision'])} | {fmt_pct(r['detection_recall'])} |")
    lines += ["", "## Data consistency checks", ""]
    lines += [f"- {message}" for message in checks]
    lines += ["", "## Interpretation notes", "",
              "- Accuracy on clean inputs alone does not characterize robustness to adversarial inputs or backdoor triggers.",
              "- Attack success rate is reported as stored by the experiment scripts; confirm its target-label definition when describing it in the final report.",
              "- The PCA and ICA sanitization results differ substantially; report both rather than selecting only the stronger result.",
              "- Some saved evaluations use subsets while others use the full test set. State sample counts when comparing results.", ""]
    if svg_path:
        lines += ["## Robustness figure", "", f"Generated chart: `{svg_path.name}`", ""]
    output = RESULTS / "summary_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS,
                        help="directory containing the JSON result files (default: repository results/)")
    args = parser.parse_args()
    RESULTS = args.results_dir.resolve()
    if not RESULTS.is_dir():
        parser.error(f"results directory does not exist: {RESULTS}")

    rows = collect_rows()
    if not rows:
        parser.error(f"no recognized JSON result files found in {RESULTS}")
    csv_path = write_csv(rows)
    checks = consistency_checks()
    svg_path = robustness_svg()
    md_path = write_markdown(rows, checks, svg_path)
    print(f"Wrote {len(rows)} metric rows to {csv_path}")
    print(f"Wrote summary report to {md_path}")
    if svg_path:
        print(f"Wrote robustness plot to {svg_path}")
    print("\nConsistency checks:")
    for message in checks:
        print(f"- {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
