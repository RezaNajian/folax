"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE
"""
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASE0_DEFAULT = "PHASE_0_PARAMETRIC_2D_Fourier_strain_25/stats_phase0_parametric.csv"
PHASE2_DEFAULT = "PHASE_2_OTF_Fourier_strain_25/stats_phase2_otf.csv"
PHASE3_DEFAULT = "PHASE_3_TL_OTF_Fourier_strain_25/stats_phase3_tl_otf.csv"
OUT_DIR_DEFAULT = "phase_metrics_comparison_plots"


KEY_ACCURACY_METRICS = [
    "ux_rms",
    "uy_rms",
    "uv_rms",
    "ux_max",
    "uy_max",
    "uv_max",
]

KEY_COST_METRICS = [
    "train_time_s",
    "fe_time_s",
    "total_time_s",
    "epochs_completed",
    "newton_total_iters",
]

KEY_TRAINING_METRICS = [
    "train_residual_final",
    "train_residual_min",
    "final_total_loss",
    "epoch_first_below_target",
]

ALL_PLOT_METRICS = [
    "train_time_s",
    "fe_time_s",
    "total_time_s",
    "epochs_completed",
    "last_epoch",
    "train_residual_final",
    "train_residual_min",
    "epoch_at_min",
    "epoch_first_below_target",
    "final_total_loss",
    "newton_total_iters",
    "newton_final_residual",
    "ux_rms",
    "ux_max",
    "uy_rms",
    "uy_max",
    "uv_rms",
    "uv_max",
]

SUMMARY_METRICS = [
    "train_time_s",
    "fe_time_s",
    "total_time_s",
    "epochs_completed",
    "newton_total_iters",
    "newton_final_residual",
    "train_residual_final",
    "train_residual_min",
    "final_total_loss",
    "ux_rms",
    "ux_max",
    "uy_rms",
    "uy_max",
    "uv_rms",
    "uv_max",
]


def _to_float(value):
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return math.nan
    if text.lower() == "true":
        return 1.0
    if text.lower() == "false":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return math.nan


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _read_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            parsed["_source_csv"] = str(path)
            parsed["sample_id"] = int(float(parsed["sample_id"]))
            rows.append(parsed)
    return rows


def _method_from_row(row: Dict[str, object]) -> str:
    phase = str(row.get("phase", "")).strip()
    skip_training = _to_bool(row.get("skip_training"), default=False)
    warmstart = _to_bool(row.get("use_fno_warmstart"), default=False)

    if phase == "PHASE_FE_ONLY":
        return "FE only"
    if phase == "PHASE_0_PARAMETRIC":
        return "Parametric + FE"
    if phase == "PHASE_2_OTF":
        return "OTF + FE"
    if phase == "PHASE_3_TL_OTF":
        if skip_training and warmstart:
            return "Parametric + FE warm-start"
        if skip_training and not warmstart:
            return "Parametric + FE (TL skip)"
        if (not skip_training) and warmstart:
            return "TL-OTF + FE warm-start"
        return "TL-OTF + FE"
    return phase or "Unknown"


def _augment_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    augmented: List[Dict[str, object]] = []
    for row in rows:
        r = dict(row)
        r["method"] = _method_from_row(r)
        r["train_time_s"] = _to_float(r.get("train_time_s"))
        r["fe_time_s"] = _to_float(r.get("fe_time_s"))
        r["total_time_s"] = (
            r["train_time_s"] + r["fe_time_s"]
            if not math.isnan(r["train_time_s"]) and not math.isnan(r["fe_time_s"])
            else (r["fe_time_s"] if math.isnan(r["train_time_s"]) else r["train_time_s"])
        )

        for key in ALL_PLOT_METRICS:
            if key not in {"train_time_s", "fe_time_s", "total_time_s"}:
                r[key] = _to_float(r.get(key))
        augmented.append(r)
    return augmented


def _group_by_method(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    return dict(grouped)


def _finite_values(rows: Sequence[Dict[str, object]], metric: str) -> List[float]:
    values = [float(row.get(metric, math.nan)) for row in rows]
    return [v for v in values if np.isfinite(v)]


def _safe_slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _write_summary_csv(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    out_path = out_dir / "summary_by_method.csv"
    headers = ["method", "n_samples"]
    for metric in SUMMARY_METRICS:
        headers.extend([
            f"{metric}_mean",
            f"{metric}_median",
            f"{metric}_std",
            f"{metric}_min",
            f"{metric}_max",
        ])

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for method, method_rows in sorted(grouped.items()):
            line: List[object] = [method, len(method_rows)]
            for metric in SUMMARY_METRICS:
                values = np.asarray(_finite_values(method_rows, metric), dtype=float)
                if values.size == 0:
                    line.extend([math.nan] * 5)
                else:
                    line.extend([
                        float(np.mean(values)),
                        float(np.median(values)),
                        float(np.std(values)),
                        float(np.min(values)),
                        float(np.max(values)),
                    ])
            writer.writerow(line)


def _write_best_method_csv(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    out_path = out_dir / "best_method_by_metric.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "optimization", "best_method", "best_value"])
        for metric in SUMMARY_METRICS:
            method_values = []
            for method, method_rows in grouped.items():
                vals = _finite_values(method_rows, metric)
                if vals:
                    method_values.append((method, float(np.mean(vals))))
            if not method_values:
                continue
            optimization = "min"
            best_method, best_value = min(method_values, key=lambda x: x[1])
            writer.writerow([metric, optimization, best_method, best_value])


def _bar_summary_plot(rows: Sequence[Dict[str, object]], metric: str, out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    methods = []
    means = []
    stds = []
    for method, method_rows in sorted(grouped.items()):
        values = np.asarray(_finite_values(method_rows, metric), dtype=float)
        if values.size == 0:
            continue
        methods.append(method)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(methods)), 5))
    ax.bar(range(len(methods)), means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Mean ± std of {metric} by method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_{_safe_slug(metric)}.png", dpi=200)
    plt.close(fig)


def _boxplot_metric(rows: Sequence[Dict[str, object]], metric: str, out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    methods = []
    data = []
    for method, method_rows in sorted(grouped.items()):
        values = _finite_values(method_rows, metric)
        if not values:
            continue
        methods.append(method)
        data.append(values)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(methods)), 5))
    ax.boxplot(data, tick_labels=methods, showfliers=True)
    ax.set_ylabel(metric)
    ax.set_title(f"Distribution of {metric} by method")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / f"box_{_safe_slug(metric)}.png", dpi=200)
    plt.close(fig)


def _paired_sample_plot(rows: Sequence[Dict[str, object]], metric: str, out_dir: Path) -> None:
    grouped_by_sample: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if np.isfinite(row.get(metric, math.nan)):
            grouped_by_sample[int(row["sample_id"])].append(row)

    if not grouped_by_sample:
        return

    methods = sorted({str(row["method"]) for row in rows})
    sample_ids = sorted(grouped_by_sample)
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(sample_ids)), 5))

    for method in methods:
        xs = []
        ys = []
        for sid in sample_ids:
            matches = [r for r in grouped_by_sample[sid] if r["method"] == method and np.isfinite(r.get(metric, math.nan))]
            if not matches:
                continue
            xs.append(sid)
            ys.append(float(matches[-1][metric]))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=method)

    ax.set_xlabel("sample_id")
    ax.set_ylabel(metric)
    ax.set_title(f"Per-sample comparison for {metric}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_lines_{_safe_slug(metric)}.png", dpi=200)
    plt.close(fig)


def _scatter_cost_vs_accuracy(rows: Sequence[Dict[str, object]], cost_metric: str, accuracy_metric: str, out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plotted = False

    for method, method_rows in sorted(grouped.items()):
        x = np.asarray(_finite_values(method_rows, cost_metric), dtype=float)
        y = np.asarray(_finite_values(method_rows, accuracy_metric), dtype=float)
        size = min(len(x), len(y))
        if size == 0:
            continue
        x = x[:size]
        y = y[:size]
        ax.scatter(x, y, label=method, alpha=0.75, s=40)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(cost_metric)
    ax.set_ylabel(accuracy_metric)
    ax.set_title(f"{accuracy_metric} vs {cost_metric}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_{_safe_slug(cost_metric)}_vs_{_safe_slug(accuracy_metric)}.png", dpi=200)
    plt.close(fig)


def _pareto_front_plot(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    points: List[Tuple[str, float, float]] = []
    for method, method_rows in sorted(grouped.items()):
        total_time = _finite_values(method_rows, "total_time_s")
        uv_rms = _finite_values(method_rows, "uv_rms")
        if total_time and uv_rms:
            points.append((method, float(np.mean(total_time)), float(np.mean(uv_rms))))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for method, x, y in points:
        ax.scatter(x, y, s=80)
        ax.annotate(method, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("mean total_time_s")
    ax.set_ylabel("mean uv_rms")
    ax.set_title("Pareto-style view: cost vs displacement error")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_total_time_vs_uv_rms.png", dpi=200)
    plt.close(fig)


def _correlation_heatmap(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    numeric_rows = []
    metrics = [m for m in SUMMARY_METRICS if any(np.isfinite(r.get(m, math.nan)) for r in rows)]
    if len(metrics) < 2:
        return

    for row in rows:
        numeric_rows.append([float(row.get(metric, math.nan)) for metric in metrics])
    arr = np.asarray(numeric_rows, dtype=float)

    valid_cols = []
    valid_names = []
    for idx, name in enumerate(metrics):
        col = arr[:, idx]
        if np.isfinite(col).sum() >= 3:
            valid_cols.append(col)
            valid_names.append(name)
    if len(valid_cols) < 2:
        return

    clean = np.column_stack(valid_cols)
    for j in range(clean.shape[1]):
        col = clean[:, j]
        mask = ~np.isfinite(col)
        if np.all(mask):
            clean[:, j] = 0.0
        else:
            clean[mask, j] = np.nanmedian(col)

    corr = np.corrcoef(clean, rowvar=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(valid_names)), max(6, 0.55 * len(valid_names))))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(valid_names)))
    ax.set_yticks(range(len(valid_names)))
    ax.set_xticklabels(valid_names, rotation=45, ha="right")
    ax.set_yticklabels(valid_names)
    ax.set_title("Correlation heatmap across numeric metrics")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=200)
    plt.close(fig)


def _method_counts_plot(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    methods = sorted(grouped)
    counts = [len(grouped[m]) for m in methods]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(methods)), 4.5))
    ax.bar(range(len(methods)), counts, color="darkslateblue", alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("row count")
    ax.set_title("Available runs per method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "method_counts.png", dpi=200)
    plt.close(fig)


def _write_readme(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    grouped = _group_by_method(rows)
    path = out_dir / "README.txt"
    with path.open("w") as f:
        f.write("Generated comparison plots for Phase 0 / Phase 2 / Phase 3 stats CSVs.\n\n")
        f.write("Detected methods:\n")
        for method, method_rows in sorted(grouped.items()):
            sample_ids = sorted({int(r["sample_id"]) for r in method_rows})
            f.write(f"- {method}: {len(method_rows)} rows, samples {sample_ids[:5]}")
            if len(sample_ids) > 5:
                f.write(" ...")
            f.write("\n")
        f.write("\nKey outputs:\n")
        f.write("- summary_by_method.csv: mean/median/std/min/max per metric and method\n")
        f.write("- best_method_by_metric.csv: lowest-mean method for each metric\n")
        f.write("- bar_*.png, box_*.png: aggregated method comparisons\n")
        f.write("- sample_lines_*.png: sample-by-sample comparisons\n")
        f.write("- scatter_*.png: cost-vs-accuracy tradeoffs\n")
        f.write("- pareto_total_time_vs_uv_rms.png: compact tradeoff view\n")
        f.write("- correlation_heatmap.png: relationships between numeric metrics\n")


def make_plots(phase0_csv: Path, phase2_csv: Path, phase3_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for csv_path in [phase0_csv, phase2_csv, phase3_csv]:
        all_rows.extend(_read_csv(csv_path))
    rows = _augment_rows(all_rows)

    _write_summary_csv(rows, out_dir)
    _write_best_method_csv(rows, out_dir)
    _write_readme(rows, out_dir)
    _method_counts_plot(rows, out_dir)

    for metric in ALL_PLOT_METRICS:
        _bar_summary_plot(rows, metric, out_dir)
        _boxplot_metric(rows, metric, out_dir)
        _paired_sample_plot(rows, metric, out_dir)

    for cost_metric in ["train_time_s", "fe_time_s", "total_time_s", "newton_total_iters"]:
        for accuracy_metric in ["ux_rms", "uy_rms", "uv_rms", "ux_max", "uy_max", "uv_max"]:
            _scatter_cost_vs_accuracy(rows, cost_metric, accuracy_metric, out_dir)

    _pareto_front_plot(rows, out_dir)
    _correlation_heatmap(rows, out_dir)


def _resolve_path(script_dir: Path, arg_path: str, default_rel: str) -> Path:
    if arg_path:
        return Path(arg_path).expanduser().resolve()
    return (script_dir / default_rel).resolve()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    phase0_csv = _resolve_path(script_dir, sys.argv[1] if len(sys.argv) > 1 else "", PHASE0_DEFAULT)
    phase2_csv = _resolve_path(script_dir, sys.argv[2] if len(sys.argv) > 2 else "", PHASE2_DEFAULT)
    phase3_csv = _resolve_path(script_dir, sys.argv[3] if len(sys.argv) > 3 else "", PHASE3_DEFAULT)
    out_dir = _resolve_path(script_dir, sys.argv[4] if len(sys.argv) > 4 else "", OUT_DIR_DEFAULT)

    for path in [phase0_csv, phase2_csv, phase3_csv]:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

    make_plots(phase0_csv, phase2_csv, phase3_csv, out_dir)
    print(f"Saved plots and summaries to: {out_dir}")
