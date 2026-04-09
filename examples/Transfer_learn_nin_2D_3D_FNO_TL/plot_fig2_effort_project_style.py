"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE
"""
#!/usr/bin/env python3
# ============================================================
# plot_fig2_effort_and_proxy_l2_project_style.py
#
# Project-style version of the uploaded-CSV effort plot script.
#
# What it does:
#   1) Reads Phase-0 / Phase-2 / Phase-3 CSV files using paths
#      defined relative to the script location.
#   2) Reproduces the "same" effort plot:
#        - NFEM   (Newton iters) from Phase-2 CSV
#        - OTF    (epochs effort) from Phase-2 CSV
#        - TL-OTF (epochs effort) from Phase-3 CSV
#   3) Also creates an extended plot including:
#        - Parametric effort (epochs) from Phase-0 CSV
#
# Effort definition:
#   epochs_effort = (epoch_first_below_target + 1) if available
#                   else epochs_completed
#   NFEM          = newton_total_iters
#
# Deduplication:
#   group by sample_id, keep MIN value per sample_id (per metric)
# ============================================================

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG (EDIT HERE ONLY)
# =========================
THIS_DIR = Path(__file__).resolve().parent   # e.g. .../folax/examples/mechanical_square
FOLAX_ROOT = THIS_DIR.parents[1]             # e.g. .../folax

# Input CSVs (project-relative)
PHASE0_CSV = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_0_PARAMETRIC_Fourier/stats_phase0_parametric.csv"
PHASE2_CSV = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_2_OTF_Fourier/stats_phase2_otf.csv"
PHASE3_CSV = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_3_TL_OTF_Fourier/stats_phase3_tl_otf.csv"

# Output files
OUT_EFFORT_SAME_PNG = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_effort_distribution_uploaded_same.png"
OUT_EFFORT_SAME_PDF = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_effort_distribution_uploaded_same.pdf"

OUT_EFFORT_EXT_PNG  = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_effort_distribution_uploaded_extended.png"
OUT_EFFORT_EXT_PDF  = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_effort_distribution_uploaded_extended.pdf"

DPI = 250

EFFORT_SAME_TITLE = "Computation effort distribution"
EFFORT_EXT_TITLE  = "Computation effort distribution 2D Fourier-based samples(100 samples(80 train + 20 test))"

EFFORT_LOG_SCALE = True
# =========================


def _to_float(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _to_int(x):
    v = _to_float(x)
    if np.isnan(v):
        return None
    try:
        return int(v)
    except Exception:
        return None


def _read_csv_dicts(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", newline="") as f:
        rows = [row for row in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _epochs_effort_from_row(row: dict):
    e_first = _to_float(row.get("epoch_first_below_target", np.nan))
    e_comp  = _to_float(row.get("epochs_completed", np.nan))
    if not np.isnan(e_first):
        return float(e_first + 1.0)
    return float(e_comp)


def _dedup_min_by_sample_id(rows, value_fn):
    best = {}
    for row in rows:
        sid = _to_int(row.get("sample_id"))
        if sid is None:
            continue
        v = value_fn(row)
        if v is None or np.isnan(v):
            continue
        if sid not in best or v < best[sid][0]:
            best[sid] = (v, row)
    return [best[sid][0] for sid in sorted(best.keys())]


def _dedup_min_value(rows, key: str):
    return _dedup_min_by_sample_id(rows, lambda r: _to_float(r.get(key, np.nan)))


def _filter_positive(values, name: str):
    out = []
    for v in values:
        if v is None or np.isnan(v):
            continue
        if v > 0:
            out.append(float(v))
    if len(out) == 0:
        raise ValueError(f"No positive values for {name}. Cannot plot on log-scale.")
    return out


def _jitter(x, n, amount=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return x + rng.uniform(-amount, amount, size=n)


def _box_scatter(ax, data, xlabels, y_label, title, log_scale=False, seed0=0):
    positions = list(range(1, len(data) + 1))

    if log_scale:
        data = [_filter_positive(d, f"{title}:{xlabels[i]}") for i, d in enumerate(data)]
        ax.set_yscale("log")

    ax.boxplot(
        data,
        positions=positions,
        widths=0.30,
        showfliers=False,
        medianprops=dict(linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    for i, y in enumerate(data):
        xs = _jitter(positions[i], len(y), amount=0.08, seed=seed0 + i)
        ax.scatter(xs, y, s=55, alpha=0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=16, pad=14)

    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.3)


def _save(fig, out_png: Path, out_pdf: Path):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight")
    print(f"[write] {out_png}")

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_pdf), bbox_inches="tight")
    print(f"[write] {out_pdf}")


def _print_summary(name, values):
    arr = np.asarray(values, dtype=float)
    print(
        f"{name:>22s} | n={len(arr):3d} | "
        f"min={np.min(arr):8.3f} | median={np.median(arr):8.3f} | max={np.max(arr):8.3f}"
    )


def main():
    p0 = _read_csv_dicts(PHASE0_CSV)
    p2 = _read_csv_dicts(PHASE2_CSV)
    p3 = _read_csv_dicts(PHASE3_CSV)

    # -----------------------
    # Same 3-group effort plot
    # -----------------------
    nfem   = _dedup_min_value(p2, "newton_total_iters")
    otf    = _dedup_min_by_sample_id(p2, _epochs_effort_from_row)
    tl_otf = _dedup_min_by_sample_id(p3, _epochs_effort_from_row)

    fig, ax = plt.subplots(figsize=(14, 5))
    _box_scatter(
        ax,
        data=[nfem, otf, tl_otf],
        xlabels=["NFEM\n(Newton iters)", "OTF\n(epochs effort)", "TL-OTF\n(epochs effort)"],
        y_label="Iterations / epochs",
        title=EFFORT_SAME_TITLE,
        log_scale=EFFORT_LOG_SCALE,
        seed0=100,
    )
    fig.tight_layout()
    _save(fig, OUT_EFFORT_SAME_PNG, OUT_EFFORT_SAME_PDF)
    plt.close(fig)

    # -----------------------
    # Extended 4-group effort plot
    # -----------------------
    param = _dedup_min_by_sample_id(p0, _epochs_effort_from_row)

    fig, ax = plt.subplots(figsize=(16, 5))
    _box_scatter(
        ax,
        data=[param, nfem, otf, tl_otf],
        xlabels=[
            "Parametric\n(epochs effort)",
            "NFEM\n(Newton iters)",
            "OTF\n(epochs effort)",
            "TL-OTF\n(epochs effort)",
        ],
        y_label="Iterations / epochs",
        title=EFFORT_EXT_TITLE,
        log_scale=EFFORT_LOG_SCALE,
        seed0=300,
    )
    fig.tight_layout()
    _save(fig, OUT_EFFORT_EXT_PNG, OUT_EFFORT_EXT_PDF)
    plt.close(fig)

    print("\nSummary statistics")
    print("-" * 78)
    _print_summary("Parametric epochs", param)
    _print_summary("NFEM Newton iters", nfem)
    _print_summary("OTF epochs", otf)
    _print_summary("TL-OTF epochs", tl_otf)


if __name__ == "__main__":
    main()
