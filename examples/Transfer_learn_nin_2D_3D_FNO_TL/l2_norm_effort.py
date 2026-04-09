#!/usr/bin/env python3
# ============================================================
# plot_fig2_effort_and_proxy_l2_revised.py
#
# Outputs:
#   1) fig2_effort_distribution.png
#        - NFEM (Newton iters) from Phase-2 CSV
#        - OTF effort (epochs) from Phase-2 CSV
#        - TL-OTF effort (epochs) from Phase-3 CSV
#
#   2) fig2_relative_l2_proxy.png
#        - Param  (Phase-0 here) proxy relative L2 using ux_rms / U_REF
#        - OTF    (Phase-2) proxy relative L2 using ux_rms / U_REF
#        - TL     (Phase-3) proxy relative L2 using ux_rms / U_REF
#
# Effort definition:
#   epochs_effort = (epoch_first_below_target + 1) if available else epochs_completed
#   NFEM          = newton_total_iters
#
# Deduplication:
#   group by sample_id, keep MIN value per sample_id (per metric)
#
# Proxy relative L2:
#   relative_L2_proxy = ux_rms / U_REF, with U_REF=0.1 by default
#
# Note:
#   True relative L2 would be ||u_pred - u_FE||2 / ||u_FE||2.
#   CSV does not include ||u_FE||2, so this is a proxy.
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
THIS_DIR = Path(__file__).resolve().parent         # .../folax/examples/mechanical_square
FOLAX_ROOT = THIS_DIR.parents[1]                   # .../folax

# Revised: use Phase-0 instead of Phase-1
PARAM_CSV  = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square_2D_3D_FNO_TL/PHASE_0_PARAMETRIC_2D_Fourier_strain_25/stats_phase0_parametric.csv"
PHASE2_CSV = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square_2D_3D_FNO_TL/PHASE_2_OTF_Fourier_strain_25/stats_phase2_otf.csv"
PHASE3_CSV = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square_2D_3D_FNO_TL/PHASE_3_TL_OTF_Fourier_evaluate_only_25/stats_phase3_tl_otf.csv"

OUT_EFFORT_PNG = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_effort_distribution_warmstart.png"
OUT_PROXY_PNG  = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_relative_l2_proxy_warmstart.png"

# Optional PDFs (set "" to disable)
OUT_EFFORT_PDF = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_effort_distribution_warmstart.pdf"
OUT_PROXY_PDF  = "/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/fig2_relative_l2_proxy_warmstart.pdf"

DPI = 250

EFFORT_TITLE = "Computation effort distribution"
PROXY_TITLE  = "Relative L2 norm (proxy; Param. from Phase-0, NFEM as reference)"

PARAM_LABEL = "Param. (P0)"

# Proxy denominator
U_REF = 0.25

# Scales
EFFORT_LOG_SCALE = True
PROXY_LOG_SCALE  = False
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
        r = csv.DictReader(f)
        rows = [row for row in r]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _epochs_effort_from_row(row: dict):
    e_first = _to_float(row.get("epoch_first_below_target", np.nan))
    e_comp  = _to_float(row.get("epochs_completed", np.nan))
    if not np.isnan(e_first):
        return float(e_first + 1.0)  # epoch index -> count
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


def _save(fig, out_png: Path, out_pdf: Path | str):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight")
    print(f"[write] {out_png}")

    if out_pdf and str(out_pdf).strip():
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_pdf), bbox_inches="tight")
        print(f"[write] {out_pdf}")


def _print_summary(name, arr):
    arr = np.asarray(arr, dtype=float)
    print(
        f"{name:>18s} | n={len(arr):3d} | "
        f"min={np.min(arr):.6f} | median={np.median(arr):.6f} | max={np.max(arr):.6f}"
    )


def main():
    p0 = _read_csv_dicts(PARAM_CSV)
    p2 = _read_csv_dicts(PHASE2_CSV)
    p3 = _read_csv_dicts(PHASE3_CSV)

    # -----------------------
    # FIG 1: effort
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
        title=EFFORT_TITLE,
        log_scale=EFFORT_LOG_SCALE,
        seed0=100,
    )
    fig.tight_layout()
    _save(fig, OUT_EFFORT_PNG, OUT_EFFORT_PDF)
    plt.close(fig)

    # -----------------------
    # FIG 2: proxy relative L2
    # -----------------------
    param_ux_rms = _dedup_min_value(p0, "ux_rms")
    otf_ux_rms   = _dedup_min_value(p2, "ux_rms")
    tl_ux_rms    = _dedup_min_value(p3, "ux_rms")

    param_rel = [v / U_REF for v in param_ux_rms if (v is not None and not np.isnan(v))]
    otf_rel   = [v / U_REF for v in otf_ux_rms   if (v is not None and not np.isnan(v))]
    tl_rel    = [v / U_REF for v in tl_ux_rms    if (v is not None and not np.isnan(v))]

    fig, ax = plt.subplots(figsize=(14, 7))
    _box_scatter(
        ax,
        data=[param_rel, otf_rel, tl_rel],
        xlabels=[PARAM_LABEL, "OTF", "TL"],
        y_label=f"Relative L2 (proxy) = ux_rms / {U_REF}",
        title=PROXY_TITLE,
        log_scale=PROXY_LOG_SCALE,
        seed0=200,
    )
    fig.tight_layout()
    _save(fig, OUT_PROXY_PNG, OUT_PROXY_PDF)
    plt.close(fig)

    print("\nEffort summary")
    print("-" * 70)
    _print_summary("NFEM", nfem)
    _print_summary("OTF", otf)
    _print_summary("TL-OTF", tl_otf)

    print("\nProxy summary")
    print("-" * 70)
    _print_summary(PARAM_LABEL, param_rel)
    _print_summary("OTF", otf_rel)
    _print_summary("TL", tl_rel)


if __name__ == "__main__":
    main()