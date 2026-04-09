#!/usr/bin/env python3
"""
compute_effort_analysis_strict.py
=================================
Research-oriented effort/accuracy analysis using only directly logged metrics.

Main comparisons
----------------
1) OTF vs TL-OTF (adaptation/training effect)
   Direct logged metrics:
     - epochs_effort      = epoch_first_below_target + 1, else epochs_completed
     - train_time_s
     - train_residual_final
     - uv_rms

2) TL-OTF vs TL-OTF+warmstart (solver effect)
   Direct logged metrics:
     - newton_total_iters
     - fe_time_s
     - total_online_time_s = train_time_s + fe_time_s
     - uv_rms

3) OTF vs TL-OTF vs TL-OTF+warmstart (practical online cost)
   Direct logged metrics:
     - total_online_time_s = train_time_s + fe_time_s
     - uv_rms

4) Parametric reported separately as an offline-trained reference
   Direct logged metrics only:
     - train_time_s
     - fe_time_s
     - newton_total_iters
     - uv_rms
     - uv_max
     - train_residual_final
     - newton_final_residual

Important
---------
- No inferred NFEM baseline.
- No amortised parametric per-sample training cost.
- No silent duplicate cherry-picking.
- No relative-L2 proxy using arbitrary reference displacement.

Duplicate policy
----------------
Default is DUPLICATE_POLICY='error'. If the same sample_id appears multiple times
inside the same CSV, the script stops and reports it.

If you intentionally want a rerun-overwrite rule, set:
    DUPLICATE_POLICY = "last"
That is less strict and should be stated explicitly in a paper/report.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
# ============================================================
# CONFIG
# ============================================================
_BASE = Path(__file__).resolve().parent

# ---- use ONLY the 3-sample CSVs for this strict study ----
PARAM_CSV = Path("/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_0_PARAMETRIC_2D_Fourier/stats_phase0_parametric_non_warm.csv")
OTF_CSV   = Path("/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_2_OTF_Fourier/stats_phase2_otf_non_warm.csv")
TL_CSV    = Path("/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_3_TL_OTF_Fourier/stats_phase3_tl_otf_non_warm.csv")

PARAM_WS_CSV = Path("/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_0_PARAMETRIC_2D_Fourier_warmstart/stats_phase0_parametric_warm_start.csv")
OTF_WS_CSV   = Path("/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_2_OTF_Fourier_warmstart/stats_phase2_otf_warm_start.csv")
TL_WS_CSV    = Path("/home/jerry-paul/Access/folax_main_jan_venv/examples/mechanical_square/PHASE_3_TL_OTF_Fourier_warmstart/stats_phase3_tl_otf_warm_start.csv")

OUT_DIR = _BASE / "effort_analysis_strict_3samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 220
DUPLICATE_POLICY = "last"   # 'error' or 'last'
MIN_SHARED_FOR_PLOTS = 1
# ============================================================
# ============================================================


# ============================================================
# IO helpers
# ============================================================
def _must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return path


def _read_csv(path: Path, tag: str) -> pd.DataFrame:
    path = _must_exist(path)
    df = pd.read_csv(path)
    df["_source_tag"] = tag
    df["_source_file"] = str(path)
    return df


def _require_columns(df: pd.DataFrame, cols: list[str], tag: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{tag}: missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _handle_duplicates(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise KeyError(f"{tag}: 'sample_id' column is required")

    dup_mask = df.duplicated(subset=["sample_id"], keep=False)
    if not dup_mask.any():
        return df.copy()

    counts = df.loc[dup_mask, "sample_id"].value_counts().sort_index()
    msg = f"{tag}: duplicate sample_id rows detected:\n{counts.to_string()}"

    if DUPLICATE_POLICY == "error":
        raise ValueError(
            msg
            + "\nThis script is strict by default. "
              "If these are intentional reruns and the newest row should overwrite older rows, "
              "set DUPLICATE_POLICY = 'last'."
        )
    if DUPLICATE_POLICY == "last":
        print("WARNING:", msg)
        return df.drop_duplicates(subset=["sample_id"], keep="last").copy()

    raise ValueError(f"Unknown DUPLICATE_POLICY={DUPLICATE_POLICY!r}")


# ============================================================
# Metric helpers
# ============================================================
def _add_direct_metrics(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    required = [
        "sample_id",
        "train_time_s",
        "fe_time_s",
        "newton_total_iters",
        "uv_rms",
        "epoch_first_below_target",
        "epochs_completed",
        "train_residual_final",
    ]
    _require_columns(df, required, tag)

    out = _coerce_numeric(
        df,
        required + ["newton_final_residual", "uv_max", "use_fno_warmstart"]
    )
    out = _handle_duplicates(out, tag)

    out["epochs_effort"] = np.where(
        out["epoch_first_below_target"].notna(),
        out["epoch_first_below_target"] + 1.0,
        out["epochs_completed"],
    )
    out["total_online_time_s"] = out["train_time_s"] + out["fe_time_s"]
    out = out.sort_values("sample_id").reset_index(drop=True)
    return out


def _shared_subset(*pairs: tuple[str, pd.DataFrame]) -> tuple[list[int], dict[str, pd.DataFrame]]:
    if not pairs:
        return [], {}

    ids = None
    for _, df in pairs:
        s = set(df["sample_id"].astype(int).tolist())
        ids = s if ids is None else ids & s

    shared_ids = sorted(ids) if ids is not None else []
    out = {}
    for name, df in pairs:
        out[name] = (
            df[df["sample_id"].isin(shared_ids)]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
    return shared_ids, out


# ============================================================
# Plot helpers
# ============================================================
def _save(fig, stem: str) -> None:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def _paired_bar_2x2(
    df_a: pd.DataFrame, name_a: str,
    df_b: pd.DataFrame, name_b: str,
    metrics: list[tuple[str, str, bool]],
    title: str,
    stem: str,
) -> None:
    sample_ids = df_a["sample_id"].astype(int).tolist()
    x = np.arange(len(sample_ids))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, ylabel, logy) in zip(axes, metrics):
        a = df_a[metric].to_numpy(dtype=float)
        b = df_b[metric].to_numpy(dtype=float)

        ax.bar(x - width / 2, a, width=width, label=name_a)
        ax.bar(x + width / 2, b, width=width, label=name_b)
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{sid}" for sid in sample_ids])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        if logy:
            positive = np.concatenate([
                a[np.isfinite(a) & (a > 0)],
                b[np.isfinite(b) & (b > 0)],
            ])
            if len(positive) > 0:
                ax.set_yscale("log")

    axes[0].legend()
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)


def _paired_scatter(
    df_a: pd.DataFrame, name_a: str,
    df_b: pd.DataFrame, name_b: str,
    metrics: list[tuple[str, str]],
    title: str,
    stem: str,
) -> None:
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    sample_ids = df_a["sample_id"].astype(int).tolist()

    for ax, (metric, label) in zip(axes, metrics):
        a = df_a[metric].to_numpy(dtype=float)
        b = df_b[metric].to_numpy(dtype=float)

        finite = np.isfinite(a) & np.isfinite(b)
        a = a[finite]
        b = b[finite]
        ids = [sid for sid, keep in zip(sample_ids, finite) if keep]

        if len(a) == 0:
            ax.set_title(label + "\n(no finite data)")
            continue

        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)

        ax.scatter(a, b, s=75)
        for sid, xi, yi in zip(ids, a, b):
            ax.annotate(str(sid), (xi, yi))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")

        ax.set_xlabel(name_a)
        ax.set_ylabel(name_b)
        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)


def _cost_accuracy_scatter(dfs: list[tuple[str, pd.DataFrame]], title: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = ["o", "s", "^", "D", "P"]

    for i, (name, df) in enumerate(dfs):
        x = df["total_online_time_s"].to_numpy(dtype=float)
        y = df["uv_rms"].to_numpy(dtype=float)

        ax.scatter(x, y, s=75, marker=markers[i % len(markers)], label=name)

        for sid, xi, yi in zip(df["sample_id"].astype(int), x, y):
            ax.annotate(str(sid), (xi, yi))

    ax.set_xlabel("total_online_time_s = train_time_s + fe_time_s")
    ax.set_ylabel("uv_rms")
    positive = np.concatenate([df["uv_rms"].to_numpy(dtype=float) for _, df in dfs])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) > 0:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)


def _reference_table(df: pd.DataFrame, method_name: str, stem: str) -> None:
    cols = [
        "sample_id",
        "epochs_effort",
        "train_time_s",
        "fe_time_s",
        "newton_total_iters",
        "uv_rms",
        "uv_max",
        "train_residual_final",
        "newton_final_residual",
    ]
    existing = [c for c in cols if c in df.columns]
    out = df[existing].copy()
    out.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    print(f"[saved] {OUT_DIR / f'{stem}.csv'}")
    print(f"\n{method_name} reference summary (direct logged metrics only):")
    print(out.to_string(index=False))


def _print_metric_summary(name: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for c in cols:
        if c not in df.columns:
            continue
        a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            print(f"  {c:22s}: NO DATA")
            continue
        print(
            f"  {c:22s}: n={len(a):3d}  "
            f"min={a.min():.4g}  median={np.median(a):.4g}  max={a.max():.4g}"
        )


# ============================================================
# Main
# ============================================================
def main() -> None:
    # Load canonical non-warm runs
    param = _add_direct_metrics(_read_csv(PARAM_CSV, "Param"), "Param")
    otf   = _add_direct_metrics(_read_csv(OTF_CSV,   "OTF"),   "OTF")
    tl    = _add_direct_metrics(_read_csv(TL_CSV,    "TL-OTF"), "TL-OTF")

    # Warm-start TL is the only warm CSV needed for the main claims
    have_ws = TL_WS_CSV.exists()
    tl_ws = None
    if have_ws:
        tl_ws = _add_direct_metrics(_read_csv(TL_WS_CSV, "TL-OTF+WS"), "TL-OTF+WS")

        # Sanity check if use_fno_warmstart exists
        if "use_fno_warmstart" in tl_ws.columns:
            vals = tl_ws["use_fno_warmstart"].dropna().astype(float).unique()
            if len(vals) > 0 and not np.all(vals == 1):
                print("WARNING: TL warm-start CSV contains rows where use_fno_warmstart != 1")

    # Save cleaned tables
    param.to_csv(OUT_DIR / "strict_parametric_clean.csv", index=False)
    otf.to_csv(OUT_DIR / "strict_otf_clean.csv", index=False)
    tl.to_csv(OUT_DIR / "strict_tl_clean.csv", index=False)
    print(f"[saved] {OUT_DIR / 'strict_parametric_clean.csv'}")
    print(f"[saved] {OUT_DIR / 'strict_otf_clean.csv'}")
    print(f"[saved] {OUT_DIR / 'strict_tl_clean.csv'}")
    if have_ws:
        tl_ws.to_csv(OUT_DIR / "strict_tl_warmstart_clean.csv", index=False)
        print(f"[saved] {OUT_DIR / 'strict_tl_warmstart_clean.csv'}")

    # --------------------------------------------------------
    # 1) TL effect: OTF vs TL-OTF
    # --------------------------------------------------------
    shared_otf_tl_ids, shared_otf_tl = _shared_subset(("OTF", otf), ("TL-OTF", tl))
    if len(shared_otf_tl_ids) < MIN_SHARED_FOR_PLOTS:
        print("WARNING: no shared sample_ids between OTF and TL-OTF. Skipping TL-adaptation plots.")
    else:
        shared_otf_tl["OTF"].to_csv(OUT_DIR / "shared_otf_clean.csv", index=False)
        shared_otf_tl["TL-OTF"].to_csv(OUT_DIR / "shared_tl_clean.csv", index=False)
        print(f"[saved] {OUT_DIR / 'shared_otf_clean.csv'}")
        print(f"[saved] {OUT_DIR / 'shared_tl_clean.csv'}")

        _paired_bar_2x2(
            shared_otf_tl["OTF"], "OTF",
            shared_otf_tl["TL-OTF"], "TL-OTF",
            metrics=[
                ("epochs_effort", "epochs_effort", False),
                ("train_time_s", "train_time_s [s]", False),
                ("train_residual_final", "train_residual_final", True),
                ("uv_rms", "uv_rms", True),
            ],
            title="OTF vs TL-OTF on shared sample_ids (direct logged metrics only)",
            stem="fig_1_tl_adaptation_strict",
        )

    # --------------------------------------------------------
    # 2) Warm-start effect: TL-OTF vs TL-OTF+WS
    # --------------------------------------------------------
    if have_ws:
        shared_tl_ws_ids, shared_tl_ws = _shared_subset(("TL-OTF", tl), ("TL-OTF+WS", tl_ws))
        if len(shared_tl_ws_ids) < MIN_SHARED_FOR_PLOTS:
            print("WARNING: no shared sample_ids between TL-OTF and TL-OTF+WS. Skipping warm-start plots.")
        else:
            shared_tl_ws["TL-OTF"].to_csv(OUT_DIR / "shared_tl_nonwarm_clean.csv", index=False)
            shared_tl_ws["TL-OTF+WS"].to_csv(OUT_DIR / "shared_tl_warmstart_clean.csv", index=False)
            print(f"[saved] {OUT_DIR / 'shared_tl_nonwarm_clean.csv'}")
            print(f"[saved] {OUT_DIR / 'shared_tl_warmstart_clean.csv'}")

            _paired_bar_2x2(
                shared_tl_ws["TL-OTF"], "TL-OTF",
                shared_tl_ws["TL-OTF+WS"], "TL-OTF+WS",
                metrics=[
                    ("newton_total_iters", "newton_total_iters", False),
                    ("fe_time_s", "fe_time_s [s]", False),
                    ("total_online_time_s", "total_online_time_s [s]", False),
                    ("uv_rms", "uv_rms", True),
                ],
                title="TL-OTF vs TL-OTF+warmstart on shared sample_ids (direct logged metrics only)",
                stem="fig_2_warmstart_solver_strict",
            )

            _paired_scatter(
                shared_tl_ws["TL-OTF"], "TL-OTF",
                shared_tl_ws["TL-OTF+WS"], "TL-OTF+WS",
                metrics=[
                    ("fe_time_s", "fe_time_s"),
                    ("newton_total_iters", "newton_total_iters"),
                ],
                title="Warm-start paired solver comparison (direct logged metrics only)",
                stem="fig_3_warmstart_paired_scatter_strict",
            )

    # --------------------------------------------------------
    # 3) Practical online cost: OTF / TL-OTF / TL-OTF+WS
    # --------------------------------------------------------
    if have_ws:
        shared_online_ids, shared_online = _shared_subset(
            ("OTF", otf), ("TL-OTF", tl), ("TL-OTF+WS", tl_ws)
        )
        if len(shared_online_ids) >= MIN_SHARED_FOR_PLOTS:
            _cost_accuracy_scatter(
                [
                    ("OTF", shared_online["OTF"]),
                    ("TL-OTF", shared_online["TL-OTF"]),
                    ("TL-OTF+WS", shared_online["TL-OTF+WS"]),
                ],
                title="Online cost vs accuracy on shared sample_ids (direct logged metrics only)",
                stem="fig_4_online_cost_vs_accuracy_strict",
            )
        else:
            print("WARNING: no shared sample_ids across OTF / TL-OTF / TL-OTF+WS. Skipping online cost-vs-accuracy plot.")
    else:
        shared_online_ids, shared_online = _shared_subset(("OTF", otf), ("TL-OTF", tl))
        if len(shared_online_ids) >= MIN_SHARED_FOR_PLOTS:
            _cost_accuracy_scatter(
                [
                    ("OTF", shared_online["OTF"]),
                    ("TL-OTF", shared_online["TL-OTF"]),
                ],
                title="Online cost vs accuracy on shared sample_ids (direct logged metrics only)",
                stem="fig_4_online_cost_vs_accuracy_strict",
            )
        else:
            print("WARNING: no shared sample_ids across OTF / TL-OTF. Skipping online cost-vs-accuracy plot.")

    # --------------------------------------------------------
    # 4) Parametric as offline-trained reference only
    # --------------------------------------------------------
    _reference_table(param, "Parametric", "parametric_reference_direct_metrics")

    # Optional reference-only warm CSV inspection, but not used in main figures
    if PARAM_WS_CSV.exists():
        try:
            param_ws = _add_direct_metrics(_read_csv(PARAM_WS_CSV, "Param+WS"), "Param+WS")
            _reference_table(param_ws, "Parametric warm-start reference", "parametric_warmstart_reference_direct_metrics")
        except Exception as e:
            print(f"WARNING: could not process PARAM_WS_CSV as reference-only: {e}")

    if OTF_WS_CSV.exists():
        try:
            otf_ws = _add_direct_metrics(_read_csv(OTF_WS_CSV, "OTF+WS"), "OTF+WS")
            _reference_table(otf_ws, "OTF warm-start reference", "otf_warmstart_reference_direct_metrics")
        except Exception as e:
            print(f"WARNING: could not process OTF_WS_CSV as reference-only: {e}")

    # Console summaries
    summary_cols = [
        "epochs_effort",
        "train_time_s",
        "fe_time_s",
        "total_online_time_s",
        "newton_total_iters",
        "uv_rms",
        "train_residual_final",
    ]
    _print_metric_summary("Parametric (reference only)", param, summary_cols)
    _print_metric_summary("OTF", otf, summary_cols)
    _print_metric_summary("TL-OTF", tl, summary_cols)
    if have_ws:
        _print_metric_summary("TL-OTF+warmstart", tl_ws, summary_cols)

    print("\nDone.")
    print(f"Outputs saved in: {OUT_DIR}")
    print(f"Duplicate handling policy: {DUPLICATE_POLICY!r}")


if __name__ == "__main__":
    main()