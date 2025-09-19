import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mechanical2d_utilities import *

case_dir = os.path.join(os.path.dirname(__file__), f"./mechanical_2d_error_data/")
mea_ux, max_ux = [], []
mea_p11, max_p11 = [], []
for pc in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    mea_ux.append(np.loadtxt(case_dir + f"err_mae_ux_PC_{pc}_cleaned.txt")[:44])
    max_ux.append(np.loadtxt(case_dir + f"err_max_ux_PC_{pc}_cleaned.txt")[:44])
    mea_p11.append(np.loadtxt(case_dir + f"err_mae_p11_PC_{pc}_cleaned.txt")[:44])
    max_p11.append(np.loadtxt(case_dir + f"err_max_p11_PC_{pc}_cleaned.txt")[:44])

# Create DataFrames correctly
mea_ux_df = pd.DataFrame({str(pc): mea for pc, mea in zip([100, 50, 20, 10, 5, 2][::-1], mea_ux[::-1])})
max_ux_df = pd.DataFrame({str(pc): mx for pc, mx in zip([100, 50, 20, 10, 5, 2][::-1], max_ux[::-1])})
mea_p11_df = pd.DataFrame({str(pc): mea for pc, mea in zip([100, 50, 20, 10, 5, 2][::-1], mea_p11[::-1])})
max_p11_df = pd.DataFrame({str(pc): mx for pc, mx in zip([100, 50, 20, 10, 5, 2][::-1], max_p11[::-1])})

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row')

# Row 1
mea_ux_df.plot.box(ax=axes[0, 0], showfliers=False, showmeans=False)
mea_p11_df.plot.box(ax=axes[0, 1], showfliers=False, showmeans=False)

axes[0, 0].set_title("MAE Ux")
axes[0, 1].set_title("MAE P11")

# Row 2
max_ux_df.plot.box(ax=axes[1, 0], showfliers=False, showmeans=False)
max_p11_df.plot.box(ax=axes[1, 1], showfliers=False, showmeans=False)

axes[1, 0].set_title("Max Ux")
axes[1, 1].set_title("Max P11")

# Highlight PC=10 box across all plots
highlight_pc = "10"
highlight_color = "red"
dfs = [mea_ux_df, mea_p11_df, max_ux_df, max_p11_df]

for df, ax in zip(dfs, axes.ravel()):
    colnames = list(df.columns)
    if highlight_pc in colnames:
        idx = colnames.index(highlight_pc)  # position of PC=10

        # boxes are patches
        boxes = [patch for patch in ax.patches if isinstance(patch, plt.Rectangle)]
        if idx < len(boxes):
            box = boxes[idx]
            box.set_facecolor(highlight_color)
            box.set_alpha(0.4)

        # match whiskers, caps, and medians
        lines = ax.lines
        ncols = len(colnames)
        # Each box has: 2 whiskers, 2 caps, 1 median (5 lines)
        start = idx * 6
        whiskers_caps_median = lines[start:start+5]
        for line in whiskers_caps_median:
            line.set_color(highlight_color)
            line.set_linewidth(2)

# Common formatting
for ax in axes.ravel():
    ax.set_xlabel("Phase Contrast")
    ax.set_ylabel("Error")
    ax.set_yscale("log")   # log scale if needed
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(case_dir + f"PC_Plot_highlight10.png", dpi=300)
plt.show()
