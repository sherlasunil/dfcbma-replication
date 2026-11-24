"""
Reproduces the figures requested by the reviewer for the manuscript
"Digital-Focused Cross-Border M&A and Enterprise Innovation: Evidence from Chinese Firms".

Figures
-------
Fig_R1_event_study.png        -> Comment 4  (event-study plot with 95% CI)
Fig_R2_parallel_trends.png    -> Comment 11 (pre-treatment parallel-trends test)
Fig_R3_mediation_effects.png  -> Comment 14 (indirect effects, bootstrap 95% CI)

Input : reviewer_response_tables.xlsx  (sheets Cmt4, Cmt11, Cmt14)
Usage : python make_figures.py reviewer_response_tables.xlsx outdir
"""
import re
import sys
import os
import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

XLSX = sys.argv[1] if len(sys.argv) > 1 else "reviewer_response_tables.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
available = {f.name for f in font_manager.fontManager.ttflist}
family = "Times New Roman" if "Times New Roman" in available else "Liberation Serif"
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams["font.family"] = family
plt.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.grid"] = False
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
FS = 18


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=FS, color="black")
    ax.set_ylabel(ylabel, fontsize=FS, color="black")
    ax.set_title(title, fontsize=FS, color="black")
    ax.tick_params(axis="both", labelsize=FS, colors="black")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
        lab.set_color("black")


def num(x):
    """'−0.012', '0.048**', 0.004 -> float (None for dashes)."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("−", "-").replace("*", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def ci(s):
    a, b = re.findall(r"-?\d+\.\d+", str(s).replace("−", "-"))
    return float(a), float(b)


# ---------------------------------------------------------------- data
wb = openpyxl.load_workbook(XLSX, data_only=True)

ws = wb["Cmt4"]
ev = []
for r in ws.iter_rows(values_only=True):
    try:
        t = int(str(r[0]).replace("−", "-"))
    except (ValueError, TypeError):
        continue
    if not -5 <= t <= 5:
        continue
    b = num(r[1])
    if t == -1:
        ev.append((t, 0.0, 0.0, 0.0, None))
    else:
        lo, hi = ci(r[5])
        ev.append((t, b, lo, hi, num(r[4])))
t = np.array([e[0] for e in ev])
b = np.array([e[1] for e in ev])
lo = np.array([e[2] for e in ev])
hi = np.array([e[3] for e in ev])
p = [e[4] for e in ev]

# ---------------------------------------------------------------- Fig R1
c1 = "#1f77b4"
fig, ax = plt.subplots()
ax.errorbar(t, b, yerr=[b - lo, hi - b], fmt="o-", color=c1, ecolor=c1,
            capsize=6, lw=2.2, ms=9, elinewidth=2)
ax.fill_between(t, lo, hi, color=c1, alpha=0.15)
ax.axhline(0, color="black", lw=1.2, ls="--")
ax.axvline(-0.5, color="black", lw=1.2, ls=":")
ax.annotate("Treatment", xy=(-0.4, 0.215), fontsize=FS, color="black")
ax.annotate("(t = −1 reference)", xy=(-1.0, -0.075), xytext=(-4.9, -0.075),
            fontsize=14, color="black")
ax.set_xticks(t)
ax.set_ylim(-0.1, 0.26)
style(ax, "Event time (years relative to DFCBMA)",
      "Coefficient on Innovation", "Event Study")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig_R1_event_study.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig R2
c2 = "#2ca02c"
pre = t < 0
tp, bp, lp, hp = t[pre], b[pre], lo[pre], hi[pre]
fig, ax = plt.subplots()
ax.errorbar(tp, bp, yerr=[bp - lp, hp - bp], fmt="s", color=c2, ecolor=c2,
            capsize=7, lw=2.2, ms=11, elinewidth=2.2)
ax.plot(tp, bp, color=c2, lw=2)
ax.axhline(0, color="black", lw=1.2, ls="--")
for xi, bi, pi in zip(tp, bp, [e[4] for e in np.array(ev, dtype=object)[pre]]):
    if pi is not None:
        ax.annotate(f"p = {pi:.3f}", xy=(xi, bi), xytext=(xi, 0.052),
                    ha="center", fontsize=14, color="black")
ax.set_xticks(tp)
ax.set_ylim(-0.06, 0.07)
ax.set_xlim(-5.6, -0.4)
style(ax, "Pre-treatment event time (years)", "Coefficient on Innovation",
      "Parallel Trends")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig_R2_parallel_trends.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig R3
c3 = "#d62728"
ws = wb["Cmt14"]
med = {}
for r in ws.iter_rows(values_only=True):
    if r[0] in ("Managerial Myopia", "Financing Capacity") and isinstance(r[1], (int, float)):
        med[r[0]] = (r[1], r[3], r[4])
names = list(med)
ind = np.array([med[n][0] for n in names])
l3 = np.array([med[n][1] for n in names])
h3 = np.array([med[n][2] for n in names])
fig, ax = plt.subplots()
x = np.arange(len(names))
ax.bar(x, ind, width=0.5, color=c3, edgecolor="black", linewidth=1.5)
ax.errorbar(x, ind, yerr=[ind - l3, h3 - ind], fmt="none", ecolor="black",
            capsize=10, elinewidth=2.2, capthick=2.2)
for xi, v, hh in zip(x, ind, h3):
    ax.text(xi, hh + 0.0006, f"{v:.4f}", ha="center", va="bottom",
            fontsize=FS, color="black")
ax.axhline(0, color="black", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 0.016)
style(ax, "Mediation pathway", "Indirect effect (a × b)",
      "Indirect Effects")
fig.tight_layout()
fig.savefig(f"{OUT}/Fig_R3_mediation_effects.png", dpi=300)
plt.close(fig)
print("saved to", OUT, "| font:", family)
