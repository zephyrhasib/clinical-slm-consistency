"""
_style.py - shared, reproducible plotting style for the Results figures.

Colorblind-safe (Okabe-Ito) model palette, IEEE column-friendly sizing, and a
save() helper that writes both vector PDF (for the report) and PNG (for preview)
to outputs/figures/. Imported by every fig_*.py script.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "figures" / "data"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# IEEE two-column geometry (inches)
COL = 3.35          # single column
DBL = 6.9           # full text width

# canonical orders
MODELS = ["Qwen-MediCare-BD", "Gemma-3-4B-it", "Phi-4-mini-instruct"]
QUANTS = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
QUANT_TICK = ["Q6", "Q5", "Q4", "Q3"]   # compact x ticks; full names in caption

# IEEE print-aware palette: conservative hues with distinct luminance, paired
# with line styles so the three models stay separable in grayscale printing.
MODEL_COLOR = {
    "Qwen-MediCare-BD":   "#2166AC",   # blue
    "Gemma-3-4B-it":      "#B2182B",   # red
    "Phi-4-mini-instruct":"#4D4D4D",   # dark grey
}
MODEL_LS = {"Qwen-MediCare-BD": "-", "Gemma-3-4B-it": "--", "Phi-4-mini-instruct": ":"}
MODEL_MARKER = {"Qwen-MediCare-BD": "o", "Gemma-3-4B-it": "s", "Phi-4-mini-instruct": "^"}
MODEL_LABEL = {
    "Qwen-MediCare-BD":    "Qwen (specialist)",
    "Gemma-3-4B-it":       "Gemma (generalist)",
    "Phi-4-mini-instruct": "Phi-4 (edge)",
}
# short names for in-figure legends (full names live in the prose)
MODEL_SHORT = {"Qwen-MediCare-BD": "Qwen", "Gemma-3-4B-it": "Gemma", "Phi-4-mini-instruct": "Phi-4"}

# five-axis colour (for the by-model dose-response and other per-metric plots)
METRIC_COLOR = {
    "BERTScore (semantic)": "#0072B2",
    "ROUGE-L (lexical)":    "#56B4E9",
    "UMLS-F1 (clinical)":   "#009E73",
    "Numeric Jaccard":      "#E69F00",
    "NLI contradiction":    "#CC79A7",
}

_RC = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,      # editable text in vector output
    "ps.fonttype": 42,
}


def apply():
    plt.rcParams.update(_RC)


def save(fig, name):
    """Write <name>.pdf and <name>.png to outputs/figures/ and report sizes."""
    pdf = FIGDIR / f"{name}.pdf"
    png = FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    w, h = fig.get_size_inches()
    print(f"  saved {name:42s} {w:.2f}x{h:.2f} in  -> {pdf.name}, {png.name}")
    plt.close(fig)
