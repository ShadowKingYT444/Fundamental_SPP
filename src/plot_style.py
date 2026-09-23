"""Publication figure house style, distilled from Chen Liu's figures4papers
repository (https://github.com/ChenLiu-1996/figures4papers), vendored at
~/workspace/figures4papers for reference.

Rules adopted:
  * Typography: sans-serif fallback stack (Arial > Helvetica > DejaVu Sans).
  * Minimalist axes: right/top spines off, frameless legends, linewidth 2.
  * Semantic palette: anchor blue for the proposed/key method (MISS, fund63),
    blue secondary for related series, red for contrasts, neutrals for
    baselines; black edges on bars for print-safe separation.
  * Layout: tight_layout(pad=2), dpi=300 PNG + vector PDF/SVG with editable
    text (svg.fonttype='none', pdf.fonttype=42).
  * No invented data: plotting scripts read only from results/metrics.json
    and results/equity/*.csv (see src/make_figures.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette ---
PALETTE = {
    "blue_main": "#0F4D92",       # proposed/key method (MISS, fund63)
    "blue_secondary": "#3775BA",  # related method/series
    "blue_light": "#9BC8FA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",      # contrast (e.g. high-frequency tech5)
    "neutral": "#CFCECE",
    "neutral_dark": "#767676",
    "ink": "#272727",
    "highlight": "#FFD700",
}

# Series assignment used consistently across all figures.
REGIME_COLORS = {
    "fund63": PALETTE["blue_main"],
    "tech63": PALETTE["blue_secondary"],
    "tech5": PALETTE["red_strong"],
}
ARCH_COLORS = {
    "miss": PALETTE["blue_main"],
    "stockmixer": PALETTE["blue_secondary"],
    "gnn": PALETTE["neutral_dark"],
    "lstm": PALETTE["red_strong"],
}
ARCH_LABELS = {
    "miss": "MISS",
    "stockmixer": "StockMixer",
    "gnn": "GNN",
    "lstm": "LSTM",
}
REGIME_LABELS = {
    "fund63": "Fundamental 63d",
    "tech63": "Technical 63d",
    "tech5": "Technical 5d",
}


def apply_style(font_size: int = 15) -> None:
    """Apply the figures4papers-like rcParams preset."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "legend.fontsize": font_size - 1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2,
            "axes.edgecolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_pub(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    """Finalize and export: PNG at dpi + vector PDF with editable text."""
    fig.tight_layout(pad=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
