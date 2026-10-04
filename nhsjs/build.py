#!/usr/bin/env python3
"""Single-source generator for the NHSJS submission package.

Produces from one content definition:
  1. main.tex  -> compiled PDF (anonymized, superscript numeric citations)
  2. Word .docx (anonymized, online ((...)) citation format, tables rebuilt,
     figures embedded)
Reference strings follow the NHSJS reference format; the bibliography is
ordered by first appearance in the text.
"""
import re, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

# ---------------------------------------------------------------- references
# NHSJS format: initials + surname. Title in sentence case. Journal. Vol. X,
# pg. Y-Z, Year, DOI.  (keys double as \cite labels)
REFS = {
 "fischer2018": "T. Fischer, C. Krauss. Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research. Vol. 270, pg. 654-669, 2018, https://doi.org/10.1016/j.ejor.2017.11.054.",
 "gkx2020": "S. Gu, B. Kelly, D. Xiu. Empirical asset pricing via machine learning. Review of Financial Studies. Vol. 33, pg. 2223-2273, 2020, https://doi.org/10.1093/rfs/hhaa009.",
 "lopez2018": "M. Lopez de Prado. \\textit{Advances in financial machine learning}. John Wiley \\& Sons, 2018.",
 "demiguel2020": "V. DeMiguel, A. Martin-Utrera, F. J. Nogales, R. Uppal. A transaction-cost perspective on the multitude of firm characteristics. Review of Financial Studies. Vol. 33, pg. 2180-2222, 2020, https://doi.org/10.1093/rfs/hhz085.",
 "harvey2016": "C. R. Harvey, Y. Liu, H. Zhu. ...and the cross-section of expected returns. Review of Financial Studies. Vol. 29, pg. 5-68, 2016, https://doi.org/10.1093/rfs/hhv059.",
 "gu2023": "A. Gu, T. Dao. Mamba: linear-time sequence modeling with selective state spaces. arXiv:2312.00752, 2023.",
 "shiller1981": "R. J. Shiller. Do stock prices move too much to be justified by subsequent dividends? American Economic Review. Vol. 71, pg. 421-436, 1981.",
 "bailey2014": "D. H. Bailey, M. Lopez de Prado. The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. Journal of Portfolio Management. Vol. 40, pg. 94-107, 2014.",
 "yahoo2026": "Yahoo Finance. Historical market data and price history. 2026, https://finance.yahoo.com.",
 "sec2025": "U.S. Securities and Exchange Commission. EDGAR application programming interfaces (APIs): submissions and XBRL data. SEC.gov, 2025, https://www.sec.gov/edgar/sec-api-documentation.",
 "hochreiter1997": "S. Hochreiter, J. Schmidhuber. Long short-term memory. Neural Computation. Vol. 9, pg. 1735-1780, 1997, https://doi.org/10.1162/neco.1997.9.8.1735.",
 "rumelhart1986": "D. E. Rumelhart, G. E. Hinton, R. J. Williams. Learning representations by back-propagating errors. Nature. Vol. 323, pg. 533-536, 1986.",
 "fan2024": "J. Fan, Y. Shen. StockMixer: a simple yet strong MLP-based architecture for stock price forecasting. Proceedings of AAAI. Vol. 38, pg. 8389-8397, 2024, https://doi.org/10.1609/aaai.v38i8.28681.",
 "feng2019": "F. Feng, X. He, X. Wang, C. Luo, Y. Liu, T.-S. Chua. Temporal relational ranking for stock prediction. ACM Transactions on Information Systems. Vol. 37, 2019, https://doi.org/10.1145/3309547.",
 "qian2024": "H. Qian, et al. MDGNN: multi-relational dynamic graph neural network for comprehensive and dynamic stock investment prediction. Proceedings of AAAI. Vol. 38, 2024, https://doi.org/10.1609/aaai.v38i13.29381.",
 "vaswani2017": "A. Vaswani, et al. Attention is all you need. Advances in Neural Information Processing Systems. Vol. 30, 2017.",
}
# docx versions: strip LaTeX markup from the ref strings
def ref_docx(k):
    s = REFS[k]
    s = s.replace("\\textit{", "").replace("\\&", "&")
    s = re.sub(r"\}", "", s)
    return s

TITLE = "Fundamental Information for Low-Turnover Equity ML"

# ------------------------------------------------------------------ content
# Paragraphs are lists of segments: ("t", text) or ("b", bold_text).
# Citations: [[key]] or [[k1,k2]] placed immediately before punctuation.
# Blocks: ("p", segs) | ("eq", latex) | ("table1",) | ("table2",) |
#         ("table3",) | ("fig", name, caption)

ABSTRACT = [
 ("b", "Background/Objective. "),
 ("t", "Most stock-prediction research targets price moves a few days ahead and evaluates models on prediction metrics, leaving open whether accuracy gains survive turnover and transaction costs as portfolios. This study asks whether slow-moving fundamental data, traded by a long-memory model as a low-turnover strategy, beats technical signals. "),
 ("b", "Methods. "),
 ("t", "A Mamba-inspired selective state space (MISS) model was trained on point-in-time SEC fundamental features (profitability, growth, valuation, balance-sheet strength) to predict 63-day sector-neutral forward returns for S&P 500 stocks. LSTM, StockMixer, and graph neural network baselines, plus matched 63-day and 5-day technical regimes under identical rules. Walk-forward retraining covered test years 2021--2025; scores were traded as a sector-demeaned top-10/bottom-10 long-short book at 15 bps one-way cost and scored on return, Sharpe ratio, turnover, cost sensitivity, a block bootstrap, and the Deflated Sharpe Ratio. "),
 ("b", "Results. "),
 ("t", "The fundamental strategy returned 17.53\\% per year at Sharpe 1.104 with about 32 trade events per year, against $-6.84\\%$ ($-0.299$) for matched long-horizon technicals and $-4.48\\%$ ($-0.276$, 605 events) for 5-day technicals. The block bootstrap gave $\\Pr(\\mathrm{Sharpe~gap} > 0) = 0.988$; the fundamental strategy was the only one of twelve configurations with Deflated Sharpe near 1.0, and MISS led every architecture on fundamentals. "),
 ("b", "Conclusions. "),
 ("t", "Fundamental information pays when the horizon matches how slowly fundamentals change, and selective state-space memory suits sparse fundamental data. The edge is regime- and architecture-dependent, and the trained technical scores carried no out-of-sample signal. "),
 ("b", "Keywords: "),
 ("t", "financial machine learning; fundamental data; state space models; Mamba; long-horizon trading; walk-forward validation; transaction costs."),
]

INTRO = [
 ("p", [("b", "Background and Context. "),
  ("t", "Stock price prediction is a core problem in financial machine learning, and neural networks are widely used for it because they fit nonlinear market structure[[fischer2018,gkx2020]]. Prices move on fundamentals, technicals, and sentiment: earnings and balance sheets set value, order flow and volatility move prices in the short run, and news shocks move investor psychology. The Sharpe ratio is the natural score for a strategy, since it prices the volatility a return costs. Financial data is noisy, nonstationary, and short---about 250 trading days a year per stock---yet nonlinear models have found useful signals in large panels[[fischer2018,gkx2020]], and algorithmic trading already relies on such models[[lopez2018]].")]),
 ("p", [("b", "Problem Statement and Rationale. "),
  ("t", "Most published models aim a few days out and are scored on prediction metrics, which says little about whether a small accuracy gain survives turnover and transaction costs once it becomes a portfolio[[fischer2018,demiguel2020]]. A model can also have lower prediction error and still lose money if it trades too much or fails in volatile stretches, and testing many variants can fake significance[[harvey2016]]. Research needs strategies evaluated as portfolios, not just predictions.")]),
 ("p", [("b", "Significance and Purpose. "),
  ("t", "This study tests a different combination: slow-moving fundamental data, a Mamba selective state space model whose input-dependent memory keeps important observations for months while letting noise fade[[gu2023]], and a sector-neutral portfolio designed to trade rarely. If the data's frequency, the model's memory, and the portfolio's trading frequency are matched, the result should be a more practical investment framework than a fancier short-term model.")]),
 ("p", [("b", "Objectives. "),
  ("t", "The objectives are (1) to test whether long-horizon fundamental signals beat matched long-horizon and short-horizon technical signals as portfolios; (2) to compare MISS against LSTM, StockMixer, and GNN architectures under identical rules; and (3) to verify robustness to transaction costs, sampling variation, and multiple testing.")]),
 ("p", [("b", "Scope and Limitations. "),
  ("t", "The study covers S&P 500 stocks over out-of-sample years 2021--2025. Point-in-time fundamental coverage is incomplete (161--201 scored names per year versus 441--474 for technicals), five years cannot cover every market regime, and sector neutrality removes industry bets but not factor exposures.")]),
 ("p", [("b", "Theoretical Framework. "),
  ("t", "The guiding idea is horizon matching: a feature's value depends on the horizon at which it is evaluated, so slow fundamentals should predict quarterly-scale returns, and the model's memory should run on the same slow clock[[shiller1981]].")]),
 ("p", [("b", "Methodology Overview. "),
  ("t", "Models are retrained yearly on expanding windows, scored daily, and traded as a concentrated long-short book; performance is reported as five-year means with bootstrap, sign-test, cost-sensitivity, and Deflated Sharpe diagnostics[[bailey2014]].")]),
]

METHODS = [
 ("p", [("b", "Research Design. "),
  ("t", "Walk-forward design. For each test year 2021--2025, the model trains on an expanding window ending two years prior, validates on the intervening year (checkpoints picked by validation RankIC), and tests on the target year, retrained at each annual boundary using only information available at that point.")]),
 ("p", [("b", "Sample. "),
  ("t", "S&P 500 constituents. The fundamental regime scores 161--201 names per year, limited by point-in-time SEC filing coverage; the technical regimes score 441--474 names per year.")]),
 ("p", [("b", "Data Collection. "),
  ("t", "Daily open-high-low-close-volume data come from Yahoo Finance[[yahoo2026]]. Fundamental accounting features (return on assets, operating margin, revenue and earnings growth, earnings and free-cash-flow yield, leverage, liquidity, accruals) are built from SEC EDGAR filings[[sec2025]] and enter only after each filing's acceptance date, never backfilled. Technical features (multi-period returns, realized volatility, moving-average distance, RSI, MACD, ATR, volume surprise, price-volume trend) are computed from prices and volume. Features are winsorized on training-data thresholds and cross-sectionally standardized per date.")]),
 ("p", [("b", "Models. "),
  ("t", "Four model types of about 0.2M parameters each see the same targets and portfolio rules: MISS (two selective state-space blocks plus a scoring head, trained with a Huber regression loss plus a pairwise ranking loss), LSTM[[hochreiter1997,rumelhart1986]], StockMixer (MLP mixing across indicators, time, and stocks)[[fan2024]], and GNN (message passing over sector and rolling return-correlation edges)[[feng2019,qian2024]]. A Transformer was not used as a baseline because its quadratic attention cost is impractical for the 252-day input sequences here[[vaswani2017]].")]),
 ("p", [("b", "Variables and Measurements. "),
  ("t", "The primary target is the 63-trading-day forward return minus the stock's sector mean return (sector-neutral stock selection); the technical benchmark uses a 5-day target. Model output is a daily score per stock. Portfolio outcomes are measured as annualized return, annualized Sharpe ratio ($\\times\\sqrt{252}$, risk-free zero), annual one-way turnover (traded notional divided by NAV), trade events per year (position entries plus exits), and maximum drawdown.")]),
 ("p", [("b", "Procedure. "),
  ("t", "Each date, scores are demeaned within sector and ranked globally, following the concentrated long-short form of Fischer and Krauss[[fischer2018]]: long the top 10 names, short the bottom 10, equal weight, 100\\% long / 100\\% short, approximately zero net market exposure. A held name is retained while its rank stays inside the top (bottom) 80, which lowers turnover. The 63-day books rebalance monthly and the 5-day book weekly; 15 basis points are charged per unit of one-way notional traded.")]),
 ("p", [("b", "Data Analysis. "),
  ("t", "Metrics are computed per test year and averaged over the five years. Costs are swept from 0 to 50 basis points one-way. A 10,000-resample moving-block bootstrap (21-day blocks) estimates the distribution of the Sharpe difference between the fundamental and short-horizon technical strategies. Year-level one-sided exact sign tests compare yearly wins. The Deflated Sharpe Ratio[[bailey2014]] adjusts the best Sharpe for the twelve configurations tested.")]),
 ("p", [("b", "Ethical Considerations. "),
  ("t", "No human participants, surveys, or personal data are involved. All inputs are public market data and regulatory filings; no informed consent or confidentiality procedures apply.")]),
]

RESULTS = [
 ("p", [("t", "Table 1 reports the five out-of-sample years for MISS. The fundamental strategy returns 17.53\\% per year at Sharpe 1.104 with about 32 trade events per year. The matched 63-day technical strategy returns $-6.84\\%$ (Sharpe $-0.299$) with 254 events; the 5-day technical strategy returns $-4.48\\%$ (Sharpe $-0.276$) with 605 events. The fundamental edge over the 63-day technical book is 24.37 percentage points of return and 1.403 of Sharpe; over the 5-day book, 22.01 points and 1.380. The edge is uneven across years: 2021 is negative ($-1.9\\%$) while 2025 ($+46.4\\%$) and 2023 ($+18.2\\%$) carry the mean, and the fundamental strategy beats the 5-day technical strategy in three of five years on both return and Sharpe. Figure 1 shows \\$1 compounding to \\$2.13 for the fundamental book while both technical books end below \\$0.71.")]),
 ("table1",),
 ("fig", "fig1_equity", "Cumulative net asset value of the three MISS information regimes, 2021--2025, net of 15 bps one-way costs. Fundamental 63d compounds \\$1 to \\$2.13; Technical 63d and 5d end at \\$0.67 and \\$0.71."),
 ("p", [("t", "Table 2 compares architectures under identical rules. MISS leads the fundamental task (Sharpe 1.104), ahead of GNN (0.839), LSTM (0.665), and StockMixer ($-0.239$). On technical tasks nothing is reliably positive: GNN (0.174) and LSTM (0.115) edge above zero on 63-day technicals, and every model is negative on the 5-day task (Figure 2).")]),
 ("table2",),
 ("fig", "fig2_arch_sharpe", "Mean out-of-sample Sharpe ratio by architecture and information regime (five-year means, net of 15 bps one-way costs). MISS leads on Fundamental 63d; no architecture is positive on Technical 5d."),
 ("p", [("t", "Table 3 gives the portfolio-independent explanation: mean out-of-sample RankIC of the scores. MISS has the strongest sector-demeaned fundamental RankIC (0.039), matching its portfolio rank. StockMixer's fundamental RankIC is negative ($-0.012$), so its last place is a signal failure rather than a backtest artifact. Every technical RankIC lies within $\\pm 0.03$ of zero, which is why those portfolios sit at or below zero under any construction.")]),
 ("table3",),
 ("p", [("t", "The fundamental book turns over about 4.7 times NAV per year against roughly 63 for the 5-day book. Figure 3 sweeps costs from 0 to 50 bps: the fundamental Sharpe moves only from 1.136 to 1.028 (return 18.02\\% to 16.40\\%), while the 5-day technical strategy falls from 0.298 to $-1.558$.")]),
 ("fig", "fig3_cost_sensitivity", "Sensitivity of the three MISS strategies to one-way transaction costs, 0--50 bps (five-year means). Dashed line: 15 bps baseline. The low-turnover Fundamental 63d book is nearly cost-insensitive; the high-turnover Technical 5d book collapses."),
 ("p", [("t", "Robustness: the fundamental strategy beats the 5-day technical strategy on annual return in 3 of 5 years (exact one-sided sign test $p = 0.50$) and the 63-day technical strategy in all 5 years ($p = 0.031$). The block bootstrap gives an observed Sharpe difference of 1.520, 95\\% interval $[0.215, 2.880]$, $\\Pr(\\mathrm{diff} > 0) = 0.988$ (Figure 4). The fundamental MISS strategy is the only one of twelve configurations with Deflated Sharpe near 1.0 (benchmark $\\mathrm{SR}_0 = 0.828$); the rest are near zero. A vol-matched variant (top-7/bottom-7 at 150\\%/150\\% gross) returns 31.07\\% at Sharpe 0.950 with 27 events per year and 24.7\\% realized volatility, showing return and volatility scale with gross exposure while Sharpe barely moves.")]),
 ("fig", "fig4_bootstrap", "Moving-block bootstrap distribution of the Sharpe difference between the fundamental and short-horizon technical MISS strategies (10,000 resamples, 21-day blocks). Observed difference 1.52; $\\Pr(\\mathrm{diff} > 0) = 0.988$."),
]

DISCUSSION = [
 ("p", [("b", "Restatement of Key Findings. "),
  ("t", "Across five out-of-sample years, long-horizon fundamental signals beat both technical benchmarks as portfolios, and MISS beat every baseline architecture on fundamentals in both Sharpe (1.104) and signal quality (demeaned RankIC 0.039).")]),
 ("p", [("b", "Implications and Significance. "),
  ("t", "The results support matching the data's frequency, the model's memory, and the portfolio's trading frequency. Mamba's selective memory fits sparse fundamental data: a few observations matter for months. The same architecture shows no edge where the signal is absent, which bounds the claim honestly.")]),
 ("p", [("b", "Connection to Objectives. "),
  ("t", "The first objective is met on means and on the 5-of-5 yearly comparison against matched technicals, though the 3-of-5 result against the 5-day strategy is not significant by the sign test alone; the bootstrap is stronger. The second is met for fundamentals; for technicals, no architecture found signal. The third is met: the edge survives 50 bps costs and multiple-testing adjustment.")]),
 ("p", [("b", "Recommendations. "),
  ("t", "Future work should widen point-in-time fundamental coverage toward the full 500 names, test longer histories, add factor regressions, and study performance by liquidity and market-cap bucket. Retraining the technical and StockMixer models is the direct route to testing whether their gaps are training artifacts.")]),
 ("p", [("b", "Limitations. "),
  ("t", "Five years cannot cover every regime, and the year path (notably 2021) differs from what a fuller universe might show. Sector neutrality does not remove factor exposures. Costs are modeled flat at 15 bps and vary in practice.")]),
 ("p", [("b", "Closing Thought. "),
  ("t", "The practical lesson is simple: trade slowly, remember longer, and let the data's own clock set the strategy.")]),
]

SECTIONS = [
 ("Abstract", None, ABSTRACT),
 ("Introduction", True, INTRO),
 ("Methods", True, METHODS),
 ("Results", True, RESULTS),
 ("Discussion", True, DISCUSSION),
]

# ------------------------------------------------------------------ tables
T1A = [
 ["Fund. 63d", "-1.9","-0.07","34", "23.0","1.49","22", "18.2","1.47","34", "1.9","0.23","36"],
 ["Tech. 63d", "-24.2","-1.53","226", "-3.5","-0.00","274", "10.6","0.77","328", "-15.1","-0.73","162"],
 ["Tech. 5d", "12.6","0.88","514", "-14.3","-0.80","658", "-31.8","-2.35","678", "24.8","1.62","578"],
]
T1B = [
 ["Fund. 63d", "46.4","2.41","34", "17.53","1.104","32"],
 ["Tech. 63d", "-2.0","-0.00","278", "-6.84","-0.299","254"],
 ["Tech. 5d", "-13.8","-0.73","598", "-4.48","-0.276","605"],
]
T2_SHARPE = [
 ["MISS / Mamba", "1.104","-0.299","-0.276"],
 ["StockMixer", "-0.239","-0.201","-0.039"],
 ["GNN", "0.839","0.174","-0.222"],
 ["LSTM", "0.665","0.115","-0.302"],
]
T2_RET = [
 ["MISS / Mamba", "17.53","-6.84","-4.48"],
 ["StockMixer", "-3.48","-1.69","0.11"],
 ["GNN", "11.92","0.60","-5.66"],
 ["LSTM", "9.71","0.80","-8.24"],
]
T3 = [
 ["Fund. 63d (raw)", "0.027","-0.012","0.027","0.036"],
 ["Fund. 63d (dem.)", "0.039","-0.002","0.033","0.033"],
 ["Tech. 63d (raw)", "0.001","0.011","0.010","0.032"],
 ["Tech. 5d (raw)", "0.008","0.003","0.002","-0.010"],
]

def num_tex(s):
    s = str(s)
    if re.fullmatch(r"-?[\d.]+", s):
        return "$%s$" % s if s.startswith("-") else s
    return s

# ------------------------------------------------------------- emitters
CITE_RE = re.compile(r"\[\[([a-z0-9_,]+)\]\]")

def cite_keys_in_order():
    seen, order = set(), []
    def scan(s):
        for m in CITE_RE.finditer(s):
            for k in m.group(1).split(","):
                if k not in seen:
                    seen.add(k); order.append(k)
    for seg in ABSTRACT:
        scan(seg[1])
    for _, _, blocks in SECTIONS:
        for b in blocks:
            if b[0] == "p":
                for _, t in b[1]:
                    scan(t)
    return order

def tex_cite(m):
    return "\\cite{%s}" % m.group(1)

def tex_escape(s):
    s = re.sub(r"(?<!\\)&", r"\\&", s)
    return s

def tex_text(s):
    s = CITE_RE.sub(tex_cite, s)
    return tex_escape(s)

def tex_par(segs):
    out = []
    for kind, t in segs:
        t = tex_text(t)
        out.append(("\\textbf{%s}" % t) if kind == "b" else t)
    return "".join(out) + "\n\n"

def tex_table1():
    def row(r):
        return " & ".join([r[0]] + [num_tex(x) for x in r[1:]]) + r" \\"
    h1 = (r"\begin{tabular}{l rrr rrr rrr rrr}" "\n\\toprule\n"
          r"& \multicolumn{3}{c}{2021} & \multicolumn{3}{c}{2022} & "
          r"\multicolumn{3}{c}{2023} & \multicolumn{3}{c}{2024} \\" "\n"
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}" "\n"
          r"Strategy & Ret. & SR & Ev. & Ret. & SR & Ev. & Ret. & SR & Ev. & Ret. & SR & Ev. \\" "\n\\midrule\n"
          + "\n".join(row(r) for r in T1A) + "\n\\bottomrule\n\\end{tabular}")
    h2 = (r"\begin{tabular}{l rrr rrr}" "\n\\toprule\n"
          r"& \multicolumn{3}{c}{2025} & \multicolumn{3}{c}{Five-year mean} \\" "\n"
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
          r"Strategy & Ret. & SR & Ev. & Ret. & SR & Ev. \\" "\n\\midrule\n"
          + "\n".join(row(r) for r in T1B) + "\n\\bottomrule\n\\end{tabular}")
    return ("\\begin{table}[t]\n\\centering\\footnotesize\n"
            "\\caption{Five-year out-of-sample results for the MISS architecture, net of 15 bps one-way costs. ``Events'' are portfolio trade/rebalance events per year.}\n"
            + h1 + "\n\n\\medskip\n" + h2 + "\n\\end{table}\n")

def tex_table2():
    def row(r):
        return " & ".join([r[0]] + [num_tex(x) for x in r[1:]]) + r" \\"
    body = ("\\begin{tabular}{l rrr}\n\\toprule\n"
            "\\multicolumn{4}{l}{\\textit{Sharpe ratio}} \\\\\n"
            "Model & Fund. 63d & Tech. 63d & Tech. 5d \\\\\n\\midrule\n"
            + "\n".join(row(r) for r in T2_SHARPE)
            + "\n\\midrule\n\\multicolumn{4}{l}{\\textit{Annual return (\\%)}} \\\\\n"
            + "\n".join(row(r) for r in T2_RET)
            + "\n\\bottomrule\n\\end{tabular}")
    return ("\\begin{table}[t]\n\\centering\\footnotesize\n"
            "\\caption{Mean out-of-sample performance by architecture and information regime, five-year means, net of 15 bps one-way costs.}\n"
            + body + "\n\\end{table}\n")

def tex_table3():
    def row(r):
        cells = [r[0]] + [("$\\mathbf{%s}$" % x if False else num_tex(x)) for x in r[1:]]
        return " & ".join(cells) + r" \\"
    rows = []
    for r in T3:
        cells = [r[0]] + [num_tex(x) for x in r[1:]]
        if r[0].startswith("Fund. 63d (dem"):
            cells[1] = "$\\mathbf{0.039}$"
        rows.append(" & ".join(cells) + r" \\")
    body = ("\\begin{tabular}{l rrrr}\n\\toprule\n"
            "Regime & MISS & StkMixer & GNN & LSTM \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    return ("\\begin{table}[t]\n\\centering\\footnotesize\n"
            "\\caption{Mean out-of-sample RankIC (score vs. forward return), 2021--2025, before any portfolio construction.}\n"
            + body + "\n\\end{table}\n")

def tex_fig(name, caption):
    return ("\\begin{figure}[t]\n\\centering\n"
            "\\includegraphics[width=0.92\\textwidth]{figures/%s.pdf}\n"
            "\\caption{%s}\n\\end{figure}\n") % (name, caption)

def build_tex(order):
    L = []
    L.append(r"""\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage[superscript]{cite}
\usepackage[hidelinks]{hyperref}
\makeatletter
\renewcommand{\@biblabel}[1]{#1.}
\makeatother
\setlength{\parskip}{4pt plus 1pt}
""")
    L.append("\\title{%s}\n\\date{October 2026}\n\n\\begin{document}\n\\maketitle\n" % TITLE)
    # abstract word count
    atext = " ".join(t for _, t in ABSTRACT)
    atext = CITE_RE.sub("", atext)
    atext = atext.replace("\\\\%", "%").replace("\\\\$", "$").replace("--", " ").replace("$", "")
    nwords = len(atext.split())
    print("abstract word count:", nwords, "(limit 200-250)" + ("" if 200 <= nwords <= 250 else "  <-- OUT OF RANGE"))
    for title, numbered, blocks in SECTIONS:
        L.append(("\\section{%s}\n" % title) if numbered else ("\\section*{%s}\n" % title))
        if title == "Abstract":
            L.append(tex_par(ABSTRACT))
            continue
        for b in blocks:
            if b[0] == "p":
                segs = b[1] if title == "Abstract" else [x for x in b[1] if x[0] != "b"]
                L.append(tex_par(segs))
            elif b[0] == "table1":
                L.append(tex_table1())
            elif b[0] == "table2":
                L.append(tex_table2())
            elif b[0] == "table3":
                L.append(tex_table3())
            elif b[0] == "fig":
                L.append(tex_fig(b[1], tex_text(b[2])))
    L.append("\\begin{thebibliography}{%d}\n" % len(order))
    for i, k in enumerate(order, 1):
        L.append("\\bibitem{%s} %s\n\n" % (k, REFS[k]))
    L.append("\\end{thebibliography}\n\n\\end{document}\n")
    return "".join(L)

# ---------------------------------------------------------------- docx
def docx_text_runs(par, s):
    """Add runs to a docx paragraph, converting [[k1,k2]] to ((...)) cites."""
    pos = 0
    for m in CITE_RE.finditer(s):
        if m.start() > pos:
            par.add_run(docx_clean(s[pos:m.start()]))
        keys = m.group(1).split(",")
        for j, k in enumerate(keys):
            if j > 0:
                r = par.add_run(",")
                r.font.superscript = True
            par.add_run("((%s))" % ref_docx(k))
        pos = m.end()
    if pos < len(s):
        par.add_run(docx_clean(s[pos:]))

def docx_clean(s):
    s = s.replace("\\\\%", "%").replace("\\\\$", "$").replace("\\\\&", "&")
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    return s

def docx_par(doc, segs, style=None):
    from docx.shared import Pt
    p = doc.add_paragraph(style=style)
    for kind, t in segs:
        t = docx_clean(t)
        pos = 0
        buf = ""
        # split text and cite markers, preserving bold for text runs
        for m in CITE_RE.finditer(t):
            if m.start() > pos:
                r = p.add_run(t[pos:m.start()])
                if kind == "b":
                    r.bold = True
            keys = m.group(1).split(",")
            for j, k in enumerate(keys):
                if j > 0:
                    rr = p.add_run(","); rr.font.superscript = True
                p.add_run("((%s))" % ref_docx(k))
            pos = m.end()
        if pos < len(t):
            r = p.add_run(t[pos:])
            if kind == "b":
                r.bold = True
    pf = p.paragraph_format
    pf.space_after = Pt(4)
    return p

def docx_add_table(doc, caption, header, rows, widths=None):
    from docx.shared import Pt, Inches
    cap = doc.add_paragraph()
    r = cap.add_run(caption)
    r.bold = True
    r.font.size = Pt(9)
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.style = "Table Grid"
    for j, h in enumerate(header):
        c = tbl.cell(0, j); c.text = ""
        rr = c.paragraphs[0].add_run(h); rr.bold = True; rr.font.size = Pt(8)
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = tbl.cell(i, j); c.text = ""
            rr = c.paragraphs[0].add_run(str(v)); rr.font.size = Pt(8)
    doc.add_paragraph()

def build_docx(order):
    from docx import Document
    from docx.shared import Pt, Inches
    doc = Document()
    st = doc.styles["Normal"]
    st.font.size = Pt(12); st.font.name = "Times New Roman"
    for hs in ("Heading 1", "Heading 2"):
        doc.styles[hs].font.size = Pt(14 if hs == "Heading 1" else Pt(12))
    t = doc.add_paragraph(); r = t.add_run(TITLE); r.bold = True; r.font.size = Pt(16)
    doc.add_paragraph()
    for title, numbered, blocks in SECTIONS:
        doc.add_heading(title, level=1)
        if title == "Abstract":
            docx_par(doc, ABSTRACT)
            continue
        for b in blocks:
            if b[0] == "p":
                segs = b[1] if title == "Abstract" else [x for x in b[1] if x[0] != "b"]
                docx_par(doc, segs)
            elif b[0] == "table1":
                docx_add_table(doc, "Table 1. Five-year out-of-sample results for the MISS architecture, net of 15 bps one-way costs. Events are portfolio trade/rebalance events per year.",
                    ["Strategy", "2021 Ret.", "SR", "Ev.", "2022 Ret.", "SR", "Ev.", "2023 Ret.", "SR", "Ev.", "2024 Ret.", "SR", "Ev."], T1A)
                docx_add_table(doc, "Table 1 (continued).",
                    ["Strategy", "2025 Ret.", "SR", "Ev.", "Mean Ret.", "SR", "Ev."], T1B)
            elif b[0] == "table2":
                docx_add_table(doc, "Table 2. Mean out-of-sample Sharpe ratio by architecture and regime (five-year means).",
                    ["Model", "Fund. 63d", "Tech. 63d", "Tech. 5d"],
                    [[r[0]] + r[1:] for r in T2_SHARPE])
                docx_add_table(doc, "Table 2 (continued). Mean annual return (%) by architecture and regime.",
                    ["Model", "Fund. 63d", "Tech. 63d", "Tech. 5d"],
                    [[r[0]] + r[1:] for r in T2_RET])
            elif b[0] == "table3":
                docx_add_table(doc, "Table 3. Mean out-of-sample RankIC (score vs. forward return), 2021-2025, before any portfolio construction.",
                    ["Regime", "MISS", "StkMixer", "GNN", "LSTM"], T3)
            elif b[0] == "fig":
                img = os.path.join(FIGDIR, b[1] + ".png")
                doc.add_picture(img, width=Inches(6.0))
                cap = doc.add_paragraph()
                rr = cap.add_run("Figure. " + docx_clean(b[2])); rr.font.size = Pt(9)
    doc.add_heading("References", level=1)
    for i, k in enumerate(order, 1):
        p = doc.add_paragraph()
        p.add_run("%d. %s" % (i, ref_docx(k)))
    return doc

def main():
    order = cite_keys_in_order()
    print("citation order:", order)
    missing = [k for k in order if k not in REFS]
    assert not missing, missing
    assert set(order) == set(REFS.keys()), "uncited refs: %s" % (set(REFS.keys()) - set(order))
    tex = build_tex(order)
    open(os.path.join(HERE, "main.tex"), "w").write(tex)
    print("wrote main.tex (%d bytes)" % len(tex))
    doc = build_docx(order)
    out = os.path.join(HERE, "Fundamental Information for Low-Turnover Equity ML.docx")
    doc.save(out)
    print("wrote", out)

if __name__ == "__main__":
    main()
