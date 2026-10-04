#!/usr/bin/env python3
"""Single-source generator for the NHSJS submission package.

Produces from one content definition:
  1. main.tex  -> compiled PDF (anonymized, superscript numeric citations)
  2. Word .docx (anonymized, online ((...)) citation format, tables rebuilt,
     figures embedded)
Reference strings follow the NHSJS reference format; the bibliography is
ordered by first appearance in the text.

Body prose follows the author's own paper (Terry Ding, "Fundamental
Information for Low-Turnover Equity ML", Oct 2026) as closely as possible:
sections reuse the author's sentences with the reproduced numbers.
"""
import re, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

# ---------------------------------------------------------------- references
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
def ref_docx(k):
    s = REFS[k]
    s = s.replace("\\textit{", "").replace("\\&", "&")
    s = re.sub(r"\}", "", s)
    return s

TITLE = "Fundamental Information for Low-Turnover Equity ML"

# ------------------------------------------------------------------ content
# Paragraphs are lists of segments: ("t", text) or ("b", bold_text).
# Citations: [[key]] or [[k1,k2]] placed immediately before punctuation.
# Blocks: ("p", segs) | ("sub", title) | ("eq", latex) | ("eq*", latex) |
#         ("table1",) | ("table2",) | ("table3",) | ("fig", name, caption)

ABSTRACT = [
 ("b", "Background/Objective. "),
 ("t", "Most stock-prediction research targets price moves a few days ahead and evaluates models on prediction metrics, leaving open whether accuracy gains survive turnover and transaction costs as portfolios. This study asks whether slow-moving fundamental data, traded by a long-memory model as a low-turnover strategy, beats technical signals. "),
 ("b", "Methods. "),
 ("t", "A Mamba-inspired selective state space (MISS) model was trained on point-in-time SEC fundamental features (profitability, growth, valuation, balance-sheet strength) to predict 63-day sector-neutral forward returns for S&P 500 stocks. LSTM, StockMixer, and graph neural network baselines, plus matched 63-day and 5-day technical regimes, ran under identical rules. Walk-forward retraining covered test years 2021--2025; scores were traded as a sector-demeaned top-10/bottom-10 long-short book at 15 bps one-way cost and scored on return, Sharpe ratio, turnover, cost sensitivity, a block bootstrap, and the Deflated Sharpe Ratio. "),
 ("b", "Results. "),
 ("t", "The fundamental strategy returned 17.53\\% per year at Sharpe 1.104 with about 32 trade events per year, against $-6.84\\%$ ($-0.299$) for matched long-horizon technicals and $-4.48\\%$ ($-0.276$, 605 events) for 5-day technicals. The block bootstrap gave $\\Pr(\\mathrm{Sharpe~gap} > 0) = 0.988$; the fundamental strategy was the only one of twelve configurations with Deflated Sharpe near 1.0, and MISS led every architecture on fundamentals. "),
 ("b", "Conclusions. "),
 ("t", "Fundamental information pays when the horizon matches how slowly fundamentals change, and selective state-space memory suits sparse fundamental data. The edge is regime- and architecture-dependent, and the trained technical scores carried no out-of-sample signal. "),
 ("b", "Keywords: "),
 ("t", "financial machine learning; fundamental data; state space models; Mamba; long-horizon trading; walk-forward validation; transaction costs."),
]

INTRO = [
 ("p", [("t", "The goal of accurately predicting stock prices has been the core purpose of financial machine learning research for decades. This research is valuable to investors, as better predictions can help create more robust and consistent investment strategies. Neural networks are particularly useful for achieving this task, as their ability to capture non-linear trends is key for navigating the complex space of stock markets. That is why individual researchers along with large asset managers and funds are increasingly using Artificial Intelligence (AI) to generate accurate stock price signals [[fischer2018,gkx2020]]. To accurately evaluate stock prediction methodologies, it is necessary to analyze each underlying variable. Indeed, AI has already transformed the financial sector by automating tasks like fraud detection, risk assessment, credit scoring, and algorithmic trading [[lopez2018]]. A growing sector of this field is stock price prediction (SPP), where models are directly used to predict future stock returns or price movements. AI is employed for this task because of its ability to recognize patterns in data that humans normally would not be able to. For example, Fischer and Krauss demonstrated that deep LSTM networks can generate statistically significant daily return signals across a universe of S&P 500 constituents over a long out-of-sample period [[fischer2018]].")]),
 ("p", [("t", "Stock prices are not just related to the company; a variety of factors influence prices. Financial machine learning models typically ingest fundamental, technical, and sentimental factors. Fundamentally, the company's earnings, products, balance sheet, and broader economic indicators like interest rates and inflation all affect how inherently valuable a company is. On a technical level, market dynamics like supply and demand, liquidity, volume, volatility, and a variety of trading patterns can move stock prices in the short term. Finally, at a sentiment level, news events like wars, political decisions, and government policies may affect investor psychology, driving price movement. To evaluate investment strategies, metrics like the Sharpe Ratio reveal how stable an investor's strategy is by comparing excess return against the volatility needed to achieve that return.")]),
 ("p", [("t", "Leveraging AI to accurately predict the future value of a stock presents a number of challenges that must be overcome. Stock datasets inherently contain a large amount of noise that models must learn to filter out while retaining useful variables [[harvey2016]]. The complex nature of how stock markets behave means that while a variety of factors affect a stock price, those factors can change at any moment. The amount of time-series data for an individual stock is also inherently limited, as there are only around 250 trading days a year. However, recent research has still shown that nonlinear ML models can identify economically useful signals from large stock panels [[fischer2018,gkx2020]].")]),
 ("p", [("t", "Currently, the majority of stock-prediction research has been oriented towards short-term price prediction, and many papers still evaluate models primarily with statistical prediction metrics instead of deploying their signals into simulated portfolios. This makes it difficult to tell whether a small improvement in prediction actually becomes a better investment strategy after turnover and transaction costs are included [[fischer2018,demiguel2020]]. Thus, we present a novel investment framework that leverages SEC fundamental data within a Mamba selective state space model to generate low-turnover, sector-neutral signals evaluated across five out-of-sample years using walk-forward validation and robustness diagnostics. To understand whether the gains come from the data or only from the architecture, the same experiment is also conducted with long-horizon technical data and short-horizon technical data, and MISS is compared against LSTM, StockMixer, and GNN models. To evaluate results, we leverage portfolio metrics like annual return, Sharpe ratio, turnover, maximum drawdown, transaction-cost sensitivity, block bootstrap testing, and the Deflated Sharpe Ratio framework [[bailey2014]].")]),
]

METHODS = [
 ("sub", "Data and Feature Construction"),
 ("p", [("t", "The experiment uses three primary categories of information: market data, fundamental data, and custom technical indicators. Daily Open-High-Low-Close-Volume data is taken from Yahoo Finance [[yahoo2026]], while fundamental accounting information is aligned using SEC filing data [[sec2025]]. The fundamental feature set includes profitability measures such as return on assets and operating margin, growth variables such as revenue and earnings growth, valuation variables such as earnings and free-cash-flow yield, balance-sheet variables such as leverage and liquidity, and quality variables such as accruals. Technical features include multi-period returns, realized volatility, moving-average distance, RSI, MACD, ATR, volume surprise, and price-volume trend measures.")]),
 ("p", [("t", "A major concern with fundamental data is look-ahead bias. A company's quarter may end before the market actually has access to the corresponding filing. Therefore, fundamental values are only added to the dataset after the public filing acceptance date instead of being backfilled to the fiscal period end. Features are winsorized using thresholds calculated from the training data and standardized cross-sectionally on each date. This prevents very large outliers from dominating the model while also avoiding the use of statistics from the future.")]),
 ("p", [("t", "The universe is the S&P 500. Because point-in-time fundamental coverage is incomplete, the fundamental regime scores 161--201 names per year against 441--474 for the technical regimes. That coverage gap matters for the portfolio results (see Discussion).")]),
 ("sub", "Target and Horizon"),
 ("p", [("t", "The main hypothesis of this paper is that the usefulness of a feature depends on the horizon at which it is evaluated. Fundamental variables change slowly, so forcing them to predict the next one or two days may create an unnecessary mismatch between the data and the target. The main long-horizon task therefore predicts a 63-trading-day forward return, which is approximately one quarter. A shorter five-day target is used for the technical benchmark. This follows the argument that research should consider market fundamentals that move prices over longer periods rather than only short-lived technical price moves [[shiller1981]]. For stock i on day t, the main target is the stock's forward return minus the average forward return of other stocks in its sector:")]),
 ("eq", "y_{i,t}^{(63)} = r_{i,t\\rightarrow t+63} - \\bar{r}_{s(i),t\\rightarrow t+63},"),
 ("p", [("t", "where s(i) represents the stock's sector. This makes the task closer to selecting better companies within an industry rather than simply predicting whether the entire technology or energy sector will rise.")]),
 ("sub", "Mamba-Inspired Selective State Model"),
 ("p", [("t", "For each stock, the daily feature vector is projected into a hidden representation and passed through stacked selective state-space blocks. At a simplified level, the model can be represented as")]),
 ("eq", "h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t,"),
 ("eq", "z_t = C_t h_t + D x_t,"),
 ("p", [("t", "where $x_t$ is the current input and $h_t$ is the model's running hidden state. Unlike a standard state-space model with the same transition at every step, Mamba allows important parameters to depend on the current input [[gu2023]]. The practical intuition is that the model can learn that some observations should strongly update memory while others should mostly be ignored. This is the main reason the architecture is tested on sparse fundamental data.")]),
 ("p", [("t", "The MISS implementation uses two selective state-space blocks, residual projections, normalization, and a final scoring head. The model is kept at approximately 0.2 million trainable parameters so that performance gains cannot be explained only by giving it substantially more capacity than the baselines. Training combines a Huber regression loss with a pairwise ranking loss:")]),
 ("eq", "L = L_{\\mathrm{Huber}}(\\hat{y}, y) + \\lambda L_{\\mathrm{rank}}(\\hat{y}, y),"),
 ("p", [("t", "with $\\lambda = 0.25$. The regression component learns the magnitude of future returns while the ranking component more directly teaches the model which stocks should be ordered above others in the portfolio.")]),
 ("sub", "Baseline Models"),
 ("p", [("t", "Four model types are evaluated using the same targets and portfolio rules. LSTM acts as the standard recurrent baseline and tests whether gated sequential memory is enough to capture the useful long-term information; its gates retain historical state while mitigating the vanishing-gradient problem of plain recurrent networks [[hochreiter1997,rumelhart1986]]. StockMixer mixes information across indicators, time, and stocks and is expected to be particularly competitive when the input is a dense technical panel [[fan2024]]. GNN adds relationships between stocks using sector links and rolling return-correlation links, allowing information to propagate between connected companies [[feng2019,qian2024]]. Finally, MISS/Mamba uses selective state propagation without an explicit graph. Parameter counts and optimization budgets are kept in the same range across models. A Transformer baseline was not used because its quadratic attention cost is impractical for the 252-day input sequences here [[vaswani2017]].")]),
 ("sub", "Walk-Forward Training and Portfolio Construction"),
 ("p", [("t", "A walk-forward design is used so that every reported test year occurs strictly after the data used to train the model. The training window expands over time, a trailing block is used for validation, and the following calendar year is used as the out-of-sample test period. At each annual boundary the model is retrained using only information that would have been available at that point in time. Model checkpoints are selected using validation Rank Information Coefficient (RankIC), which measures whether higher model scores correspond to higher future stock returns.")]),
 ("p", [("t", "The model output is converted into a sector-neutral long-short portfolio in the concentrated form used by Fischer and Krauss [[fischer2018]]. First, scores are demeaned within sector on each date,")]),
 ("eq*", "\\tilde{s}_{i,t} = s_{i,t} - \\frac{1}{|s(i)|}\\sum_{j\\in s(i)} s_{j,t},"),
 ("p", [("t", "so that selection compares companies against their own sector rather than betting on whole sectors rising or falling. All stocks are then ranked globally by $\\tilde{s}$; the portfolio goes long the top k = 10 names and short the bottom k = 10 names with equal monetary weight (100\\% of net asset value long and 100\\% short, approximately zero net market exposure). Long-horizon signals use a wider exit band so that a position does not need to be replaced every time its rank changes slightly: a held name is retained while its rank stays within the top (bottom) 80 names. This directly lowers turnover. The 63-day regimes rebalance monthly and the 5-day regime weekly. Net returns subtract 15 basis points for each unit of one-way notional traded.")]),
 ("sub", "Evaluation and Robustness"),
 ("p", [("t", "The primary metrics are annual return, annualized Sharpe ratio, turnover, trade events, and drawdown. Because high backtest returns can be misleading when many strategies have been tested, the experiment also uses a moving-block bootstrap and year-level sign tests. The block bootstrap resamples groups of consecutive returns instead of treating every day as independent, which better preserves time-series dependence. Transaction costs are also varied from 0 to 50 basis points to test whether the strategy's advantage survives less favorable execution assumptions. Finally, the Deflated Sharpe Ratio framework is used as a guide for controlling the effect of repeated model testing [[bailey2014]].")]),
 ("p", [("t", "The study uses only public market data and regulatory filings; no human participants or personal data are involved.")]),
]

RESULTS = [
 ("sub", "Fundamental vs. Technical Information"),
 ("p", [("t", "Table 1 reports the five out-of-sample years for the MISS architecture. The most important result is not simply that the fundamental model achieves a higher average return. It does so while trading far less often. The long-horizon fundamental strategy records a mean annual return of 17.53\\% and a mean Sharpe ratio of 1.104 with approximately 32 portfolio trade events per year. The matched long-horizon technical strategy records $-6.84\\%$ annual return and $-0.299$ Sharpe with 254 events per year, while the five-day technical strategy records $-4.48\\%$ annual return and $-0.276$ Sharpe with approximately 605 trade events per year.")]),
 ("p", [("t", "The advantage is not consistent in every year. The fundamental strategy is negative in 2021 ($-1.9\\%$) before its strongest relative years, and its two best absolute years are 2025 (46.4\\%) and 2023 (18.2\\%). This variation is important because it shows that fundamental information is not simply a guaranteed source of excess return. Rather, it appears to become more valuable when market conditions reward persistent company-level information. Across all five years, the fundamental model outperforms the short-horizon technical strategy in three years on both annual return and Sharpe, and it outperforms the matched long-horizon technical strategy in all five years on annual return.")]),
 ("p", [("t", "Relative to the matched 63-day technical model, the long-horizon fundamental model improves mean annual return by 24.37 percentage points and Sharpe by 1.403. Relative to the short-horizon technical strategy, the improvement is 22.01 percentage points in annual return and 1.380 in Sharpe. Figure 1 also shows that the higher average result is not created by a smooth constant gain; performance varies substantially across years, with the fundamental curve compounding to a net asset value of 2.13 over the five years while both technical curves end below 0.71.")]),
 ("table1",),
 ("fig", "fig1_equity", "Cumulative net asset value of the three MISS information regimes, 2021--2025, net of 15 bps one-way costs. Fundamental 63d compounds \\$1 to \\$2.13; Technical 63d and 5d end at \\$0.67 and \\$0.71."),
 ("sub", "Model Architecture Comparison"),
 ("p", [("t", "The next experiment tests whether the result comes only from choosing Mamba. Table 2 compares MISS against StockMixer, GNN, and LSTM under the same three information regimes. MISS performs best on the long-horizon fundamental task, where it records Sharpe 1.104 and annual return 17.53\\%, ahead of GNN (0.839), LSTM (0.665), and StockMixer ($-0.239$). On the technical regimes no architecture is reliably profitable in this reproduction: GNN (0.174) and LSTM (0.115) are mildly positive on 63-day technical information while MISS is negative ($-0.299$), and every architecture has a negative mean Sharpe on the five-day task.")]),
 ("table2",),
 ("fig", "fig2_arch_sharpe", "Mean out-of-sample Sharpe ratio by architecture and information regime (five-year means, net of 15 bps one-way costs). MISS leads on Fundamental 63d; no architecture is positive on Technical 5d."),
 ("p", [("t", "This comparison suggests that architecture should be matched to the information being processed, but it also shows the limit of that claim. Table 3 reports the mean out-of-sample RankIC of each score file, a portfolio-independent measure of signal. MISS has the strongest sector-demeaned fundamental RankIC (0.039), which is exactly the ranking the portfolio in Table 2 produces. StockMixer's fundamental RankIC is negative ($-0.012$ raw), so its last-place portfolio result reflects an absent signal rather than a poor portfolio rule. Most strikingly, every technical RankIC is within $\\pm 0.03$ of zero, and MISS's 63-day technical RankIC is 0.001: in this reproduction the technical scores carry almost no cross-sectional signal for any architecture, which is why the technical portfolios in Table 2 hover around or below zero regardless of construction. Mamba's selective state updates appear useful when an important fundamental observation may need to remain in memory for a long period [[gu2023]]; no comparable advantage is available where the underlying signal itself is absent.")]),
 ("table3",),
 ("sub", "Turnover and Transaction Costs"),
 ("p", [("t", "The original motivation for using long-horizon fundamental information is not only prediction accuracy but also the ability to generate a strategy that trades less frequently. Figure 3 increases one-way transaction costs from 0 to 50 basis points. The long-horizon fundamental portfolio turns over only about 4.7 times its net asset value per year, against roughly 63 times for the five-day technical portfolio, and its mean Sharpe declines only from 1.136 at zero cost to 1.028 at 50 bps (return 18.02\\% to 16.40\\%). The short-horizon technical strategy, which is only mildly positive before costs (Sharpe 0.298 at 0 bps), collapses to $-1.558$ at 50 bps. As costs increase, the short-horizon strategy loses return much faster because its signal requires substantially more trading. This supports the central idea of the paper: a model should not be evaluated independently from the amount of trading required to use its predictions.")]),
 ("fig", "fig3_cost_sensitivity", "Sensitivity of the three MISS strategies to one-way transaction costs, 0--50 bps (five-year means). Dashed line: 15 bps baseline. The low-turnover Fundamental 63d book is nearly cost-insensitive; the high-turnover Technical 5d book collapses."),
 ("sub", "Robustness Testing"),
 ("p", [("t", "The main limitation of a five-year experiment is that five yearly observations do not provide enough statistical power for a strong significance claim. The fundamental strategy beats the short-horizon technical strategy on annual return in three of five years. A one-sided exact sign test for three or more wins out of five gives p = 0.50. Against the matched long-horizon technical strategy, the fundamental strategy wins all five years, corresponding to p = 0.031. The block bootstrap is more informative than the year count: the observed Sharpe difference between the fundamental and short-horizon technical strategies is 1.520 (bootstrap mean 1.530, 95\\% interval [0.215, 2.880]), and the probability that the fundamental strategy has a positive Sharpe difference over the short-horizon technical strategy is 0.988 (Figure 4). This provides stronger directional support than the five-year sign test, but it still does not prove that the same advantage will remain stable in every future market regime.")]),
 ("fig", "fig4_bootstrap", "Moving-block bootstrap distribution of the Sharpe difference between the fundamental and short-horizon technical MISS strategies (10,000 resamples, 21-day blocks). The observed difference is 1.52 and $\\Pr(\\mathrm{diff} > 0) = 0.988$."),
 ("p", [("t", "Two further checks harden the result. First, under the Deflated Sharpe Ratio framework of Bailey and L\\'{o}pez de Prado [[bailey2014]], which adjusts for the twelve configurations evaluated, the MISS fundamental strategy is the only configuration with a Deflated Sharpe Ratio of approximately 1.0 (benchmark SR0 = 0.828); every other architecture-regime combination is approximately zero. Second, the result is not an artifact of one gross-exposure choice: scaling the same signals to a vol-matched book (top-7/bottom-7 at 150\\%/150\\% gross) produces a mean annual return of 31.07\\%, a Sharpe of 0.950, 27 trade events per year, and realized volatility of 24.7\\%, i.e. return and volatility scale with gross exposure while the Sharpe ratio moves only modestly (0.950 versus 1.104). The robustness results therefore support a narrower conclusion. The data suggests that long-horizon fundamentals add economic value in the tested framework, but the magnitude of the effect varies materially from year to year. This is also why model architecture and portfolio turnover should be evaluated together instead of reporting only the highest backtest return.")]),
]

DISCUSSION = [
 ("p", [("t", "This paper evaluates whether fundamental information can be more useful for stock prediction when the prediction and trading horizon is intentionally extended. A Mamba-Inspired Selective State Space model is used because its selective memory mechanism is designed to retain important information across long sequences while filtering less useful observations. Across the five out-of-sample years, the long-horizon fundamental MISS strategy records a mean annual return of 17.53\\% and a Sharpe ratio of 1.104, compared with $-6.84\\%$ and $-0.299$ for matched long-horizon technical information and $-4.48\\%$ and $-0.276$ for the five-day technical strategy. At the same time, the fundamental strategy requires only about 32 portfolio trade events per year compared with approximately 605 for the short-horizon technical strategy, and it is the only configuration whose performance survives both a 50 bps cost stress and a multiple-testing (Deflated Sharpe) adjustment.")]),
 ("p", [("t", "The model comparison also shows that the conclusion is not simply that Mamba is better than every other neural-network architecture. MISS performs best on the sparse long-horizon fundamental regime, while in this reproduction StockMixer does not reproduce its expected strength and no architecture extracts a reliable technical signal. This supports a more specific interpretation: architecture and information type interact, and the interaction is bounded by the signal actually present in the scores (Table 3). Mamba's selective state mechanism is most useful when a small number of important observations need to remain relevant for a long period. GNNs provide another competitive direction (Sharpe 0.839 on the fundamental task) because stock prices are not independent and relationships between firms can carry predictive information.")]),
 ("p", [("t", "There are still several limitations. Sector neutrality removes large industry bets, but it does not fully remove exposures to market beta, size, value, momentum, or liquidity. Fundamental data also creates difficult data-engineering problems because accounting tags vary across companies and older filings must be reconstructed exactly as they were known at the time. Transaction costs can also differ significantly between large liquid stocks and smaller companies. Finally, five out-of-sample years are not enough to cover every possible market regime. Future research should test a larger historical period, add explicit factor regressions, study performance by liquidity and market-cap bucket, and isolate how much of MISS's performance comes from selectivity itself rather than other architectural choices.")]),
 ("p", [("t", "Because this edition reports an independent reproduction, three differences from the values circulated with the original study deserve explicit statement. First, the fundamental Sharpe ratio is close to the 1.221 reported previously (1.104 here), but the annual return is lower at the faithful, unlevered gross exposure (17.53\\% versus 32.72\\%) because realized portfolio volatility is 13.8\\% rather than roughly 25\\%: the point-in-time fundamental universe contains 161--201 names per year rather than the full S&P 500, so the selected tails are less extreme, and no leverage is applied. Scaling gross exposure to match that volatility recovers a 31.07\\% mean return at a Sharpe of 0.950 (see Robustness Testing above), which locates the difference in exposure and universe breadth rather than in the signal. Second, the previously reported strength of StockMixer and of the technical regimes does not reproduce: the retrained technical scores carry near-zero out-of-sample RankIC for every architecture, and StockMixer's fundamental RankIC is negative (Table 3), so those portfolio results cannot be recovered by any portfolio construction applied to these scores. Third, the year-by-year path differs: 2021 is the weakest fundamental year here (RankIC approximately zero) rather than the strongest, so year-level agreement should not be expected even where five-year means are close. These are limitations of the reproduction's data coverage and training runs, and they are stated here so the results can be judged on what was actually measured.")]),
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
    body = ("\\begin{tabular}{l rrrr}\n\\toprule\n"
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

def tex_eq(latex, numbered=True):
    env = "equation" if numbered else "equation*"
    return "\\begin{%s}\n%s\n\\end{%s}\n" % (env, latex, env)

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
            elif b[0] == "sub":
                L.append("\\subsection{%s}\n" % tex_escape(b[1]))
            elif b[0] == "eq":
                L.append(tex_eq(tex_escape(b[1]), numbered=True))
            elif b[0] == "eq*":
                L.append(tex_eq(tex_escape(b[1]), numbered=False))
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

def docx_add_table(doc, caption, header, rows):
    from docx.shared import Pt
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
            elif b[0] == "sub":
                doc.add_heading(docx_clean(b[1]), level=2)
            elif b[0] in ("eq", "eq*"):
                p = doc.add_paragraph(); p.add_run("$ " + docx_clean(b[1]) + " $")
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
