# NHSJS submission package

Built from the 6-page condensed paper (`../paper/`), restructured to the
NHSJS submission guidelines (Title, Abstract, Introduction, Methods,
Results, Discussion, References; 12pt, single spacing).

## Files for the Google Form

1. **`Fundamental Information for Low-Turnover Equity ML.pdf`**
   Upload 1 — the paper, anonymized (no author names, affiliations, or
   acknowledgments), with standard superscript numeric citations and a
   references section. Name matches the manuscript title.
2. **`Fundamental Information for Low-Turnover Equity ML.docx`**
   Upload 2 — Word version for online publication. Citations use the
   double-parenthesis format `((full citation))`, with a superscript comma
   between multiple citations. Tables are rebuilt as Word tables; the four
   figures are embedded (and also available as PNGs in `figures/`).
3. **`main.tex`** — the LaTeX source, uploaded as supplementary information.

All three are generated from a single content definition by `build.py`,
so the PDF and Word versions cannot drift apart:

```bash
cd nhsjs
~/workspace/venv/bin/python build.py   # regenerates main.tex + .docx
~/workspace/tools/tectonic main.tex    # rebuilds main.pdf
```

Notes:
- Abstract is 248 words (limit 200-250), with the required
  Background/Objective, Methods, Results, Conclusions, Keywords subheads.
- Citations are superscript numbers placed before punctuation; references
  are numbered in order of first appearance, in NHSJS format.
- The abstract pasted in the submission form must use the reproduced
  numbers from this paper (17.53% / Sharpe 1.104 / 32 events), not the
  original paper's figures.
