# Paper (LaTeX reproduction edition)

`main.tex` is a reproduction edition of the paper, typeset in LaTeX with all
tables and Figures 1–4 generated from the real experimental results in
`../results/` (Option A construction; see `../REPORT.md`). It mirrors the
original paper's structure (Sections 1–5, Tables 1–2, Figures 1–4,
references) with the reproduced numbers, plus a RankIC signal table
(Table 3) and a replication discussion in Section 5.

- `figures/` — the four figure PDFs (same files as `../results/figures/`).
- `main.pdf` — the rendered paper (10 pages).

Build (either toolchain works):

```bash
# Tectonic (self-contained)
tectonic main.tex

# or pdfLaTeX
pdflatex main.tex && pdflatex main.tex
```
