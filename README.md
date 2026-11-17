# From Signal to Style

**Code, intermediate artefacts and audit trail for the manuscript**

<https://github.com/ZhaoyongChao-Blip/from-signal-to-style> · release `v1.0` · MIT

Archived with a permanent DOI: **10.5281/zenodo.24297971**

> *The bet was directional; the summary statistic was not.*

This archive accompanies a paper submitted to *Quantitative Finance*. It contains the
complete analysis pipeline for a study of a gradient-boosted cross-sectional model on
Chinese A-shares (3,317 stocks, 2014-01-01 → 2025-11-10), **including the six
falsification steps that destroyed the paper's original conclusion**.

The negative results are the contribution. The pipeline is released because the audit
trail — feature allow-lists, a stride correction for overlapping labels, a
randomized-score placebo, a quarterly decomposition, and a conditioning test on the sign
of the benchmark return — is more useful to a reader than the original finding was.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python scripts/00_bootstrap.py --check               # stages every script into work/
cd work && python t4b_daily_ic.py                     # 30 s, needs panel.parquet
cd work && python sign_split.py                       # 10 s, reproduces Table 7 exactly
```

`00_bootstrap.py` rebuilds the flat `work/` tree the scripts expect: they each compute
`ROOT = dirname(__file__)` and look for `data/`, `output/`, `models/`, `figs/` beside
themselves. It hardlinks, never copies, and never modifies a released file.

**Everything except a from-scratch panel rebuild runs offline** from the shipped
`assets/` (model scores) and `results/` (88 intermediate files).

## What is where

```
scripts/01_fetch/        t1_fetch.py        stock list + daily bars (akshare / East Money)
                         t1b_fetch_direct.py unadjusted path  (fqt=0) — gaps only, NOT returns
                         t1c_fetch_adj.py    adjustment factor (fqt=2) — returns come from here
scripts/02_features/     t3_features.py     46 features + 3 label variants -> panel.parquet
scripts/03_models/       t4_lgb.py          LightGBM, 9 configurations
                         t4b_daily_ic.py     daily IC, overlap-corrected  <-- start here
                         t7*_neut*.py        style neutralisation (4 factors, 8 cells)
scripts/04_backtest/     t5_backtest.py     NAV construction, 3 accounting bases
                         t8h_cost.py         cost grid, break-even ≈24 bp
                         t8l_final_grid.py   liquidity x holding-period grid
scripts/05_diagnostics/  t6_diagnose.py      limit-hit censor check
                         t7e_placebo.py      randomized-score placebo
                         t9d_excess_q.py     quarterly excess decomposition
scripts/06_walkforward/  t10*_walkforward.py rolling annual re-fit
scripts/07_attribution/  sign_split.py       up-day vs down-day split  <-- reproduces Table 7
                         beta_check.py       beta to EW market, alpha, R2
scripts/08_figures/      paper_figs.py       Figures 1-6 (600 dpi PNG)
                         fig_roadmap.py      Figure 7/8
                         figcheck.py         dpi / on-page font size / aspect audit
scripts/09_audit/        paper_numbers.py    every number -> numbers.json
                         xcheck.py           numbers.json vs manuscript, all assertions
                         verify_all_dois.py  Crossref title+author+year for all 37 DOIs
scripts/10_build/        build_paper.py      markdown -> .docx (python-docx)
                         split_anonymous.py  double-blind version
```

```
paper/manuscript.md        the manuscript source (markdown, 12 tables, 37 refs)
paper/From_Signal_to_Style_*.pdf   built PDF as submitted
paper/numbers.json         every number in the paper with its source file
figures/fig*.png           600-1551 dpi, real resolution (no upsampling)
assets/lgb_y_*.txt         model scores for 4 label variants (2.3 MB)
assets/feat_meta.json      stock universe + factor list + panel shape
results/                   all 88 intermediate CSV/MD files cited by the paper
logs/                      training and diagnostic logs, unedited
docs/PITFALL_LEAKAGE.md    the three leakage bugs, in detail
docs/ASSERTIONS.md         number -> file -> expression table
results/REPLICATION_CHECK.md   what was re-run at release, and what was not
```

## The claim, in one paragraph

A LightGBM model on 46 price/volume features earns a real out-of-sample IC
(0.0865 after overlap correction, t = 5.25) and, inside style cells, a 5-day return of
0.713% (t = 3.47). Under realized-execution accounting the long book still underperforms
the equal-weighted market by 9.43 percentage points a year. Conditioning on the sign of
the benchmark reveals why: the strategy is **short beta**. It loses 0.50% a day when the
market rises and gains 0.66% when it falls, and the two conditional means sit 0.8–1.2
percentage points apart while the time-weighted average is indistinguishable from zero.
The model learned a position, not a forecast.

## Reproducibility, honestly stated

- Seed: `20241111` everywhere. Placebo draws reproduce exactly. LightGBM does not
  reproduce bit-for-bit across BLAS builds; expect the third decimal of IC to move.
- The 2.1 GB `panel.parquet` is **not** shipped — 4,800 parquet files do not belong in a
  git repo. It rebuilds from `scripts/01` → `scripts/02` with network access. Historical
  adjustment factors are revised upstream, so a fresh pull will not be byte-identical.
- `data/raw/list.csv` and `feat_meta.json` are shipped to pin the stock universe. Without
  them the universe drifts as listings and delistings change, and the sample stops being
  the one in the paper.
- Anything that could not be reproduced at release is listed in
  `results/REPLICATION_CHECK.md`, not omitted.

## Data source and licence

Daily prices from East Money (`push2his.eastmoney.com`, `klt=101`) via the open-source
`akshare` package, forward-adjusted (`fqt=1/2`). No new data were created; all inputs are
publicly available from their original source. Code is released under the MIT licence
(see `LICENSE`). Text and figures © the authors.

## Citation

```bibtex
@misc{signal2style2026,
  title  = {From Signal to Style: A Directional Bet Disguised as Selection Skill},
  author = {Zhao, Yongchao},
  year   = {2026},
  note   = {Code and replication archive, version 1.0},
  doi    = {10.5281/zenodo.24297971}
}
```

Archived record: <https://zenodo.org/records/24297971> · DOI **10.5281/zenodo.24297971**
(version 1.0, published 2026-11-17, citable as-is; later revisions get their own version DOI
under the same concept DOI).
