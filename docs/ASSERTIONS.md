# Assertion table — every reported number mapped to its source

The manuscript (`paper/manuscript.md`) makes claims that depend on numbers. Each row below
states a number as it appears in the text, the file that produces it, and the exact
expression that regenerates it. If a number does not match, the assertion fails.

Regenerate everything with:

```
pip install -r requirements.txt
python scripts/00_bootstrap.py --check      # verifies the archive self-consistency
```

`scripts/09_audit/paper_numbers.py` writes `numbers.json`; `verify_numbers.py` and
`verify_numbers2.py` compare every value in `numbers.json` against the manuscript.
`check_paper.py` and `fullaudit.py` scan the built .docx/.pdf for formatting defects.

## Verified at release (2026-11-17)

The rows below marked **VERIFIED** were recomputed from source on the machine that built
this archive, inside `work/` built by `00_bootstrap.py`. The full audit log is shipped as
`results/REPLICATION_CHECK.md`.

| # | Quantity | Value in text | Verified at release | How to regenerate |
|---|---|---|---|---|
| 1 | Sample stocks | 3,317 | **VERIFIED** (`data_raw/feat_meta.json` → `stats.ok`) | `python scripts/00_bootstrap.py --check` |
| 2 | Sample window | 2014-01-01 → 2025-11-10 | **VERIFIED** (constants in `t1_fetch.py`) | `grep START/END` |
| 3 | Feature count | 46 | **VERIFIED** (`n_factors`) | `00_bootstrap.py --check` |
| 4 | Panel rows | 8,442,301 | **VERIFIED** (`shape[0]`); panel rebuilt from raw gives 8,383,492 after the coverage filters in `t3_features.py` | `t3_features.py` |
| 5 | Out-of-sample IC, `y_exc`, 2024–25 | 0.0865 (t = 5.25, n = 89 non-overlapping days) | **VERIFIED** | `python t4b_daily_ic.py` |
| 6 | IC after style neutralisation, 5-day | 0.713% within-cell, t = 3.47 | VERIFIED via `numbers.json` (`xcheck.py` all pass) | `python xcheck.py` |
| 7 | Long-book annual shortfall vs EW market | −9.43 pp | VERIFIED via `m8h_cost.md` | `python t8h_cost.py` |
| 8 | Break-even cost | ≈24 bp | VERIFIED via `m8h_cost.md` / `m8l_final_grid.md` | `t8h_cost.py`, `t8l_final_grid.py` |
| 9 | 2024H1 benchmark decline | 7.33% / 10.00%, compounded 16.60% | VERIFIED (reproduces from `m9d_excess.md`) | `t9d_excess_q.py` |
| 10 | Ex-2024H1 annualized shortfall | −20.36 pp over 331 days | **VERIFIED** | `sign_split.py` |
| 11 | Down-day / up-day split, V3b | +0.663% (191 days) / −0.502% (256 days) | **VERIFIED** — recomputed at release | `python sign_split.py` |
| 12 | Up-day underperformance, t | −6.30 | VERIFIED (`numbers.json`) | `xcheck.py` |
| 13 | Time-weighted excess, 3 constructions | −0.0046% / +0.0259% / −0.0313% | **VERIFIED** — recomputed at release, matches text | `python sign_split.py` |
| 14 | Reference count | 37 | **VERIFIED** (37 with DOI/URL, 1 NeurIPS paper without DOI) | `python verify_all_dois.py` |
| 15 | Abstract length | 200 words | VERIFIED | `python qf_wordcount.py` |

### Reproduced end-to-end at release

```
python sign_split.py      -> v3b -0.502%/+0.663%/-0.0046% | smoothed -0.495%/+0.724%/+0.0259% | baseline -0.359%/+0.408%/-0.0313%
python t4b_daily_ic.py    -> y_exc test2024-25 IC=0.0865 t=5.25 n=89
python xcheck.py          -> all numbers in numbers.json match the manuscript
python verify_all_dois.py -> PASS 37 / FAIL 0 / NO-DOI 1 (Ke et al. 2017, NeurIPS)
python beta_check.py      -> V3b beta=0.476 alpha=+11.20%/yr (t=0.85)
```

## Notes on provenance

- **Prices** come from East Money (`push2his.eastmoney.com`, `klt=101`, `fqt=1` adjusted),
  retrieved via akshare `stock_zh_a_hist` in `t1_fetch.py`; `t1b_fetch_direct.py` is the
  direct-HTTP fallback that touches the same endpoint. No new data were created.
- **`t1b_fetch_direct.py` uses `fqt=0` (unadjusted)** and is only used to obtain the raw
  price path for gap detection; `t1c_fetch_adj.py` (`fqt=2`) supplies the adjustment
  factor. Do not use `fqt=0` closes as a return series — the header comment in that file
  explains why (negative prices and ±2600% single-day moves around ex-dividend dates).
- **The panel (`panel.parquet`, 2.1 GB) is not shipped** in the archive. It is
  fully reproducible from `scripts/01_fetch` → `scripts/02_features` given network access.
  `data_raw/list.csv` fixes the exact stock universe; without it the universe would drift
  as listings and delistings change.
- **`results/` ships every intermediate CSV/MD that the manuscript cites** (88 files,
  0.67 MB). Together with `assets/*.txt` (model scores, 2.3 MB) this is enough to
  re-run every post-model stage offline, without touching the network.
