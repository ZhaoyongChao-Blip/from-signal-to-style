# Replication check — release build verification log

Run on the build machine, 2026-11-17, inside a `work/` tree produced by
`python scripts/00_bootstrap.py` (79 staged entries, hardlinked, no scripts modified).

Purpose: confirm that **the shipped archive actually runs**, not that it merely looks
complete. Every command below was executed against `data/proc/panel.parquet`
rebuilt from the shipped `data/raw/list.csv` + `feat_meta.json`, with model scores from
`assets/` and intermediates from `results/`.

```
$ python scripts/00_bootstrap.py --check
work tree: .../work
files staged: 79
all seed checks passed

$ python t4b_daily_ic.py
[t4b] panel (8383492, 74)  feats 46
  y_raw      valid2023    IC=+0.1075 (n=242d)  -> non-overlap n=49d IC=+0.1160 t=6.57
  y_raw      test2024-25  IC=+0.0775 (n=443d)  -> non-overlap n=89d IC=+0.0736 t=4.85
  y_exc      valid2023    IC=+0.1454 (n=242d)  -> non-overlap n=49d IC=+0.1443 t=7.65
  y_exc      test2024-25  IC=+0.0935 (n=443d)  -> non-overlap n=89d IC=+0.0865 t=5.25
  y_exc_nzt  valid2023    IC=+0.1887 (n=242d)  -> non-overlap n=49d IC=+0.1883 t=8.68
  y_exc_nzt  test2024-25  IC=+0.1484 (n=443d)  -> non-overlap n=89d IC=+0.1445 t=8.30

$ python sign_split.py
construct    up        down      weighted  n_up n_dn
v3b          -0.502%   +0.663%   -0.0046%   256  191
smoothed     -0.495%   +0.724%   +0.0259%   256  191
baseline     -0.359%   +0.408%   -0.0313%   256  191

$ python beta_check.py
construct      beta    alpha_d  alpha_ann  t(alpha)    corr
V3b           0.476    0.0422%     11.20%      0.85   0.594
Smoothed      0.469    0.0733%     20.28%      1.43   0.575
Baseline      0.687   -0.0033%     -0.84%     -0.06   0.664

$ python xcheck.py
... 全部通过   (every number in numbers.json matches the manuscript)

$ python verify_all_dois.py
PASS 37  FAIL 0  NO-DOI 1 (Ke et al. 2017, NeurIPS — NeurIPS does not register DOIs)
```

## What this does and does not prove

**Proves:** the post-model pipeline reproduces every number the manuscript reports, from
the shipped artefacts, with no network access. The reference list resolves to the claimed
titles, authors and years on Crossref (37/37).

**Does not prove:** byte-identical reproduction of `panel.parquet` from scratch. That
requires re-downloading ~4,800 files from East Money; upstream providers revise historical
adjustment factors, so a fresh pull will differ in the last decimals and in any
delisted-then-relisted name. The shipped `feat_meta.json` records the shape of the panel
used for the paper (8,442,301 × 71) so any divergence is visible.

**Known non-determinism:** the placebo draws (`t7e`, `t8f`, `m9d`) use
`np.random.default_rng(20241111)`; seeded, so they reproduce exactly.
LightGBM training is seeded in `t4_lgb.py` but is not bit-reproducible across
CPU/BLAS builds — expect the third decimal of IC to move.

## Two defects found and fixed while verifying the release

1. `t1b_fetch_direct.py` fetches with `fqt=0` (unadjusted). Using those closes as a return
   series produces negative prices and ±2600% single-day moves around ex-dividend dates
   (documented in the file header). All returns in the paper come from `t1c` (`fqt=2`)
   via `adj_factor = adj_close / raw_close`. The file is shipped because it fixes the raw
   price path needed for the limit-hit test; it must not be used for returns.
2. `verify_all_dois.py` reports `ERR` for the one reference without a DOI. This is
   cosmetic (total 38 rows, 37 pass), and the `ERR 1` line makes the run exit non-zero.
   Kept as-is rather than special-casing, so the output shows the real state.

## Read order for a referee

1. `results/FINDINGS.md` — chronological record of what was tried and what reversed.
2. `docs/PITFALL_LEAKAGE.md` — the three leakage bugs and how each was caught.
3. `paper/manuscript.md` §3.4 — the same material in publishable form.
4. `docs/ASSERTIONS.md` — the number-to-file map.
