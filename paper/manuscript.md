# From Signal to Style: A Full-Chain Reassessment of Technical-Factor Stock Selection in the Chinese A-Share Market

**Authors**: Yongchao Zhao^1

^1 School of Physics and Electronic Engineering, Nanning Normal University, Nanning, Guangxi 530000, China

**Corresponding author**: Yongchao Zhao (1792552210@qq.com)

**Funding**: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Abstract

Technical factors are widely used in equity selection, yet they often fail. We audit the chain on the Chinese A-share market with 46 factors, LightGBM and 3,317 stocks over twelve years. Overlapping 5-day labels inflate the daily IC t-statistic about sqrt(5); non-overlapping striding leaves it significant (t = 8.30), though all factors have absolute IC below 0.10. Most of the edge is style: four factors explain 16.6% of forward excess return, and the long book is low-beta (beta 0.48, CAPM alpha 11.2%, t = 0.85). Neutralizing against them raises test AUC from 0.5353 to 0.5525, and the score earns 0.713% per five days within style cells (t = 3.47); a randomized placebo shows those cells unbiased, while nearest-neighbour matching is biased by 3.3 points. The alpha does not monetize: smoothing lifts the break-even cost from 10 to 24 basis points, but the +18.17 point excess arises entirely in 2024H1, when the benchmark fell 7.33% and 10.00%, and the six other quarters sum to -13.3. The edge is signed by market direction: -0.50% on up days, +0.66% on down days (t = -6.30, +8.10). It is avoidance of a small-cap crash, not skill.

**Keywords**: technical factors; LightGBM; style neutralization; label overlap; regime dependence; negative results

## 1. Introduction

### 1.1 Motivation

Machine-learning stock selection from price-volume features is a standard recipe in quantitative finance: engineer a few dozen technical factors, train a gradient-boosted model to predict short-horizon cross-sectional returns, sort into a long book, and report the backtest. The recipe is easy to execute and hard to falsify. Three failure modes recur in published and unpublished work alike. Labels are built from overlapping windows, so independent observations are over-counted and significance is inflated. Features encode well-known style exposures such as size, value, volatility and liquidity, so the model is rewarded for rediscovering a risk premium rather than for forecasting. Backtests are evaluated against a benchmark that itself contains the style premium, so a long book holding the right style looks skilful.

This paper reports a full-chain audit designed to expose exactly these failure modes on the Chinese A-share market. We do not propose a new architecture. We take a deliberately conventional pipeline of 46 technical factors, LightGBM and a 5-day forward excess return, and subject every link to an independent check, discarding conclusions as soon as they fail. The contribution is the resulting chain of falsifications, each quantified, together with the methodology that produced it.

### 1.2 Contributions

This paper reports six findings that are ordered, not parallel: each one is a consequence of distrusting the one before it.

1. The signal is real but small. Forty-six technical factors carry statistically significant but economically thin information: the strongest absolute IC is 0.0995 and every other factor is below 0.08. A reader who stops here would conclude that technical factors work and that the task is to combine them well.

2. Part of that significance is manufactured by the measurement. The daily IC t-statistic is inflated by a factor of 2.25, close to sqrt(5), because forward 5-day labels overlap. Non-overlapping striding reduces t from 18.68 to 8.30. Significance survives; the impression of overwhelming evidence does not.

3. What remains is style, and neutralizing it raises accuracy. Four style factors explain 16.6% of cross-sectional forward excess return. Removing them raises out-of-sample AUC from 0.5353 to 0.5525 and the improvement holds in seven of seven test years under an expanding-window protocol (mean AUC 0.5451 over 5,468,264 stock-days). Accuracy improves precisely because predictable but uninformative variance is removed.

4. The control portfolio was itself the source of a false positive. Matching each pick to its nearest neighbour in style space is biased: with randomized scores it earns 19.28% annually against 15.97% for pure random selection, an artefact of 3.3 percentage points large enough to manufacture an alpha. Style-cell bucketing removes the bias and a randomized-score placebo confirms it. A benchmark can be more dangerous than no benchmark, because it looks like rigour.

5. The alpha is real within cells and does not monetize across the market. The neutralized model earns +0.713% per 5 days within style cells (t = 3.47), yet the long book underperforms the equal-weighted market by 9.43 percentage points annually under realized-execution accounting. Score smoothing closes that gap, but a quarterly decomposition assigns the entire cumulative excess to 2024H1. Both statements are true; they measure against different peers.

6. The edge is signed, not absent. Conditioning on the sign of the benchmark shows the strategy loses 0.50% on days the market rises (t = -6.30) and gains 0.66% on days it falls (t = +8.10), across three constructions; excluding 2024H1 the smoothed strategy trails by 20.36 percentage points annualized over the 331 remaining trading days. The strategy is not failing to select. It is a defensive position, and averaging over both regimes reports the mean of two opposite trades.

Each step was cheap. Each one, once applied, changed or reversed the conclusion of the step before it. The assembled sequence, together with the three implementation errors of Section 3.4 that were caught for the same reason, is the paper's actual claim: that in empirical asset pricing the distance between a publishable result and a null result can be entirely accounted for by checks that cost less than a day each. Figure 8 draws the chain: each check is run only because the previous result made it necessary, and the verdict of the last three is reversal rather than refinement.

### 1.3 Related work

*Machine learning in the cross-section.* Empirical asset pricing has moved decisively toward high-dimensional forecasting. Gu, Kelly and Xiu (2020) show that tree ensembles and neural networks dominate linear specifications on US equities, and Chen, Pelger and Zhu (2024) extend the argument to deep factor models. Avramov, Cheng and Metzker (2023) caution that unrestricted machine learning can exploit market conditions rather than fundamental signals, while Feng, Giglio and Xiu (2020) argue that most candidate factors are redundant once a small number are conditioned on. For the Chinese market specifically, Leippold, Wang and Zhou (2022) provide the most comprehensive comparison and report that liquidity and turnover variables carry most of the incremental signal, a conclusion our univariate results independently reach. Our interest is not in the architecture but in the audit, since each of these papers reports predictive accuracy without conditioning on the sign of the benchmark.

*Style exposure as a confound.* Fama and French (1992, 1993) established size and value as priced factors and later extended the taxonomy to profitability and investment (Fama and French, 2015); Carhart (1997) added momentum. Asness, Moskowitz and Pedersen (2013) document that value and momentum premia co-move across asset classes, and Frazzini and Pedersen (2014) show that funding constraints generate a beta premium, so that a portfolio tilted toward high-beta names earns a mechanical return. Jegadeesh and Titman (1993) is the origin of the momentum factor we include. In the A-share context, Liu, Stambaugh and Yuan (2019) show that the size premium is the dominant characteristic and that the shell-value of small caps distorts the standard three-factor model, which is directly relevant to the 2024H1 episode we analyse in Section 6. Barberis and Shleifer (2003) provide the behavioural account of why style-level flows can dominate stock-level information, and Bollerslev, Li and Todorov (2016) show that betas estimated on continuous returns understate exposure to jump risk, a concern that applies to any daily-frequency factor model on a market with price limits.

*Fund performance measurement.* The question of whether an apparent edge reflects skill or exposure has a long history in the fund literature. Jensen (1968) introduced the alpha regression we use in Section 4.3; Fama and French (2010) show that once a factor model is correctly specified, the cross-section of fund alphas is close to zero; and Wermers (2000) decomposes fund returns into stock-picking, style and cost components. Daniel, Grinblatt, Titman and Wermers (1997) proposed characteristic-based benchmarks precisely because factor-model alphas conflate selection with style, and Ferson and Schadt (1996) condition on public information to separate timing from selection. Our style-cell benchmark in Section 5.3 is a granular version of the same idea, and our finding that within-cell selection can coexist with negative market-relative performance is a case the fund literature anticipates: a manager can pick well relative to peers and still lose to a capitalization-agnostic basket. Malkiel (1995) and Carpenter and Lynch (1999) document how survivorship and attrition bias inflate measured persistence, which is why we report all constructions rather than the best one.

*Overlapping horizons.* The label-overlap problem is a special case of inference with overlapping horizons. Hansen and Hodrick (1980) and Newey and West (1987) provide corrected standard errors, Politis and Romano (1994) give the stationary bootstrap we would use if we resampled, and Lo (2002) shows that Sharpe ratio inference is itself sensitive to serial correlation. Our contribution is not the correction but its magnitude here: the naive t-statistic is inflated by 2.25, essentially the square root of the overlap, and the corrected statistic survives.

*Data snooping and backtest overfitting.* White (2000) introduced the reality check for data snooping; Romano and Wolf (2005) formalized stepwise multiple testing and provide the framework our placebo most closely resembles. Bailey and López de Prado (2014, 2017) show that the number of trials alone can manufacture an apparently significant Sharpe ratio. Harvey, Liu and Zhu (2015) argue that a t-statistic of 2.0 is far too lenient given the volume of factors tested in the literature, a threshold we take seriously: our strongest single factor reaches only |IC| = 0.0995. Harvey and Liu (2015) survey the corrective procedures. Our randomized-score placebo is a deliberately low-cost member of this family, and Section 5.4 shows why it is not optional: a control portfolio that looks more rigorous than random selection can itself carry a 3.3 percentage point bias.

*Trading costs and anomaly decay.* Novy-Marx and Velikov (2015) show that anomaly profitability is highly sensitive to turnover and that many published anomalies do not survive realistic costs; McLean and Pontiff (2016) document post-publication decay; Chordia, Roll and Subrahmanyam (2011) record the secular rise in trading activity that changes the cost landscape. These are the benchmarks against which our Section 6 cost frontier should be read. Ang and Chen (2002) is the antecedent for conditioning on market direction, although their interest is in correlation asymmetries rather than in the sign of a strategy's conditional alpha.

The closest antecedent for our final result is the literature on regime dependence in factor performance: a strategy that earns excess only when one style crashes has not demonstrated skill. Ang and Chen (2002) establish that equity correlations rise in downturns, which is the mechanical channel through which a low-beta, liquidity-floored book appears to add value exactly when the market falls.

## 2. Data and Pipeline

### 2.1 Universe

We use all A-share common stocks listed on the Shanghai and Shenzhen exchanges, obtained from public daily endpoints. The raw universe contains 5,580 securities; after requiring a contiguous price history and valid forward-return labels, 3,317 enter the panel. The panel spans 2014-01-02 to 2025-11-03 for training and construction. Out-of-sample evaluation uses two protocols: a single split from 2024-01-02 to 2025-11-10 comprising 447 trading days, used for all portfolio results, and an expanding-window protocol covering 2019 to 2025 comprising 1,657 trading days, used to establish that the predictive accuracy is not specific to one split (Section 5.2). The panel contains 8,442,301 stock-day rows and 2,882 distinct trading dates. Prices are stored in both unadjusted and back-adjusted form; the unadjusted series is used exclusively for trade-feasibility tests and the adjusted series exclusively for return computation.

### 2.2 Factors

We construct 46 cross-sectional technical factors from daily open, high, low, close, volume and turnover-value series, in five families: (i) momentum and reversal over 1, 5, 10, 20 and 60 days, in the tradition of Jegadeesh and Titman (1993); (ii) volatility, including 5-, 20- and 60-day realized volatility, high-low range and downside semi-deviation; (iii) liquidity, including log turnover value, its 5- and 20-day moving averages, log turnover rate and an Amihud-style illiquidity measure; (iv) trend, including moving-average deviations at 5, 10, 20 and 60 days and moving-average crossovers; (v) microstructure proxies, including overnight and intraday return decomposition, close-to-open gaps and turnover concentration. All factors are winsorized at the 1st and 99th percentile within each trading day and standardized cross-sectionally to zero mean and unit variance.

### 2.3 Labels

Three labels are defined on the 5-day forward horizon. y_raw indicates a positive forward raw return. y_exc indicates a forward return exceeding the equal-weighted market return on the same day, so the label is market-neutral by construction. y_exc_nzt additionally drops observations whose forward window contains a limit-up day, motivated by the fact that such observations constitute 5.12% of the pooled forward sample and are essentially unpredictable from technical features.

The exclusion is not cosmetic: a limit-up day is a trading halt in all but name, so the observed return is censored from above and an order at the close cannot be filled at the limit price. Liu, Wu and Zhu (2022) document price overreaction after up-limit events in this market and show that momentum performance changes once those events are handled explicitly. Our diagnostic agrees: where the forward window contains a limit-up day the mean return is 11.84% and the median 10.30%, against -0.27% and -0.21% elsewhere. Those observations add no signal a model could have seen; they add a censoring artefact that any model will learn by memorizing the censoring.

A fourth label, y_neut, is the object of Sections 5 and 6. At each trading day we regress forward excess return cross-sectionally on an intercept and the four style factors,

  r_{i,t+5}^{exc} = a_t + b_t' s_{i,t} + e_{i,t},   (1)

and define y_neut as an indicator for a positive residual e_{i,t}. The four style factors are log turnover value, log closing price, 20-day realized volatility and 20-day mean turnover rate. The mean daily cross-sectional R-squared of regression (1) is 0.1655 (median 0.0901), establishing that style accounts for a substantial and time-varying share of forward excess return.

### 2.4 Model and evaluation protocol

We train LightGBM (Ke et al., 2017) with 400 trees, learning rate 0.03, 31 leaves, minimum child weight 20, feature and row subsampling of 0.8, and L1/L2 regularization of 0.1 and 1.0. Training uses 2018-2022, validation 2023 for early stopping, and the test period is 2024-01-02 to 2025-11-10 (447 trading days). Section 5.2 reports an expanding-window alternative covering 2019 to 2025 in which every year is forecast by a model that has never seen it; that protocol is more demanding than the single split and does not alter any conclusion. Features are restricted by an explicit allow-list; any column whose name contains a forward-looking prefix, or which is itself a label, is excluded. This restriction is not cosmetic: during development an exclusion-based selector admitted both forward index returns and the label itself, inflating test AUC from 0.53 to 0.94.

### 2.5 Backtest conventions

Signals are computed at the close of day t and executed at the open of day t+1. A stock is skipped if its unadjusted open is at or above the previous unadjusted close multiplied by (1 + L - 0.005), where L is 0.20 for ChiNext and STAR-market stocks and 0.10 otherwise; this models the practical impossibility of buying a limit-up open. Returns are computed from back-adjusted open to back-adjusted close. Portfolios hold the top K names at equal weight with a 5-day rebalancing cycle, and each rebalance charges a one-way cost c on the traded fraction, a convention consistent with the cost elasticities estimated by Novy-Marx and Velikov (2015) for high-turnover strategies. The benchmark is the equal-weighted portfolio of all panel stocks, rebalanced daily, which we regard as the honest hurdle: a selection strategy that cannot beat a capitalization-agnostic basket has not demonstrated value.

Three distinct excess-return definitions appear in this paper and we state them once here to avoid the ambiguity that commonly corrupts such comparisons. The *long-only* excess is the portfolio's annualized return minus the benchmark's annualized return, and answers whether more money was made. The *long-short* excess is the annualized return of the daily return-difference series r_p - r_b, which models a 1:1 daily-rebalanced long-short book and is what most performance-attribution software reports. These two diverge substantially, by 3.5 to 5.4 percentage points in Table 10, and can even disagree in sign. The *cumulative* excess is the difference of cumulative returns over a sub-period and is used only in Table 11, where the question is where the excess arises rather than how large its annualized rate is. Unless stated otherwise, Sections 4 through 7 use the long-only definition.

### 2.6 Audit protocol

Three practices were enforced throughout. First, every derived quantity is recomputed from raw data and compared against the persisted intermediate with a tolerance of 1e-10. Second, any parameter selected with reference to the evaluation period is flagged; the smoothing window is examined explicitly in Section 6.2. Third, every conclusion is subjected to a randomized placebo before being reported, in the spirit of the reality check of White (2000) and the stepwise procedure of Romano and Wolf (2005). A machine-readable assertion table maps each reported number to the script and artefact producing it.

## 3. Results I: The Signal Is Real but Weak

### 3.1 Univariate predictive power is uniformly small

Table 1 reports the mean daily cross-sectional Spearman correlation between each factor and the 5-day forward return. The strongest factor is log turnover value with IC = -0.0995, meaning that stocks with lower turnover value subsequently outperform.

**Table 1. Univariate factor IC, top 8 of 46 by magnitude.**

| Rank | Factor | Family | Mean daily IC |
|--|--|--|--|
| 1 | log_amt | liquidity | -0.0995 |
| 2 | amt_ma_20 | liquidity | -0.0792 |
| 3 | log_turn | liquidity | -0.0789 |
| 4 | turn_ma_5 | liquidity | -0.0724 |
| 5 | hl_range | volatility | -0.0710 |
| 6 | vol_20 | volatility | -0.0685 |
| 7 | ma_dev_60 | trend | -0.0676 |
| 8 | rev_60 | momentum | -0.0643 |

Every one of the 46 factors satisfies |IC| < 0.10 and none exceeds 0.15; Figure 1 displays the full distribution. The ordering is informative: the three strongest factors all measure trading activity and all carry a negative sign. Whatever the model learns begins from the observation that illiquid, low-turnover names outperform over this horizon.

### 3.2 Overlapping labels inflate significance by sqrt(5)

Because the forward 5-day return is shared by five consecutive observations, the daily IC series is autocorrelated by construction and the naive t-statistic overstates precision. The standard remedy is the heteroskedasticity- and autocorrelation-consistent estimator of Newey and West (1987), or the overlapping-horizon correction of Hansen and Hodrick (1980); Politis and Romano (1994) provide the bootstrap alternative. We take the simplest route, retaining one observation in every five trading days, which sidesteps the choice of kernel and bandwidth at the cost of a smaller sample. Table 2 reports both the naive statistic and the statistic obtained by this stride.

**Table 2. IC and t-statistics under overlapping and non-overlapping evaluation.**

| Label | Period | IC (overlap) | Naive t | IC (stride 5) | Non-overlap t | n (stride 5) |
|--|--|--|--|--|--|--|
| y_raw | 2023 valid | 0.1075 | 13.36 | 0.1160 | 6.57 | 49 |
| y_raw | 2024-25 test | 0.0775 | 10.88 | 0.0736 | 4.85 | 89 |
| y_exc | 2023 valid | 0.1454 | 17.09 | 0.1443 | 7.65 | 49 |
| y_exc | 2024-25 test | 0.0935 | 12.80 | 0.0865 | 5.25 | 89 |
| y_exc_nzt | 2023 valid | 0.1887 | 20.58 | 0.1883 | 8.68 | 49 |
| y_exc_nzt | 2024-25 test | 0.1484 | 18.68 | 0.1445 | 8.30 | 89 |

The ratio of naive to corrected t-statistics is 2.25, essentially sqrt(5) = 2.24, confirming that the overlap accounts for the entire inflation. Significance survives: even for the weakest label the corrected t-statistic is 4.85. Figure 2 visualizes the ratio.

The mechanism behind the rise in IC when limit-up observations are removed is worth noting: these observations constitute 5.12% of the forward sample and their mean forward return is -0.27%, so removing them makes the residual distribution closer to symmetric and raises rank correlation, without adding economic value.

### 3.3 What the model actually holds

Decile analysis of the test-period score reveals an unambiguous style tilt. Table 3 contrasts the top and bottom deciles.

**Table 3. Characteristics of the top and bottom score deciles, test period, median values.**

| Characteristic | Q1 (lowest score) | Q10 (highest score) |
|--|--|--|
| Daily turnover value, CNY 100M | 4.34 | 0.37 |
| Closing price, CNY | 10.9 | 7.1 |
| Free-float market cap, CNY 100M | n/a | 27.8 |

The model systematically buys small, cheap, low-liquidity stocks. This is the same dimension identified in Section 3.1, now expressed as a portfolio.

### 3.4 Three implementation errors that changed conclusions

Three errors occurred during development. Each was found only by an independent recomputation, and each moved the headline result in the same direction: *towards* apparent success. We report them not as war stories but because they share a structure, and that structure is what makes a research pipeline untrustworthy in a way its author cannot feel from the inside.

The three errors, and where each one flattered the result:

*(i) Exclusion-based feature selection leaked labels.* Selecting features by excluding a blacklist admitted two groups of columns, the forward index returns and the label itself. Test AUC rose from 0.53 to 0.94. The leak did not produce a suspicious number so much as a *plausible* one: a strong cross-sectional signal looks like success, and the exclusion list had felt more conservative than an allow-list.

*(ii) Portfolio valuation used the wrong reference price.* Valuing the book at the closing price of the day before the rebalance produced net values of order 1e75. This error, unlike the other two, was loud enough to be caught by inspection.

*(iii) Label shifting across an irregular calendar.* Removing anomalous rows before applying a 5-day forward shift misaligned labels for stocks with missing days, a 0.037 discrepancy on one stock. The error was invisible in aggregate statistics and visible only in a per-stock reindexing check.

What the three share is not carelessness. Each arose from a defensible local decision: a blacklist is a normal way to select features, the prior close is a normal valuation price, removing bad rows is normal hygiene. The failure is that each decision was made in a different part of the pipeline from the one that could reveal its consequence. A researcher inspecting any single step finds nothing wrong, because nothing *is* wrong at that step. The error exists only in the composition.

This is why the audit protocol of Section 2.6 is structured around recomputation rather than review: reviewing a step cannot detect an error that is correct at that step. It also sets the prior we apply to our own later results. When a check made the result worse, we kept it; when a check made it better, we looked for the leak.

## 4. Results II: Baseline Backtest and Its Regime Dependence

### 4.1 Absolute performance

The equal-weighted market returns 21.11% annually over the test period with volatility 25.67% and Sharpe ratio 0.764. The Top-30, 5-day strategy under the realized-execution accounting achieves 11.68% with volatility 26.55% and Sharpe ratio 0.383, underperforming the benchmark by 9.43 percentage points annually with an information ratio of -0.37. Under an optimistic accounting that ignores limit-up unfillability the same strategy reports 18.37%, illustrating the sensitivity of such figures to execution assumptions; the fraction of intended buys unfillable at the open has median 3.3% and 90th percentile 16.7%. Throughout, Sharpe ratios are annualized using a risk-free rate of 1.5% and excess return divided by the annualized standard deviation of daily returns.

### 4.2 Regime dependence

We decompose the test period into half-years.

**Table 4. Regime dependence of the raw-label strategy.**

| Period | Equal-weighted market | Strategy | Difference |
|--|--|--|--|
| 2024H1 | -16.60% | -14.91% | +1.69 pp |
| 2024H2 to 2025 | +66.80% | +49.60% | -17.20 pp |

The reversal is stark, but not in the direction one might expect. In 2024H1, when the equal-weighted market fell 16.60%, the raw-label strategy fell 14.91% and therefore *outperformed* by 1.69 percentage points. In the second half and through 2025 the market gained 66.80% while the strategy gained 49.60%, giving up 17.20 percentage points. The strategy is therefore not uniformly weak: it cushions a crash slightly and then fails to participate in the rebound. A conclusion drawn from either half alone would be wrong. This asymmetry is the primary reason we refuse to report a backtest number without a regime decomposition, and it recurs, amplified, in Section 6.3.

### 4.3 Return attribution confirms zero alpha

Two regressions are needed here, and they answer different questions. Regressing the strategy's daily return on the equal-weighted market, the CSI 300 and the CSI 1000 explains 75 to 81% of its variance, with a loading of 1.84 on the equal-weighted market and -0.11 on the CSI 300: the book is, mechanically, a levered small-cap portfolio. That regression's own intercept is negative, with t-statistics of -0.72 and -1.39 for the two label constructions. Regressing instead on the *excess* return over the equal-weighted market removes that exposure by construction, so the intercept there measures selection alone: t-statistics of 0.03 and -0.63 over 447 trading days, tested against Newey-West standard errors. Both regressions point the same way. The first establishes that the returns are market and size exposure rather than selection; the second shows that once that exposure is subtracted there is nothing left to find.

## 5. Results III: Neutralization Strengthens the Signal

### 5.1 Neutralized labels improve out-of-sample accuracy

Our initial conclusion, that the model had no alpha, proved premature. If the score mainly encodes style, removing style from the target should isolate the residual forecast. We therefore retrain on y_neut and compare.

**Table 5. Model performance by label construction, test period.**

| Label | Test AUC | IC vs. own target | Non-overlap t |
|--|--|--|--|
| y_exc | 0.5353 | n/a | n/a |
| y_neut | 0.5525 | 0.0699 | 6.33 |

Removing style improves accuracy by 1.7 percentage points of AUC. The mechanism matches Section 3.2: style contributes a large, highly predictable but economically uninformative component to forward excess return, and removing it raises the signal-to-noise ratio of what remains. Table 7 shows the factor-level effect. Mean absolute IC falls from 0.0443 to 0.0188, yet 21 of 46 factors retain neutralized absolute IC above 0.02 and 35 of 46 retain a non-overlapping t-statistic above 2.

### 5.2 The result survives seven independent out-of-sample years

The 2024–2025 evaluation window is short, and a two-year test period invites the objection that the accuracy is a property of one market regime. We therefore re-estimate the model under an expanding-window protocol in which every year from 2019 to 2025 is forecast by a model that has never seen it. For test year Y the training set runs from 2014-01-01 to (Y-2)-12-31, the validation year Y-1 is used for early stopping, and the test year is left untouched; the training set therefore grows from 2.18 million to 6.92 million stock-days. The label and the feature set are identical to those of Section 5.1.

**Table 6. Expanding-window out-of-sample accuracy, one model per test year.**

| Test year | Train rows | Test rows | Test days | Best iter. | AUC | IC | IC (stride 5) | t (stride 5) |
|--|--|--|--|--|--|--|--|--|
| 2019 | 2,182,763 | 793,931 | 244 | 463 | 0.5534 | 0.0824 | 0.0776 | 7.81 |
| 2020 | 2,915,228 | 802,431 | 243 | 781 | 0.5473 | 0.0725 | 0.0746 | 5.38 |
| 2021 | 3,709,159 | 803,623 | 243 | 1,131 | 0.5364 | 0.0213 | 0.0155 | 1.18 |
| 2022 | 4,511,590 | 800,736 | 242 | 537 | 0.5403 | 0.0358 | 0.0439 | 3.00 |
| 2023 | 5,315,213 | 801,511 | 242 | 588 | 0.5316 | 0.0185 | 0.0238 | 2.24 |
| 2024 | 6,115,949 | 801,287 | 242 | 788 | 0.5498 | 0.0689 | 0.0758 | 3.28 |
| 2025 | 6,917,460 | 664,745 | 201 | 354 | 0.5569 | 0.0490 | 0.0484 | 2.71 |

The expanded sample covers 5,468,264 out-of-sample stock-days and 1,657 trading days. The AUC exceeds the coin flip in all seven years, with a mean of 0.5451 and a range of [0.5316, 0.5569]; the IC is positive in all seven years, with a mean of 0.0498 and a median of 0.0490; and the non-overlapping t-statistic exceeds 2 in six of the seven years. The weakest year is 2021, with an AUC of 0.5364 and a non-overlapping t of 1.18.

The two protocols agree: the 2024 and 2025 rows reproduce the 0.5525 and 0.0699 reported in Table 5 to within sampling variation (0.5498 and 0.5569, 0.0689 and 0.0490). This matters because it rules out the possibility that the head-line accuracy is an artefact of a single split. It does not, however, rule out the possibility that the signal is economically worthless, which is the question the remainder of the paper addresses: an AUC of 0.545 is a weak discriminator by construction, and the reader should keep in mind that the seven-year record establishes the existence of a signal, not its magnitude.

**Table 7. Factor-level effect of style neutralization.**

| Quantity | Raw | Neutralized |
|--|--|--|
| Mean absolute IC, 46 factors | 0.0443 | 0.0188 |
| Factors with absolute IC > 0.02 | 46 | 21 |
| Factors with non-overlap abs(t) > 2 | 41 | 35 |

### 5.3 Within style cells, the score earns +0.71% per five days

To test whether the neutralized score selects well, we compare picks against peers with the same style exposure. On each rebalance date we rank all stocks independently on each of the four style factors into quintiles and form the Cartesian product, yielding up to 625 cells (median 467 non-empty cells per day). For each pick we take the equal-weighted forward return of the other stocks in its cell as the benchmark and weight cells by the pick distribution.

**Table 8. Excess return within style cells.**

| Model | Picks' 5-day return | Cell benchmark | Excess | t (non-overlap) |
|--|--|--|--|--|
| y_exc | 0.859% | 0.702% | +0.157% | 0.70 |
| y_neut | 1.292% | 0.579% | +0.713% | 3.47 |

The neutralized model earns 0.713% per 5 days relative to same-style peers, with t = 3.47 on 89 non-overlapping observations. The raw model earns 0.157% with t = 0.70. Figure 4(b) shows the quarterly breakdown: positive in 8 of 8 quarters, with per-quarter hit rates between 56.2% and 72.2%.

### 5.4 The placebo that saved the conclusion

Our first version of this test used nearest-neighbour matching: for each pick, select the 30 stocks with the smallest standardized Euclidean distance in style space to the picks' mean style vector. That control portfolio returned 27.87% annually, 9.5 percentage points more than the model at 18.37%, which we initially read as evidence that the model selected adversely.

Replacing the model's scores with random numbers exposed the artefact. Running the identical procedure with random scores 100 times gives a mean annualized return of 19.28%, against 15.97% for pure random selection under the same eligibility filters. The matching construction carries a 3.3 percentage point positive bias. The mechanism is geometric: when the picks' mean style vector lies away from the cross-sectional mean, the nearest candidates are those that shrink toward the centroid in all dimensions simultaneously, so the control is systematically shifted toward middling style rather than toward the picks' style. Cell-based bucketing lacks this property, and its placebo passes: with random scores the cell-based excess is -0.056%, +0.002%, +0.004%, +0.052% and -0.059% across five draws, with t-statistics between -1.18 and 0.82.

We regard this as the paper's most transferable lesson. A control portfolio that appears more rigorous can reverse a conclusion, and the only reliable test is a randomized-score placebo.

## 6. Results IV: The Alpha Does Not Monetize

### 6.1 Execution constraints and the cost frontier

Realized-execution accounting changes the picture materially. Using unadjusted prices for the fill test and adjusted prices for returns, the neutralized Top-30 strategy earns 18.37% under naive accounting but 11.68% under realized-execution accounting, a gap of 6.7 percentage points concentrated in small caps where limit-up frequency is higher.

A liquidity filter helps. Restricting the universe to names with daily turnover value above CNY 300 million raises the annualized return to 21.17% with volatility 20.59% and Sharpe ratio 0.955, essentially matching the benchmark's return at lower volatility. A same-universe placebo, random selection of 30 names from the identical eligibility pool over 100 draws, yields mean annualized -12.46% with standard deviation 5.73% and mean Sharpe -0.500. The model's 21.17% therefore sits at the 100th percentile of its own null distribution, confirming that selection rather than pool choice generates the return.

### 6.2 Score smoothing raises the cost frontier

The remaining obstacle is turnover. At a 5-day cycle one-way turnover is 56.7% per rebalance, 28.6 times annually, costing 8.57% per year at 15 basis points. Smoothing the score with a 5-day moving average reduces turnover to 43.3% per rebalance and 21.8 times annually.

**Table 9. Effect of score smoothing. CNY 300M liquidity floor, 5-day holding, one-way cost in bp.**

| Signal | 0 bp | 10 bp | 15 bp | 20 bp | 30 bp |
|--|--|--|--|--|--|
| Single-day score | 32.76% | 24.92% | 21.17% | 17.53% | 10.56% |
| 5-day smoothed | 39.20% | 32.80% | 29.71% | 26.69% | 20.86% |
| 5-day smoothed, lag 1 | 33.86% | 27.67% | 24.68% | 21.76% | 16.12% |

Smoothing is not merely a cost saving. At zero cost the smoothed signal already outperforms by 6.4 percentage points, establishing that smoothing removes noise from the forecast rather than merely reducing trading. The lagged version, which excludes the current day's score and therefore cannot benefit from contemporaneous information, lies between the two, confirming that the current score contributes genuine information.

Two further checks support the mechanism. Window sensitivity in Table 10 is unimodal rather than spiky: windows of 2, 3 and 5 days all deliver annualized returns between 28.55% and 30.11%, decaying smoothly at 10 and 20 days. And the Top-K choice is not knife-edge: K = 20 yields 31.22% with +4.61 percentage points of excess, K = 30 yields 29.71% with +3.35, and K = 50 yields 22.74% with -1.90.

**Table 10. Smoothing window sensitivity.** CNY 300M liquidity floor, 5-day holding, 15 bp one-way cost. Excess is reported on two definitions, since they diverge materially: the long-only view (portfolio annualized return minus benchmark annualized return) and the 1:1 daily-rebalanced long-short view (annualized return of the return-difference series). Sections 6.3 and 6.4 use the long-only definition throughout.

| Window (days) | 1 | 2 | 3 | 5 | 10 | 20 |
|--|--|--|--|--|--|--|
| Annualized return | 21.17% | 28.55% | 30.11% | 29.71% | 26.20% | 23.93% |
| Sharpe ratio | 0.955 | 1.302 | 1.313 | 1.346 | 1.199 | 1.058 |
| Excess, long-only | +0.06% | +7.44% | +9.00% | +8.60% | +5.09% | +2.82% |
| Excess, 1:1 long-short | -3.41% | +2.42% | +3.79% | +3.35% | +0.59% | -0.95% |
| One-way turnover | 57.6% | 53.3% | 50.0% | 43.3% | 30.0% | 20.0% |

The two excess definitions differ by 3.5 to 5.4 percentage points and even disagree in sign at the 1-day window, because the long-short construction adds the benchmark's full volatility to the numerator while the long-only view does not. A paper reporting one number without stating which definition it uses is not reporting a falsifiable claim. We note the discrepancy explicitly rather than choosing the flattering column.

Top-K sensitivity under the same conventions: K = 20 yields 31.22% annualized, K = 30 yields 29.71%, and K = 50 yields 22.74%.

### 6.3 Quarterly decomposition overturns the conclusion

Taken at face value the smoothed strategy delivers 29.71% against the benchmark's 21.11%, a Sharpe ratio of 1.398 against 0.764, and a positive long-only excess of 8.60 percentage points. The decomposition in Table 11 shows why it is not a strategy.

**Table 11. Quarterly returns and excess, smoothed strategy, 15 bp one-way cost.** Excess is the difference of cumulative quarterly returns (strategy minus benchmark), which is neither the annualized difference of Section 6.3 nor the long-short series of Table 10; it is used here because the question is where the cumulative excess comes from, not what its annualized rate is.

| Quarter | Strategy | Equal-weighted market | Excess |
|--|--|--|--|
| 2024Q1 | +12.08% | -7.33% | +19.41% |
| 2024Q2 | +1.97% | -10.00% | +11.97% |
| 2024Q3 | +15.76% | +18.88% | -3.11% |
| 2024Q4 | -6.01% | +7.19% | -13.19% |
| 2025Q1 | +12.06% | +4.54% | +7.52% |
| 2025Q2 | +3.83% | +8.02% | -4.19% |
| 2025Q3 | +12.71% | +12.50% | +0.21% |
| 2025Q4 | -2.72% | +4.05% | -6.77% |
| Full period | +58.64% | +40.46% | +18.17% |

Four quarters are positive and four negative. The cumulative excess of +18.17 percentage points is almost entirely attributable to 2024H1, where the two quarters sum to +31.4 percentage points, and those are precisely the two quarters in which the equal-weighted market fell 7.33% and 10.00%. The remaining six quarters sum to -13.3 percentage points. Table 11 reports the long-only definition; the quarterly return-difference series is not shown because it inherits the same concentration.

The interpretation is direct. The strategy holds high-liquidity names because of the liquidity floor; in a small-cap crash it does not participate. That is a style-avoidance benefit, not selection skill. Figure 6 displays the pattern. A selection strategy is credibly skilful only if it earns excess when its benchmark is not collapsing, and on that test the present strategy fails.

### 6.4 What survives the audit

Three statements survive. The 46 technical factors carry genuine, if small, predictive information: after neutralizing style the model reaches test AUC 0.5525 and IC 0.0699 against the neutralized target with a non-overlapping t-statistic of 6.33. Within style cells that information translates into a spread of 0.713% per 5 days, t = 3.47, with consistent sign across all eight quarters. And the liquidity-floored, smoothed construction genuinely dominates its own randomized null, at the 100th percentile on both return and Sharpe ratio.

The statement that does not survive is that this constitutes an investable strategy. The excess is not earned throughout the sample but concentrated in a single regime, and the cost frontier, while improved, remains close enough to realistic transaction costs that a modest increase in assumed impact erases it.

### 6.5 Conditioning on the benchmark: the edge is signed, not merely absent

A sharper test splits the sample by the sign of the benchmark return. If the strategy carried genuine selection skill, it should earn excess on days when the benchmark rises as well as on days when it falls. Table 12 shows the opposite, and does so under all three constructions.

**Table 12. Excess conditional on the sign of the benchmark return.** Daily excess is the portfolio return minus the equal-weighted market return; monthly excess is the difference of cumulative monthly returns. t-statistics are computed on the respective frequency.

| Construction | Up days (n=256) | t | Down days (n=192) | t | Up months (n=15) | Down months (n=8) |
|--|--|--|--|--|--|--|
| V3b, 3e8 floor, 5d | -0.502% | -6.30 | +0.659% | +8.10 | -1.91 pp | +3.64 pp |
| Smoothed, 3e8 floor, 5d | -0.495% | -6.22 | +0.721% | +8.37 | -0.70 pp | +3.15 pp |
| Baseline, 2e7 floor, 5d | -0.359% | -4.24 | +0.406% | +4.67 | -2.53 pp | +2.84 pp |

On days when the benchmark rises the strategy underperforms by roughly half a percentage point, with t-statistics between -4.2 and -6.3. On days when it falls the strategy outperforms by a comparable magnitude. Every construction wins eight of eight down-months and four or fewer of fifteen up-months. Excluding 2024H1 entirely, the smoothed strategy returns 28.36% against a benchmark of 48.72%, an annualized shortfall of 20.36 percentage points.

Figure 7 displays both panels. This reframes the finding. The earlier sections established that the raw long book fails to beat the equal-weighted market; Table 12 shows that its conditional edge is negative precisely where selection skill would have to appear. The positive within-cell spread of Section 5.3 is therefore not a weak version of skill that fails to cover costs. It is a defensive tilt whose sign flips with the market, and the cumulative excess of Table 11 is the integral of that tilt over one crash rather than the accumulation of forecast accuracy. We regard this as the paper's central negative result, and it is only visible because the sample was split by benchmark direction rather than averaged over it.

## 7. Discussion

### 7.1 Why neutralization increases accuracy

A repeated pattern in this paper is that removing a component of the target or of the forecast increases apparent predictive power. This happens three times: removing limit-up observations, removing style from the label, and smoothing the score. The common mechanism is variance reduction in the uninformative part. Style contributes an R-squared of 0.1655 to forward excess return, and much of that contribution is predictable but uninformative: a model that forecasts style well has learned nothing about which stock will outperform its peers. Removing it concentrates the learning problem on the residual, where the signal-to-noise ratio is higher.

The corollary for reporting practice is that an increase in AUC or IC following a neutralization step should not be read as evidence that new alpha has been discovered. It may simply be that the same information is now visible against a quieter background.

### 7.2 Benchmark construction is not a technical detail

Our results are sensitive to benchmark construction in three distinct ways. The nearest-neighbour control was biased by 3.3 percentage points. The equal-weighted market benchmark contains a large small-cap premium over this sample, which a liquidity-floored strategy mechanically avoids, so the comparison conflates selection with style avoidance. And the choice of evaluation window determines the sign of the conclusion, because the excess is concentrated in 2024H1. A paper reporting one benchmark and one window reports a number, not a finding. This is the same problem Daniel, Grinblatt, Titman and Wermers (1997) confronted when they introduced characteristic-based benchmarks, and the fund literature has repeatedly shown that the choice of benchmark dominates the reported alpha (Wermers, 2000; Fama and French, 2010).

### 7.3 The role of low-beta exposure

Section 6.5 shows that the strategy loses on days the market rises and gains on days it falls. This is the signature of a low-beta book, and the regression confirms it: against the equal-weighted market the long book has a beta of 0.48 (0.47 when smoothed, 0.69 for the pre-smoothing baseline) and a CAPM alpha of 11.2% annualized with t = 0.85, so the return stream is well described as leveraged exposure to the market, scaled down. Frazzini and Pedersen (2014) provide the canonical account of why such a book earns a positive risk-adjusted return in the presence of leverage constraints, and Ang and Chen (2002) document the complementary fact that correlations rise in downturns, which amplifies the apparent benefit of a defensive tilt exactly when it is measured.

We do not claim that this mechanism is new. The low-beta anomaly and its funding-constraint explanation are established. What the audit contributes is the observation that the mechanism can pass for selection skill when the evaluation window contains a single crash, and that the quarterly decomposition of Section 6.3, the natural first diagnostic, does not expose it: it localises the anomaly in 2024H1 without identifying it. Splitting on the sign of the benchmark does identify it, because it maps the return stream onto the state variable the mechanism predicts. We therefore present the sign split not as a novel test but as the cheapest diagnostic that would have caught what the quarterly decomposition let through.

### 7.4 Limitations

The sample covers one market. The out-of-sample accuracy is not confined to a single split: Section 5.2 forecasts seven consecutive years, 2019 through 2025, with widening training windows, and accuracy exceeds the coin flip in all seven. The portfolio results, however, are evaluated on the 2024–2025 window only. This asymmetry is deliberate: re-mining the portfolio layer across seven years would invite a second round of specification search on the same data, and the sign reversal of Section 6.5 already explains why a longer window would be unlikely to rescue the strategy, since the mechanism it identifies is structural rather than temporal. A reader who wants the portfolio evidence extended to 2019–2023 should treat it as out of sample in the strict sense and not as a robustness check.

Impact costs are not modelled from order-book data, which daily bars cannot support, so our cost frontier is a scenario analysis rather than an estimate. The evaluation window contains a single style-regime break, which is precisely what allows the sign decomposition of Table 12 to be identified, but it also means we observe one such break rather than a distribution of them. Finally, the factor set is confined to price-volume information; incorporating fundamentals or analyst data could change the residual signal, although it would not change the methodological conclusions of Sections 3 through 6.

## 8. Conclusion

The result of this audit can be stated in one line: every check that made the answer worse was correct, and together they turned a working strategy into a defensive tilt that a passive index beats.

What the pipeline actually contained was a real statistical signal. After correcting for overlapping labels, out-of-sample IC remains significant; after neutralizing four style factors the residual signal is stronger still, earning 0.713% per 5 days within style cells with t = 3.47 and consistent sign across eight quarters. None of this survived contact with the question of where the money comes from. Under realized-execution accounting the long book underperforms the equal-weighted market by 9.43 percentage points annually. Liquidity filtering and score smoothing bring it to parity and lift the break-even cost to approximately 24 basis points, but a quarterly decomposition attributes the entire cumulative excess to a single half-year in which the equal-weighted market fell 7.33% and 10.00% in its two quarters (a compounded 16.60%). The within-cell spread and the annual shortfall are both true, and they coexist because they are measured against different things: style peers and the market.

The final test is the one we would urge most strongly on others. Conditioning on the sign of the benchmark return shows that the strategy underperforms on days the market rises (-0.50% daily, t = -6.30) and outperforms on days it falls (+0.66%, t = +8.10), identically across three constructions; removing 2024H1 leaves an annualized shortfall of 20.36 percentage points over the 331 remaining trading days. What looked like selection skill is a position: the model is long the stocks that hurt least in a crash, and it holds that position on the days when it helps and on the days when it costs. Averaging over a sample that contains both directions is not wrong in itself, but it returns a number that describes neither regime. Across the three constructions the time-weighted average excess is -0.005%, +0.026% and -0.031% per day, indistinguishable from zero while the two conditional means are 0.8 to 1.2 percentage points apart. The bet was directional; the summary statistic was not.

The methodological contribution is the chain of falsifications and the tools that produced them: an allow-list for features, a stride correction for overlapping labels, style-cell bucketing in place of nearest-neighbour matching, a randomized-score placebo required before any control portfolio is used, a quarterly decomposition reported alongside any cumulative excess, and a sign split on the benchmark. Each is inexpensive, and each reversed a conclusion in this study. That asymmetry is worth stating plainly: none of the tools that *supported* the original finding changed anything, and every one that *contradicted* it was right.

We release the complete pipeline, all intermediate artefacts and the assertion table mapping every reported number to its source, in the belief that the audit trail is more useful than the result. A reader who trusts none of our numbers can regenerate all of them, and the errors we document in Section 3.4 are the reason we think that matters more than the finding itself.

## Data and code availability

All data used in this study are publicly available and were obtained from their original
source; no new data were created. Daily price and volume data for 3,317 Chinese A-share
stocks over 2014-01-01 to 2025-11-10 were retrieved through the open-source Python package
akshare (function stock_zh_a_hist, forward-adjusted prices), which redistributes publicly
available market data from East Money. The complete analysis pipeline, including panel
construction, factor IC tables, model scores, net value series, placebo draws and the
assertion table mapping every reported number in this paper to its source file, is openly
available at https://github.com/ZhaoyongChao-Blip/from-signal-to-style (release v1.0) under
an MIT licence, and is archived with a permanent DOI on Zenodo (to be inserted at
acceptance). Any figure in this paper can be regenerated from those files by running the
scripts in the order documented in the archive README.

## References

Ang, A., & Chen, J. (2002). Asymmetric correlations of equity portfolios. *Journal of Financial Economics*, 63(3), 443–494. https://doi.org/10.1016/s0304-405x(02)00068-5

Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *The Journal of Finance*, 68(3), 929–985. https://doi.org/10.1111/jofi.12021

Avramov, D., Cheng, S., & Metzker, L. (2023). Machine learning vs. economic restrictions: Evidence from stock return predictability. *Management Science*, 69(5), 2587–2619. https://doi.org/10.1287/mnsc.2022.4449

Bailey, D. H., & López De Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management*, 40(5), 94–107. https://doi.org/10.3905/jpm.2014.40.5.094

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–69. https://doi.org/10.21314/jcf.2016.322

Barberis, N., & Shleifer, A. (2003). Style investing. *Journal of Financial Economics*, 68(2), 161–199. https://doi.org/10.1016/s0304-405x(03)00064-3

Bollerslev, T., Li, S. Z., & Todorov, V. (2016). Roughing up beta: Continuous versus discontinuous betas and the cross section of expected stock returns. *Journal of Financial Economics*, 120(3), 464–490. https://doi.org/10.1016/j.jfineco.2016.02.001

Carhart, M. M. (1997). On persistence in mutual fund performance. *The Journal of Finance*, 52(1), 57–82. https://doi.org/10.1111/j.1540-6261.1997.tb03808.x

Carpenter, J. N., & Lynch, A. W. (1999). Survivorship bias and attrition effects in measures of performance persistence. *Journal of Financial Economics*, 54(3), 337–374. https://doi.org/10.1016/s0304-405x(99)00040-9

Chen, L., Pelger, M., & Zhu, J. (2024). Deep learning in asset pricing. *Management Science*, 70(2), 714–750. https://doi.org/10.1287/mnsc.2023.4695

Chordia, T., Roll, R., & Subrahmanyam, A. (2011). Recent trends in trading activity and market quality. *Journal of Financial Economics*, 101(2), 243–263. https://doi.org/10.2139/ssrn.1700191

Daniel, K., Grinblatt, M., Titman, S., & Wermers, R. (1997). Measuring mutual fund performance with characteristic-based benchmarks. *The Journal of Finance*, 52(3), 1035. https://doi.org/10.2307/2329515

Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *The Journal of Finance*, 47(2), 427. https://doi.org/10.2307/2329112

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56. https://doi.org/10.1016/0304-405x(93)90023-5

Fama, E. F., & French, K. R. (2010). Luck versus skill in the cross‐section of mutual fund returns. *The Journal of Finance*, 65(5), 1915–1947. https://doi.org/10.1111/j.1540-6261.2010.01598.x

Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010

Feng, G., Giglio, S., & Xiu, D. (2020). Taming the factor zoo: A test of new factors. *The Journal of Finance*, 75(3), 1327–1370. https://doi.org/10.1111/jofi.12883

Ferson, W. E., & Schadt, R. W. (1996). Measuring fund strategy and performance in changing economic conditions. *The Journal of Finance*, 51(2), 425–461. https://doi.org/10.1111/j.1540-6261.1996.tb02690.x

Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1–25. https://doi.org/10.1016/j.jfineco.2013.10.005

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

Hansen, L. P., & Hodrick, R. J. (1980). Forward exchange rates as optimal predictors of future spot rates: An econometric analysis. *Journal of Political Economy*, 88(5), 829–853. https://doi.org/10.1086/260910

Harvey, C. R., & Liu, Y. (2015). Backtesting. *The Journal of Portfolio Management*, 42(1), 13–28. https://doi.org/10.3905/jpm.2015.42.1.013

Harvey, C. R., Liu, Y., & Zhu, H. (2015). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *The Journal of Finance*, 48(1), 65–91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

Jensen, M. C. (1968). The performance of mutual funds in the period 1945–1964. *The Journal of Finance*, 23(2), 389. https://doi.org/10.2307/2325404

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

Leippold, M., Wang, Q., & Zhou, W. (2022). Machine learning in the Chinese stock market. *Journal of Financial Economics*, 145(2), 64–82. https://doi.org/10.1016/j.jfineco.2021.08.017

Liu, C., Wu, Y., & Zhu, D. (2022). Price overreaction to up-limit events and revised momentum strategies in the Chinese stock market. *Economic Modelling*, 114, 105910. https://doi.org/10.1016/j.econmod.2022.105910

Liu, J., Stambaugh, R. F., & Yuan, Y. (2019). Size and value in China. *Journal of Financial Economics*, 134(1), 48–69. https://doi.org/10.1016/j.jfineco.2019.03.008

Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52. https://doi.org/10.2469/faj.v58.n4.2453

Malkiel, B. G. (1995). Returns from investing in equity mutual funds 1971 to 1991. *The Journal of Finance*, 50(2), 549–572. https://doi.org/10.1111/j.1540-6261.1995.tb04795.x

McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability?. *The Journal of Finance*, 71(1), 5–32. https://doi.org/10.1111/jofi.12365

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703. https://doi.org/10.2307/1913610

Novy-Marx, R., & Velikov, M. (2015). A taxonomy of anomalies and their trading costs. *Review of Financial Studies*, 29(1), 104–147. https://doi.org/10.1093/rfs/hhv063

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870

Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237–1282. https://doi.org/10.1111/j.1468-0262.2005.00615.x

Wermers, R. (2000). Mutual fund performance: An empirical decomposition into stock‐picking talent, style, transactions costs, and expenses. *The Journal of Finance*, 55(4), 1655–1695. https://doi.org/10.1111/0022-1082.00263

White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152

## Figure captions

**Fig. 1** Univariate cross-sectional Spearman IC of all 46 technical factors against the 5-day forward return. Every factor satisfies absolute IC below 0.10; the three strongest are liquidity measures and all carry a negative sign.

**Fig. 2** t-statistics for the daily IC series under overlapping and non-overlapping (stride 5) evaluation. The ratio is 2.25, close to sqrt(5) = 2.24, confirming that label overlap accounts for the entire inflation.

**Fig. 3** Net value of three label constructions against the equal-weighted market, January 2024 to November 2025. The raw label underperforms throughout; the style-neutral label and its smoothed counterpart improve monotonically.

**Fig. 4** Effect of style neutralization. (a) Test-set AUC rises from 0.5353 for the market-relative label to 0.5525 for the style-neutral label. (b) Excess return within style cells by quarter; positive in eight of eight quarters.

**Fig. 5** Cost sensitivity under a CNY 300 million liquidity floor. The single-day score breaks even against the equal-weighted market at about 10 basis points one way; 5-day smoothing moves the break-even point to about 24 basis points. The lagged variant bounds the contribution of contemporaneous information. Returns are long-only and excess is measured as the annualized difference, consistent with Section 6.3.

**Fig. 6** Quarterly return decomposition. Excess return concentrates in 2024H1, the two quarters in which the equal-weighted market fell 7.33% and 10.00%; the following six quarters sum to -13.3 percentage points. Excess here is the difference of cumulative quarterly returns.

**Fig. 7** The edge is signed. (a) Mean daily excess return conditional on the sign of the equal-weighted market return, for three portfolio constructions; the strategy loses on up days and gains on down days in every case. (b) Annualized returns with and without 2024H1; once the crash is removed the smoothed strategy returns 28.36% against 48.72% for the benchmark over the 331 remaining trading days, a gap of 20.36 percentage points.

**Fig. 8** The audit chain. Six checks, each run only because the previous result made it necessary, with the verdict each returned. The first three leave the conclusion standing, the second more weakly; the last three reverse it. The combined cost of all six is under one analyst-week, which is the paper's practical claim: the gap between a publishable result and a null result in this literature is often smaller than the audit that would separate them.
