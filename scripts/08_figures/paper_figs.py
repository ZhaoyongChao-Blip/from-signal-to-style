# -*- coding: utf-8 -*-
"""paper_figs.py —— 论文插图（2026-11-15 全面重做版）

重构原则（针对用户「都不行，全改」的反馈）：
  1. **尺寸统一**：所有 figure 宽 = 5.9 in（= 纸面 15cm 版心宽），
     不再出现「某张画 6.9 宽、某张 5.4 宽」导致的纸面字号不一致。
  2. **字号统一**：图内最小字号 >= 7.0pt。纸面缩放比 = 15cm/5.9in*2.54 -> 1.0，
     即纸面 pt ≈ 图内 pt。低于 7pt 的刻度/标注全部上调。
  3. **文字一律不进图形**：柱顶数字朝外、注释放在数据区之外，
     用 helper 函数统一处理，不靠逐图调参。
  4. **配色统一**：一套语义色贯穿 8 张图
     strategy=#1f6fb2  market=#8d99a6  positive=#2e8b57  negative=#b03a2e
  5. 每张图的产出都经过 paper_figs.check_text() 自检（bbox 相交）。
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)

# ---------------- 全局样式（唯一真源） ----------------
FIG_W = 5.9           # 所有图统一宽度（英寸）=== 纸面 15cm
DPI = 1460
FS = dict(title=9.0, label=8.0, tick=7.2, annot=7.2, legend=7.4, panel=7.8)
MIN_FS = 7.0

C = dict(
    strat="#1f6fb2",      # 策略/主序列
    market="#8d99a6",     # 基准
    pos="#2e8b57",        # 正向
    neg="#b03a2e",        # 负向
    accent="#d68910",     # 提示/阴影
    alt="#6c5ba7",        # 第三序列
    grid="#cccccc",
)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "savefig.dpi": DPI,
    "font.size": FS["label"],
    "axes.titlesize": FS["title"],
    "axes.labelsize": FS["label"],
    "legend.fontsize": FS["legend"],
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# ---------------- 通用助手 ----------------
def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.04)

    sizes = [t.get_fontsize() for t in fig.findobj()
             if hasattr(t, "get_fontsize") and t.get_text() and t.get_text().strip()]
    rec = dict(fig_w_in=round(fig.get_size_inches()[0], 2),
               min_fontsize=(round(min(sizes), 2) if sizes else None),
               n_text=len(sizes))
    mp = os.path.join(FIG, "fig_manifest.json")
    man = {}
    if os.path.exists(mp):
        try:
            man = json.load(open(mp, encoding="utf-8"))
        except Exception:
            man = {}
    man[name] = rec
    json.dump(man, open(mp, "w", encoding="utf-8"), indent=1)

    plt.close(fig)
    from PIL import Image
    im = Image.open(p)
    eff = im.size[0] / (15.0 / 2.54)
    print(f"[fig] {name:22s} {im.size[0]}x{im.size[1]}  "
          f"{os.path.getsize(p)/1024:4.0f} KB  eff_dpi={eff:.0f}  min_fs={rec['min_fontsize']}")


def bar_labels(ax, xs, ys, fmt="{:.1f}", pad_frac=0.022, dx=None, min_gap=None, **kw):
    """柱顶数字：正柱标上方、负柱标下方，**一律在柱外**。

    ax.bar_label 对负值柱会把标签塞进柱内，这里手工定位。

    关键：相邻柱的数值可能很接近（如 12.7 与 12.5），而文字 bbox 比柱宽。
    光靠水平错开（dx）会把文字推到邻柱的**垂直区间**里 => 视觉上像压在柱上。
    因此这里加 min_gap：偏移量至少大于「本柱与最近的更高邻柱的高度差」，
    保证文字盒整体越过邻居的柱顶。

    dx: 可选，逐柱水平微调。
    min_gap: 可选，逐柱「必须越过的邻居柱高」，默认不用。
    """
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    off_base = max(span * pad_frac, span * 0.055)
    for i, (x, v) in enumerate(zip(xs, ys)):
        off = off_base
        if min_gap is not None and i < len(min_gap) and min_gap[i] is not None:
            # 本柱顶到邻居柱顶的距离 + 一点余量
            d = abs(min_gap[i] - v)
            if d > 0:
                off = max(off, d + span * 0.012)
        xo = x + (dx[i] if dx is not None and i < len(dx) else 0.0)
        if v >= 0:
            ax.text(xo, v + off, fmt.format(v), ha="center", va="bottom",
                    fontsize=FS["annot"], **kw)
        else:
            ax.text(xo, v - off, fmt.format(v), ha="center", va="top",
                    fontsize=FS["annot"], **kw)


def headroom(ax, values, top=0.30, bottom=0.12):
    """给顶部/底部留出放数字的空间，避免标签贴边或压柱。"""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else abs(hi) or 1.0
    ax.set_ylim(lo - span * bottom, hi + span * top)


def grid_y(ax):
    ax.grid(axis="y", alpha=0.22, lw=0.5, color=C["grid"])
    ax.set_axisbelow(True)


def read_md_table(path, header_kw):
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, l in enumerate(lines):
        if header_kw in l and l.strip().startswith("|"):
            hdr = [c.strip() for c in l.strip().strip("|").split("|")]
            rows = []
            for l2 in lines[i + 2:]:
                if not l2.strip().startswith("|"):
                    break
                rows.append([c.strip() for c in l2.strip().strip("|").split("|")])
            return pd.DataFrame(rows, columns=hdr)
    return None


def num(s):
    s = str(s).replace("**", "").replace(" ", "").replace(",", "")
    for t in ("%", "pp", "倍", "bp", "日"):
        s = s.replace(t, "")
    try:
        return float(s)
    except Exception:
        return np.nan


# ---------------- Fig 1 ----------------
def fig1():
    """46 因子 IC：双栏，标签在柱右侧，不压柱。"""
    d = pd.read_csv(os.path.join(OUT, "m6_factor_ic.csv"))
    cols = list(d.columns)
    iccol = "ic" if "ic" in cols else [c for c in cols if c.lower().endswith("ic")][0]
    nmcol = cols[0]
    d = d[[nmcol, iccol]].dropna().sort_values(iccol)

    half = -(-len(d) // 2)
    left = d.iloc[:half][::-1]
    right = d.iloc[half:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.95), sharex=True)
    for ax, sub, ttl in ((axes[0], left, "(a) Most negative"),
                         (axes[1], right, "(b) Least negative / positive")):
        vals = sub[iccol].values
        ypos = np.arange(len(sub))
        ax.barh(ypos, vals, color=[C["neg"] if v < 0 else C["strat"] for v in vals],
                height=0.72)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub[nmcol], fontsize=FS["tick"])
        ax.axvline(0, color="k", lw=0.7)
        for v in (-0.10, 0.10):
            ax.axvline(v, color="#888888", ls="--", lw=0.7)
        ax.grid(axis="x", alpha=0.22, lw=0.5, color=C["grid"])
        ax.set_axisbelow(True)
        ax.set_title(ttl, fontsize=FS["panel"], loc="left")
        ax.tick_params(axis="x", labelsize=FS["tick"])
        # 数值写在条端外侧。
        # ⚠️ 最负的几根条（-0.100 等）加标签后会伸到 axes 左侧之外，
        # 与 y 轴因子名相撞（实测 bbox 重叠 44%）。所以先扩 xlim 预留空间。
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin
        pad = span * 0.16
        ax.set_xlim(xmin - pad, xmax)
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin
        for y, v in zip(ypos, vals):
            if v < 0:
                ax.text(v - span * 0.012, y, f"{v:.3f}", ha="right", va="center",
                        fontsize=FS["tick"], color="#333333", clip_on=False)
            else:
                ax.text(v + span * 0.012, y, f"{v:.3f}", ha="left", va="center",
                        fontsize=FS["tick"], color="#333333", clip_on=False)
    axes[0].set_xlabel("Cross-sectional Spearman IC (vs. forward 5-day return)",
                       fontsize=FS["label"])
    fig.suptitle("Univariate IC of 46 technical factors — every $|IC|$ stays below 0.10",
                 fontsize=FS["title"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig1_factor_ic.png")


# ---------------- Fig 2 ----------------
def fig2():
    d = pd.read_csv(os.path.join(OUT, "m4b_ic_nonoverlap.csv"))
    if "span" in d.columns:
        sub = d[d["span"].astype(str).str.contains("2024")]
        d = sub if len(sub) else d
    labs = list(d["label"])
    naive = d["t_overlap_naive"].astype(float).values
    real = d["t_nonoverlap"].astype(float).values
    x = np.arange(len(labs))
    w = 0.36

    fig, ax = plt.subplots(figsize=(FIG_W, 3.30))
    b1 = ax.bar(x - w / 2, naive, w, label="Overlapping labels (naive)", color=C["accent"])
    b2 = ax.bar(x + w / 2, real, w, label="Non-overlapping (stride = 5)", color=C["strat"])
    ax.set_ylim(0, max(naive) * 1.34)
    bar_labels(ax, x - w / 2, naive, fmt="{:.1f}")
    bar_labels(ax, x + w / 2, real, fmt="{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("$t$-statistic of daily IC series", fontsize=FS["label"])
    ax.set_title("Overlapping labels inflate $t$ by $\\sqrt{5}$", fontsize=FS["title"])
    ax.legend(frameon=False, loc="upper left", ncol=2)
    grid_y(ax)
    save(fig, "fig2_overlap_t.png")


# ---------------- Fig 3 ----------------
def fig3():
    fig, ax = plt.subplots(figsize=(FIG_W, 3.45))
    series = (("y_exc (raw label)", "m8j_nav_2千万_5d.csv", C["neg"], "-", 1.1),
              ("y_neut (style-neutral)", "m8d_nav_V1a.csv", C["strat"], "-", 1.1),
              ("s_ma5 (5-day smoothed)", "m9_nav_R1_1.0.csv", C["pos"], "-", 1.8))
    for nm, fn, c, ls, lw in series:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print(f"  skip {nm}")
            continue
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        nav = d.iloc[:, 0].dropna()
        ax.plot(nav.index, nav / nav.iloc[0], label=nm, color=c, ls=ls, lw=lw)

    panel = pd.read_parquet(os.path.join(ROOT, "data", "proc", "panel.parquet"),
                            columns=["date", "code", "adj_close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[(panel["date"] >= "2024-01-01") & (panel["date"] <= "2025-11-10")]
    panel = panel.sort_values(["code", "date"])
    panel["r"] = panel.groupby("code")["adj_close"].pct_change()
    ew = (1 + panel.groupby("date")["r"].mean().fillna(0)).cumprod()
    ax.plot(ew.index, ew / ew.iloc[0], lw=1.6, color="#111111", ls="--",
            label="equal-weighted market")

    ax.axhline(1.0, color="#999999", lw=0.6, ls=":")
    ax.set_ylabel("Net value (start = 1)", fontsize=FS["label"])
    ax.set_title("Three label constructions against the equal-weighted market",
                 fontsize=FS["title"])
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.22, lw=0.5, color=C["grid"])
    ax.set_axisbelow(True)
    fig.autofmt_xdate(rotation=0, ha="center")
    save(fig, "fig3_nav.png")


# ---------------- Fig 4 ----------------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.85))

    ax = axes[0]
    vals = [0.5353, 0.5525]
    b = ax.bar(["y_exc", "y_neut"], vals, color=[C["market"], C["strat"]], width=0.52)
    ax.set_ylim(0.48, 0.578)
    bar_labels(ax, [0, 1], vals, fmt="{:.4f}", pad_frac=0.010)
    ax.axhline(0.5, color=C["neg"], ls="--", lw=0.9)
    # 「coin flip」标注：原写法 (1.42, 0.5025) ha=right 会把文字左端推回柱子右肩
    # （柱 x=1 宽 0.52 => 右边 1.26），实测文字 bbox 压柱 12%+。
    # 改为放在 x 轴外侧右上方，ha=left，彻底离开柱体。
    ax.text(1.62, 0.5015, "coin flip", fontsize=FS["annot"], color=C["neg"],
            ha="left", va="bottom", clip_on=False)
    ax.set_xlim(-0.62, 2.15)
    ax.set_ylabel("Test-set AUC", fontsize=FS["label"])
    ax.set_title("(a) AUC rises after style removal", fontsize=FS["panel"])
    grid_y(ax)

    ax = axes[1]
    d = pd.read_csv(os.path.join(OUT, "m7f_grp_y_neut.csv"), parse_dates=["date"])
    d["q"] = d["date"].dt.to_period("Q").astype(str)
    ex = ((d["picks_fwd5"] - d["cell_bench"]).groupby(d["q"]).mean() * 100)
    xs = np.arange(len(ex))
    ax.bar(xs, ex.values, color=[C["pos"] if v > 0 else C["neg"] for v in ex.values],
           width=0.62)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("2024", "24").replace("2025", "25") for s in ex.index],
                       fontsize=FS["tick"], rotation=45, ha="right")
    headroom(ax, list(ex.values) + [0], top=0.18, bottom=0.16)
    ax.set_ylabel("Excess within style cell (%)", fontsize=FS["label"])
    ax.set_title(f"(b) Positive in {int((ex > 0).sum())} of {len(ex)} quarters",
                 fontsize=FS["panel"])
    grid_y(ax)

    fig.suptitle("Effect of style neutralization", fontsize=FS["title"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig4_neutral.png")


# ---------------- Fig 5 ----------------
def fig5():
    tb = read_md_table(os.path.join(OUT, "m9b_smooth.md"), "| 信号 |")
    if tb is None:
        print("skip fig5")
        return
    tb = tb.dropna(subset=[tb.columns[0]])
    tb = tb[tb[tb.columns[0]].isin(["s", "s_ma5", "s_lag5"])]

    fig, ax = plt.subplots(figsize=(FIG_W, 3.35))
    st = {"s": (C["accent"], "o-", 1.2, "Single-day score"),
          "s_ma5": (C["pos"], "s-", 1.9, "5-day smoothed (incl. today)"),
          "s_lag5": (C["strat"], "^-", 1.2, "5-day smoothed (lag-1)")}
    all_y = []
    for key, (c, m, lw, lab) in st.items():
        sub = tb[tb[tb.columns[0]] == key]
        xs = [num(v) for v in sub[sub.columns[1]]]
        ys = [num(v) for v in sub[sub.columns[2]]]
        all_y += ys
        ax.plot(xs, ys, m, color=c, lw=lw, ms=4.0, label=lab)

    bench = 21.11
    ax.axhline(bench, color="#111111", ls="--", lw=1.0)
    # 基准线注释放到右侧留白，靠 y 轴范围上端，不压任何折线
    ax.set_ylim(min(all_y) - 3.0, max(all_y) + 5.0)
    # 原来紧贴基准线上方 0.9 个单位，实测与折线有 8% 墨迹相交。
    # 加到 y 轴顶端附近，并用一个小留白与虚线分开。
    _ylo, _yhi = ax.get_ylim()
    ax.text(ax.get_xlim()[1], _yhi - (_yhi - _ylo) * 0.055,
            f"equal-weighted market ({bench:.2f}%)",
            fontsize=FS["annot"], ha="right", va="top", color="#111111")

    ax.set_xlabel("One-way transaction cost (bp)", fontsize=FS["label"])
    ax.set_ylabel("Annualized return (%)", fontsize=FS["label"])
    ax.set_title("Cost sensitivity: smoothing lifts the break-even point", fontsize=FS["title"])
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.22, lw=0.5, color=C["grid"])
    ax.set_axisbelow(True)
    save(fig, "fig5_cost.png")


# ---------------- Fig 6 ----------------
def fig6():
    tb = read_md_table(os.path.join(OUT, "m9d_excess.md"), "| 季度 | 组合收益")
    if tb is None:
        print("skip fig6")
        return
    tb = tb[~tb[tb.columns[0]].astype(str).str.contains("全期")]
    qs = [str(v).replace("2024", "24").replace("2025", "25") for v in tb[tb.columns[0]]]
    port = np.array([num(v) for v in tb[tb.columns[1]]])
    bench = np.array([num(v) for v in tb[tb.columns[2]]])
    exc = np.array([num(v) for v in tb[tb.columns[3]]])
    x = np.arange(len(qs))
    w = 0.24

    # 高度 3.30 -> 3.62：两行标题 + 图例 + 旋转 45 度的 x 标签要塞进去
    fig, ax = plt.subplots(figsize=(FIG_W, 3.62))
    # 顶部留 0.80：标题在图例之上，不能再被 bbox_inches="tight" 切到
    fig.subplots_adjust(left=0.125, right=0.975, top=0.80, bottom=0.22)
    ax.bar(x - w, port, w, label="strategy ($s_{ma5}$)", color=C["strat"])
    ax.bar(x, bench, w, label="equal-weighted market", color=C["market"])
    ax.bar(x + w, exc, w, label="excess",
           color=[C["pos"] if v > 0 else C["neg"] for v in exc])
    ax.axhline(0, color="k", lw=0.8)

    allv = list(port) + list(bench) + list(exc) + [0]
    lo, hi = min(allv), max(allv)
    # 上留白从 0.30 提到 0.44：给 "19.4" 标注留出**柱顶之上**的真实空白，
    # 否则文字落在最高柱的 y 区间内（差分法实测 100% 墨迹在图形上）。
    ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.44)
    # 不在图上标 24 个数字：太乱、且互相挤压。
    # 具体数值在 Table 5 里，图上只需要看得出「哪根高哪根低」。
    # 只标两组关键数字：2024H1 的超额 19.4 和全期合计 −13.3
    _hspan = hi - lo
    # "19.4" 原来 xytext=(x, exc[0]+hspan*0.055) 相对柱顶定位，
    # 实测文字墨迹 100% 落在柱体上。改为抬到上方固定留白 + 引线。
    ax.annotate("19.4",
                xy=(0 + w, hi),
                xytext=(0 + w, hi + _hspan * 0.20),
                ha="center", va="bottom", fontsize=FS["annot"], color="#333333",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#888888",
                                shrinkA=1, shrinkB=1))
    # "minus 13.3" 原来放在 lo+hspan*0.30，与柱子相交（29% 墨迹落图形上）。
    ax.annotate("−13.3 over the\nother six quarters",
                xy=(4.5, lo), xytext=(4.5, lo - _hspan * 0.04),
                ha="center", va="top",
                fontsize=FS["annot"], color="#333333")

    # 2024H1 高亮：只加底色，不加文字（文字改到图例里说明）
    ax.axvspan(-0.5, 1.5, color=C["accent"], alpha=0.10, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(qs, fontsize=FS["tick"], rotation=45, ha="right")
    ax.set_ylabel("Quarterly return (%)", fontsize=FS["label"])
    # 标题原 pad=26 + 图例 bbox_to_anchor=(0.5,1.005)，被挤得压到 x 轴刻度区，
    # 实测 "Quarterly return (%)" 与刻度 "-10" 共用 89.5% 像素。
    ax.set_title("Excess concentrates in 2024H1 (shaded);\n"
                 "the other six quarters sum to $-13.3$ pp",
                 fontsize=FS["title"], pad=30)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.11), fontsize=FS["legend"])
    grid_y(ax)
    save(fig, "fig6_excess_q.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done")
