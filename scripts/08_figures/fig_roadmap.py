# -*- coding: utf-8 -*-
"""fig_roadmap.py —— Fig.8 研究思路图（横向六卡，极简版）

2026-11-15 18:28 用户：「图8也有同样问题」（太乱、数字挤、文字压图形）
按 Fig.6/7 同样的原则处理：**删掉卡片里的所有结果数字**，
每张卡片只留三样东西：序号 / 动作标题 / 判决词。

数字去哪了：全部在正文 2.6 节的六步审计里逐条给出，
图的作用是让读者一眼看到「六步、三步否决、结论怎么塌的」，
不是替代表格。图上堆数字只会互相挤压。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# 六步：号 / 动作（两行）/ 判决 / 颜色
# 只用红绿两色（用户 2026-11-15 18:29 要求去掉琥珀色）：
#   green = 结论站得住（含「站得住但变弱」）
#   red   = 结论被推翻或被解释掉
STEPS = [
    ("1", "46 technical\nfactors",             "holds",          "g"),
    ("2", "Overlap\ncorrection",               "holds, weaker",  "g"),
    ("3", "Style\nneutralization",             "improves",       "g"),
    ("4", "Benchmark\naudit",                  "reversed",       "r"),
    ("5", "Execution\naccounting",             "reversed",       "r"),
    ("6", "Sign of\nbenchmark",                "explains",       "r"),
]

COL = {"g": "#2e7d32", "r": "#c62828"}
BG = {"g": "#eef7ef", "r": "#fdeeee"}

fig, ax = plt.subplots(figsize=(6.2, 2.55))
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.axis("off")

W, GAP = 0.945, 0.055
Y_TOP, Y_H = 0.815, 0.560      # 卡片带
Y_BOT = Y_TOP - Y_H

FS_ACT, FS_VER, FS_NUM = 9.0, 8.2, 10.5

for i, (num, act, verdict, c) in enumerate(STEPS):
    x = i * (W + GAP) + 0.02
    ax.add_patch(FancyBboxPatch((x, Y_BOT), W, Y_H,
                                boxstyle="round,pad=0.008,rounding_size=0.03",
                                linewidth=1.0, edgecolor=COL[c], facecolor=BG[c]))

    # 序号靠上，动作居中，判决贴底 —— 三样东西各有固定带，不会互撞
    ax.text(x + W / 2, Y_TOP - 0.105, num, fontsize=FS_NUM, fontweight="bold",
            color=COL[c], ha="center", va="center")
    ax.text(x + W / 2, Y_TOP - 0.290, act, fontsize=FS_ACT, fontweight="bold",
            ha="center", va="center", color="#1a1a1a", linespacing=1.30)
    ax.plot([x + 0.10, x + W - 0.10], [Y_BOT + 0.155] * 2,
            color=COL[c], linewidth=0.6, alpha=0.55)
    ax.text(x + W / 2, Y_BOT + 0.080, verdict, fontsize=FS_VER,
            fontweight="bold", style="italic", ha="center", va="center",
            color=COL[c])

for i in range(len(STEPS) - 1):
    x0 = i * (W + GAP) + 0.02 + W + 0.005
    x1 = (i + 1) * (W + GAP) + 0.02 - 0.005
    ax.add_patch(FancyArrowPatch((x0, Y_TOP - Y_H / 2), (x1, Y_TOP - Y_H / 2),
                                 arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.8, color="#666666"))

ax.text(0.02, 0.935,
        "Each check is run only because the previous result made it necessary.",
        fontsize=8.0, style="italic", color="#444444", va="center", ha="left")
ax.text(0.02, 0.085,
        "green = the conclusion survives    red = it is reversed or explained"
        "    (figures for each step are given in Section 2.6)",
        fontsize=7.4, color="#444444", va="center", ha="left")

p = os.path.join(FIG, "fig8_roadmap.png")
fig.savefig(p, dpi=1400, bbox_inches="tight", facecolor="white", pad_inches=0.10)

# sidecar
import json as _json
_fs = [t.get_fontsize() for t in fig.findobj()
       if hasattr(t, "get_fontsize") and t.get_text() and t.get_text().strip()]
_mp = os.path.join(FIG, "fig_manifest.json")
_man = {}
if os.path.exists(_mp):
    try:
        _man = _json.load(open(_mp, encoding="utf-8"))
    except Exception:
        _man = {}
_man["fig8_roadmap.png"] = dict(fig_w_in=round(fig.get_size_inches()[0], 2),
                                min_fontsize=round(min(_fs), 2) if _fs else None,
                                n_text=len(_fs))
_json.dump(_man, open(_mp, "w", encoding="utf-8"), indent=1)

plt.close(fig)

from PIL import Image
im = Image.open(p)
print(f"[fig] fig8_roadmap.png  {im.size[0]}x{im.size[1]}  {os.path.getsize(p)/1024:.0f} KB")
print(f"  15cm wide -> {15.0*im.size[1]/im.size[0]:.1f} cm  min_fs={min(_fs) if _fs else 0}")
