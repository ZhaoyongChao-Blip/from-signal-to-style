# -*- coding: utf-8 -*-
"""reg43.py —— 把 4.3 节的数字登记进 numbers.json，并**带行标识**

背景：4.3 曾经把 m5d_attrib.md 的「原始收益」行的 beta/R2 与「超额 vs EW」行的
t 统计量拼成一句话。所以这里每个 key 的名字都带 row 前缀，
断言时也能检验「同一行」的配对关系。
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "paper", "numbers.json")
SRC = "output/m5d_attrib.md"

new_nums = {
    # —— row=raw（因变量 = 原始收益，三因子回归）——
    "attr_raw_beta_ew_yraw": 1.84,
    "attr_raw_beta_ew_yexc": 1.83,
    "attr_raw_beta_hs300_yraw": -0.11,
    "attr_raw_beta_hs300_yexc": -0.05,
    "attr_raw_r2_yraw": 0.807,
    "attr_raw_r2_yexc": 0.746,
    "attr_raw_t_alpha_yraw": -0.72,
    "attr_raw_t_alpha_yexc": -1.39,
    # —— row=excess（因变量 = 超额 vs EW，此时 beta 无定义）——
    "attr_exc_t_alpha_yraw": 0.03,
    "attr_exc_t_alpha_yexc": -0.63,
    "attr_exc_r2_yraw": 0.067,
    "attr_exc_r2_yexc": 0.145,
    # —— 共同 ——
    "attr_n_days": 447,
}
new_src = {k: SRC for k in new_nums}

d = json.load(open(P, encoding="utf-8"))
d["numbers"].update(new_nums)
d["source"].update(new_src)
json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("已登记 %d 个数字，总数 %d" % (len(new_nums), len(d["numbers"])))
