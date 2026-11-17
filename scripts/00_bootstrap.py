# -*- coding: utf-8 -*-
"""
00_bootstrap.py — make this archive runnable with the exact layout the scripts expect.

The original scripts were written against a single flat project root:

    <root>/t1_fetch.py, t3_features.py, ...
    <root>/data/raw/{list.csv,daily/,adj/,index/}
    <root>/data/proc/panel.parquet

In this release the scripts are grouped into stage subfolders. Every script computes
    ROOT = dirname(abspath(__file__))
so they would look for their data inside e.g. scripts/01_fetch/data/.

This bootstrap creates a working tree `work/` in which each script is flattened back
to `work/` and `work/data/` is populated — **without copying the scripts**, using
hardlinks where possible. It never modifies the released files.

Usage:
    python scripts/00_bootstrap.py            # build work/
    python scripts/00_bootstrap.py --check    # build + report what is missing
"""
import os
import sys
import shutil
import json
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
RELROOT = os.path.dirname(HERE)
WORK = os.path.join(RELROOT, "work")

STAGES = ["01_fetch", "02_features", "03_models", "04_backtest", "05_diagnostics",
          "06_walkforward", "07_attribution", "08_figures", "09_audit", "10_build"]

DIRS = ["data/raw", "data/raw/daily", "data/raw/adj", "data/raw/index",
        "data/proc", "figs", "models", "output", "logs", "paper"]


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)          # same volume: no extra bytes
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def build():
    os.makedirs(WORK, exist_ok=True)
    for d in DIRS:
        os.makedirs(os.path.join(WORK, d), exist_ok=True)

    report = {}
    for stage in STAGES:
        sdir = os.path.join(HERE, stage)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".py"):
                continue
            report[fn] = link_or_copy(os.path.join(sdir, fn), os.path.join(WORK, fn))

    # shipped seeds
    seeds = [
        ("data_raw/list.csv",      "data/raw/list.csv"),
        ("data_raw/feat_meta.json", "data/proc/feat_meta.json"),
        ("paper/numbers.json",      "paper/numbers.json"),
        ("paper/numbers.json",      "output/numbers.json"),
        ("paper/manuscript.md",     "paper/paper.md"),
        ("paper/refs_final.json",   "paper/refs_final.json"),
    ]
    for src, dst in seeds:
        s = os.path.join(RELROOT, src)
        if os.path.exists(s):
            report["seed:" + dst] = link_or_copy(s, os.path.join(WORK, dst))

    # shipped intermediates: results/* -> work/output/*  (post-model stages read these)
    res = os.path.join(RELROOT, "results")
    if os.path.isdir(res):
        for fn in sorted(os.listdir(res)):
            if fn.endswith((".csv", ".md", ".json", ".txt")):
                report["output/" + fn] = link_or_copy(
                    os.path.join(res, fn), os.path.join(WORK, "output", fn))

        # PDFs referenced by the audit scripts live under paper/ in the original tree
    papers = os.path.join(RELROOT, "paper")
    if os.path.isdir(papers):
        for fn in sorted(os.listdir(papers)):
            if fn.endswith(".pdf"):
                report["paper/" + fn] = link_or_copy(
                    os.path.join(papers, fn), os.path.join(WORK, "paper", fn))

    # figures + model scores
    for sub, pattern_dir in (("figures", "figs"), ("assets", "models")):
        d = os.path.join(RELROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            report["%s/%s" % (pattern_dir, fn)] = link_or_copy(
                os.path.join(d, fn), os.path.join(WORK, pattern_dir, fn))

    return report


def check(report):
    meta_path = os.path.join(WORK, "data/proc/feat_meta.json")
    problems = []
    if not os.path.exists(meta_path):
        problems.append("feat_meta.json missing — run without --check first")
    else:
        metadata = json.load(open(meta_path, encoding="utf-8"))
        if metadata["stats"]["ok"] != 3317:
            problems.append("stock universe != 3317 (%s)" % metadata["stats"]["ok"])
        if metadata["n_factors"] != 46:
            problems.append("n_factors != 46 (%s)" % metadata["n_factors"])

    panel = os.path.join(WORK, "data/proc/panel.parquet")
    if not os.path.exists(panel):
        problems.append("panel.parquet absent — this is expected in a fresh checkout; "
                        "rebuild it with scripts 01_fetch -> 02_features (needs network)")

    refs = os.path.join(RELROOT, "paper/manuscript.md")
    if os.path.exists(refs):
        txt = open(refs, encoding="utf-8").read()
        n = txt.split("## References")[-1].count("https://doi.org") + \
            txt.split("## References")[-1].count("https://openreview.net")
        if n != 37:
            problems.append("reference count %d != 37" % n)

    print("=" * 60)
    print("work tree:", WORK)
    print("files staged:", len(report))
    if problems:
        print("WARNINGS:")
        for p in problems:
            print("  -", p)
    else:
        print("all seed checks passed")
    print("=" * 60)
    return 0 if not problems else 1


if __name__ == "__main__":
    rep = build()
    if "--check" in sys.argv:
        sys.exit(check(rep))
    print("work/ built:", len(rep), "entries")
