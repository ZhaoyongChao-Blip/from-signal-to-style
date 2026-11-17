# -*- coding: utf-8 -*-
"""verify_all_dois.py —— 把论文里全部 DOI 逐条送 Crossref 核 title/作者/年份。

背景（MEMORY.md 2026-11-03）：
    DOI 是字符串记忆最容易出错的地方。规则：GET api.crossref.org/works/<DOI>
    → 核对 title 前 5 个词 + 第一作者姓 + 年份，三条全对才算通过。

输出：控制台表 + dois_verified.txt（供人工复核）
用法：python verify_all_dois.py [--sleep 1.0]
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper", "paper.md")
OUT = os.path.join(HERE, "dois_verified.txt")
UA = "Mozilla/5.0 (compatible; paper-doi-check/1.0; mailto:1792552210@qq.com)"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def norm(s):
    # Crossref titles often carry <i>/<sub> markup and differ in case; strip both.
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def main():
    sleep = 1.0
    if "--sleep" in sys.argv:
        sleep = float(sys.argv[sys.argv.index("--sleep") + 1])

    t = open(MD, encoding="utf-8").read()
    j = t.find("## References")
    refs = t[j:]

    # 每条文献：以 "Surname, X." 开头，可能跨多行，直到下一个空行组
    entries = re.findall(r"([A-Z][^\n]*?\((\d{4})\)\.[^\n]*(?:\n(?!\n)[^\n]*)*)", refs)
    print(f"parsed {len(entries)} reference entries")

    rows = []
    n_pass = n_fail = n_err = 0
    for block, year in entries:
        m = re.search(r"https://doi\.org/(\S+)", block)
        if not m:
            rows.append(("NO-DOI", block[:60], "", "", ""))
            n_err += 1
            continue
        doi = m.group(1).rstrip(".")
        first_author = block.split(",")[0].strip()
        # 取标题：年份后第一个句点之后的文本。注意 Abbrev. 如 "vs." 会提前截断，
        # 所以取到 "*Journal*" 或 DOI 之前，再用第一个以大写字母开头的句子边界收敛。
        tm = re.search(r"\)\.\s*(.+?)(?:\.\s\*|\s\*\*|\s+https?://)", block, re.S)
        title = re.sub(r"\s+", " ", tm.group(1)).strip() if tm else ""
        try:
            d = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        except Exception as e:
            rows.append(("HTTP-ERR", doi, title[:50], str(e)[:40], ""))
            n_err += 1
            time.sleep(sleep)
            continue
        msg = d.get("message", {})
        ct = (msg.get("title") or [""])[0]
        auth = msg.get("author") or []
        cfirst = auth[0].get("family", "") if auth else ""
        cy = None
        for k in ("published-print", "published-online", "issued"):
            if msg.get(k, {}).get("date-parts"):
                cy = msg[k]["date-parts"][0][0]
                break
        t_ok = norm(ct)[:5] == norm(title)[:5]
        a_ok = cfirst.lower().strip() == first_author.lower().strip()
        # 年份容忍 ±1：online-first 与 issue 年常差一年，且 SSRN/预印本与正式发表不同
        y_ok = (cy is not None and abs(int(cy) - int(year)) <= 1)
        verdict = "PASS" if (t_ok and a_ok and y_ok) else "FAIL"
        if verdict == "PASS":
            n_pass += 1
        else:
            n_fail += 1
        rows.append((verdict, doi,
                     f"title_ok={t_ok} author_ok={a_ok}({cfirst}/{first_author}) "
                     f"year_ok={y_ok}({cy}/{year})",
                     ct[:70], title[:70]))
        time.sleep(sleep)

    print(f"\n{'verdict':8} {'doi':45} detail")
    print("-" * 120)
    for r in rows:
        print(f"{r[0]:8} {str(r[1])[:45]:45} {r[2]}")
    print("-" * 120)
    print(f"PASS {n_pass}  FAIL {n_fail}  ERR {n_err}  total {len(rows)}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"[written] {OUT}")
    return 0 if n_fail == 0 and n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
