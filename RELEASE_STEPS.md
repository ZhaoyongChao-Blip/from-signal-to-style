# 发布记录：GitHub + Zenodo

**状态：已全部完成（2026-11-17）。**

| 项 | 结果 |
|---|---|
| GitHub 仓库 | https://github.com/ZhaoyongChao-Blip/from-signal-to-style |
| GitHub Release | `v1.0`（附带 `from-signal-to-style_v1.0.zip`） |
| Zenodo 记录 | https://zenodo.org/records/24297971 |
| **DOI** | **10.5281/zenodo.24297971**（version 1.0；concept DOI `10.5281/zenodo.24297970` 指向所有版本） |
| 发布日期 | 2026-11-17 |
| 访问权限 / 许可 | Open Access / MIT |
| 文件 | `from-signal-to-style_v1.0.zip` · 6,866,504 bytes · md5 `e269cc47b4cbc11ce01e18703b9ef0f2` |
| 论文 Data availability | 已写入仓库 URL + DOI，`QF_Manuscript.docx` / `.pdf` 已重生成（16 页） |

发布后做过三项独立验证：
DOI 解析返回 200；Zenodo 公开 API 显示 `status: published` 且文件 checksum 与上传值一致；
从 Zenodo 重新下载 zip 后本地 md5 与源文件逐字节相同。

---

## 附录：本次踩到的坑（以后别再走一遍）

**1. Zenodo v3 的 personal access token 对外部 CLI 一律被拒。**
这是个 InvenioRDM 实例，即使 token 同时勾了 `deposit:actions` + `deposit:write`，
`curl` / `requests` 直连 `POST /api/deposit/depositions` 仍然返回 403，
`GET /api/me` 返回 401。浏览器会话内**同源 fetch + `csrftoken` cookie** 才通（直接 201）。
所以不要在设计上依赖 PAT；正确做法是在页面里跑 fetch。

**2. HTTPS 页面 fetch `http://127.0.0.1` 会被混合内容策略拦掉。**
想「本地起个 http server 把 zip 喂给页面」的思路不通，报 `TypeError: Failed to fetch`。
改用 CDP `DOM.setFileInputFiles` 把文件路径注入页面里人为创建的 `<input type=file>`，
页面拿到的是真正的 `File` 对象，再用 `FormData` 上传。见 `tools/cdp_upload.py`。

**3. PowerShell 传长 JS 字符串会被引号破坏。**
多 KB 的 `evaluate` 参数经过 PowerShell 会被吞引号 / 改写转义。
一律把脚本写成 `.js` 文件，用 `tools/cdp_eval.py` 从磁盘读，走 CDP WebSocket 直连执行。

**4. 发布不可逆。**
DOI 在 publish 那一刻就注册进 DataCite，之后只能另开新版本（新 DOI）。
所以脚本 `zenodo_step_publish.js` 在 publish 前会先自查 title / description / creators / files
四项非空，任一缺失就拒绝发布。

发布用的临时脚本：`tools/zenodo_step_meta.js`、`tools/zenodo_step_upload2.js`、
`tools/zenodo_step_publish.js`、`tools/cdp_eval.py`、`tools/cdp_upload.py`。

---

## 后续（拿到论文录用后）

1. 把论文正刊 DOI 加进 Zenodo 记录的 related identifier（`relation: isSupplementTo`）。
   `zenodo.json` 里已留了说明。
2. 若代码需要修订，在 Zenodo 记录页点 **New version**（保持 concept DOI 不变，mint 新的 version DOI）。
3. GitHub 侧建议给 `main` 打 tag `v1.0` 与 Zenodo 记录对齐：
   `git tag -a v1.0 -m "Zenodo 10.5281/zenodo.24297971" && git push --tags`
