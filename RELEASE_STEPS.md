# 发布操作清单（Gitee + Zenodo 双保险）

归档已构建完成并通过验证，**只差账号动作**。以下每一步都标明了「谁做」——
凡是要密码/验证码/点击的，只能你做；凡是我能做的，已经做完了。

---

## 已完成（我这边）

| 项 | 结果 |
|---|---|
| 归档目录 | `C:\Users\17925\.openclaw\workspace\_release_ts_stock_lgb\`（221 文件 / 20 MB） |
| 发布包 | `C:\Users\17925\.openclaw\workspace\from-signal-to-style_v1.0.zip`（7.5 MB） |
| git 仓库 | 已 `git init`，2 个提交，`main` 分支，含 `.gitignore` / `LICENSE`(MIT) |
| 自检 | 解压后 `python scripts/00_bootstrap.py --check` 通过；补上 `panel.parquet` 后 `sign_split.py` 与 `xcheck.py` 全过 |
| 复算证据 | `results/REPLICATION_CHECK.md`（实测输出，非承诺） |
| 数字溯源 | `docs/ASSERTIONS.md`（15 项，逐项标注已验证） |
| Zenodo 元数据 | `zenodo.json`（占位，需你填作者信息） |

---

## 步骤 1：Gitee 建仓（你做，约 5 分钟）

1. 打开 https://gitee.com → 注册（手机号 + 验证码，国内无墙）
2. 右上角 `+` → **新建仓库**
   - 仓库名称：`from-signal-to-style`
   - 路径：自动
   - **开源**（选「公开」；Zenodo 要公开才能长期存档，Gitee 公开仓也能被引用）
   - **不要**勾选「使用 Readme 初始化」/「.gitignore 模板」/「开源许可证」——三样都别勾，否则推送会冲突
3. 建完后页面会显示仓库地址，形如 `https://gitee.com/<你的用户名>/from-signal-to-style.git`

## 步骤 2：推送（我做，你只需给我地址）

把仓库地址发我，我执行：

```
cd _release_ts_stock_lgb
git remote add origin https://gitee.com/<用户名>/from-signal-to-style.git
git push -u origin main
```

**推送时会弹窗要账号密码**——Gitee 现在要求用**私人令牌（Private Token）**而不是登录密码：
Gitee 网页 → 设置 → 私人令牌 → 生成新令牌 → 勾 `projects` 权限 → 复制那串字符。

---

## 步骤 3：Zenodo（你做，约 10 分钟）

Zenodo 在中国大陆访问正常（已实测本机 `https://zenodo.org` 返回 200，不需要梯子）。

1. https://zenodo.org → 右上 `Sign up` → 用邮箱注册（**建议直接用 ORCID 登录**，一步到位且会自动带作者身份）
2. 验证邮件 → 激活
3. 右上 `New upload`
4. **Files**：上传 `from-signal-to-style_v1.0.zip`
5. **Metadata**：
   - 把 `_release_ts_stock_lgb/zenodo.json` 里的内容对应填入，或直接点 Zenodo 页面的 `Import` 导入该 JSON
   - **必须改**：`creators` → 填 `Zhao, Yongchao`（名/姓分开填），affiliation 填 `Nanning Normal University`
   - `Upload type` 选 **Software**
   - `License` 选 **MIT License**
6. **Publish** → Zenodo 立刻生成 DOI（形如 `10.5281/zenodo.1234567`）
7. 把 DOI 发我，我改三处：
   - `README.md` 的 citation 块
   - `zenodo.json` 的 related_identifiers
   - **论文 Data availability 段落**（现在是 "available ... on request"，要改成仓库 URL + DOI）
   - 改完重新生成 QF 的 docx/pdf 附件

---

## ⚠️ 一个先后顺序的坑（必须知道）

Zenodo 的 DOI **一旦发布就永久固定**，不能重新指向另一个文件。所以：

- **先传 GitHub/Gitee 版本？** → Zenodo 记录里可以后加 `related_identifiers` 指向仓库 URL，没问题
- **但反过来不行**：先发 Zenodo 再改 zip 内容，只能发**新版本**（`v1.1`），且旧 DOI 永久存在

**建议顺序：先推 Gitee（步骤 1-2）→ 确认仓库可访问 → 再发 Zenodo（步骤 3）**，
这样 Zenodo 记录里能直接写上仓库链接，审稿人看到的是「代码仓库 + 永久 DOI」两条腿。

---

## 论文里的原文（等你给 DOI 后我改）

现在 `submission/QF_Declarations.md` 与 `paper/paper.md` 写的是：

> is available as a versioned code archive from the corresponding author on request
> and will be deposited in a public repository upon acceptance.

改成：

> is openly available at https://gitee.com/<user>/from-signal-to-style and archived
> with a permanent DOI at https://doi.org/10.5281/zenodo.XXXXXXX.

这一改，Data availability 就从「承诺」变成「事实」——**审稿人对 negative result
论文最挑的一点就是「你说能复现，那在哪儿」**，有 DOI 直接堵住这条。

---

## 如果 Gitee 公开仓审核不过（备选）

Gitee 公开仓库偶有内容审核。若被卡，按此降级：

1. **Atomgit**（开放原子基金会，https://atomgit.com）——国内注册，背景更硬，审核宽松
2. **GitCode**（CSDN 系，https://gitcode.com）
3. **Zenodo 单独用**——DOI 本身已经是合规的代码存档，仓库只是「好浏览」的附加项

三种情况都不影响论文结论与投稿，只影响审稿人点链接的体验。
