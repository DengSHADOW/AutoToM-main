# Partner TODO — Tengyang Deng

> 截止：5月3日（最终报告 due）
> 我（Zhuoxuan）这两天已完成的部分见底部 "Already Done"。

---

## 🎯 核心任务（必须完成）

### 1. ToMi 泛化实验（最高优先级）
**目标**：用 ToMi 作为第三个数据集，验证 PAWM 的 Φᵢ 抽象不局限于 BigToM/FANToM。

- 数据集：ToMi (Le et al. 2019)，重点跑 **first-order false belief** 和 **second-order false belief** 子集（各 50 条即可）
- 实现要点：
  - ToMi 的 Φᵢ = "agent 离开房间之前能看到的 location moves"
  - 复用 `model/pawm.py` 里的 `filter_story_for_agent()` 模式，但需要写一个 `_filter_tomi_story()`，按句子级别检测 "X left the room" / "X entered the room" 切分窗口
  - 在 `ProbSolver.py` 的 PAWM 入口处加一个 `"ToMi" in dataset_name` 分支
- 跑两组：**baseline AutoToM (no PAWM) vs AutoToM + PAWM**
- 期望：first-order ≥ +5pp，second-order 至少不掉点
- 输出：把 ToMi 的数字补到 final report Table 1（和 BigToM/FANToM 并列）

### 2. 最终报告整合
**位置**：把 [final_report_additions.md](final_report_additions.md) 的 §6 / §7 / §8 / §9 整合进主报告 [PAWM_report.md](PAWM_report.md)。

- §6 Generalization → 插在 Results 之后
- §7 Failure Modes → §6 之后
- §8 Beyond Text → §7 之后
- §9 Revised Conclusion → 替换原 §5 Conclusion
- 把 ToMi 实验结果插到 §6.2 末尾作为 "empirical validation of Φᵢ generality"
- 检查 §7 的 12 个错误数字是否需要根据你的 ToMi 跑出来后的总错误数更新

### 3. BigToM 48-case 回归
**目的**：确认 FANToM 的改动没影响 BigToM 86%。

```bash
cd /Users/lzx/AutoToM-main
python eval_pawm_48.py 2>&1 | tee bigtom_regression.log
```
- 期望：48 / 48 跑完，准确率 ≥ 85%（86% 是上次的数字，允许 ±1 LLM 噪声）
- 如果掉了 ≥ 2pp：检查 `ProbSolver.py:703-731` 的 epistemic prior 分支是否真的被 `"FANToM" in dataset_name` gate 住了

---

## 📝 报告辅助任务

### 4. Failure mode §7 配图
- §7.5 的失败模式 4 类计数表 → 做成一个 stacked bar 或饼图（matplotlib 即可）
- 配色用 PPT 里的 NAVY/AMBER/CRIMSON 系，保持一致

### 5. §8 多模态部分的引用
- §8.1 提到的 AI2-THOR / RoboTHOR / MMToM-QA / Social-IQ-2.0 需要补正式 BibTeX 引用
- 找一下原 paper 加到 references.bib

---

## ⛔ 不要做（明确 future work）

- **不要实现 second-order FANToM PAWM** — 在报告 §6.2.1 里作为 future work 提一句即可
- **不要实际跑多模态实验** — §8 全部是 conceptual sketch，不需要任何代码
- **不要再调 epistemic prior** — 当前 60.0% 是 conservative 的稳定点，再调会引入 LLM 噪声方差

---

## ✅ Already Done（我这边完成的）

- [x] FANToM 1st-order PAWM (mechanical agent-window filter) — 50% → 60%
- [x] BigToM PAWM (fork-detector) — 76% → 86%
- [x] Mode A epistemic prior 实现并接入 BIP（[pawm.py](model/pawm.py) 的 `epistemic_prior_reweight()` + [ProbSolver.py:703-731](model/ProbSolver.py#L703-L731)）
- [x] focal-agent override 修复 multi-agent 歧义（Mode B 部分缓解）
- [x] 12 个 FANToM 错误案例分类成 Mode A/B/C/D
- [x] [final_report.docx](final_report.docx) 演讲稿 + [final_report.pptx](final_report.pptx) 学术风格 PPT
- [x] [final_report_additions.md](final_report_additions.md) §6/§7/§8/§9 草稿（共 ~1900 字）

---

## 📞 沟通

- 跑完 ToMi 实验直接微信发我数字，我合并到 Table 1
- 报告整合完先互相 review 一遍再交
- 有 bug 直接看 `git log --oneline` 找 fantom 分支最近 commit
