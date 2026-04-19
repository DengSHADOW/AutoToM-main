# PAWM 项目进展总结

## 当前完成工作

### 1. FANToM 支持（Conversational Mode）

**问题**：AutoToM 原始代码不支持 FANToM benchmark，存在两个 bug：
- 文件名 `:` 分隔符在 Windows 下不兼容 → 改为 `_`
- `determine_higher_order_belief()` 将 FANToM 1st-order 问题误判为 2nd-order → 在 ProbSolver 里对 FANToM-1st 数据集硬编码 `nested=False`

**实现**：新增 `_apply_conv_pawm()` 函数，处理对话型场景下的信息不对称：

| | BigToM（Narrative Mode） | FANToM（Conversational Mode） |
|---|---|---|
| 信息不对称来源 | Agent 错过了物理状态变化 | Agent 缺席了部分对话 |
| 修正目标变量 | `State.possible_values` | `Observation.possible_values` |
| LLM 检测内容 | fork timestep（状态分叉点） | absent timestep indices（缺席时间步） |
| 修正内容 | State → S_fork（agent 最后观察到的状态） | Observation → 缺席标记 |

两种模式共享统一入口 `apply_pawm()`，通过 `dataset_name` 自动分发。

---

### 2. BigToM false-belief 评估结果

在 48 个已知错误案例（`eval_pawm_48.py`）上评估：

| | 正确数 | 准确率 |
|---|---|---|
| AutoToM baseline | 0 / 48 | 0.0% |
| AutoToM + PAWM | **25 / 48** | **52.1%** |

换算到全量 200 题：

| | 正确数 | 准确率 |
|---|---|---|
| AutoToM baseline | 152 / 200 | 76.0% |
| AutoToM + PAWM | **177 / 200** | **88.5%** |

---

### 3. 剩余 23 道错题的失败原因分析（BigToM bbfb）

| 类型 | 题数 | 根本原因 |
|---|---|---|
| BIP 卡在 50/50（Action 无区分力） | 15 | Agent 的行为在两种信念假设下概率相同，BIP 无法通过行为反推信念 |
| S_fork 时间步选错 | 2 | PAWM 从已含真实状态的 timestep 0 取 S_fork，方向反了 |
| S_fork 方向错误 | 2 | LLM 生成的 S_fork 指向了错误的信念方向 |
| 数字在提取阶段丢失 | 1 | Information Extraction 把 "5 AM" 提取为 "at AM"，具体数值丢失 |
| Perspective header 不够强 | 2 | 故事原文中真实状态描述太显眼，LLM 无法完全忽视 |
| 假设-选项映射 bug | 1 | BIP 内部计算正确但映射到 choices 时顺序反了（AutoToM 本身 bug） |

**最大瓶颈是第1类（15题）**，这是 BIP 算法本身的局限：当 agent 在两种信念下都会采取相同行动时，贝叶斯更新无法区分。

---

### 4. FANToM inaccessible first-order 评估结果

评估范围：30 题（idx 0–29），`eval_fantom_baseline.py`。

| | 正确数 | 准确率 |
|---|---|---|
| AutoToM baseline | 15 / 30 | **50.0%** |
| AutoToM + PAWM | 15 / 30 | **50.0%** |

两者总分相同，但答对/错的题目有差异：PAWM 在 4 道题上退步（idx 8, 9, 16, 21），在 3 道题上进步（idx 20, 25, 27），净效果为零。

#### FANToM 错误类型分析

**类型 1：全知视角污染（Curse of Knowledge）——约 6/15 错误**

```
预测：X believes [对话内容的具体细节]
正确：X is unaware of [对话内容]
```

模型读了完整对话，即使 agent 当时缺席，仍把对话内容赋给 agent 的 belief。
典型案例：idx 14, 15, 22, 25, 28。与 BigToM 的 "perspective header 不够强" 同类，
但在对话型场景里更严重，因为对话内容直接出现在故事文本中，视角隔离更难。

**类型 2：BIP 50/50 死锁——约 5/15 错误**

FANToM 的 `time_variables` 里 action 几乎全是 `NONE`，缺乏区分信念的行为信号。
BIP 从第 0 步到最后一步概率始终是 `[0.5, 0.5]`，最终靠随机打破平局。
典型案例：idx 2（Veronica，28 个 timestep 全部 50/50）、idx 8, 9（Silas，5 个 timestep 全部 50/50）。

**类型 3：信息提取/解析崩溃——约 4/30（13%）**

选项文本含 apostrophe 或过长对话字符串，导致 `eval()` 解析失败。
idx 13（空概率列表）、idx 19、idx 27（baseline）、idx 29 属此类。
AutoToM 原始代码 bug，与 PAWM 无关。

#### PAWM 对 FANToM 失效的根本原因

**结构性错配**：PAWM 的 `_apply_conv_pawm()` 预设信息不对称体现在 `Observation` 变量，
但 AutoToM 对对话型数据的信息提取将对话内容编码进 `State` 变量，`Observation` 变量极少出现：

```
[PAWM-Conv] Absent timestep indices: [0, 1, 2, 3, 4]
[PAWM-Conv] Timestep 0: no Observation variable found, skipping.
...
[PAWM-Conv] No Observation variables were changed.
```

每道题都是此结果——PAWM 的缺席检测逻辑正确，但修正目标不存在，实际上是 **no-op**。

**Perspective Header 副作用**：PAWM 唯一起效的部分是添加 perspective header。
这在 BigToM 有效，但在 FANToM 有时干扰了 BIP 对 State 链的正常推断，
导致 idx 8, 9, 16, 21 四道 baseline 原本答对的题被 PAWM 改错。

**BigToM vs FANToM 对比**：

| | BigToM bbfb | FANToM inaccessible |
|---|---|---|
| 信息不对称类型 | 物理状态变化（agent 缺席时物体移动）| 对话内容（agent 缺席时发生的讨论）|
| `time_variables` 主要变量 | State + Observation 均有 | 主要是 State，Observation 极少 |
| PAWM 修正目标 | State（S_fork）→ 有效 | Observation → 目标变量不存在 |
| BIP 的 action 信号 | 有（agent 去哪里找物体）| 无（action 基本全为 NONE）|
| BIP 50/50 比例 | 少数题 | 多数题 |

---

### 5. 理论定位（Gweon 2021）

Gweon (2021) *"Inferential Social Learning"* (Trends in Cognitive Sciences) 指出：

> "the traditional notion of belief–desire psychology is insufficient to explain how humans make sense of others' behaviors... humans assume other agents are **utility-maximizers**, and interpret others' actions in terms of their underlying **utilities** (costs and rewards)."

论文进一步区分三种 action 的**信息生成方式**（sampling process）：

| Action 类型 | 含义 | 对信念的诊断力 |
|---|---|---|
| Incidental | 偶然发生，不反映意图 | 极低 |
| Instrumental | 为达成目标而做 | 中等（但可能与 belief 无关）|
| Communicative/Pedagogical | 特意展示以传递信息 | 高，可做 closure inference |

PAWM 的设计与此一致：通过修正 agent 视角下的 world model，使 BIP 在有限信息视角下推断信念。
但当前 PAWM 尚未处理 action 类型对推断权重的影响，是主要瓶颈所在。

---

## 未来工作：Goal Correction + Action Intent Prior 组合方案

### 动机

BigToM bbfb 15 道 50/50 错题的根本问题有两层：

1. **Goal 未修正**：PAWM 修正了 `State = S_fork`（agent 最后看到的状态），
   但 `Goal` 变量仍从真实 State 派生，而非从 S_fork 派生。
   这导致 BIP 在 "false belief" 假设下使用了基于真实状态的目标，
   使得 action 在两种假设下的概率接近相同。

2. **Incidental action 噪声未过滤**：即使 Goal 被正确修正，
   如果 action 与 belief-relevant object 无关（如"开始编织"、"预热烤箱"），
   BIP 仍会把这个无意义的 action 当作等权重的证据来源，
   稀释 PAWM 修正好的 State/Observation 信号。

### 方案：两步组合修正

#### Step 1：Goal Correction（PAWM 预处理阶段新增）

在现有 `_apply_narrative_pawm()` 完成 S_fork 替换后，追加一次 LLM 调用：

```python
# 已有：
time_variables[fork_t]["State"].possible_values = [s_fork_value]

# 新增：基于 S_fork 派生 false belief 下的 goal
goal_under_false_belief = llm_call(f"""
  Story context: {story}
  Agent: {inf_agent_name}
  Agent's believed state (what they last saw): {s_fork_value}
  
  Given this belief, what goal would {inf_agent_name} be pursuing?
  Describe the goal in one sentence.
""")

# 替换 fork 之后所有 timestep 的 Goal 变量
for t in range(fork_t, len(time_variables)):
    if "Goal" in time_variables[t]:
        time_variables[t]["Goal"].possible_values = [goal_under_false_belief]
```

**原理**：BIP 的推断链是 `State → Belief → Goal → Action`。
PAWM 修正了 State，Goal Correction 进一步确保 Goal 也与 false belief 一致，
使 `P(action | goal_false, belief_false)` 与 `P(action | goal_true, belief_true)` 产生差异。

**适用条件**：action 与 belief-relevant object 存在关联时有效（如 "goes to find X"）。
对于 action 完全无关的情况（"starts knitting"），Goal Correction 单独无法帮助。

#### Step 2：Action Intent Prior（BIP 运行前的权重调整）

在 Goal Correction 完成后，基于修正后的 Goal 重新评估 action 类型，
并在 `probs.py` 的 action utility 计算中乘以对应权重：

```python
# 在 PAWM 预处理阶段（BIP 运行前）
# 注意：需在 Goal Correction 之后运行，以便用修正后的 goal 来判断 action 类型

action_type = llm_call(f"""
  Agent: {inf_agent_name}
  Agent's goal (given false belief): {goal_under_false_belief}
  Observed action: {action}
  
  Classify this action:
  - "instrumental": action is directly driven by the goal above
  - "incidental": action is unrelated to the goal above
  - "communicative": action is intended to convey information
  
  Output only one word.
""")

action_weight = {
    "instrumental":   1.0,
    "incidental":     0.1,   # 大幅压低，减少噪声
    "communicative":  2.0,
}[action_type]

# BIP 里 action utility 乘以此权重（修改 probs.py 第137-148行）
utility_action *= action_weight
```

**原理（来自 Gweon 2021）**：人类解读他人行为时不只看"这个 action 有多可能"，
而是推断"agent 为什么选择展示这个 action，而非其他 action"（sampling process awareness）。
Incidental action 对信念的诊断力接近零；将其权重压低后，
BIP 主要依赖 State/Observation chain（已由 PAWM 修正），正确信号得以传播。

**关键**：Action Intent Prior 必须在 Goal Correction **之后**运行。
Goal Correction 可能使某个原本看似 incidental 的 action 变为与修正后的 goal 相关（instrumental），
若先压低权重再修正 goal，会错误地丢弃本可利用的信号。

### 两种方法的适用范围与协同

| 场景 | Goal Correction 单独 | Action Intent Prior 单独 | 组合 |
|---|---|---|---|
| action 与 belief object 相关 | 有效 | 可能误压低 | 有效（Step 2 识别为 instrumental，不压低）|
| action 与 belief object 无关 | 无效 | 有效（降噪，让 PAWM 的 State 修正传播）| 有效 |
| action = NONE | 无效 | 无效 | 无效（无 action 可操作）|
| FANToM（action 基本全 NONE）| 无效 | 无效 | 无效（需先解决 Observation 变量缺失问题）|

### 需要改动的文件

- `model/pawm.py`：在 `_apply_narrative_pawm()` 中，S_fork 替换后追加 Goal Correction 逻辑
- `model/probs.py`：在 Action utility 计算的 else 分支（约第137行）中，
  乘以来自 PAWM 预处理阶段传入的 `action_weight`
- `model/ProbSolver.py`：在调用 BIP 前，将 `action_weight` 作为参数传入

### 预期收益（BigToM bbfb 48道错题）

| 方案 | 预期额外修复题数 | 说明 |
|---|---|---|
| 仅 Goal Correction | ~4–6 | 仅对 action-goal 相关的题有效 |
| 仅 Action Intent Prior | ~8–10 | 适用范围更广，但不改变 BIP 输入 |
| 两者组合 | ~10–13 | Goal Correction 修正输入 + Action Intent Prior 过滤噪声 |

风险：若 fork 检测在 true-belief 题上误报，Goal Correction 会错误替换 Goal，
比单独 State 修正影响更大（两个变量同时出错）。实现后需在 true-belief 题上做消融实验验证。

### 参考

- Gweon, H. (2021). Inferential social learning: cognitive foundations of human social learning and teaching. *Trends in Cognitive Sciences*. https://doi.org/10.1016/j.tics.2021.07.008
- Jara-Ettinger, J., Gweon, H., Schulz, L. E., & Tenenbaum, J. B. (2016). The naïve utility calculus: computational principles underlying commonsense psychology. *Trends in Cognitive Sciences, 20*(10), 589–604.
- AutoToM: Ying, Z., et al. (2025). AutoToM: Automated Bayesian Inverse Planning and Theory of Mind. *NeurIPS 2025*.
