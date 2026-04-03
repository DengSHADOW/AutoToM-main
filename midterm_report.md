# Midterm Progress Report
## PAWM: Perspective-Aware World Models for Bayesian Theory of Mind Inference
### Course Project | Spring 2026

---

## 1. Summary of Progress

We have completed the core PAWM system and achieved a significant accuracy improvement on the primary evaluation set (BigToM bbfb). Additionally, we have identified a new evaluation frontier — the FANToM conversational ToM benchmark — where AutoToM currently fails and PAWM's perspective-tracking principle can be extended.

### 1.1 PAWM Implementation and Results (Weeks 1–4, COMPLETE)

**Root Cause Confirmed.** We evaluated AutoToM (GPT-4o) on the full BigToM Backward Belief False-Belief dataset (200 questions), establishing a baseline accuracy of 76.0% (152/200). Manual inspection of all 48 wrong cases confirmed a single, consistent failure pattern: in every case (48/48, 100%), the world state changed while the focal agent was not observing, and AutoToM assigned the true post-change state directly to the agent's belief. Structural analysis of AutoToM's internal Bayesian network variable files revealed the mechanism: the `State` variable is uniformly marked `is_observed=True` with no per-agent observability constraint, meaning omniscient state information contaminates belief inference before BIP begins.

**PAWM Implemented.** We implemented the three-step PAWM pipeline as proposed:

| Step | Description | Implementation |
|------|-------------|----------------|
| Fork Detection | Identify timestep where focal agent stops observing | 1 LLM call per episode; lexical + contextual signals |
| State Correction | Replace `State.possible_values` with S\_fork (agent's last known state) | In-place modification of `time_variables` |
| Observation & Story Correction | Fix Observation variable and prepend perspective header to story text | Covers both BIP inference paths |

The system is implemented as a single module (`pawm.py`, 206 lines) with three hook points in `ProbSolver.py`. No changes were made to AutoToM's core BIP algorithm.

**Primary Results:**

| Setting | Correct / Total | Accuracy |
|---------|----------------|----------|
| AutoToM baseline (our replication) | 152 / 200 | 76.0% |
| **AutoToM + PAWM** | **172 / 200** | **86.0%** |
| **Improvement** | **+20 cases** | **+10.0 pp** |

For reference, the AutoToM paper reports 86.92% on the full BigToM dataset (all subsets combined). PAWM brings the hardest subset (bbfb) to parity with the paper's overall average.

**Cost:** Each episode requires only 1 additional LLM call for fork detection (~$0.001–0.003). For true-belief cases, PAWM detects no fork and passes through with zero overhead.

**Residual Error Analysis.** Of the 48 original error cases, PAWM repaired 20 (net). The 28 remaining failures break down as:
- **Action text leakage** (primary): Story text describes the agent's action using omniscient language (e.g., "unknowingly pours **hot** sauce"). PAWM corrects State and Observation but cannot rewrite the Action variable text, which BIP reads directly.
- **S\_fork estimation failures** (minor): In single-timestep stories, S\_fork is LLM-generated rather than read from a prior timestep, occasionally producing format mismatches.

### 1.2 Cross-Subset Validation

We also evaluated AutoToM baseline on BigToM Forward Belief False-Belief (fbfb, 200 questions):

| Setting | Correct / Total | Accuracy |
|---------|----------------|----------|
| AutoToM baseline (fbfb) | 191 / 200 | 95.5% |

The error profile differs from bbfb: only 3/9 errors are information asymmetry; the remainder involve benchmark ambiguity or observation interpretation errors. This confirms that PAWM's targeted intervention is appropriate — it addresses the dominant failure mode in backward belief while forward belief is already well-handled.

### 1.3 New Finding: AutoToM Fails on Conversational ToM (FANToM)

Beyond the original proposal scope, we investigated AutoToM's performance on FANToM — a multi-party conversational ToM benchmark already included in the AutoToM repository but never successfully evaluated. This investigation yielded an important finding.

**Technical Blockers Fixed.** AutoToM's codebase contained two bugs preventing FANToM evaluation: (1) incorrect filenames in `DataLoader.py` (using `:` instead of `_`), and (2) the nested belief detector (`determine_higher_order_belief`) misclassifying FANToM first-order questions as second-order due to multi-agent phrasing, causing `check_nested` to reject them. We fixed both issues.

**Baseline Results (first 10 questions, FANToM inaccessible first-order):**

| Setting | Correct / Total | Accuracy |
|---------|----------------|----------|
| AutoToM baseline | 6 / 10 | 60.0% |
| Random chance (binary) | — | 50.0% |

AutoToM performs barely above chance. Diagnostic analysis of the verbose output reveals three fundamental mismatches between AutoToM's pipeline and conversational ToM:

1. **State extraction yields NONE.** Conversations have no physical state changes; the `State` variable — designed for "the coral is damaged" — extracts irrelevant information like "Richard is receiving a delivery" when the question asks about Silas's belief about a parenting discussion.

2. **Excessive timesteps.** A 30-turn conversation produces ~28 timesteps. With back-inference iterating from each starting timestep, a single question requires 271+ API calls ($0.40), compared to ~30 calls ($0.04) for BigToM.

3. **Utterance matching is uninformative.** BIP attempts to infer Belief from Utterance, but casual replies ("Will do, Kobe") have no semantic connection to specific topic-level beliefs, producing [0.5, 0.5] at most timesteps.

**Key Insight:** The information asymmetry in FANToM is structurally identical to BigToM — an agent misses information while absent — but the medium is conversational rather than physical. PAWM's core principle (show the model only what the agent could have perceived) applies directly: filter the conversation to exclude turns that occurred during the agent's absence.

---

## 2. Updated Timeline

| Week | Original Plan | Actual Status |
|------|--------------|---------------|
| 1–2 | Error analysis on BigToM bbfb | **COMPLETE** (76.0% baseline; 48/48 = info asymmetry) |
| 3–4 | Implement PAWM; validate on 48 error cases | **COMPLETE** (86.0%; +10 pp; residual errors classified) |
| 5–6 | Full evaluation: bbfb + fbtb + fbfb; ablation | **REVISED** (see below) |
| 7–8 | Write-up and report | **REVISED** (see below) |

**Revised Plan for Weeks 5–8:**

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 5 | Implement Conversation-PAWM for FANToM: detect agent absence periods in conversations; filter story to agent-visible turns only | Working conv-PAWM module; pilot results on 20 FANToM questions |
| 6 | Evaluate on FANToM inaccessible subsets (1st-order full + short context, ~640 questions each); compare baseline vs. conv-PAWM | Accuracy tables; error analysis |
| 7 | Ablation: PAWM-only vs. conv-PAWM-only vs. both; BigToM fbtb control evaluation | Complete comparison tables |
| 8 | Final report: unified narrative (BigToM narrative PAWM + FANToM conversational PAWM) | Technical report |

---

## 3. Roadblocks and Discussion

### 3.1 Remaining Challenge: Action Text Leakage (BigToM)

PAWM corrects State and Observation variables but cannot address cases where the story's Action description itself contains omniscient information. For example, "Farhan unknowingly pours **hot** sauce" — the Action variable extracted by AutoToM includes "hot sauce," which leaks the true state through a pathway PAWM does not modify. Extending PAWM to also rewrite Action text is possible but risks introducing errors through aggressive text modification. We plan to quantify this ceiling effect rather than attempt to fix it.

### 3.2 Key Design Decision: Conversation-PAWM Architecture

For FANToM, we are considering two approaches:

**Approach A: Story-level filtering.** Before any AutoToM processing, identify the focal agent's absence period and remove those conversation turns from the story text entirely. This is simple and ensures all downstream components (State extraction, Utterance matching, Initial Belief estimation) only see agent-accessible information.

**Approach B: Variable-level correction.** Similar to the original PAWM, let AutoToM extract all variables first, then correct them. This preserves more of the original pipeline but is harder to implement for conversations where "State" is already meaningless.

We favor Approach A for its simplicity and because the diagnostic results show that the variable extraction stage itself is the bottleneck — correcting variables after extraction cannot help when the variables were never meaningful.

### 3.3 Cost Concern

AutoToM's per-question cost on FANToM ($0.15–0.40) is 4–10x higher than on BigToM ($0.04), driven by the large number of conversation timesteps. Conversation-PAWM's story filtering should reduce timestep count substantially (from ~28 to ~10–15), which would also reduce cost. We will track cost reduction as a secondary metric.

---

## 4. Next Steps

1. **Implement Conversation-PAWM** (Approach A): a preprocessing module that detects agent absence in multi-party conversations and filters the story to agent-visible turns only.
2. **Establish FANToM baseline** on a larger sample (50–100 questions) to get a stable baseline accuracy number.
3. **Evaluate Conversation-PAWM** on FANToM inaccessible first-order (642 questions) and compare against baseline.
4. **Unified evaluation**: present BigToM PAWM and FANToM Conversation-PAWM as two instantiations of the same perspective-tracking principle applied to different information modalities (narrative vs. conversational).

---

## References

Zhang, Z., Jin, C., Jia, M. Y., Zhang, S., & Shu, T. (2025). AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling. *NeurIPS 2025 Spotlight*. arXiv:2502.15676.

Kim, H., Sclar, M., Zhou, X., Bras, R. L., & Choi, Y. (2023). FANToM: A Benchmark for Stress-Testing Machine Theory of Mind in Interactions. *EMNLP 2023*.

Hou, G., Zhang, W., Shen, Y., Wu, L., & Lu, W. (2024). TimeToM: Temporal Space is the Key to Unlocking the Door of Large Language Models' Theory-of-Mind. *ACL 2024 Findings*.

Wilf, A., Lee, S., Liang, P. P., & Morency, L.-P. (2023). Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities. arXiv:2311.10227.
