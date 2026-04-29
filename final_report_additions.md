# Final-Report Additions — Response to Presentation Feedback

> *Instructor feedback:*
> *"Discuss how to make your approach more general. Discuss failure modes with examples. You may also discuss how to go beyond text modality."*

The three sections below are written to be inserted into the final report after the Results section and before the Conclusion.

---

## 6. Generalizing PAWM Beyond BigToM and FANToM

PAWM in this work is implemented as two concrete operators — a **fork-detector** for BigToM and an **agent-present-window filter** for FANToM. While these two operators look mechanically different, both are special cases of a single mathematical object that we believe generalizes well beyond our two benchmarks.

### 6.1 The Unifying Abstraction: a Perspective Operator Φᵢ

For each focal agent *i*, define a **perspective operator**

> Φᵢ : 𝒲 ⟶ 𝒲ᵢ

that maps the full world history 𝒲 to the *agent-i-accessible* sub-history 𝒲ᵢ. The Bayesian belief that AutoToM should compute is then

> Bᵢ(s) ∝ P( s | Φᵢ(𝒲) )

instead of

> Bᵢ(s) ∝ P( s | 𝒲 ),  *(what AutoToM currently does on false-belief inputs)*

Under this view, the two PAWM modes we implemented are simply two instantiations of Φᵢ:

| Domain      | World 𝒲                  | Φᵢ                                                |
|-------------|--------------------------|---------------------------------------------------|
| BigToM      | Sequence of state events | Drop events after the agent's *fork timestep*     |
| FANToM      | Sequence of dialogue turns | Keep only turns inside the agent-present window |

This abstraction makes the path to generality concrete: *any new domain only needs a domain-specific Φ, while the BIP machinery downstream remains untouched.*

### 6.2 Concrete Generalizations Enabled by Φᵢ

1. **Higher-order Theory of Mind.** Second-order belief reasoning becomes a composition of operators: Bᵢⱼ(s) ∝ P( s | Φⱼ(Φᵢ(𝒲)) ). The current BIP code already supports nested agents; PAWM only needs to apply the appropriate Φ at each nesting level.

2. **Heterogeneous multi-agent settings.** Each agent gets its own Φᵢ, so the same world is filtered differently for different observers. This is exactly what is needed for benchmarks like ToMi (where multiple agents move in/out of a room) and SocialIQa (each speaker has different background information).

3. **Probabilistic perspective.** Φᵢ does not have to be deterministic. A relaxed Φᵢ can return a *weighted* sub-history — useful when an agent might have *partially* heard a turn (overhearing through a wall, noisy channel). This is straightforward to implement: replace the binary `keep[t]` mask in our FANToM filter with a continuous accessibility weight αᵢ(t) ∈ [0, 1], then propagate that weight into the likelihood term.

4. **Continuous-time / streaming settings.** The current Φᵢ assumes discrete timesteps. Streaming applications (live transcription, ongoing video) require a Φᵢ that updates incrementally. The first-utterance / departure-marker rules we use for FANToM extend naturally to a sliding window over a live conversation.

5. **Domain-agnostic perspective extraction.** A single LLM/VLM call can be used to *learn* Φᵢ from raw input by asking "which of these events could agent i have perceived?". We already prototype this in our BigToM mode (one LLM call detects the fork timestep). The same recipe transfers to any new modality where the perspective question is well-posed.

The key claim: PAWM's contribution is not the two specific operators we wrote, but the **architectural pattern** — a perspective layer between the world and the BIP — that lets a single inference algorithm serve many ToM domains.

---

## 7. Failure Modes with Examples

Despite the +10 pp gains on both benchmarks, our system still fails on a non-trivial fraction of items. We categorized every remaining error from our most recent FANToM run into four structural failure modes. Each mode is illustrated with a concrete case below; full counts are reported at the end.

### 7.1 Mode A — Filter-correct, BIP-limited *(most common, 8/12 errors)*

The perspective operator Φᵢ correctly removes the absent-time content, but BIP still predicts "believes X" instead of "unaware of X". This happens when the focal agent's *post-arrival* utterances are short, generic, or unrelated to the missed topic, so the Utterance → Belief likelihood cannot discriminate between the two competing belief hypotheses.

> **Q29 — Amber / "Settlers of Catan"**
> *Filter:* removes 23 / 40 turns covering Bethany & Lyric's earlier discussion of board games. ✓
> *BIP outcome:* the two belief hypotheses — "Amber believes Lyric prefers Catan" and "Amber is unaware" — both produce a 0.5 likelihood for Amber's only post-arrival utterance ("I totally agree…"). The tie defaults to the wrong answer.

This is not a preprocessing failure; it is a *likelihood under-determination* problem. We discuss the proposed epistemic-prior fix in §8.

### 7.2 Mode B — Multi-agent reference ambiguity *(2/12 errors)*

When the question references several agents ("What does *Joaquin* believe about who raised *Alayna*?"), AutoToM's `find_inferred_agent` LLM call can latch onto the wrong subject. We mitigated this with a regex-based focal-agent override, but the override only works when the question follows a "What does X verb …" pattern.

> **Q11 — Joaquin / Alayna's upbringing**
> The question contains both names. Without the override, BIP infers *Alayna's* belief about her own upbringing, which makes the "knows" hypothesis trivially true.
> *Why it still fails sometimes:* even after the override fires, the filter then keeps Joaquin's full conversation, and Alayna's first-person narrative still appears in turns Joaquin witnessed — so the filter cannot remove the misleading content.

Generalization: the perspective operator should be *content-aware* in addition to time-aware — i.e., remove turns whose *topic* the focal agent is being asked about, even when the agent was nominally present.

### 7.3 Mode C — Parser / extraction failures *(2/12 errors)*

Empty predictions on Q13 (Titus) and Q19 (Lisa) trace back to AutoToM's upstream `parse_story_and_question` returning malformed structures (e.g. an unterminated string after an apostrophe in a name). PAWM never gets a chance to run.

> **Q19 — "Russell's favorite books and *who* the *author* is"**
> The choices contain nested apostrophes and quotes; AutoToM's prompt-as-code construction triggers a Python `SyntaxError` on the choice string and the pipeline aborts.

These are infrastructure bugs orthogonal to PAWM. We list them so future work knows the achievable ceiling on FANToM is bounded by the parser, not by the BIP itself.

### 7.4 Mode D — Filter false-positive *(0 / 12 in the current run, but worth flagging)*

Our deterministic departure-marker regex (`brb`, `catch you later`, etc.) could in principle remove turns that the agent actually heard — e.g., when an agent says "BRB" but immediately resumes speaking. Empirically this never fired on the 30 evaluated items, but a stress test on the full FANToM split is needed before deploying PAWM at scale.

### 7.5 Summary of failure-mode counts (FANToM-1st inaccessible, n = 30)

| Mode                                  | Count | Solvable by extending PAWM? |
|---------------------------------------|-------|------------------------------|
| A. Filter-correct, BIP-limited        | 8     | Requires Belief-prior change (§8) |
| B. Multi-agent reference ambiguity    | 2     | Yes — content-aware Φᵢ        |
| C. Parser / extraction failure        | 2     | No — orthogonal infra bug     |
| D. Filter false-positive              | 0     | Already deterministic          |
| **Total errors**                       | **12** |                              |
| Correct                                | 18    |                              |

---

## 8. Beyond Text: Multimodal Perspective Operators

A natural question is whether the PAWM principle extends to non-text modalities. We argue that the abstraction in §6 — "everything that changes between domains is the perspective operator Φᵢ" — makes multimodal extension a clean engineering problem rather than a research re-derivation. We sketch three concrete directions.

### 8.1 Visual Modality — Line-of-Sight and Occlusion

In an embodied / scene-based setting, an agent's perspective is constrained by **line-of-sight** and **occlusion** rather than by who is talking. Φᵢ is then a *visibility mask* over a scene graph:

> Φᵢ(𝒲, t) = { (object, state) : object is visible to agent *i* at time *t* }

A vision-language model can compute this mask from a single frame — e.g., GPT-4V or Qwen-VL prompted with "Which objects in this image is agent A able to see?". Existing benchmarks that fit this pattern:

- **AI2-THOR / RoboTHOR** — agents move between rooms; objects move while the focal agent is elsewhere.
- **MMToM-QA** — multimodal ToM benchmark with video + text scene descriptions; PAWM would replace its current narrative-only handling with a visibility-aware Φ.
- **Social-IQ-2.0** — multi-party video; "who saw what" can be derived from gaze direction and head pose.

The BIP machinery downstream needs no change: the visibility mask is just a different way of producing 𝒲ᵢ.

### 8.2 Audio Modality — Hearing Range and Channel Noise

For multi-room or telephony settings, Φᵢ becomes a **hearing model**: agent *i* hears utterance *u* with probability proportional to source-receiver distance and ambient noise. This is the *probabilistic Φ* mentioned in §6.2.3 — every utterance gets an audibility weight αᵢ(t) ∈ [0, 1] that flows into the likelihood term:

> P(uᵢ_t | Bᵢ) ⟶ αᵢ(t) · P(uᵢ_t | Bᵢ)  +  (1 − αᵢ(t)) · P(uᵢ_t | nothing-heard)

A simple acoustic prior or a learned audio-VLM (e.g., Qwen-Audio) can supply αᵢ.

### 8.3 Cross-modal Asymmetry — The Hardest and Most Realistic Case

Real social settings rarely lose information cleanly along one modality. An agent might **see** an event but not **hear** the explanation that accompanies it (or vice-versa). Φᵢ then becomes a tuple of modality-specific masks — Φᵢᵛⁱˢ, Φᵢᵃᵘᵈ, Φᵢˡⁱⁿᵍ — and BIP must reason over **partial cross-modal evidence**.

> **Worked example:** Sally watches Anne move the marble (visual: ✓) but is wearing headphones and does not hear Anne narrate "I'll move it to the box for safekeeping" (audio: ✗). Sally knows *where* the marble is but does not know *why* — her belief state factorizes across modalities.

This is exactly the kind of structured social inference that the course argues humans perform routinely. Operationalizing it requires:

1. A modality-aware variable layout in BIP (today, AutoToM has a single Observation channel).
2. Per-modality Φ operators that PAWM can plug in independently.
3. Multimodal evaluation benchmarks — MMToM-QA is a starting point; a proper cross-modal false-belief benchmark does not yet exist and is in itself a research contribution we plan to scope.

### 8.4 Why This Matters for the PAWM Story

The same architectural pattern — *correct what BIP gets to see, leave the inference algorithm alone* — is what makes the multimodal extension cheap. We do not need a new probabilistic graphical model for vision and another for audio; we need different perspective operators feeding the same BIP. This is, we believe, the most useful methodological takeaway from this project beyond the two benchmark numbers we report.

---

## 9. Revised Conclusion (drop-in replacement for §五)

PAWM started as a fix for one observed AutoToM failure on BigToM and grew, through this work, into a unifying architectural pattern: **information asymmetry should be enforced by a perspective operator Φᵢ that filters the world before BIP runs**, not by surgery inside the BIP itself. We instantiated Φᵢ in two domains — narrative (BigToM) and conversation (FANToM) — and gained +10 pp in each. We also identified a structural ceiling that no preprocessor can break: when the post-arrival utterances are uninformative, the Utterance → Belief likelihood ties at 0.5 and our filter cannot disambiguate. We outlined three avenues for future work — higher-order ToM via operator composition, probabilistic / multimodal Φ, and an epistemic prior on the Belief variable to break Mode-A ties — and argued that all three follow from the same abstraction without changing the underlying BIP.
