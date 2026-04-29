"""
PAWM: Perspective-Aware World Model
A unified preprocessing module that fixes AutoToM's false-belief reasoning failure
across both narrative (BigToM) and conversational (FANToM) benchmarks.

Unified principle:
    Both benchmarks share the same information asymmetry structure — a focal agent
    misses information while absent. PAWM corrects the extracted time_variables to
    only reflect what the focal agent could have perceived, then passes the corrected
    variables to BIP.

Two modes, one interface:
    Narrative mode (BigToM): detects the fork timestep where the agent stops
        observing, then replaces State.possible_values with S_fork (the agent's
        last known state) and prepends a perspective header to the story.
    Conversational mode (FANToM): detects which timestep indices correspond to
        the agent's absence period, overwrites those timesteps' Observation
        variables to mark non-presence, and prepends a perspective header to
        the story.

Design:
    - Single LLM call per episode in both modes (~1 API call overhead)
    - Both modes operate on time_variables directly (variable-level correction)
    - No changes to core BIP algorithm
    - No-op when no absence is detected (true-belief cases pass through unchanged)
    - Dispatcher auto-selects mode from dataset_name
"""

import json
from utils import llm_request


# ---------------------------------------------------------------------------
# Narrative mode (BigToM) — fork detection in physical-state stories
# ---------------------------------------------------------------------------

FORK_DETECTION_PROMPT = """Analyze this story to detect information asymmetry for a specific agent.

Story: {story}

Agent being analyzed: {agent}

The story has been parsed into {num_timesteps} timestep(s). The world state at each timestep:
{state_summary}

The CURRENT (final) state description is:
  {current_state}

Your task:
Determine whether {agent} was absent, distracted, or otherwise unable to observe a state change.

Common signals that {agent} missed a state change:
- A third party changed something while {agent} was not watching
- "suddenly" something changed without {agent} being aware
- "{agent} unknowingly" did something
- "{agent} was preparing / busy / away" when a change happened

If {agent} missed a state change:
1. last_observed_timestep: last timestep index (0-based) at which {agent} was still observing.
   For single-timestep stories, always use 0.
2. s_fork: what the state WOULD HAVE BEEN from {agent}'s perspective (before the missed change).
   CRITICAL: Write s_fork in EXACTLY the same format and style as the current state description above.
   Only change the part that describes the thing that changed. Keep everything else identical.

Example: if current state is "John is in Paris. The vase is broken." and John missed the vase breaking,
then s_fork = "John is in Paris. The vase is intact."

Respond in JSON only (no other text):
{{
    "fork_detected": true or false,
    "reasoning": "one sentence explanation",
    "last_observed_timestep": integer (0 for single-timestep stories),
    "s_fork": "state description in same format as current state (empty string if fork_detected is false)"
}}"""


def _apply_narrative_pawm(time_variables, story, inf_agent_name, llm):
    """
    Narrative PAWM (BigToM): detect fork timestep, correct State variables in-place,
    and return a perspective-filtered story string. Returns False if no fork detected.
    """
    if not time_variables:
        return False

    # Build state summary for the prompt
    state_lines = []
    for i, tv in enumerate(time_variables):
        if "State" in tv:
            state_lines.append(f"  Timestep {i}: {tv['State'].possible_values[0]}")
        else:
            state_lines.append(f"  Timestep {i}: (no state extracted)")

    # Use the last timestep's State as the "current state" reference for format matching
    current_state = time_variables[-1]["State"].possible_values[0] if "State" in time_variables[-1] else "(unknown)"

    prompt = FORK_DETECTION_PROMPT.format(
        story=story,
        agent=inf_agent_name,
        num_timesteps=len(time_variables),
        state_summary="\n".join(state_lines),
        current_state=current_state,
    )

    resp, _ = llm_request(prompt, temperature=0.0, hypo=True, model=llm)

    # Parse JSON response
    try:
        resp_clean = resp.strip()
        # Strip markdown code fences if present
        if "```" in resp_clean:
            parts = resp_clean.split("```")
            for part in parts:
                if "{" in part:
                    resp_clean = part.lstrip("json").strip()
                    break
        result = json.loads(resp_clean)
    except Exception as e:
        print(f"[PAWM] Failed to parse LLM response: {e}\nResponse was: {resp}")
        return False

    if not result.get("fork_detected", False):
        print(f"[PAWM] No fork detected for {inf_agent_name} — passing through unchanged.")
        return False

    last_obs_t = result.get("last_observed_timestep", -1)
    s_fork_from_llm = result.get("s_fork", "").strip()
    reasoning = result.get("reasoning", "")

    print(f"[PAWM] Fork detected! Reasoning: {reasoning}")

    # Determine S_fork: prefer looking it up from time_variables when possible,
    # fall back to the LLM-provided string for single-timestep cases.
    s_fork = None

    if 0 <= last_obs_t < len(time_variables) - 1:
        # Multi-timestep: use the State extracted at the last observed timestep
        if "State" in time_variables[last_obs_t]:
            s_fork = time_variables[last_obs_t]["State"].possible_values[0]
            print(f"[PAWM] S_fork from timestep {last_obs_t}: '{s_fork}'")

    if s_fork is None:
        # Single-timestep or no prior observed timestep: use LLM-provided S_fork
        if s_fork_from_llm:
            s_fork = s_fork_from_llm
            print(f"[PAWM] S_fork from LLM (single-timestep case): '{s_fork}'")
        else:
            print(f"[PAWM] Fork detected but could not determine S_fork — skipping.")
            return False

    # Apply correction: replace State at all post-fork timesteps.
    # Special case: single-timestep stories have no "previous" timestep.
    # The entire timestep 0 contains S_final and must be corrected.
    if len(time_variables) == 1:
        start_t = 0
    else:
        start_t = last_obs_t + 1 if last_obs_t >= 0 else 0
    corrected = 0
    for t in range(start_t, len(time_variables)):
        if "State" in time_variables[t]:
            original = time_variables[t]["State"].possible_values[0]
            if original == s_fork:
                print(f"[PAWM] Timestep {t}: State already matches S_fork, no change needed.")
                continue
            time_variables[t]["State"].possible_values = [s_fork]
            print(f"[PAWM] Timestep {t}: State '{original}' → '{s_fork}'")
            corrected += 1

    if corrected == 0:
        print("[PAWM] No State variables were changed.")
        return False

    print(f"[PAWM] Applied S_fork correction to {corrected} timestep(s).")

    # Fix Observation-path: BIP sometimes infers Belief from Observation (not Initial Belief).
    # If the focal agent's Observation is "unknown" or "none", replace it with a perspective
    # view derived from s_fork so that BIP can distinguish the two belief hypotheses.
    obs_key = f"{inf_agent_name}'s Observation"
    _unknown_markers = ("unknown", "none", "n/a", "not stated", "not clearly")
    obs_corrected = 0
    for t in range(start_t, len(time_variables)):
        if obs_key in time_variables[t]:
            obs_val = time_variables[t][obs_key].possible_values[0]
            if any(m in obs_val.lower() for m in _unknown_markers):
                obs_perspective = f"{inf_agent_name} observes: {s_fork}"
                time_variables[t][obs_key].possible_values = [obs_perspective]
                print(f"[PAWM] Timestep {t}: Observation '{obs_val}' → '{obs_perspective}'")
                obs_corrected += 1
    if obs_corrected:
        print(f"[PAWM] Applied Observation perspective fix to {obs_corrected} timestep(s).")

    # Also build a perspective-corrected story for the Initial Belief code path.
    # BIP's Initial Belief computation uses self.story directly (not the State variable).
    # We PREPEND a strong perspective header so it is the first thing BIP reads,
    # making it harder for the true-state story text to override the agent's belief.
    perspective_header = (
        f"[IMPORTANT — Belief inference for {inf_agent_name}: "
        f"{inf_agent_name} did NOT witness the state change described below. "
        f"From {inf_agent_name}'s perspective the state is: {s_fork}. "
        f"When inferring {inf_agent_name}'s belief, use this perspective only — "
        f"do NOT use omniscient story events {inf_agent_name} could not have observed.]\n"
    )
    corrected_story = perspective_header + story
    print(f"[PAWM] Perspective header prepended to story.")

    return corrected_story


# ---------------------------------------------------------------------------
# Conversational mode (FANToM) — absence detection in multi-party dialogues
# ---------------------------------------------------------------------------

CONV_FORK_DETECTION_PROMPT = """You are analyzing extracted timesteps from a multi-party conversation.

Original conversation:
{story}

Focal agent: {agent}

The conversation has been parsed into {num_timesteps} timestep(s).
The extracted variables at each timestep:
{timestep_summary}

Task: Determine whether {agent} was absent from the conversation at any timestep(s).

Signs of absence:
- The observation variable says {agent} is NOT in the conversation / did not hear the exchange
- The state indicates {agent} has left or has not yet joined
- The story narration says "{agent} left", "stepped away", "joined later", etc.

If {agent} was absent during some timesteps, list the 0-based TIMESTEP indices (matching
the list above) where {agent} could NOT hear the conversation.

Respond in JSON only (no other text):
{{
    "absence_detected": true or false,
    "reasoning": "one sentence explanation",
    "absent_timestep_indices": [list of 0-based integers, empty list if no absence]
}}"""


def _apply_conv_pawm(time_variables, story, inf_agent_name, llm):
    """
    Conversational PAWM (FANToM): detect which extracted timesteps correspond to
    the agent's absence, overwrite those timesteps' Observation variables to mark
    non-presence, and return a perspective-annotated story string.
    Returns False if no absence detected.

    Mirrors _apply_narrative_pawm but for conversational structure:
      BigToM: corrects State variables  (agent missed a physical state change)
      FANToM: corrects Observation vars (agent was absent from the conversation)
    """
    if not time_variables:
        return False

    obs_key = f"{inf_agent_name}'s Observation"

    # Build a timestep summary from time_variables for the LLM prompt
    timestep_lines = []
    for i, tv in enumerate(time_variables):
        state_val = tv["State"].possible_values[0] if "State" in tv else "(no state extracted)"
        obs_val = tv[obs_key].possible_values[0] if obs_key in tv else "(no observation extracted)"
        timestep_lines.append(f"  Timestep {i}: State={state_val} | {obs_key}={obs_val}")

    prompt = CONV_FORK_DETECTION_PROMPT.format(
        story=story,
        agent=inf_agent_name,
        num_timesteps=len(time_variables),
        timestep_summary="\n".join(timestep_lines),
    )

    resp, _ = llm_request(prompt, temperature=0.0, hypo=True, model=llm)

    try:
        resp_clean = resp.strip()
        if "```" in resp_clean:
            parts = resp_clean.split("```")
            for part in parts:
                if "{" in part:
                    resp_clean = part.lstrip("json").strip()
                    break
        result = json.loads(resp_clean)
    except Exception as e:
        print(f"[PAWM-Conv] Failed to parse LLM response: {e}\nResponse was: {resp}")
        return False

    if not result.get("absence_detected", False):
        print(f"[PAWM-Conv] No absence detected for {inf_agent_name} — passing through.")
        return False

    absent_indices = set(result.get("absent_timestep_indices", []))
    reasoning = result.get("reasoning", "")
    print(f"[PAWM-Conv] Absence detected! Reasoning: {reasoning}")
    print(f"[PAWM-Conv] Absent timestep indices: {sorted(absent_indices)}")

    # Correct Observation variables for absent timesteps — parallel to how
    # _apply_narrative_pawm corrects State variables for post-fork timesteps.
    absence_marker = (
        f"{inf_agent_name} was not present in the conversation at this point "
        f"and did not hear these utterances."
    )
    obs_corrected = 0
    for t in sorted(absent_indices):
        if not (0 <= t < len(time_variables)):
            continue
        if obs_key in time_variables[t]:
            original = time_variables[t][obs_key].possible_values[0]
            if original == absence_marker:
                print(f"[PAWM-Conv] Timestep {t}: Observation already marks absence, no change.")
                continue
            time_variables[t][obs_key].possible_values = [absence_marker]
            print(f"[PAWM-Conv] Timestep {t}: Observation '{original[:60]}' → absence marker")
            obs_corrected += 1
        else:
            print(f"[PAWM-Conv] Timestep {t}: no Observation variable found, skipping.")

    if obs_corrected == 0:
        print("[PAWM-Conv] No Observation variables were changed.")
        return False

    print(f"[PAWM-Conv] Applied absence marker to {obs_corrected} timestep(s).")

    # Prepend perspective header to story for the Initial Belief computation path.
    # (BIP uses self.story directly for Initial Belief, so the header steers it
    # to reason from the focal agent's limited vantage point.)
    perspective_header = (
        f"[IMPORTANT — Belief inference for {inf_agent_name}: "
        f"{inf_agent_name} was absent from this conversation during timestep(s) "
        f"{sorted(absent_indices)}. "
        f"Infer {inf_agent_name}'s belief based only on what they could have heard — "
        f"do NOT use information from turns {inf_agent_name} could not have observed.]\n"
    )
    corrected_story = perspective_header + story
    print(f"[PAWM-Conv] Perspective header prepended to story.")

    return corrected_story


# ---------------------------------------------------------------------------
# Story-level filter for FANToM (pre-extraction hook)
# ---------------------------------------------------------------------------

STORY_FILTER_PROMPT = """You are given a multi-party conversation and a focal agent.

Conversation:
{story}

Focal agent: {agent}

Task:
1. Determine whether {agent} left and later rejoined the conversation.
2. If yes, remove from the conversation all turns that occurred WHILE {agent} was absent
   (from the moment they left until the moment they rejoined).
3. Return the filtered conversation — only the turns {agent} could have heard.

Rules:
- Keep all turns BEFORE {agent} leaves and AFTER {agent} returns.
- Remove all turns BETWEEN their departure and return.
- Keep {agent}'s own departure statement if they explicitly said goodbye.
- Keep {agent}'s own return statement when they come back.
- If {agent} never left, return the original conversation unchanged.
- Return ONLY the filtered conversation text, no explanations or extra text.

Filtered conversation:"""


_DEPARTURE_PATTERNS = [
    r"\bcatch you (?:all )?later\b",
    r"\bi'?ll be back\b",
    r"\bi have to (?:go|leave|run|head)\b",
    r"\bi need to (?:go|leave|run|head)\b",
    r"\bgot to (?:go|run)\b",
    r"\bgotta (?:go|run)\b",
    r"\bsee you (?:all |guys )?(?:later|soon)\b",
    r"\btalk to you later\b",
    r"\bbbl\b",
    r"\bbrb\b",
    r"\bexcuse me\b.*\b(?:moment|minute|second)\b",
    r"\bbe right back\b",
    r"\bstep (?:out|away)\b",
]


def _segment_turns(story):
    """Split conversation into (speaker, content, raw_turn) tuples.

    Handles both newline-separated and space-separated formats by splitting
    at every "Name:" or "Name1 & Name2:" token that looks like a speaker tag.
    """
    import re
    # Find start index of every speaker tag
    speaker_re = re.compile(
        r"(?:(?<=\n)|(?<=^)|(?<=[\.\?\!]\s))"
        r"([A-Z][a-zA-Z]+(?:\s*&\s*[A-Z][a-zA-Z]+)?)\s*:",
        re.MULTILINE,
    )
    matches = list(speaker_re.finditer(story))
    if not matches:
        return []
    turns = []
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(story)
        raw = story[start:end].rstrip()
        content = raw[m.end() - m.start():].strip()
        turns.append((speaker, content, raw))
    return turns


def filter_story_for_agent(story, inf_agent_name, llm=None):
    """Mechanical perspective filter for FANToM (pre-extraction).

    Uses deterministic rules instead of an LLM call:
      1. Keep only turns within [first_utterance, last_utterance] of the focal agent.
         Turns before their first utterance = they hadn't arrived yet.
         Turns after their last utterance = they had already left.
      2. Inside that window, if the agent explicitly signals a departure
         ("catch you later", "have to go", etc.) and then speaks again later,
         drop the turns strictly between those two points (treated as absence).

    Returns filtered story string (possibly the full story if the agent is
    present throughout) or the original story if parsing fails.
    """
    import re
    turns = _segment_turns(story)
    if not turns:
        print(f"[PAWM-StoryFilter] No parsed turns, keeping original story.")
        return story

    agent_indices = [
        i for i, (sp, _, _) in enumerate(turns)
        if inf_agent_name in [s.strip() for s in sp.split("&")]
    ]

    if not agent_indices:
        print(f"[PAWM-StoryFilter] Focal agent {inf_agent_name} never speaks — keeping original story.")
        return story

    first_i, last_i = agent_indices[0], agent_indices[-1]

    # Detect inner absence windows from departure markers
    keep = [False] * len(turns)
    for i in range(first_i, last_i + 1):
        keep[i] = True

    # Walk through agent utterances; if one is a departure marker, drop turns
    # until the next agent utterance.
    departure_re = re.compile("|".join(_DEPARTURE_PATTERNS), re.IGNORECASE)
    for idx_pos, i in enumerate(agent_indices):
        _, content, _ = turns[i]
        if departure_re.search(content):
            # find next agent index
            if idx_pos + 1 < len(agent_indices):
                j = agent_indices[idx_pos + 1]
                for k in range(i + 1, j):
                    keep[k] = False

    filtered_lines = [turns[i][2] for i in range(len(turns)) if keep[i]]
    filtered = "\n".join(filtered_lines)

    kept = sum(keep)
    removed = len(turns) - kept
    if removed == 0:
        print(f"[PAWM-StoryFilter] {inf_agent_name} present throughout ({kept} turns) — story unchanged.")
        return story

    print(
        f"[PAWM-StoryFilter] Mechanical filter for {inf_agent_name}: "
        f"kept {kept}/{len(turns)} turns, removed {removed} "
        f"(agent first turn={first_i}, last turn={last_i})."
    )
    return filtered


# ---------------------------------------------------------------------------
# Epistemic prior — Mode A fix for FANToM 0.5/0.5 likelihood ties
# ---------------------------------------------------------------------------
#
# Failure mode A (most common in FANToM-1st inaccessible): the perspective
# filter correctly removes absent-time content, but BIP's Utterance->Belief
# likelihood ties at ~0.5 because the agent's post-arrival utterances do not
# discriminate between "believes <concrete content>" and "is unaware".
#
# The fix: reweight final belief probabilities by an epistemic prior that
# checks whether the choice's content actually appears in the agent's
# perspective-filtered story.
#
#   - "X is unaware of Y" is more plausible if Y's keywords are NOT in the
#     filtered story (the agent literally never heard about Y).
#   - "X believes <specific Y>" is more plausible if Y's keywords ARE in the
#     filtered story.

_UNAWARE_PATTERNS = [
    r"\bis unaware\b",
    r"\bdoes not know\b",
    r"\bdoesn'?t know\b",
    r"\bdid not (?:know|hear|see|witness)\b",
    r"\bdidn'?t (?:know|hear|see|witness)\b",
    r"\bnot involved in the conversation\b",
    r"\bwas not (?:present|involved|aware)\b",
    r"\bhas no (?:knowledge|idea|information)\b",
]

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "doing", "have", "has",
    "had", "having", "this", "that", "these", "those", "it", "its",
    "they", "them", "their", "there", "what", "when", "where", "how",
    "who", "whom", "whose", "which", "why", "about", "his", "her", "him",
    "she", "he", "we", "us", "our", "you", "your", "i", "me", "my",
    "would", "could", "should", "will", "shall", "may", "might", "must",
    "can", "not", "no", "yes", "than", "then", "if", "because",
    "believe", "believes", "believed", "knows", "know", "known",
    "unaware", "involved", "conversation", "discussed", "discussion",
    "group", "people", "person", "thing", "things", "way", "ways",
}


def _content_keywords(text):
    """Extract content keywords (3+ char alphabetic tokens, not stop words)."""
    import re
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text)
    return [t.lower() for t in tokens if t.lower() not in _STOP_WORDS]


def _is_unaware_choice(choice_text):
    import re
    pattern = re.compile("|".join(_UNAWARE_PATTERNS), re.IGNORECASE)
    return bool(pattern.search(choice_text))


def epistemic_prior_reweight(final_probs, choices, filtered_story,
                             tie_threshold=0.05, prior_strength=1.0):
    """Reweight tied belief probabilities by content-presence in filtered story.

    Args:
        final_probs:    list of probabilities (one per choice)
        choices:        list of choice strings (same order as final_probs)
        filtered_story: story after PAWM perspective filter (the agent's view)
        tie_threshold:  apply prior only if max-min prob spread < threshold
        prior_strength: in [0, 1]; 0 = no effect, 1 = fully replace prob

    Returns:
        Reweighted probability list (same order). Returns input unchanged if
        no tie is detected or no content can be scored.
    """
    if not final_probs or len(final_probs) != len(choices):
        return final_probs

    spread = max(final_probs) - min(final_probs)
    if spread >= tie_threshold:
        # BIP was already confident — don't second-guess
        return final_probs

    story_kw = set(_content_keywords(filtered_story))
    if not story_kw:
        return final_probs

    # Identify the believes-vs-unaware pair (only meaningful for binary choices).
    if len(choices) != 2:
        return final_probs

    unaware_idx = None
    for i, c in enumerate(choices):
        if _is_unaware_choice(c):
            unaware_idx = i
            break
    if unaware_idx is None:
        return final_probs
    concrete_idx = 1 - unaware_idx

    # Distinctive keywords = words that appear in the CONCRETE choice but not
    # in the UNAWARE choice. These are the actual content claims being made.
    concrete_kw = set(_content_keywords(choices[concrete_idx]))
    unaware_kw  = set(_content_keywords(choices[unaware_idx]))
    distinctive = concrete_kw - unaware_kw
    if not distinctive:
        return final_probs

    # How many of the distinctive (content) keywords appear in the agent's
    # filtered story?
    presence = sum(1 for k in distinctive if k in story_kw)
    fraction_present = presence / len(distinctive)

    # Build the prior:
    #   - high fraction_present  -> concrete-content belief is supported
    #   - low  fraction_present  -> agent never heard the content, "unaware" wins
    prior = [0.0, 0.0]
    prior[concrete_idx] = fraction_present
    prior[unaware_idx]  = 1.0 - fraction_present

    total = sum(prior)
    if total == 0:
        return final_probs
    prior = [p / total for p in prior]

    # Mix prior with BIP probabilities
    mixed = [
        (1 - prior_strength) * p + prior_strength * pr
        for p, pr in zip(final_probs, prior)
    ]
    z = sum(mixed)
    if z == 0:
        return final_probs
    mixed = [m / z for m in mixed]

    # Telemetry
    print(
        f"[PAWM-EpistemicPrior] tie detected (spread={spread:.2f}); "
        f"prior={[round(x,3) for x in prior]}, "
        f"BIP={[round(x,3) for x in final_probs]} -> mixed={[round(x,3) for x in mixed]}"
    )
    return mixed


# ---------------------------------------------------------------------------
# Unified entry point — auto-dispatches based on dataset_name
# ---------------------------------------------------------------------------

def apply_pawm(time_variables, story, inf_agent_name, llm, dataset_name=""):
    """
    Unified PAWM entry point.

    Dispatches to the appropriate mode based on dataset_name:
      - FANToM datasets  → conversational mode (_apply_conv_pawm)
      - All other datasets → narrative mode (_apply_narrative_pawm)

    Both modes return a perspective-filtered story string on success,
    or False if no information asymmetry is detected.

    Args:
        time_variables: list of dicts (one per timestep) with Variable objects
        story:          raw story/conversation text (str)
        inf_agent_name: focal agent whose belief is being inferred (str)
        llm:            LLM model name string (e.g. "gpt-4o")
        dataset_name:   dataset identifier used to select the mode (str)
    """
    if "FANToM" in dataset_name:
        # Story was already pre-filtered by filter_story_for_agent() before extraction.
        # Just prepend a perspective header to steer Initial Belief computation.
        print(f"[PAWM] Conversational mode — adding perspective header (dataset: {dataset_name})")
        perspective_header = (
            f"[IMPORTANT — Belief inference for {inf_agent_name}: "
            f"the conversation below shows only what {inf_agent_name} could have heard. "
            f"Infer {inf_agent_name}'s belief based solely on this filtered view.]\n"
        )
        return perspective_header + story
    else:
        print(f"[PAWM] Narrative mode (dataset: {dataset_name})")
        return _apply_narrative_pawm(time_variables, story, inf_agent_name, llm)
