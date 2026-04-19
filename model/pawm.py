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
        print(f"[PAWM] Conversational mode (dataset: {dataset_name})")
        return _apply_conv_pawm(time_variables, story, inf_agent_name, llm)
    else:
        print(f"[PAWM] Narrative mode (dataset: {dataset_name})")
        return _apply_narrative_pawm(time_variables, story, inf_agent_name, llm)
