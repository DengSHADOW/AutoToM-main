from utils import *
from ElementExtractor import *

def get_nested_states(
    self,
    time_variables,
    orig_time_variables,
    variable_values_with_time,
    orig_variable_values_with_time,
):
    from ProbSolver import ProblemSolver, argmax, argmin
    # time_variables: state, ..., (variables).
    # orig_time_variables: consist of the ground truth states, but the timestamps are not aligned.
    # variable_values with time: values in story of chunks, states, ....
    values = deepcopy(variable_values_with_time)
    gt_values = deepcopy(orig_variable_values_with_time)
    variables = deepcopy(time_variables)
    gt_variables = deepcopy(orig_time_variables)
    altered_variables = deepcopy(time_variables)
    now_init_state = ""

    # j => Pointer for ground truth state timestamps, j >= i
    j = 0
    last_align = -1
    for i, vars in enumerate(variables):
        # Align the timestamps
        while (
            values[i][f"{self.inf_agent_name}'s Action"]
            != gt_values[j][f"{self.inf_agent_name}'s Action"]
        ):
            if self.verbose:
                print(
                    "Aligning: different, ",
                    values[i][f"{self.inf_agent_name}'s Action"],
                    gt_values[j][f"{self.inf_agent_name}'s Action"],
                )
            j += 1
            if j > len(gt_variables) - 1:
                break
        if j > len(gt_variables) - 1:
            j = len(gt_variables) - 1
        altered_variables[i]["Ground Truth State"] = gt_variables[j]["State"]
        if self.verbose:
            enh_print(
                f"Ground Truth State at time {i}: {altered_variables[i]['Ground Truth State']}",
                "yellow",
            )
        # Set the initial state for this story span, picking up from where we aligned last time

        last_state = gt_variables[last_align]["State"].possible_values[0]
        if last_align != -1:
            if f"{self.inf_agent_name}'s Action" in gt_values[last_align]:
                # print(gt_values[last_align][f"{self.inf_agent_name}'s Action"], last_state)
                now_init_state = update_state(
                    last_state,
                    gt_values[last_align][f"{self.inf_agent_name}'s Action"],
                    self.llm,
                    self.verbose,
                    self.dataset_name,
                )
                # print(now_init_state)
        elif j > 0: # has not aligned before but skipped an action of the main agent. We should keep track of the state after the action.
            # For example, B comes in the room second. B does not observe that A entered the room. But B should know that A is in the room. It is an initial state.
            # Specifically, when B comes into the room at time 1 while A comes into the room at time 0, we need to know that before time1, the initial state is like A is in the room.
            if f"{self.inf_agent_name}'s Action" in gt_values[0]:
                now_init_state = update_state(
                    gt_variables[0]["State"].possible_values[0],
                    gt_values[0][f"{self.inf_agent_name}'s Action"],
                    self.llm,
                    self.verbose,
                )
        story_now = now_init_state + " " + values[i]["Chunk"]
        last_align = j
        if self.verbose:
            enh_print(
                f"Story for nested state reasoning step {i}: {story_now}", "yellow"
            )
        states = gt_variables[j]["State"].possible_values[0]
        new_states = ""
        belief_of_states = get_belief_of_states(
            states, self.first_agent_name, self.orig_choices, self.llm, self.relevant_entities
        )
        if self.verbose:
            enh_print(f"Guided beliefs of states: {belief_of_states}", "yellow")
        for s in belief_of_states:
            # get probs
            if i == len(variables) - 1:
                full = True
            else:
                full = False
                
            nested_episode_name = f"{self.episode_name}_nestedChunk_{i}_{s[0]}"
            nested_choices = []
            for _ in s:
                nested_choices.append(f'{self.first_agent_name} believes that {_}')
            nested_question = f"Which one of the following is more likely to be {self.first_agent_name}'s belief?"
            save_ipomdp_intermediate_story(
                story_now, nested_question, nested_choices, self.model_name, nested_episode_name
            )
            prob, _= ProblemSolver(
                story=story_now,
                question=nested_question,
                choices=nested_choices,
                K=self.K,
                assigned_model=["State", "Observation", "Belief"], # Reconstructing Belief of State using sob model
                model_name="sob",
                episode_name=nested_episode_name,
                llm=self.llm,
                hypo_llm=self.hypo_llm,
                verbose=self.verbose,
                dataset_name=self.dataset_name,
                hypo_method=self.hypo_method,
                nested=False,
                tab="  ",
                full=full,
                back_inference=self.back_inference,
                reduce_hypotheses=self.reduce_hypotheses,
                no_model_adjustment=self.no_model_adjustment,
            ).solve()

            if self.verbose:
                enh_print(f"Prob of belief of State: {s}: {prob}")
            print(prob)
            if max(prob) >= 0.6 or (len(prob) > 2 and max(prob) >= 0.5):  # must be sure about one of the beliefs
                new_states += ". " + s[argmax(prob)]
            self.save_nested_results(self, i, s, prob, story_now, nested_question, nested_choices)

        new_states = new_states + ". "
        new_states = new_states[1:].strip()
        if new_states == "":
            new_states = "NONE"
        if self.verbose:
            enh_print(f"nested state reasoning step i: {new_states}", "yellow")
        altered_variables[i]["State"].possible_values = [new_states]
    return altered_variables

def save_nested_results(self, i, s, prob, story_now, nested_question, nested_choices):
    """
    Helper function to save nested reasoning results to CSV.n
    """
    os.makedirs('../results/nested_results', exist_ok=True)
    
    data = {
        'timestamp': i,
        'state': s,
        'probability': prob,
        'story': story_now,
        'question': nested_question,
        'choices': nested_choices
    }
    df = pd.DataFrame([data])
    
    csv_path = f'../results/nested_results/{self.episode_name}_{self.first_agent_name}.csv'
    
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

