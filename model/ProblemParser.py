from utils import *
from ElementExtractor import *

def load_parsed_result_into_self(self, parsed_result):
    needed_list = [
        "story", "question", "choices", "inf_agent_name", "first_agent_name",
        "inf_var_name", "orig_choices", "orig_story", "initial_state", "nested_agents_list",
        "full"
    ]
    for key, value in parsed_result.items():
        if key in needed_list:
            setattr(self, key, value)

def parse_story_and_question(self):
    """
    Extracts and processes information from the story, question and choices.
    Sets up necessary instance variables and saves the extracted information.
    
    Returns:
        info: Dictionary containing the extracted and processed information,
                or None if no model is assigned
    """
    if self.assigned_model == None:
        print("Model Not Assigned")
        return None
    
    # Extract info from question
    new_info = get_info_from_question(self.question, self.llm, self.dataset_name)
    if self.verbose:
        (
            enh_print(f"{self.tab}Info from question: {new_info}")
            if new_info != ""
            else enh_print(self.tab + "No new info from question")
        )

    if "MuMa" in self.dataset_name:
        video_story = video_extracted_actions(self.video_id)
        if "belief_of_goal" in self.dataset_name:
            video_story = correct_visual_actions(video_story, self.choices, self.llm)
        self.story = story_fusion(video_story, self.story, self.llm)
    
    if self.dataset_name in ["MuMaToM_belief", "MuMaToM_social_goal"]:
        self.initial_state = get_initial_state(self.story, self.llm)
        self.story = find_inference_timestep(self.story, self.choices, self.llm)
        if new_info != "":
            new_info = rewrite_belief_info(new_info, self.initial_state, self.llm)

    self.story = get_rid_of_number_starts(self.story)
    self.story += " " + new_info

    if "HiToM" in self.dataset_name:
        self.story = split_sentences(self.story, self.llm)

    if self.nested:
        self.orig_story = deepcopy(self.story)
        self.orig_choices = rephrase_choices(self.question, self.orig_choices, self.hypo_llm)
        self.nested_agents_list = find_nested_agent_list(self.question, self.choices, self.hypo_llm)
        self.first_agent_name = self.nested_agents_list[0]
        self.story, vis = reconstruct_story_nested(
            self.orig_story, self.first_agent_name, self.llm, self.dataset_name
        )
        self.story = " ".join(self.story)
        save_reconstructed_story(vis, self.model_name, self.episode_name, self.first_agent_name)
        
        if self.verbose:
            print("New story:", self.story)

        self.question = rephrase_question_nested(
            self.question, self.nested_agents_list[-1], self.llm, self.dataset_name
        )
        if self.verbose:
            print("New question:", self.question)

    self.choices = rephrase_choices(self.question, self.choices, self.hypo_llm)
    if self.verbose:
        print(self.tab + f"{self.choices}")

    if self.model_name == "automated":
        possible_vars = ["Belief", "Goal"]
        if "BigToM" in self.dataset_name:
            possible_vars.append("Action")
        if "ToMi" in self.dataset_name or "HiToM" in self.dataset_name:
            possible_vars = ["Belief"]
        self.inf_var_name = get_inf_var(
            self.question,
            self.choices,
            possible_vars,
            self.hypo_llm,
            self.dataset_name,
        )
        print("inf_var_name", self.inf_var_name)
    else:
        self.inf_var_name = get_inf_var(
            self.question,
            self.choices,
            self.assigned_model,
            self.hypo_llm,
            self.dataset_name,
        )

    if self.nested == True:
        self.inf_agent_name = self.nested_agents_list[-1]
    else:
        self.inf_agent_name = find_inferred_agent(
            self.question, self.choices, self.hypo_llm
        )

    if self.verbose:
        enh_print(self.tab + f"Inferring: {self.inf_agent_name}'s {self.inf_var_name}", "green")

    needed_list = [
        "story", "question", "choices", "inf_agent_name", "first_agent_name",
        "inf_var_name", "orig_choices", "orig_story", "initial_state", "nested_agents_list",
        "full"
    ]
    info = {k: v for k, v in self.__dict__.items() if k in needed_list}
    save_parsed_result(info, self.model_name, self.episode_name)
    
    return info

