import re
import random
import csv
import json
import pandas as pd

def disambiguate_hitom(story):
    story = story.strip()
    # print(story)
    ss = story.split('\n')
    for s in ss:
        if 'entered the ' in s:
            loc = s.split('entered the ')[1][:-1]
            break
    fixed_ss = []
    for i, s in enumerate(ss):
        fixed_ss.append(s)
        if "is in the " in s:
            container = s.split("is in the ")[1][:-1]
            fixed_ss.append(f"The {container} is in the {loc}.")
        elif "to the " in s:
            container = s.split("to the ")[1][:-1]
            fixed_ss.append(f"The {container} is in the {loc}.")

    return '\n'.join(fixed_ss)


with open('../data/Hi-ToM_data.json') as f:
    data = json.load(f)["data"]

entries = {"tell": {0:[], 1:[], 2:[], 3:[], 4:[]}, 
           "no_tell": {0:[], 1:[], 2:[], 3:[], 4:[]},}

for entry in data:
    if entry["prompting_type"] == "CoTP":
        continue
    if entry["story_length"] > 1:
        continue
    story = entry["story"]
    question = entry["question"]
    answer = entry["answer"]
    choices = entry["choices"]
    order = entry["question_order"]
    tell = entry["deception"]

    story = story.split('Directly output the answer without explanation.\n')[1]
    story = disambiguate_hitom(story)
    choices = choices.split('. ')[1:]
    for i, c in enumerate(choices):
        choices[i] = c.split(', ')[0]
        
    record = {
        "story": story,
        "question": question,
        "gt_answer": answer,
        "answer_choices": choices,
    }

    if tell is True:
        entries["tell"][order].append(record)
    else:
        entries["no_tell"][order].append(record)


output_path_base = "../full_data_formatted/"
for label, label_entries in entries.items():
    for order in range(0, 5):
        df = pd.DataFrame(label_entries[order])
        df.to_csv(f"{output_path_base}HiToM_len1_{label}_{order}.csv", index=False)

