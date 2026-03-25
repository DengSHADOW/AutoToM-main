import re
import random
import csv
import json
import pandas as pd

with open('../data/ToMi/test_balanced.jsonl') as f:
    data = [json.loads(line) for line in f]

entries = {"first_order": [], 
           "second_order": [],
           "reality": [],
           "memory": []}

for entry in data:
    story = entry["story"]
    question = entry["question"]
    answer = entry["answer"]
    choices = ','.join(entry["containers"])
    record = {
        "story": story,
        "question": question,
        "gt_answer": answer,
        "answer_choices": choices,
    }
    
    story_type = entry["story_type"]
    question_type = entry["question_type"]

    if "first_order" in question_type:
        entries["first_order"].append(record)
    if "second_order" in question_type:
        entries["second_order"].append(record)
    if "memory" in question_type:
        entries["memory"].append(record)
    if "reality" in question_type:
        entries["reality"].append(record)


output_path_base = "../full_data_formatted/"
for label, label_entries in entries.items():
    df = pd.DataFrame(label_entries)
    df.to_csv(f"{output_path_base}tomi_{label}.csv", index=False)

