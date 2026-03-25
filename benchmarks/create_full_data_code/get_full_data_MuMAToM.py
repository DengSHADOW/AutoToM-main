import json
import pandas as pd
import random

data_path = "../data/MuMa/questions.json"

try:
    with open(data_path, "r") as file:
        data = json.load(file)
    print("JSON file has been successfully loaded.")
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
except FileNotFoundError:
    print(f"No file found at {data_path}")
except Exception as e:
    print(f"An error occurred: {e}")


import cv2
import os

def extract_frames(video_path, output_dir, num_frames=8):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    interval = total_frames // num_frames
    
    frame_indices = [i * interval for i in range(num_frames)]
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"frame_{idx+1:02d}.png")
            cv2.imwrite(output_path, frame)
    
    cap.release()


entries = {"belief": [], "social_goal": [], "belief_of_goal": []}

if "data" in locals():
    # sampled_entries = random.sample(list(data), 100)
    sampled_entries = list(data)
    for episode_id in sampled_entries:
        entry = data[episode_id]
        description = entry["description"][2:-1].replace("\\n", " ").replace('"', "'")
        if "filtered_description" in entry:
            description = entry["filtered_description"].replace("\\n", " ").replace('"', "'")
        for q_key, question in entry["questions"].items():
            
            question_parts = question.split("\n")
            prompt = question_parts[0].replace('"', "'")
            answer_choices = [choice.replace('"', "'") for choice in question_parts[1:]]

            gt_answer = entry["answers"][q_key].replace('"', "'")

            record = {
                "story": description,
                "question": prompt,
                "answer_choices": answer_choices,
                "gt_answer": gt_answer,
                "video_id": episode_id
            }
            # relabel the question types
            ans = entry["answers"][q_key]

            if "When giving information" in ans:

                if "believed that there" in ans:
                    entries["belief"].append(record)

                else:
                    entries["social_goal"].append(record)

            else:
                entries["belief_of_goal"].append(record)

            # if entry["labels"][q_key] == "belief":
            #     entries["belief"].append(record)
            # elif entry["labels"][q_key] == "social_goal":
            #     entries["social_goal"].append(record)
            # elif entry["labels"][q_key] == "belief_of_goal":
            #     entries["belief_of_goal"].append(record)

output_path_base = "../full_data_formatted/"
for label, label_entries in entries.items():
    df = pd.DataFrame(label_entries)
    df.to_csv(f"{output_path_base}MuMA_{label}.csv", index=False)
    print(f'Length of {label}: {len(label_entries)}')

print("Data has been saved to respective files.")
