import csv
import random


def load_full_dataset(dataset_name):
    random.seed(42)

    if "HiToM_len1" in dataset_name:
        order = dataset_name.split('order')[1]
        tell = dataset_name.split('tell')[1][0]
        if tell == "1":
            tell = "tell"
        else:
            tell = "no_tell"
        with open(f"../benchmarks/full_data_formatted/HiToM_len1_{tell}_{order}.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = eval(choices)
            # choices[1] = choices[1][1:]
            random.shuffle(choices)
            # print(choices[0], true_answer)
            answer = 0
            for i, c in enumerate(choices):
                if c == true_answer:
                    answer = i
            # print(choices, answer)
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "MuMaToM_belief":
        with open("../benchmarks/full_data_formatted/MuMA_belief.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        return data

    elif dataset_name == "MuMaToM_social_goal":
        with open("../benchmarks/full_data_formatted/MuMA_social_goal.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        # print(data)
        return data

    elif dataset_name == "MuMaToM_belief_of_goal":
        with open(
            "../benchmarks/full_data_formatted/MuMA_belief_of_goal.csv", "r"
        ) as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        # print(data)
        return data

    if dataset_name == "MMToM-QA":
        with open("../benchmarks/full_data_formatted/MMToM-QA.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices, answer = eval(row[2]), row[3]
            states, actions = eval(row[4]), eval(row[5])

            random.shuffle(choices)
            
            if choices[0] == answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer, states, actions))
        return data

    if dataset_name == "ToMi-1st":
        with open("../benchmarks/full_data_formatted/tomi_first_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return data
    if dataset_name == "ToMi-2nd":
        with open("../benchmarks/full_data_formatted/tomi_second_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return data
    
    if dataset_name == "ToMi-memory":
        with open("../benchmarks/full_data_formatted/tomi_memory.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return data
    
    if dataset_name == "ToMi-reality":
        with open("../benchmarks/full_data_formatted/tomi_reality.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return data

    elif "FANToM" in dataset_name:
        if dataset_name == "FANToM-1st_TB_short":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:accessible_short_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "FANToM-1st_TB_full":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:accessible_full_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "FANToM-1st_FB_short":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:inaccessible_short_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "FANToM-1st_FB_full":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:inaccessible_full_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "FANToM-2nd_TB_short":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:accessible_short_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "FANToM-2nd_TB_full":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:accessible_full_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "FANToM-2nd_FB_short":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:inaccessible_short_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "FANToM-2nd_FB_full":
            with open(
                "../benchmarks/full_data_formatted/FANToM_tom:belief:inaccessible_full_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[3], row[2]

            choices = eval(choices)
            random.shuffle(choices)
            if choices[0] == true_answer:
                answer = "A"
            elif choices[1] == true_answer:
                answer = "B"
            else:
                print("ERROR")
                
            data.append((story, question, choices, answer))
        return data

    elif "BigToM" in dataset_name:
        if dataset_name == "BigToM_fbfb":
            with open(
                "../benchmarks/full_data_formatted/BigToM_forward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_bbfb":
            with open(
                "../benchmarks/full_data_formatted/BigToM_backward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_bbtb":
            with open(
                "../benchmarks/full_data_formatted/bigToM_backward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fafb":
            with open(
                "../benchmarks/full_data_formatted/bigToM_forward_action_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fatb":
            with open(
                "../benchmarks/full_data_formatted/bigToM_forward_action_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fbfb":
            with open(
                "../benchmarks/full_data_formatted/bigToM_forward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fbtb":
            with open(
                "../benchmarks/full_data_formatted/bigToM_forward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[3], row[2]
            if dataset_name == "BigToM":
                choices = choices.replace("[", "").replace("]", "").split(",")
            else:
                choices = choices.replace("[", "").replace("]", "").split(";")
                
            cleaned_choices = []
            for c in choices:
                cleaned_choices.append(c.strip())
            choices = cleaned_choices

            random.shuffle(choices)
            if choices[0].replace(".", "") == true_answer.replace(".", ""):
                answer = "A"
            elif choices[1].replace(".", "") == true_answer.replace(".", ""):
                answer = "B"
            else:
                print("ERROR")

            data.append((story, question, choices, answer))
        return data


def load_dataset(dataset_name):
    random.seed(42)
    if "BigToM" in dataset_name:
        if dataset_name == "BigToM":
            with open("../benchmarks/mini-data/bigToM.csv", "r") as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_bbfb":
            with open(
                "../benchmarks/mini-data/bigToM_backward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_bbtb":
            with open(
                "../benchmarks/mini-data/bigToM_backward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fafb":
            with open(
                "../benchmarks/mini-data/bigToM_forward_action_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fatb":
            with open(
                "../benchmarks/mini-data/bigToM_forward_action_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fbfb":
            with open(
                "../benchmarks/mini-data/bigToM_forward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "BigToM_fbtb":
            with open(
                "../benchmarks/mini-data/bigToM_forward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            if dataset_name == "BigToM":
                choices = choices.replace("[", "").replace("]", "").split(",")
            else:
                choices = choices.replace("[", "").replace("]", "").split(";")
                
            cleaned_choices = []
            for c in choices:
                cleaned_choices.append(c.strip())
            choices = cleaned_choices

            random.shuffle(choices)
            print(choices)
            if choices[0].replace(".", "") == true_answer.replace(".", ""):
                answer = "A"
            elif choices[1].replace(".", "") == true_answer.replace(".", ""):
                answer = "B"
            else:
                print("ERROR")

            data.append((story, question, choices, answer))
        return data

    if dataset_name == "ToMi-1st":
        with open("../benchmarks/mini-data/tomi_1st_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            choices[1] = choices[1][1:]
            random.shuffle(choices)
            # print(choices[0], true_answer)
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
            # print(choices, answer)
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "FANToM-1st":
        with open("../benchmarks/mini-data/FANToM_1st.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[3], row[2]

            choices = eval(choices)
            random.shuffle(choices)
            if choices[0] == true_answer:
                answer = "A"
            elif choices[1] == true_answer:
                answer = "B"
            else:
                print("ERROR")
                
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "MuMaToM_belief":
        with open("../benchmarks/mini-data/MuMa_belief.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        return data

    elif dataset_name == "MuMaToM_social_goal":
        with open("../benchmarks/mini-data/MuMa_social_goal.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
            
        return data

    elif dataset_name == "MuMaToM_belief_of_goal":
        with open("../benchmarks/mini-data/MuMa_belief_of_goal.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
            
        return data

    if dataset_name == "ToMi-2nd":
        with open("../benchmarks/mini-data/tomi_2nd_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            choices[1] = choices[1][1:]
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "MMToM-QA":
        with open("../benchmarks/mini-data/MMToM-QA.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story_and_question, answer = row[0], row[1]
            story = story_and_question.split("Question: ")[0].strip()
            question = (
                story_and_question.split("Question: ")[1].split(" (a)")[0].strip()
            )
            choices = [
                story_and_question.split("(a) ")[1].split(" (b)")[0],
                story_and_question.split("(b) ")[1].split(" Please")[0],
            ]
            answer = answer.capitalize()
            data.append((story, question, choices, answer))
        return data
