import os
import json
import pandas as pd
from datasets import Dataset

path_dev = "datasets/quality/raw/QuALITY.v1.0.1.htmlstripped.dev"
path_train = "datasets/quality/raw/QuALITY.v1.0.1.htmlstripped.train"
# path_test = "dataset/QuALITY.v1.0.1.htmlstripped.test"
paths = [path_dev, path_train]


for path in paths:
    questionlist = []
    # Open the JSON file
    with open(path, "r", encoding="utf8") as file:
        # Load the JSON data
        for line in file:
            line = line.strip()

            json_obj = json.loads(line)

            article = json_obj["article"]

            for jquestion in json_obj["questions"]:
                # getting the relevant properties
                row = {
                    "article": article,
                    "question": jquestion["question"],
                    "options": jquestion["options"],
                    "gold_label": jquestion["gold_label"],
                }
                questionlist.append(row)

    df = pd.DataFrame(questionlist)
    # example of converting it into hugging face dataset
    split = path.split(".")[-1]
    print(split)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(f"datasets/quality/{split}")
