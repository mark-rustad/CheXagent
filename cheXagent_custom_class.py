#!/usr/bin/env python3
"""CheXagent script"""

# ###### Imports and definitions

import csv
import json
import os
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from rich import print
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


class CheXpertDataset(Dataset):
    def __init__(
        self,
        images,
        finding_labels,
        prompt_keys,
        prompts_dict,
        processor_pretrained_model="StanfordAIMI/CheXagent-8b",
    ):
        self.images = images
        self.finding_labels = finding_labels
        self.prompt_keys = prompt_keys
        self.prompts_dict = prompts_dict
        self.processor = AutoProcessor.from_pretrained(
            processor_pretrained_model, trust_remote_code=True
        )

    def __len__(self):
        return len(self.finding_labels)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        prompt_key = self.prompt_keys[index]
        prompt = self.prompts_dict[prompt_key]
        finding_label = self.finding_labels[index]
        inputs = self.processor(
            images=[image], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Adjust as necessary

        return inputs, image_path, prompt_key, finding_label


# ### Prepare input data for analysis
#
# ###### NIH Chest X-ray dataset
#

os.chdir(os.path.dirname(__file__))

column_names = ["image_index", "finding_labels", "follow_up_number", "patient_id", "patient_age", "patient_gender", "view_position", "original_image_width", "original_image_height", "original_image_pixel_spacing_x", "original_image_pixel_spacing_y"]  # fmt: skip # nopep8

data = pd.read_csv(
    "./data/NIH_Chest_X-ray_Dataset/Data_Entry_2017.csv",
    names=column_names,
    header=0,
    index_col=False,
)

# #### IRF-relevant cases
#
# ###### Prompt dictionary
#

prompt_file_list = Path("data").glob("prompts_ref*")
# prompt_file_list = Path("data").glob("prompts.json")
prompts = {}
for prompt_file in prompt_file_list:
    with open(prompt_file.as_posix(), "r") as json_file:
        prompt_i = json.load(json_file)
        prompts.update(prompt_i)

# ###### NIH Chest X-ray dataset: define subset to analyze
#

labels = (
    "Atelectasis",  # relevant to IRF work
    # "Cardiomegaly",
    "Consolidation",  # relevant to IRF work
    # "Edema",
    "Effusion",  # relevant to IRF work
    # "Emphysema",
    # "Fibrosis",
    # "Hernia",
    "Infiltration",  # relevant to IRF work
    # "Mass",
    # "Nodule",
    # "Pleural_Thickening",
    "Pneumonia",  # relevant to IRF work
    # "Pneumothorax",
    "No Finding",  # relevant to IRF work
)
labels = "|".join(labels)

labels_ignore = (
    # "Atelectasis", # relevant to IRF work
    "Cardiomegaly",
    # "Consolidation", # relevant to IRF work
    "Edema",
    # "Effusion", # relevant to IRF work
    "Emphysema",
    "Fibrosis",
    "Hernia",
    # "Infiltration", # relevant to IRF work
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    # "Pneumonia", # relevant to IRF work
    "Pneumothorax",
    # "No Finding", # relevant to IRF work
)
labels_ignore = "|".join(labels_ignore)

nih_cases_to_analyze = data.loc[
    data["finding_labels"].str.contains(labels)
    & ~data["finding_labels"].str.contains(labels_ignore),
    "image_index",
].to_numpy()

pairs = list(product(nih_cases_to_analyze, prompts.keys()))
df_inputs = pd.DataFrame(pairs, columns=["image_index", "prompt_key"])
df_inputs["finding_labels"] = df_inputs["image_index"].map(
    data.set_index("image_index")["finding_labels"].to_dict()
)

# ###### Load analyzed image/prompt combinations
#

filelist = [x.as_posix() for x in Path("output").rglob("disease_classification_QA*.csv")]
# filelist = [x.as_posix() for x in Path("output").rglob("CheXagent_results_on_NIH_CXR.csv.gz")]

if filelist == []:
    df_results_prev = pd.DataFrame(
        columns=["image_index", "finding_labels", "prompt_key", "response"]
    )
else:
    df_results_prev = pd.DataFrame()
    for f in filelist:
        df_i = pd.read_csv(
            f, usecols=["image_index", "finding_labels", "prompt_key", "response"], dtype=str
        )
        df_results_prev = pd.concat([df_results_prev, df_i])

    df_results_prev = df_results_prev.drop_duplicates(ignore_index=True)

# ###### Input image/prompt combinations that have not been analyzed
#

df_not_analyzed = pd.merge(left=df_inputs, right=df_results_prev, how="left").pipe(
    lambda x: x.loc[x["response"].isna(), ["image_index", "finding_labels", "prompt_key"]]
)

# ###### sort the input dataframe
#

# show least analyzed finding_label first
count_map = df_results_prev["finding_labels"].value_counts().to_dict()
df_not_analyzed["label_count"] = df_not_analyzed["finding_labels"].map(count_map)

# create int dtype of prompt key column
df_not_analyzed["prompt_key_int"] = df_not_analyzed["prompt_key"].astype(int)

df_not_analyzed = df_not_analyzed.sort_values(
    ["label_count", "image_index", "prompt_key_int"], ignore_index=True
).drop(columns=["label_count", "prompt_key_int"])
# ###### add image filepaths
#

# import filepaths
df_full_paths = (
    pd.read_csv("data/NIH_Chest_X-ray_image_filepaths.csv")
    .set_index("image_index")["image_path"]
    .to_dict()
)

df_not_analyzed["image_path"] = df_not_analyzed["image_index"].map(df_full_paths)

# ###### delete unused objects
#

del (
    column_names,
    count_map,
    data,
    df_full_paths,
    df_i,
    df_inputs,
    df_results_prev,
    f,
    filelist,
    json_file,
    labels,
    labels_ignore,
    nih_cases_to_analyze,
    pairs,
    prompt_file,
    prompt_file_list,
    prompt_i,
)

# ###### define filepath for results
#

csv_file_parent = Path("output/NIH_Chest_Xray_IRFrelevant_findings")

# ### Using CheXagent
#
# ###### initialize `CheXpertDataset`
#

dataset = CheXpertDataset(
    df_not_analyzed["image_path"].values,
    df_not_analyzed["finding_labels"].values,
    df_not_analyzed["prompt_key"].values,
    prompts,
)
# keys=[key for key in prompts.keys()]
# dataset = CheXpertDataset(
#     ["data/DRR.jpg"] * len(keys),
#     ["No Finding"] * len(keys),
#     [key for key in prompts.keys()],
#     prompts,
# )
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size set to 1 for simplicity
processor = dataset.processor

# Load the model and set it to evaluation mode
device = "cuda"
dtype = torch.float16

model_name = "StanfordAIMI/CheXagent-8b"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, trust_remote_code=True
).to(device)
model.eval()

# Load generation config if needed
generation_config = GenerationConfig.from_pretrained(model_name)

# ###### run the analysis
#

now = datetime.now()
datetime_str = now.strftime("%Y%m%d_%H%M%S")
csv_file_path = csv_file_parent.joinpath(f"disease_classification_QA_{datetime_str}.csv").as_posix()

print(f"Start datetime: {now}")
print(f"Working directory: {os.getcwd()}")

warnings.filterwarnings("ignore", category=UserWarning)

with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_index", "finding_labels", "prompt_key", "response"])

    for batch in tqdm(data_loader, total=len(data_loader), desc="Processing images"):
        inputs = batch[0]
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        image_path, prompt_key, finding_label = [x[0] for x in batch[1:]]
        image_index = Path(image_path).name

        # Generate text; adjust depending on your model's API
        output = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=processor.tokenizer.eos_token_id,
        )[0]

        # Decode and print the generated text
        generated_text = processor.tokenizer.decode(output, skip_special_tokens=True)

        row_data = [image_index, finding_label, prompt_key, generated_text]
        writer.writerow(row_data)

warnings.resetwarnings()
