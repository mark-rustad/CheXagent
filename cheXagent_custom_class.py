import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


class CheXpertDataset(Dataset):
    def __init__(self, dataframe, prompts, processor_pretrained_model="StanfordAIMI/CheXagent-8b"):
        self.dataframe = dataframe
        self.prompts = prompts
        self.processor = AutoProcessor.from_pretrained(processor_pretrained_model, trust_remote_code=True)

        # Create a list of tuples, each containing an image path and a prompt
        self.image_prompt_pairs = [
            (row["image_path"], prompt) for _, row in self.dataframe.iterrows() for _, prompt in self.prompts.items()
        ]

    def __len__(self):
        return len(self.image_prompt_pairs)

    def __getitem__(self, index):
        image_path, prompt = self.image_prompt_pairs[index]
        full_image_path = f"{image_path}"
        image = Image.open(full_image_path).convert("RGB")

        inputs = self.processor(images=[image], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Adjust as necessary

        return inputs


column_names = ["image_index", "finding_labels", "follow_up_number", "patient_id", "patient_age", "patient_gender", "view_position", "original_image_width", "original_image_height", "original_image_pixel_spacing_x", "original_image_pixel_spacing_y"]  # fmt: skip # nopep8

data = pd.read_csv(
    "./data/NIH_Chest_X-ray_Dataset/Data_Entry_2017.csv",
    names=column_names,
    header=0,
    index_col=False,
)

# import prompts dictionary
with open("output/prompts.json", "r") as json_file:
    prompts = json.load(json_file)


results_prev = pd.read_csv(
    "output/disease_classification_QA.csv",
    usecols=["image_index", "finding_labels", "prompt_key", "response"],
    dtype=str,
)


# images with missing prompt cases
images_incomplete = (
    results_prev.groupby(["image_index"], as_index=True, sort=False)["prompt_key"]
    .apply(lambda x: x.nunique())
    .pipe(lambda x: x[x != len(prompts)])
    .index.to_list()
)

# images that have been analyed
images_analyzed = set(results_prev["image_index"].values) - set(images_incomplete)

# images to input into CheXagent
images_not_analyzed = set(data["image_index"].values) - images_analyzed
subset = (
    data[data["image_index"].isin(images_not_analyzed) & ((data["finding_labels"].str.count("\\|") + 1) == 1)]
    .sample(5)
    .copy()
)

subset["image_path"] = [
    next(Path("data/NIH_Chest_X-ray_Dataset").rglob(x)).as_posix() for x in subset["image_index"].values
]

print(f"nRows: {subset.shape[0]:,}\tnColumns: {subset.shape[1]}")


# Assuming `dataframe`, `image_folder_path`, and `prompts` are already defined
dataset = CheXpertDataset(subset, prompts)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size set to 1 for simplicity
processor = dataset.processor

# Load the model and set it to evaluation mode
device = "cuda"
dtype = torch.float16

model_name = "StanfordAIMI/CheXagent-8b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).to(device)
model.eval()

# Load generation config if needed
generation_config = GenerationConfig.from_pretrained(model_name)

# Perform text generation
for batch in data_loader:
    inputs = {k: v.to("cuda") for k, v in batch.items()}

    # Generate text; adjust depending on your model's API
    outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode and print the generated text
    generated_text = [processor.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print(generated_text)
