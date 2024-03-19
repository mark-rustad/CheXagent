import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from torchvision import transforms

device = "cuda"
dtype = torch.float16

# Load the processor, model, and generation configuration
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True).to(
    device
)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(
            csv_file,
            usecols=["image_path", "finding_labels"],
            nrows=1e4,
        )
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = f"{self.data_frame.iloc[idx, 1]}"
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(images=images[:2], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(
        device=device, dtype=dtype
    )
    output = model.generate(
        **inputs,
        generation_config=generation_config,
        # this silences "Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation."
        pad_token_id=processor.tokenizer.eos_token_id,
    )[0]
    response = processor.tokenizer.decode(output, skip_special_tokens=True)
    return response


# Initialize the dataset and DataLoader
dataset = ChestXrayDataset(
    csv_file="./output/disease_classification_QA_prompts.csv",
    img_dir="./data/NIH_Chest_X-ray_Dataset/images_001/images",
)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fixed prompt for all images
prompt = """A single finding is present in the given chest X-ray. Identify the finding by selecting one option from the following list:
    A) Atelectasis
    B) Cardiomegaly
    C) Consolidation
    D) Edema
    E) Effusion
    F) Emphysema
    G) Fibrosis
    H) Hernia
    I) Infiltration
    J) Mass
    K) Nodule
    L) Pleural_Thickening
    M) Pneumonia
    N) Pneumothorax
    O) No Finding
Do not select multiple findings."""

# Iterate over the DataLoader
for images in data_loader:
    # Generate responses for each image using the fixed prompt
    responses = [
        generate(image.unsqueeze(0), prompt, processor, model, device, dtype, generation_config) for image in images
    ]
    # Do something with the responses, like printing or logging them
    for response in responses:
        print(response)
