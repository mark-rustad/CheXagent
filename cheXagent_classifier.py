import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm

device = "cuda"
dtype = torch.float16

# Load the processor, model, and generation configuration
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True).to(
    device
)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")

# TODO: fix how inputs are fed to generate() when using batches and DataLoader

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file, nrows=1e4)
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = f"{self.data_frame.iloc[idx, -1]}"
        img_name = f"{self.data_frame.iloc[idx, 0]}"
        label = f"{self.data_frame.iloc[idx, 1]}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name, label


def generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(
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
dataset = ChestXrayDataset(csv_file="./output/single_finding_datalist.csv")
data_loader = DataLoader(dataset, batch_size=1)

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

writer = SummaryWriter("runs/responses_log")
total_responses = 0
for batch in tqdm(data_loader, desc="Processing images"):
    images, img_names, labels = batch
    responses = [
        generate(image.unsqueeze(0), prompt, processor, model, device, dtype, generation_config) for image in images
    ]

    for img_name, label, response in zip(img_names, labels, responses):
        print(f"Image: {img_name}, Label: {label}, Response: {response}")
        writer.add_text("Info", f"Image: {img_name}, Label: {label}, Response: {response}", total_responses)
        total_responses += 1

writer.close()
