import os
from glob import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, TrainingArguments, Trainer, ViTConfig
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
 
# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# Load the main dataframe
train_df_main = pd.read_csv('train_df.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('images*', '*', '*.png'))}
train_df_main["FilePath"] = train_df_main["Image Index"].map(all_image_paths)
train_df_main.drop(['No Finding'], axis=1, inplace=True)
 
# Filter for only Cardiomegaly and Effusion
selected_labels = ['Cardiomegaly', 'Effusion']
train_df_main = train_df_main[['Image Index', 'FilePath'] + selected_labels]
train_df_main = train_df_main[(train_df_main[selected_labels] == 1).any(axis=1)]
 
# Function to get a subset of the data with specific number of images per label
def create_label_subset(df, labels, num_samples=100):
    subsets = []
    for label in labels:
        label_subset = df[df[label] == 1].sample(n=num_samples, random_state=42)
        subsets.append(label_subset)
    subset_df = pd.concat(subsets).drop_duplicates().reset_index(drop=True)
    return subset_df
 
# Create subset
subset_df = create_label_subset(train_df_main, selected_labels, num_samples=100)
 
# Check if the subset was created correctly
print(subset_df.head())
print(f"Total images in subset: {len(subset_df)}")
 
# Specify the directory where you want to save the file
output_directory = './output_directory'
 
# Ensure the directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
 
# Define the output file path
output_file_path = os.path.join(output_directory, 'subset_df_new.csv')
 
# Save to a new CSV (optional)
try:
    subset_df.to_csv(output_file_path, index=False)
    print(f"Subset saved to {output_file_path}")
except PermissionError as e:
    print(f"PermissionError: {e}. Could not save the file at {output_file_path}")
 
# Define custom dataset for ViT
class CustomImageDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform
        self.image_paths = df['FilePath'].values
        self.label_values = df[labels].values
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.label_values[idx], dtype=torch.float32)
        return {"pixel_values": image, "labels": label}
 
# Custom transform function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# Create dataset and dataloaders
train_dataset = CustomImageDataset(subset_df, selected_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
 
# Initialize the model from scratch
config = ViTConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    image_size=224,
    patch_size=16,
    num_labels=len(selected_labels)
)
model = ViTForImageClassification(config).to(device)
 
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=10,
    eval_steps=10,
)
 
# Define Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        outputs = model(**inputs)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
 
    def compute_metrics(self, p):
        preds = torch.sigmoid(p.predictions).cpu().numpy()
        labels = p.label_ids
        preds = (preds > 0.5).astype(int)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}
 
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Use train dataset for evaluation as placeholder
)
 
# Train the model
trainer.train()
 
# Save the model
model.save_pretrained('./trained_model')
 
# Evaluation on test set
def evaluate_model(trainer, dataset):
    trainer.model.eval()  # Set the model to evaluation mode
    predictions, labels = [], []
    for batch in DataLoader(dataset, batch_size=4):
        inputs = {"pixel_values": batch["pixel_values"].to(device)}
        with torch.no_grad():
            outputs = trainer.model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        predictions.append(logits)
        labels.append(batch["labels"].numpy())
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    predictions = (predictions > 0.5).astype(int)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return acc, f1
 
# Create test dataset and evaluate
test_dataset = CustomImageDataset(subset_df, selected_labels, transform=transform)
accuracy, f1 = evaluate_model(trainer, test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")