import os
from torch.utils.data import DataLoader

from Model import SpeechTextModel
from SpeechTextDataset import SpeechTextDataset
from Train import train_model, plot_training_history, evaluate

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
SAVE_MODEL_DIR = os.path.join(BASE_DIR, "Saved_Models")
PLOT_MODEL_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_MODEL_DIR, exist_ok=True)

# --- PATHS ---
save_model_path = os.path.join(SAVE_MODEL_DIR, "fastapi_trial_model.pth")
train_plot_path = os.path.join(PLOT_MODEL_DIR, "training_history.png")

# --- DATASETS & DATALOADERS ---
train_ds = SpeechTextDataset(os.path.join(DATA_DIR, "Train-small.xlsx"))
val_ds   = SpeechTextDataset(os.path.join(DATA_DIR, "Validation_small.xlsx"))
test_ds  = SpeechTextDataset(os.path.join(DATA_DIR, "Test_small.xlsx"))

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=2, shuffle=False)

# --- MODEL ---
model = SpeechTextModel(
    whisper_dim=768,
    distilbert_dim=768,
    num_classes=2
)

# --- TRAIN ---
history = train_model(
    model,
    num_epochs=2,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    save_path=save_model_path
)

# --- SAVE TRAINING HISTORY PLOT ---
plot_training_history(model, history, save_path=train_plot_path)

# --- FINAL TEST EVALUATION ---
test_metrics = evaluate(model, test_loader)
print("Test Results:", test_metrics)
