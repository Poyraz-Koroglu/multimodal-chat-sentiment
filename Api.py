import os
import io
import torch
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from SpeechTextDataset import SpeechTextDataset
from Model import SpeechTextModel

app = FastAPI(title="Multimodal Sentiment Classifier")

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load trained model ---
MODEL_PATH = os.path.join("Saved_Models", "fastapi_trial_model.pth")
model = SpeechTextModel(whisper_dim=768, distilbert_dim=768, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Dummy dataset just for tokenizer & preprocessing ---
dummy_dataset = SpeechTextDataset(
    excel_path=os.path.join("Data", "Train-small.xlsx"),
    validate_files=False
)

@app.post("/predict")
async def predict(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Predict sentiment from text and/or audio.
    Returns: 0 for neutral, 1 for negative
    """

    if not text and not audio:
        return {"error": "Please provide either text or audio."}

    # --- Process text ---
    input_ids, attention_mask = None, None
    if text:
        encoded = dummy_dataset.tokenizer(
            text,
            max_length=dummy_dataset.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
    else:
        input_ids = None
        attention_mask = None
    # --- Process audio ---
    audio_tensor = None
    if audio:
        audio_bytes = await audio.read()
        audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        audio_tensor = dummy_dataset.normalize_audio_length_dynamic(audio_tensor)
        audio_tensor = audio_tensor.to(device)

    # --- Run inference ---
    with torch.no_grad():
        outputs = model(audio_tensor, input_ids, attention_mask)
        sentiment = torch.argmax(outputs["logits"], dim=1).item()

    return {"prediction": sentiment}
